import json
import torch
import numpy as np

import pytorch_tools as pt
from pytorch_tools.utils.box import decode
from pytorch_tools.utils.box import clip_bboxes_batch
import pytorch_tools.utils.misc as utils

# huge extension over default TB logger to support logging box / class losses separately and log mAP/mAR
# inherit from existing TB logger to avoid rewriting extra code
# I don't know a cleaner way to do this
class CocoEvalClbTB(pt.fit_wrapper.callbacks.TensorBoard):
    def __init__(self, log_dir, coco_api, anchors, log_every=20, **decode_params):
        super().__init__(log_dir, log_every=log_every)
        self.coco_api = coco_api
        self.anchors = anchors
        self.decode_params = decode_params

        self.cls_loss_meter = utils.AverageMeter("cls_loss")
        self.box_loss_meter = utils.AverageMeter("box_loss")
        self.train_cls_loss = None
        self.train_box_loss = None
        self.val_cls_loss = None
        self.val_box_loss = None

        self.all_img_ids_set = set()
        self.all_results = []
        self.batch_img_ids = None
        self.batch_ratios = None

    def on_batch_begin(self):
        if self.state.is_train:
            return

        # remove image id and ratio from target
        data, (pyt_targets, img_ids, ratios) = self.state.input
        self.state.input = data, pyt_targets

        # save img_ids for later
        self.all_img_ids_set.update(img_ids.flatten().tolist())
        self.batch_ratios = ratios
        self.batch_img_ids = img_ids

    def on_batch_end(self):
        super().on_batch_end()
        # log cls & box loss. I'm using a very dirty way to do it by accessing
        # protected attributes of loss class
        self.cls_loss_meter.update(utils.to_numpy(self.state.criterion._cls_loss))
        self.box_loss_meter.update(utils.to_numpy(self.state.criterion._box_loss))

        # skip creating coco res for train
        if self.state.is_train:
            return

        cls_out, box_out = self.state.output
        res = decode(cls_out, box_out, self.anchors, **self.decode_params)

        # rescale to image size. don't really need to clip after that
        res[..., :4] *= self.batch_ratios.view(-1, 1, 1).to(res)
        # xyxy -> xywh
        res[..., 2:4] = res[..., 2:4] - res[..., :2]

        for batch in range(res.size(0)):
            for one_res in res[batch]:
                coco_result = dict(
                    image_id=self.batch_img_ids[batch, 0].tolist(),
                    bbox=one_res[:4].tolist(),
                    score=one_res[4].tolist(),
                    category_id=int(one_res[5].tolist()),
                )
                self.all_results.append(coco_result)

    def on_loader_end(self):
        # reduce between gpus
        for meter in (self.cls_loss_meter, self.box_loss_meter):
            meter = utils.reduce_meter(meter)

        if self.state.is_train:
            self.train_cls_loss = deepcopy(self.cls_loss_meter)
            self.train_box_loss = deepcopy(self.box_loss_meter)
            # reset to also capture val loss
            self.cls_loss_meter.reset()
            self.box_loss_meter.reset()
            return
        self.val_cls_loss = deepcopy(self.cls_loss_meter)
        self.val_box_loss = deepcopy(self.box_loss_meter)

        # to avoid errors with pycocotools we first dump result for every process in separate file
        # then gather them, dump collected file to disc and finally use it for evaluation
        # to avoid doing expensive evaluation in every process i do it only in master one
        json.dump(self.all_results, open(f"/tmp/temp_{utils.env_rank()}.json", "w"), indent=4)
        json.dump(list(self.all_img_ids_set), open(f"/tmp/temp_ids_{utils.env_rank()}.json", "w"), indent=4)
        if utils.env_rank() != 0:
            return
        res, img_ids = [], []
        for i in range(utils.env_world_size()):
            res.extend(json.load(open(f"/tmp/temp_{i}.json")))
            img_ids.extend(json.load(open(f"/tmp/temp_ids_{i}.json")))

        json.dump(res, open(f"/tmp/temp.json", "w"), indent=4)
        # print("LENS:", len(self.all_results), len(res))
        # print("LENS:", len(self.all_img_ids_set), len(img_ids))

        results = self.coco_api.loadRes("/tmp/temp.json")
        coco_eval = COCOeval(self.coco_api, results, "bbox")
        coco_eval.params.imgIds = list(set(img_ids))  # score only ids we've predicted
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # log metric to TB
        metric_names = [
            "mAP@0.5:0.95",
            "mAP@0.5",
            "mAP@0.75",
            "mAP@0.5:0.95_small",
            "mAP@0.5:0.95_medium",
            "mAP@0.5:0.95_large",
            "mAR@0.5:0.95",
            "mAR@0.5",
            "mAR@0.75",
            "mAR@0.5:0.95_small",
            "mAR@0.5:0.95_medium",
            "mAR@0.5:0.95_large",
        ]
        for metric, m_name in zip(coco_eval.stats, metric_names):
            self.writer.add_scalar(f"val_coco/{m_name}", metric, self.current_step)

    def on_epoch_end(self):
        super().on_epoch_end()
        for m in (self.train_box_loss, self.train_cls_loss):
            self.writer.add_scalar(f"train/{m.name}", m.avg, self.current_step)

        for m in (self.val_box_loss, self.val_cls_loss):
            self.writer.add_scalar(f"val/{m.name}", m.avg, self.current_step)

    def reset(self):
        self.all_img_ids_set = set()
        self.all_results = []
        self.batch_img_ids = None
        self.batch_ratios = None
        self.cls_loss_meter.reset()
        self.box_loss_meter.reset()


class CocoEvalClb(pt.fit_wrapper.callbacks.Callback):
    """
    Accumulates validation results and computes mAP & mAR for them
    This Callback also consumes img id and ratio from input, so it should go before any other
    callback which messes with input data
    """

    def __init__(self, coco_api, anchors, **decode_params):
        super().__init__()
        self.coco_api = coco_api
        self.anchors = anchors
        self.decode_params = decode_params
        self.all_img_ids_set = set()
        self.all_img_ids = []
        self.all_results = []
        self.batch_img_ids = None
        self.batch_ratios = None

    def on_batch_begin(self):
        if self.state.is_train:
            return

        data, (pyt_targets, img_ids, ratios) = self.state.input

        self.state.input = data, pyt_targets
        # save img_ids for later
        self.all_img_ids_set.update(img_ids.flatten().tolist())
        self.batch_ratios = ratios
        self.batch_img_ids = img_ids

    def on_batch_end(self):
        if self.state.is_train:
            return

        cls_out, box_out = self.state.output
        res = decode(cls_out, box_out, self.anchors)  #  TODO: return **self.decode_params

        # rescale to image size. don't really need to clip after that
        res[..., :4] *= self.batch_ratios.view(-1, 1, 1).to(res)
        # xyxy -> xywh
        res[..., 2:4] = res[..., 2:4] - res[..., :2]

        for batch in range(res.size(0)):
            for one_res in res[batch]:
                self.all_img_ids.append(self.batch_img_ids[batch, 0].tolist())
                coco_result = dict(
                    image_id=self.batch_img_ids[batch, 0].tolist(),
                    bbox=one_res[:4].tolist(),
                    score=one_res[4].tolist(),
                    category_id=int(one_res[5].tolist()),
                )
                self.all_results.append(coco_result)

    def on_loader_end(self):
        if self.state.is_train:
            return

        # dump and load to avoid error with pycocotools. it's eaiser than installing it from master
        json.dump(self.all_results, open("/tmp/temp.json", "w"), indent=4)
        results = self.coco_api.loadRes("/tmp/temp.json")

        coco_eval = COCOeval(self.coco_api, results, "bbox")
        # score only ids we've predicted
        coco_eval.params.imgIds = list(self.all_img_ids_set)
        coco_eval.params.imgIds = self.all_img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # print(coco_eval.stats)
        metric = coco_eval.stats[0]  # mAP 0.5-0.95
        # TODO: reset
        pass
