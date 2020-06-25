import json
import torch
import numpy as np

import pytorch_tools as pt
from pytorch_tools.utils.box import decode
from pytorch_tools.utils.box import clip_bboxes_batch
from pytorch_tools.utils.misc import to_numpy
from pycocotools.cocoeval import COCOeval


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
