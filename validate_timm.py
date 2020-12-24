#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import sys
import json
import time
import logging
import torch
import torch.nn.parallel
from apex import amp

sys.path.append("/home/zakirov/repoz/efficientdet-pytorch/")
from effdet import create_model
from data import create_loader, CocoDetection
from timm.utils import AverageMeter, setup_default_logging

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import pytorch_tools as pt
import pytorch_tools.utils.box as box_utils

from src.dali_loader import DaliLoader

has_amp = True

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--anno", default="val2017", help="mscoco annotation set (one of val2017, train2017, test-dev2017)"
)
parser.add_argument(
    "--model",
    "-m",
    metavar="MODEL",
    default="efficientdet_d0",
    help="model architecture (default: tf_efficientdet_d1)",
)
parser.add_argument(
    "--no-redundant-bias",
    action="store_true",
    default=None,
    help="remove redundant bias layers if True, need False for official TF weights",
)
parser.add_argument(
    "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
)
parser.add_argument(
    "-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size (default: 128)"
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--mean", type=float, nargs="+", default=None, metavar="MEAN", help="Override mean pixel value of dataset"
)
parser.add_argument(
    "--std", type=float, nargs="+", default=None, metavar="STD", help="Override std deviation of of dataset"
)
parser.add_argument(
    "--interpolation",
    default="bilinear",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "--fill-color",
    default="mean",
    type=str,
    metavar="NAME",
    help='Image augmentation fill (background) color ("mean" or int)',
)
parser.add_argument(
    "--log-freq", default=10, type=int, metavar="N", help="batch logging frequency (default: 10)"
)
parser.add_argument(
    "--checkpoint", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)"
)
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument("--no-prefetcher", action="store_true", default=False, help="disable fast prefetcher")
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--torchscript", dest="torchscript", action="store_true", help="convert model torchscript for inference"
)
parser.add_argument(
    "--results",
    default="./results.json",
    type=str,
    metavar="FILENAME",
    help="JSON filename for evaluation results",
)


def validate(args):
    setup_default_logging()

    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    if args.no_redundant_bias is None:
        args.redundant_bias = None
    else:
        args.redundant_bias = not args.no_redundant_bias

    # create model
    bench2 = pt.detection_models.__dict__[args.model](
        match_tf_same_padding=True, encoder_norm_act="swish_hard"
    )
    bench2 = bench2.eval().requires_grad_(False).cuda()
    input_size2 = bench2.pretrained_settings["input_size"]

    param_count = sum([m.numel() for m in bench2.parameters()])
    print("Model %s created, param count: %d" % (args.model, param_count))

    if has_amp:
        print("Using AMP mixed precision.")
        bench2 = amp.initialize(bench2, opt_level="O1")
    else:
        print("AMP not installed, running network in FP32.")

    if "test" in args.anno:
        annotation_path = os.path.join(args.data, "annotations", f"image_info_{args.anno}.json")
        image_dir = "test2017"
    else:
        annotation_path = os.path.join(args.data, "annotations", f"instances_{args.anno}.json")
        image_dir = args.anno

    # loader = create_loader(
    #     dataset,
    #     input_size=input_size,
    #     batch_size=args.batch_size,
    #     use_prefetcher=args.prefetcher,
    #     interpolation=args.interpolation,
    #     fill_color=args.fill_color,
    #     num_workers=args.workers,
    #     pin_mem=args.pin_mem,
    # )
    print(f"Input size: {input_size2[0]}")
    loader = DaliLoader(False, args.batch_size, args.workers, input_size2[0])
    img_ids = []
    results = []
    batch_time = AverageMeter()
    end = time.time()
    start_time = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            output2 = bench2.predict(input)

            _, batch_ids, ratios = target
            target = {"img_scale": ratios, "img_id": batch_ids}
            # rescale to image size and clip
            output2[..., :4] *= target["img_scale"].view(-1, 1, 1).to(output2)
            # works even without clipping
            # output2[..., :4] = box_utils.clip_bboxes_batch(output2[..., :4], target["img_size"][..., [1, 0]])
            # xyxy => xywh
            output2[..., 2:4] = output2[..., 2:4] - output2[..., :2]

            output = output2

            output = output.cpu()
            sample_ids = target["img_id"].cpu()
            for index, sample in enumerate(output):
                image_id = int(sample_ids[index])
                for det in sample:
                    score = float(det[4])
                    if score < 0.001:  # stop when below this threshold, scores in descending order
                        break
                    coco_det = dict(
                        image_id=image_id, bbox=det[0:4].tolist(), score=score, category_id=int(det[5])
                    )
                    img_ids.append(image_id)
                    results.append(coco_det)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                print(
                    "Test: [{0:>4d}/{1}]  "
                    "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  ".format(
                        i, len(loader), batch_time=batch_time, rate_avg=input.size(0) / batch_time.avg,
                    )
                )
            # if i > 10:
            # break
    print(f"Full eval took: {time.time() - start_time:.2f}s")
    json.dump(results, open(args.results, "w"), indent=4)
    if "test" not in args.anno:
        coco_api = COCO("data/annotations/instances_val2017.json")
        coco_results = coco_api.loadRes(args.results)
        coco_eval = COCOeval(coco_api, coco_results, "bbox")
        coco_eval.params.imgIds = img_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return results


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
