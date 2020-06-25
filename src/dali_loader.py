"""
Here you can find definition of DALI Loader. It's very fast but writing augmentations as a graph is not very
convenient. Would replace with kornia / albumentation in the future
"""
import math
import torch

from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import feed_ndarray

from pytorch_tools.utils.misc import env_rank
from pytorch_tools.utils.misc import env_world_size

from src.coco_ids import COCO_80_TO_90_ARR

DATA_DIR = "data/"

# inspired by
# https://github.com/NVIDIA/retinanet-examples/blob/master/retinanet/dali.py
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/coco_pipeline.py
class COCOPipeline(Pipeline):
    """COCO Pipeline
    It implements the following augmentation strategy:
    Given: output_size (OS)
    
    For training:
        1. crop random part of the image with area in [0.3, 1]
    For validation:
        1. just decode
    2. scale longest image size to OS
    3. pad image to OS (in case H or W is less)
    """

    def __init__(self, train=False, batch_size=16, workers=4, size=512):
        # TODO: support size as tuple
        local_rank, world_size = env_rank(), env_world_size()
        super().__init__(batch_size, workers, local_rank, seed=42)

        split_str = "train" if train else "val"
        self.input = ops.COCOReader(
            file_root=f"{DATA_DIR}/{split_str}2017",
            annotations_file=f"{DATA_DIR}/annotations/instances_{split_str}2017.json",
            shard_id=local_rank,
            num_shards=world_size,
            ratio=True,  # want bbox in [0, 1]
            ltrb=True,  #
            random_shuffle=train,
            save_img_ids=True,  # Need ids for evaluation
            # skip_empty=True # skips images without objects. not sure if we want to do so
        )

        self.bbox_crop = ops.RandomBBoxCrop(
            device="cpu",  # gpu is not supported (and not needed actually)
            bbox_layout="xyXY",  # same as 'ltrb'
            scaling=[0.3, 1.0],
            # adding 0.0 to thr instead of `allow_no_crop`
            thresholds=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        )
        if train:
            self.decode = ops.ImageDecoderSlice(device="mixed", output_type=types.RGB)
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.resize = ops.Resize(
            device="gpu", interp_type=types.INTERP_CUBIC, resize_longer=size, save_attrs=True
        )

        self.bbox_flip = ops.BbFlip(device="cpu", ltrb=True)
        self.img_flip = ops.Flip(device="gpu")

        # color augmentations
        self.bc = ops.BrightnessContrast(device="gpu")
        self.hsv = ops.Hsv(device="gpu")

        # pad to match output stride
        self.pad = ops.Pad(device="gpu", fill_value=0, axes=(1, 2), shape=(size, size))
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            # Imagenet mean and std
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            # mean=[0, 0, 0],
            # std=[1, 1, 1],
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
        )

        # TODO: add Jitter aug

        # Random number generation for augmentation
        self.coin_flip = ops.CoinFlip(probability=0.5)
        self.rng1 = ops.Uniform(range=[0, 1])
        self.rng2 = ops.Uniform(range=[0.85, 1.15])
        self.rng3 = ops.Uniform(range=[-15, 15])
        self.train = train

    def define_graph(self):
        images, bboxes, labels, img_ids = self.input()

        if self.train:
            # crop bbox first and then decode only part of the image
            crop_begin, crop_size, bboxes, labels = self.bbox_crop(bboxes, labels)
            images = self.decode(images, crop_begin, crop_size)

            # maybe flip
            flip = self.coin_flip()
            bboxes = self.bbox_flip(bboxes, horizontal=flip)
            images = self.img_flip(images, horizontal=flip)

            # color augs
            images = self.bc(images, brightness=self.rng2(), contrast=self.rng2())
            images = self.hsv(images, hue=self.rng3(), saturation=self.rng2(), value=self.rng2())

        else:
            images = self.decode(images)

        # resize longest size to size
        images, attrs = self.resize(images)

        # need size before pad to un normalize bboxes lagter
        before_pad = images
        images = self.normalize(images)
        # pad to have the same shape
        images = self.pad(images)

        # labels are in ltrb
        return images, bboxes, labels, before_pad, img_ids, attrs


class DaliLoader:
    """Wrapper around Dali to process raw batches """

    "Data loader for data parallel using Dali"

    def __init__(self, train, batch_size, workers, size):
        self.train = train
        self.batch_size = batch_size

        self.pipe = COCOPipeline(train, batch_size, workers, size)
        self.pipe.build()

    def __len__(self):
        return math.ceil(next(iter(self.pipe.epoch_size().values())) / self.batch_size)

    def __iter__(self):
        for _ in range(self.__len__()):

            data, ratios = [], []
            dali_data, dali_boxes, dali_labels, dali_before_pad, dali_img_ids, dali_attr = self.pipe.run()

            # convert images from dali tensors to pytorch
            data = feed_ndarray(
                dali_data.as_tensor(),
                torch.zeros(dali_data.as_tensor().shape(), dtype=torch.float, device=torch.device("cuda")),
            )

            max_detections = max(*(dali_boxes[i].shape()[0] for i in range(len(dali_boxes))), 1)
            pyt_targets = -1 * torch.ones([len(dali_boxes), max_detections, 5])

            # get image ids. only needed for evaluation
            img_ids = torch.tensor(dali_img_ids.as_array())

            prior_size = dali_attr.as_cpu().as_array()

            # target has different size for each image so need to treat them separately
            for batch in range(self.batch_size):

                # Calculate image resize ratio to rescale boxes
                resized_size = dali_before_pad[batch].shape()[:2]

                # in this formulation to get true bbox you need to **multiply** prediction by ratio
                ratios.append(max(prior_size[batch]) / max(resized_size))

                # Rescale boxes
                pyt_bbox = feed_ndarray(dali_boxes[batch], torch.zeros(dali_boxes[batch].shape()))
                num_dets = pyt_bbox.size(0)
                if num_dets > 0:
                    pyt_bbox[:, 0::2] *= float(resized_size[1])
                    pyt_bbox[:, 1::2] *= float(resized_size[0])
                    pyt_targets[batch, :num_dets, :4] = pyt_bbox

                # Arrange labels in target tensor
                np_label = feed_ndarray(
                    dali_labels[batch], torch.empty(dali_labels[batch].shape(), dtype=torch.int32)
                )
                # DALI CocoReader maps existing 90 classes to 80 unique classes. Need to map to 90 again
                # this is done by indexing numpy array of new (90) labels with old (80) labels
                pyt_label = torch.tensor(COCO_80_TO_90_ARR[np_label.squeeze().numpy()])
                if num_dets > 0:
                    pyt_label -= 1  # [0, 90] => [-1, 89]. Removes background
                    pyt_targets[batch, :num_dets, 4] = pyt_label

            pyt_targets = pyt_targets.cuda(non_blocking=True)
            ratios = torch.tensor(ratios)

            if self.train:
                yield data, pyt_targets
            else:
                yield data, (pyt_targets, img_ids, ratios)
