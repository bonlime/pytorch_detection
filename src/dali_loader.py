# My imports
from nvidia.dali import ops 
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline

from pytorch_tools.utils.misc import env_rank
from pytorch_tools.utils.misc import env_world_size

## original imports

from contextlib import redirect_stdout
from math import ceil
import ctypes
import torch
from pycocotools.coco import COCO

import torch
import ctypes
import logging

import numpy as np

# DALI imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import time


DATA_DIR="data/"

class COCOPipeline(Pipeline):
    "My version of Dali COCO Pipeline"
    def __init__(self, train=False, batch_size=32, workers=4, size=512, max_size=1333):
        # TODO: double check size and max_size
        local_rank, world_size = env_rank(), env_world_size()
        super().__init__(batch_size, workers, local_rank, seed=42)

        split_str = "train" if train else "val"
        self.input = ops.COCOReader(
            file_root = f"{DATA_DIR}/{split_str}2017", 
            annotations_file = f"{DATA_DIR}/annotations/instances_{split_str}2017.json",
            shard_id = local_rank, 
            num_shards = world_size,
            ratio=True, # want bbox in [0, 1]
            ltrb=True, # 
            random_shuffle=train, # only shuffle train images
            # skip_empty=True # skips images without objects. not sure if we want to do so
            # save_img_ids=True, # used in RetinaNet repo for some reasons. don't know why 
        )

        if train:
            self.bbox_crop = ops.RandomBBoxCrop(
                device='cpu', # gpu is not supported (and not needed actually) 
                # ltrb=True, # deprecated (?) 
                bbox_layout="xyXY",
                scaling=[0.3, 1.0],
                # adding 0.0 to thr instead of `allow_no_crop`
                thresholds=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
            )
            self.decode = ops.ImageDecoderSlice(device="mixed", output_type=types.RGB)
            # need attributes to resize bboxes outside DALI
            self.resize = ops.Resize(device='gpu', interp_type=types.INTERP_CUBIC, save_attrs=True)
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            self.resize = ops.Resize(
                device='gpu', interp_type=types.INTERP_CUBIC, resize_longer=max_size, save_attrs=True
            )

        self.bbox_flip = ops.BbFlip(device='cpu', ltrb=True)
        self.img_flip = ops.Flip(device='gpu')

        # color augmentations
        self.bc = ops.BrightnessContrast(device='gpu')
        self.hsv = ops.Hsv(device='gpu')

        # pad to match output stride
        STRIDE = 128
        padded_size = round(max_size + STRIDE // 2) * STRIDE # 1333 => 1408
        self.pad = ops.Paste(device='gpu', fill_value=0, ratio=1.1, min_canvas_size=padded_size, paste_x=0, paste_y=0)
        self.normalize = ops.CropMirrorNormalize(
            device='gpu',
            # Imagenet mean and std
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            output_dtype=dali.types.FLOAT,
            output_layout=dali.types.NCHW,
            crop=(padded_size, padded_size),
            crop_pos_x=0, crop_pos_y=0
        )
        
        # TODO: add Jitter aug
        # TODO: add RGB => BGR and RGB => Gray aug
        
        # Random number generation for augmentation
        self.coin_flip = ops.CoinFlip(probability=0.5)
        self.rng1 = ops.Uniform(range=[0, 1])
        self.rng2 = ops.Uniform(range=[0.85, 1.15])
        self.rng3 = ops.Uniform(range=[-15, 15])
        self.rand_resize = ops.Uniform(range=[size, max_size])
        self.train = train


    def define_graph(self):
        images, bboxes, labels = self.input()

        if self.train:
            # crop bbox first and then decode only part of the image
            crop_begin, crop_size, bboxes, labels = self.bbox_crop(bboxes, labels)
            images = self.decode(images, crop_begin, crop_size)

            # resize to some random size
            resize = self.rand_resize()
            images, attrs = self.resize(images, resize_longer=resize)

            # maybe flip
            flip = self.coin_flip()
            bboxes = self.bbox_flip(bboxes, horizontal=flip)
            images = self.img_flip(images, horizontal=flip)

            # color augs
            images = self.bc(images, brightness=self.rng2(), contrast=self.rng2())
            images = self.hsv(images, hue=self.rng3(), saturation=self.rng2(), value=self.rng2())
        else:
            images = self.decode(images)
            images, attrs = self.resize(images)

        resized_images = images
        images = self.normalize(self.pad(images))

        return images, bboxes, labels, attrs, resized_images


class DaliLoader():
    """Wrapper around Dali to process raw batches """
    'Data loader for data parallel using Dali'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False,
                 rotate_augment=False, augment_brightness=0.0,
                 augment_contrast=0.0, augment_hue=0.0, augment_saturation=0.0):
        self.training = training
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.batch_size = batch_size // world

        self.world = world
        self.path = path

        # Setup COCO
        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = {k: i for i, k in enumerate(self.coco.getCatIds())}

        self.pipe = COCOPipeline(batch_size=self.batch_size, num_threads=2,
                                 path=path, training=training, annotations=annotations, world=world,
                                 device_id=torch.cuda.current_device(), mean=self.mean, std=self.std, resize=resize,
                                 max_size=max_size, stride=self.stride, rotate_augment=rotate_augment,
                                 augment_brightness=augment_brightness,
                                 augment_contrast=augment_contrast, augment_hue=augment_hue,
                                 augment_saturation=augment_saturation)

        self.pipe.build()

    def __repr__(self):
        return '\n'.join([
            '    loader: dali',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return ceil(len(self.ids) // self.world / self.batch_size)

    def __iter__(self):
        for _ in range(self.__len__()):

            data, ratios, ids, num_detections = [], [], [], []
            dali_data, dali_boxes, dali_labels, dali_ids, dali_attrs, dali_resize_img = self.pipe.run()

            for l in range(len(dali_boxes)):
                num_detections.append(dali_boxes.at(l).shape[0])

            pyt_targets = -1 * torch.ones([len(dali_boxes), max(max(num_detections), 1), 5])

            for batch in range(self.batch_size):
                id = int(dali_ids.at(batch)[0])

                # Convert dali tensor to pytorch
                dali_tensor = dali_data.at(batch)
                tensor_shape = dali_tensor.shape()

                datum = torch.zeros(dali_tensor.shape(), dtype=torch.float, device=torch.device('cuda'))
                c_type_pointer = ctypes.c_void_p(datum.data_ptr())
                dali_tensor.copy_to_external(c_type_pointer)

                # Calculate image resize ratio to rescale boxes
                prior_size = dali_attrs.as_cpu().at(batch)
                resized_size = dali_resize_img.at(batch).shape()
                ratio = max(resized_size) / max(prior_size)

                if self.training:
                    # Rescale boxes
                    b_arr = dali_boxes.at(batch)
                    num_dets = b_arr.shape[0]
                    if num_dets is not 0:
                        pyt_bbox = torch.from_numpy(b_arr).float()

                        pyt_bbox[:, 0] *= float(prior_size[1])
                        pyt_bbox[:, 1] *= float(prior_size[0])
                        pyt_bbox[:, 2] *= float(prior_size[1])
                        pyt_bbox[:, 3] *= float(prior_size[0])
                        # (l,t,r,b) ->  (x,y,w,h) == (l,r, r-l, b-t)
                        pyt_bbox[:, 2] -= pyt_bbox[:, 0]
                        pyt_bbox[:, 3] -= pyt_bbox[:, 1]
                        pyt_targets[batch, :num_dets, :4] = pyt_bbox * ratio

                    # Arrange labels in target tensor
                    l_arr = dali_labels.at(batch)
                    if num_dets is not 0:
                        pyt_label = torch.from_numpy(l_arr).float()
                        pyt_label -= 1  # Rescale labels to [0,79] instead of [1,80]
                        pyt_targets[batch, :num_dets, 4] = pyt_label.squeeze()

                ids.append(id)
                data.append(datum.unsqueeze(0))
                ratios.append(ratio)

            data = torch.cat(data, dim=0)

            if self.training:
                pyt_targets = pyt_targets.cuda(non_blocking=True)

                yield data, pyt_targets

            else:
                ids = torch.Tensor(ids).int().cuda(non_blocking=True)
                ratios = torch.Tensor(ratios).cuda(non_blocking=True)

                yield data, ids, ratios