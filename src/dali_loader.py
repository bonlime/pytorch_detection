import torch
import ctypes

from nvidia.dali import ops 
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline

from pytorch_tools.utils.misc import env_rank
from pytorch_tools.utils.misc import env_world_size

DATA_DIR="data/"

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
            file_root = f"{DATA_DIR}/{split_str}2017", 
            annotations_file = f"{DATA_DIR}/annotations/instances_{split_str}2017.json",
            shard_id = local_rank, 
            num_shards = world_size,
            ratio=True, # want bbox in [0, 1]
            ltrb=True, # 
            random_shuffle=train,
            # skip_empty=True # skips images without objects. not sure if we want to do so
            # save_img_ids=True, # used in RetinaNet repo for some reasons. don't know why 
        )
        
        self.bbox_crop = ops.RandomBBoxCrop(
            device='cpu', # gpu is not supported (and not needed actually) 
            bbox_layout="xyXY", # same as 'ltrb'
            scaling=[0.3, 1.0],
            # adding 0.0 to thr instead of `allow_no_crop`
            thresholds=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        )
        if train:
            self.decode = ops.ImageDecoderSlice(device="mixed", output_type=types.RGB)
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.resize = ops.Resize(device='gpu', interp_type=types.INTERP_CUBIC, resize_longer=size)
        
        self.bbox_flip = ops.BbFlip(device='cpu', ltrb=True)
        self.img_flip = ops.Flip(device='gpu')

        # color augmentations
        self.bc = ops.BrightnessContrast(device='gpu')
        self.hsv = ops.Hsv(device='gpu')

        # pad to match output stride
        self.pad = ops.Pad(device="gpu", fill_value=0, axes=(1, 2), shape=(size, size)) 
        self.normalize = ops.CropMirrorNormalize(
            device='gpu',
            # Imagenet mean and std
#             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#             std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mean=[0, 0, 0],
            std=[1, 1, 1],
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
        images, bboxes, labels = self.input()

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
        images = self.resize(images)
        
        # need size before pad to un normalize bboxes lagter
        before_pad = images
        images = self.normalize(images)
        # pad to have the same shape
        images = self.pad(images)
        
        # labels are in ltrb
        return images, bboxes, labels, before_pad

class DaliLoader():
    """Wrapper around Dali to process raw batches """
    'Data loader for data parallel using Dali'

    def __init__(self, train, batch_size, workers, size):
        self.train = train
        self.batch_size = batch_size
        
        self.pipe = COCOPipeline(train, batch_size, workers, size)
        self.pipe.build()

    # TODO: somehow get proper len of loader
    def __len__(self):
        return 100
    #     return ceil(len(self.ids) // self.world / self.batch_size)

    def __iter__(self):
        for _ in range(self.__len__()):
            
            data, num_detections = [], []
            dali_data, dali_boxes, dali_labels, dali_before_pad = self.pipe.run()

            for l in range(len(dali_boxes)):
                num_detections.append(dali_boxes.at(l).shape[0])

            pyt_targets = -1 * torch.ones([len(dali_boxes), max(*num_detections, 1), 5])

            for batch in range(self.batch_size):

                # Convert dali tensor to pytorch
                dali_tensor = dali_data.at(batch)

                datum = torch.zeros(dali_tensor.shape(), dtype=torch.float, device=torch.device('cuda'))
                c_type_pointer = ctypes.c_void_p(datum.data_ptr())
                dali_tensor.copy_to_external(c_type_pointer)

                # Calculate image resize ratio to rescale boxes
                resized_size = dali_before_pad.at(batch).shape()[:2]

                # Rescale boxes
                b_arr = dali_boxes.at(batch)
                num_dets = len(b_arr)
                if num_dets > 0 :
                    pyt_bbox = torch.from_numpy(b_arr)
                    pyt_bbox[:, 0::2] *= float(resized_size[1])
                    pyt_bbox[:, 1::2] *= float(resized_size[0])
                    pyt_targets[batch, :num_dets, :4] = pyt_bbox

                # Arrange labels in target tensor
                l_arr = dali_labels.at(batch)
                if num_dets > 0:
                    pyt_label = torch.from_numpy(l_arr).float()
                    pyt_label -= 1  # Rescale labels to [0,79] instead of [1,80]
                    pyt_targets[batch, :num_dets, 4] = pyt_label.squeeze()

                data.append(datum.unsqueeze(0))

            data = torch.cat(data, dim=0)
            pyt_targets = pyt_targets.cuda(non_blocking=True)

            yield data, pyt_targets