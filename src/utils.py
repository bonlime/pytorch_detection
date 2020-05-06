# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import random
import itertools
import torch
import torch.nn.functional as F
import json
import time
import bz2
import pickle
from math import sqrt

# TODO: make this function usable
def draw_patches(img, bboxes, labels, order="xywh", label_map={}):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # Suppose bboxes in fractional coordinate:
    # cx, cy, w, h
    # img = img.numpy()
    img = np.array(img)
    labels = np.array(labels)
    bboxes = bboxes.numpy()

    if label_map:
        labels = [label_map.get(l) for l in labels]

    if order == "ltrb":
        xmin, ymin, xmax, ymax = bboxes[:, 0],  bboxes[:, 1],  bboxes[:, 2],  bboxes[:, 3]
        cx, cy, w, h = (xmin + xmax)/2, (ymin + ymax)/2, xmax - xmin, ymax - ymin
    else:
        cx, cy, w, h = bboxes[:, 0],  bboxes[:, 1],  bboxes[:, 2],  bboxes[:, 3]

    htot, wtot,_ = img.shape
    cx *= wtot
    cy *= htot
    w *= wtot
    h *= htot

    bboxes = zip(cx, cy, w, h)

    plt.imshow(img)
    ax = plt.gca()
    for (cx, cy, w, h), label in zip(bboxes, labels):
        if label == "background": continue
        ax.add_patch(patches.Rectangle((cx-0.5*w, cy-0.5*h),
                                        w, h, fill=False, color="r"))
        bbox_props = dict(boxstyle="round", fc="y", ec="0.5", alpha=0.3)
        ax.text(cx-0.5*w, cy-0.5*h, label, ha="center", va="center", size=15, bbox=bbox_props)
    plt.show()

