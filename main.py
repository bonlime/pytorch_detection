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

import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed

from src.model import SSD300, ResNet, Loss
from src.utils import dboxes300_coco, Encoder
from src.logger import Logger, BenchLogger
from src.evaluate import evaluate
from src.train import train_loop, tencent_trick, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
from src.data import get_train_loader, get_val_dataset, get_val_dataloader, get_coco_ground_truth

# Apex imports
try:
    from apex.parallel.LARC import LARC
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")



def train(train_loop_func, logger, args):
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda

    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1

    if args.seed is None:
        args.seed = np.random.randint(1e4)

    if args.distributed:
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)


    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    train_loader = get_train_loader(args, args.seed - 2**31)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path))
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    if use_cuda:
        ssd300.cuda()
        loss_func.cuda()

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    if args.amp:
        ssd300, optimizer = amp.initialize(ssd300, optimizer, opt_level='O2')

    if args.distributed:
        ssd300 = DDP(ssd300)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.distributed else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    total_time = 0

    if args.mode == 'evaluation':
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))

        return
    mean, std = generate_mean_std(args)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        scheduler.step()
        iteration = train_loop_func(ssd300, loss_func, epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                    logger, args, mean, std)
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            torch.save(obj, './models/epoch_{}.pt'.format(epoch))
        train_loader.reset()
    print('total training time: {}'.format(total_time))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    torch.backends.cudnn.benchmark = True

    if args.mode == 'benchmark-training':
        train_loop_func = benchmark_train_loop
        logger = BenchLogger('Training benchmark')
        args.epochs = 1
    elif args.mode == 'benchmark-inference':
        train_loop_func = benchmark_inference_loop
        logger = BenchLogger('Inference benchmark')
        args.epochs = 1
    else:
        train_loop_func = train_loop
        logger = Logger('Training logger', print_freq=1)

    train(train_loop_func, logger, args)
