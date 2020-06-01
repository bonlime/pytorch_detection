import os
import sys
import yaml
import copy
import math
import json
import time
import shutil
import warnings
import subprocess
from pathlib import Path
from loguru import logger
from datetime import datetime
import configargparse as argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist

# for fp16
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import pytorch_tools as pt
import pytorch_tools.detection_models as det_models
import pytorch_tools.fit_wrapper.callbacks as pt_clb
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClbk

from pytorch_tools.utils.misc import listify
from pytorch_tools.optim import optimizer_from_name

from src.dali_dataloader import DaliLoader
from src.arg_parser import parse_args

def main():

    ## get config for this run
    FLAGS = parse_args()
    os.makedirs(FLAGS.outdir, exist_ok=True)
    config = {
        "handlers": [
            {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
            {"sink": f"{FLAGS.outdir}/logs.txt", "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
        ],
    }
    if FLAGS.is_master:
        logger.configure(**config)
        ## dump config and diff for reproducibility
        yaml.dump(vars(FLAGS), open(FLAGS.outdir + '/config.yaml', 'w'))
        kwargs = {"universal_newlines": True, "stdout": subprocess.PIPE}
        with open(FLAGS.outdir + '/commit_hash.txt', 'w') as fp:
            fp.write(subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs).stdout)
        with open(FLAGS.outdir + '/diff.txt', 'w') as fp:
            fp.write(subprocess.run(["git", "diff"], **kwargs).stdout)
    else:
        logger.configure(handlers=[])
    logger.info(FLAGS)

    
    ## makes it slightly faster
    cudnn.benchmark = True
    if FLAGS.deterministic:
        pt.utils.misc.set_random_seed(42) # fix all seeds

    ## setup distributed
    if FLAGS.distributed:
        logger.info("Distributed initializing process group")
        torch.cuda.set_device(FLAGS.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=FLAGS.world_size)

    ## get dataloaders
    train_loader = DaliLoader(True, FLAGS.batch_size, FLAGS.workers, FLAGS.size)
    val_loader = DaliLoader(False, FLAGS.batch_size, FLAGS.workers, FLAGS.size)
    
    ## get model
    logger.info(f"=> Creating model '{FLAGS.arch}'")
    model = det_models.__dict__[FLAGS.arch](**FLAGS.model_params)
    if FLAGS.weight_standardization:
        model = pt.modules.weight_standartization.conv_to_ws_conv(model)
    model = model.cuda()
    
    ## get optimizer
    # want to filter BN from weight decay by default. It never hurts
    optim_params = pt.utils.misc.filter_bn_from_wd(model)
    # start with 0 lr. Scheduler will change this later
    optimizer = optimizer_from_name(FLAGS.optim)(
        optim_params, lr=0, weight_decay=FLAGS.weight_decay, **FLAGS.optim_params
    )
    if FLAGS.lookahead:
        optimizer = pt.optim.Lookahead(optimizer, la_alpha=0.5)

    ## load weights from previous run if given
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume, map_location=lambda s, loc: s.cuda()) # map for multi-gpu
        model.load_state_dict(checkpoint["state_dict"]) # strict=False
        FLAGS.start_epoch = checkpoint["epoch"]
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:  # may raise an error if another optimzer was used or no optimizer in state dict
            logger.info("Failed to load state dict into optimizer")

    ## init mixed precision
    model, optimizer = amp.initialize(model, optimizer, opt_level=FLAGS.opt_level, verbosity=0)

    # Important to create EMA Callback after cuda() and AMP but before DDP wrapper
    ema_clb = pt_clb.ModelEma(model, FLAGS.ema_decay) if FLAGS.ema_decay else NoClbk()
    if FLAGS.distributed:
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion)
    criterion = None # TODO: add later

    model_saver = pt_clb.CheckpointSaver(FLAGS.outdir, save_name="model.chpn") if FLAGS.is_master else NoClbk()
    sheduler = pt.fit_wrapper.callbacks.PhasesScheduler(FLAGS.phases)
    # common callbacks
    callbacks = [
        sheduler,
        pt_clb.FileLogger(FLAGS.outdir, logger=logger), # need in every process to be able to sync in DDP mode 
        pt_clb.Mixup(FLAGS.mixup, 1000) if FLAGS.mixup else NoClbk(),
        pt_clb.Cutmix(FLAGS.cutmix, 1000) if FLAGS.cutmix else NoClbk(),
        model_saver, # need to have CheckpointSaver before EMA so moving it here
        ema_clb, # ModelEMA MUST go after checkpoint saver to work, otherwise it would save main model instead of EMA
    ]
    if FLAGS.is_master:  # callback for master process
        master_callbacks = [
                pt_clb.Timer(),
                pt_clb.ConsoleLogger(),
                pt_clb.TensorBoard(FLAGS.outdir, log_every=25),
        ]
        callbacks.extend(master_callbacks)

    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion,
        # metrics=[pt.metrics.Accuracy(), pt.metrics.Accuracy(5)],
        callbacks=callbacks,
    )
    if FLAGS.evaluate:
        return None, (42, 42)
        return runner.evaluate(val_loader)

    runner.fit(
        train_loader,
        steps_per_epoch=(None, 10)[FLAGS.short_epoch],
        val_loader=val_loader,
        val_steps=(None, 20)[FLAGS.short_epoch],
        epochs=sheduler.tot_epochs,
        # start_epoch=FLAGS.start_epoch, # TODO: maybe want to continue from epoch  
    )

    # TODO: maybe return best loss?
    return runner.state.val_loss.avg, [m.avg for m in runner.state.val_metrics]



if __name__ == "__main__":
    start_time = time.time()  # Loading start to after everything is loaded
    _, res = main()
    acc1, acc5 = res[0], res[1]
    if FLAGS.is_master:
        logger.info(f"Acc@1 {acc1:.3f} Acc@5 {acc5:.3f}")
        m = (time.time() - start_time) / 60
        logger.info(f"Total time: {int(m / 60)}h {m % 60:.1f}m")