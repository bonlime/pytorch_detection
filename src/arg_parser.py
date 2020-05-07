from loguru import logger
import pytorch_tools as pt
import configargparse as argparse

def make_parser():
    parser = argparse.ArgumentParser(
        description="COCO Object Detection Training",
        default_config_files=["configs/base.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )
    add_arg = parser.add_argument

    ## MODEL
    add_arg("--backbone", default="resnet18", help="model architecture: (default: resnet18)")
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")


    ## OPTIMIZER
    add_arg("--optim", type=str, default="SGD", help="Optimizer to use (default: sgd)")
    add_arg("--optim_params", type=eval, default={}, help="Additional optimizer params as kwargs")
    add_arg("--lookahead", action="store_true", help="Flag to wrap optimizer with Lookahead wrapper")
    add_arg(
        "--weight_decay", "--wd", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)",
    )


    ## DATALOADER
    add_arg("-j", "--workers", default=4, type=int, help="number of data loading workers (default: 4)")
    add_arg(
        "--mixup", type=float, default=0, help="Alpha for mixup augmentation. If 0 then mixup is diabled",
    )
    add_arg(
        "--cutmix", type=float, default=0, help="Alpha for cutmix augmentation. If 0 then cutmix is diabled",
    )
    add_arg("--cutmix_prob", type=float, default=0.5)
    add_arg("--ctwist", action="store_true", help="Turns on color twist augmentation")
    add_arg("--batch_size", "-bs", type=int, default=8, help="Batch size for training")
    add_arg("--size", type=int, default=512, help="Input size for training")



    ## CRITERION 
    add_arg("--focal_alpha", type=float, default=0.25, help="Alpha for focal loss")
    add_arg("--focal_gamma", type=float, default=2, help="Gamma for focal loss")
    add_arg("--huber_delta", type=float, default=0.1, help="Delta for SmoothL1loss")
    add_arg("--box_weight", )


    ## TRAINING
    add_arg("--short_epoch", action="store_true", help="make epochs short (for debugging)")
    add_arg(
        "--phases",
        type=eval,
        action="append",
        help="Specify epoch order of data resize and learning rate schedule:"
        '[{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]',
    )
    add_arg("--resume", default="", type=str, help="path to latest checkpoint (default: none)")
    add_arg("--evaluate", "-e", action="store_true", help="Only evaluate model on validation set")
    add_arg(
        "--opt_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2", "O3"],
        help='optimizatin level for apex. (default: "00")',
    )

    ## OTHER
    add_arg(
        "--local_rank",
        "--gpu",
        default=0,
        type=int,
        help="Used for multi-process training. Can either be manually set or automatically set by using 'python -m multiproc'.",
    )

    ## LOGGING 
    add_arg("--logdir", default="logs", type=str, help="where logs go")
    add_arg(
        "-n",
        "--name",
        type=str,
        default="",
        dest="name",
        help="Name of this run. If empty it would be a timestamp",
    )

    args, not_parsed = parser.parse_known_args()
    logger.info(f"Not parsed args: {not_parsed}")

    # detect distributed
    args.world_size = pt.utils.misc.env_world_size()
    args.distributed = args.world_size > 1

    # Only want master rank logging to tensorboard
    args.is_master = not args.distributed or args.local_rank == 0
    timestamp = pt.utils.misc.get_timestamp()
    args.name = args.name + "_" + timestamp if args.name else timestamp
    return args