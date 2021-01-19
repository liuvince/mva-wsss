import numpy as np
import torch
import random
import argparse
import os

import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

# Most part of this code was taken from https://github.com/alipay/cvpr2020-plant-pathology/blob/master/utils.py

def mkdir(path):
    """Create directory.
     Create directory if it is not exist, else do nothing.
     Parameters
     ----------
     path: str
        Path of your directory.
     Examples
     --------
     mkdir("data/raw/train/")
     """
    try:
        if path is None:
            pass
        else:
            os.stat(path)
    except Exception:
        os.makedirs(path)

def seed_reproducer(seed=2020):
    """Reproducer for pytorch experiment.
    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.
    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

def init_params():
    parser = argparse.ArgumentParser(add_help=False)

    # General parameters
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--gpus", nargs="+", default=[0])
    parser.add_argument("--precision", type=int, default=16)

    parser.add_argument("--image_size", nargs="+", default=[300, 300])
    parser.add_argument("--n_clusters", type=int, default=93)

    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wt_dec", type=float, default=5e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=1)
    parser.add_argument("--max_epochs", type=int, default=15)

    # Project speficic parameters
    parser.add_argument("--mixup", type=int, default=0)

    parser.add_argument("--manifold_mixup", type=int, default=0)

    parser.add_argument("--fmix", type=int, default=0)

    parser.add_argument("--ent", type=int, default=0)
    parser.add_argument("--ent_loss_weight", type=float, default=0.02)

    parser.add_argument("--concent", type=int, default=0)
    parser.add_argument("--concent_loss_weight", type=float, default=2e-4)

    parser.add_argument("--knowledge_distillation", type=int, default=0)
    parser.add_argument("--knowledge_distillation_loss_weight", type=float, default=2)

    # Load model
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--pretrained_weights", type=str, default="network/ilsvrc-cls_rna-a1_cls1000_ep-0001.params")

    # Data dir
    parser.add_argument("--data_train", type=str, default="experiment/data_train.csv")
    parser.add_argument("--data_train_aug", type=str, default="experiment/data_train_aug.csv")
    parser.add_argument("--data_val", type=str, default="experiment/data_test.csv")
    parser.add_argument("--data_cam_generation", type=str, default="experiment/data_train.csv")

    # Utils dir
    parser.add_argument("--log_dir", type=str, default="experiment/log_dir")
    parser.add_argument("--distill_dir", type=str, default="experiment/distill_dir")
    parser.add_argument("--cam_dir", type=str, default="experiment/cam_dir")
    parser.add_argument("--vit_features_dir", type=str, default="experiment/vit_features_dir")

    parser.add_argument("--compute_vit_features", type=int, default=1)
    parser.add_argument("--cam_eval_thres", type=float, default=0.15)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    print(type(args.gpus), args.gpus)
    if len(args.gpus) == 1:
        args.gpus = [int(args.gpus[0])]
    else:
        args.gpus = [int(gpu) for gpu in args.gpus]

    args.image_size = [int(size) for size in args.image_size]
    return args

def init_logger(log_name, log_dir=None):
    try:
        if log_dir is None:
            pass
        else:
            os.stat(log_dir)
    except Exception:
        os.makedirs(log_dir)

    if log_name not in Logger.manager.loggerDict:
        logging.root.handlers.clear()
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s] %(filename)s[%(lineno)4s] : %(levelname)s  %(message)s"
        formatter = logging.Formatter(format_str, datefmt)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            file_info_handler = TimedRotatingFileHandler(
                filename=os.path.join(log_dir, "%s.log" % log_name), when="D", backupCount=7
            )
            file_info_handler.setFormatter(formatter)
            file_info_handler.setLevel(logging.INFO)
            logger.addHandler(file_info_handler)

    logger = logging.getLogger(log_name)
    return logger
