import os

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import math
import torch
import random
import numpy as np
from PIL import Image
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def random_seed(seed_value=42, deter=False):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    cudnn.enabled = not deter
    cudnn.benchmark = not deter
    cudnn.deterministic = deter
    torch.use_deterministic_algorithms(deter)


class SchedulerByIter(object):
    def __init__(self, optimizer, lr, num_iter, **kwargs):
        self.optimizer = optimizer
        self.lr = lr
        self.mode = kwargs['mode']
        self.decay_steps = num_iter * (kwargs['total_epochs'] - kwargs['warmup_epochs'])
        self.warmup_steps = num_iter * kwargs['warmup_epochs']

    def step(self, idx, iter=0):
        if iter < self.warmup_steps:
            # warm up lr schedule
            new_lr = iter * self.lr / self.warmup_steps
        else:
            if self.mode == 'poly':
                new_lr = self.lr * pow((1 - 1.0 * (iter - self.warmup_steps) / self.decay_steps), 0.9)
            else:
                raise NotImplemented

        assert new_lr >= 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_cpt(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")


def save_cpt(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def to_cuda(data, non_blocking=True):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = v.cuda(non_blocking=non_blocking)
    else:
        data = data.cuda(non_blocking=non_blocking)
    return data


def log_msg(msg, mode="INFO"):
    color_map = {
        "PROCESS": 38,
        "INFO": 34,
        "TRAIN": 36,
        "EVAL": 32,
        "TEST": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)

    return msg


class Convert2Color(object):
    def __init__(self, labels_dict):
        self.palette = {}
        for sub_dict in labels_dict.values():
            self.palette.update(sub_dict)

    def __call__(self, arr_2d):

        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in self.palette.items():
            m = arr_2d == c
            arr_3d[m] = i

        return arr_3d.transpose(2, 0, 1)
