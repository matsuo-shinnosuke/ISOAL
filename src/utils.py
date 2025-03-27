import os
import pickle
from datetime import datetime
import sys
import numpy as np
import random
import torch

def set_device(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    return device

def set_reproductibility(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm
    torch.backends.cudnn.benchmark = False

def get_date():
    now = datetime.now()
    date = now.strftime('%Y%m%d_%H%M%S')
    return date

def save_pickle(save_path, value):
    with open(save_path, 'wb') as f:
        pickle.dump(value, f)

class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.console = sys.stdout
        self.date = get_date()

    def write(self, message):
        with open(self.file_path, "a") as f:
            f.write(message)
        self.console.write(message)

    def flush(self):
        self.console.flush()

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