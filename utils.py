import math
import numpy as np

class LRScheduler:
    def __init__(self, optimizer, base_lr):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.lr_decay = 0.1
        self.decay_step = 100

    def step(self, epoch):
        self.optimizer.lr = self.base_lr * (self.lr_decay ** (epoch // self.decay_step))

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)