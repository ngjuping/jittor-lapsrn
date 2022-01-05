import math
import numpy as np

class LRScheduler:
    def __init__(self, optimizer, base_lr):
        self.optimizer = optimizer

        self.basic_lr = base_lr
        self.lr_decay = 0.6
        self.decay_step = 15000

    def step(self, step):
        lr_decay = self.lr_decay ** int(step / self.decay_step)
        lr_decay = max(lr_decay, 2e-5)
        self.optimizer.lr = lr_decay * self.basic_lr

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)