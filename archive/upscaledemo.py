import numpy as np
import jittor as jt
from jittor import nn
import matplotlib.pyplot as plt
from scipy import signal

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    # print(og)
    # print(np.array(og).shape)
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return jt.Var(filter).float()

bilinear_kernel = get_upsample_filter(64)
print(bilinear_kernel.shape)
plt.imshow(bilinear_kernel, interpolation='none')
plt.savefig("BilinearKernel.png")

def gkern(kernlen, std=10):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = jt.Var(gkern2d).float()
    return gkern2d

gaussian_kernel = gkern(64)
print(gaussian_kernel.shape)
plt.imshow(gaussian_kernel, interpolation='none')
plt.savefig("GaussianKernel.png")