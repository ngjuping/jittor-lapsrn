import jittor as jt
from jittor import Module
from jittor import nn
from jittor import init
import numpy as np
import math
from scipy import signal

# 2D bilinear kernel, provided in original implementation
def get_upsample_filter(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return jt.Var(filter).float()

# 2D Gaussian kernel, adjust std for larger kernlen
def gaussian_filter(kernlen, std=10):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = jt.Var(gkern2d).float()
    return gkern2d

class _Conv_Block(Module):    
    def __init__(self):
        super(_Conv_Block, self).__init__()
        
        self.cov_block = nn.Sequential(
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        
    def execute(self, x):  
        output = self.cov_block(x)
        return output 

class LapSRN(Module):
    def __init__(self):
        super(LapSRN, self).__init__()
        
        self.conv_input = nn.Conv(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        
        self.convt_I1 = nn.ConvTranspose(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)
  
        self.convt_I2 = nn.ConvTranspose(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)        
        
        for m in self.modules():
            if isinstance(m, nn.Conv):
                m.weight.kaiming_normal_() # init weights with Gaussian(0, 2/no. of inputs)
                if m.bias is not None:
                    m.bias.zero_() # init biases as zero if bias=True

            if isinstance(m, nn.ConvTranspose): # upsampling
                c1, c2, h, w = m.weight.size()
                weight = get_upsample_filter(h) # [h, w]
                weight = weight.view(1, 1, h, w) # [1, 1, h, w]
                m.weight = weight.repeat(c1, c2, 1, 1) # [c1, c2, h, w]
                if m.bias is not None:
                    m.bias.zero_() # init biases as zero if bias=True
                    
    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def execute(self, x):    
        out = self.relu(self.conv_input(x))
        
        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1
        
        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2
       
        return HR_2x, HR_4x
        
class L1_Charbonnier_loss(Module):
    """L1 Charbonnier loss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def execute(self, X, Y):
        diff = jt.add(X, -Y)
        error = jt.sqrt( diff * diff + self.eps )
        loss = jt.sum(error) 
        return loss