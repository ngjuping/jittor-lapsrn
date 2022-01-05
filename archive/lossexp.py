import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np

class L1_Charbonnier_loss(Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def execute(self, X, Y):
        diff = jt.add(X, -Y)
        error = jt.sqrt( diff * diff + self.eps )
        loss = jt.sum(error) 
        return loss

criterion = L1_Charbonnier_loss()
res = criterion(jt.Var([1,2,3]), jt.Var([1.1,2.2,3.3]))
print(res)