import torch.nn as nn
from fake_quant import *
from QParam import *

class QConv(nn.Module):
    def __init__(self, qi, qo, conv):
        super().__init__()
        self.qi = qi
        self.qo = qo
        self.conv = conv
        self.qw = Qparam()
    
    def forward(self, x):
        x = fake_quant(self.qi, x)
        out = self.conv(x)
