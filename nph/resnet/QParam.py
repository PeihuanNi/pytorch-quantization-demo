import torch

class Qparam:
    def __init__(self, qmin=0, qmax=255):
        self.qmin = qmin
        self.qmax = qmax
        self.rmin = None
        self.rmax = None

    def update(self, r):
        self.rmin = r.min()
        self.rmax = r.max()
        self.S = (self.rmax - self.rmin) / (self.qmax - self.qmin)
        self.Z = torch.round(self.qmax - self.rmax / self.S)
        return self.S, self.Z