import torch
def fake_quant(q, x):
    qx = torch.round(x / q.S + q.Z)
    x = q.S * (qx - q.Z)
    return x