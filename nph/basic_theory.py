import numpy
import torch

weight1 = torch.randn([5, 5], dtype=torch.float, requires_grad=False, device='cuda')
weight2 = torch.randn([5, 5], dtype=torch.float, requires_grad=False, device='cuda')
input = torch.randn([5, 5], dtype=torch.float, requires_grad=False, device='cuda')
# print("weight1")
# print(weight1)
# print("weight2")
# print(weight2)
# print("input")
# print(input)

def quant(r, qmin=-128, qmax=127):
    rmin = r.min()
    rmax = r.max()
    S = (rmax - rmin) / (qmax - qmin)
    # Z = torch.round(qmin - rmin / S)
    Z = torch.round(qmax -rmax / S)
    q = torch.round(r / S + Z).to(torch.int8).to('cuda')
    return q, S, Z

def dequant(q, S, Z):
    r = S * (q - Z).to(torch.float).to('cuda')
    return r

def MSE(origin, dequant):
    return ((origin-dequant)**2).sum() / origin.numel()

a1 = torch.matmul(weight1, input)
qw1, Sw1, Zw1 = quant(weight1)
qw2, Sw2, Zw2 = quant(weight2)
qi, Si, Zi = quant(input)

qa1 = torch.matmul(qw1.to(torch.float), qi.to(torch.float))
_, Sa1, Za1 = quant(a1, qa1.min(), qa1.max())

ra1 = dequant(qa1, Sa1, Za1)
mse = MSE(a1, ra1)

print('qa1')
print(qa1)
print('Sa1, Za1')
print(Sa1, Za1)
print('a1')
print(a1)
print('ra1')
print(ra1)
print('mse')
print(mse)