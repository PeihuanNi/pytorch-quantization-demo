import torch

res = []
for i in range(1000000):

    num1 = torch.randn([3, 3], dtype=torch.float, requires_grad=False, device='cuda')
    num2 = torch.randn([3, 3], dtype=torch.float, requires_grad=False, device='cuda')
    mul = torch.matmul(num1, num2)
    # print(mul)
    res.append(mul)

rmax=0
rmin=0
for tensor in res:
    tmax = tensor.max().item()
    tmin = tensor.min().item()
    if tmax > rmax:
        rmax = tmax
    if tmin < rmin:
        rmin = tmin

print('rmax')
print(rmax)
print('rmin')
print(rmin)