import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Residual(nn.Module):
    def __init__(self, input_channels, num_channesls, use1x1conv=False, strides=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channesls, kernel_size=3, padding=1, stride=strides)

        self.conv2 = nn.Conv2d(num_channesls, num_channesls, kernel_size=3, padding=1)

        if use1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channesls, kernel_size=1, stride=strides)
        
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channesls)
        self.bn2 = nn.BatchNorm2d(num_channesls)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.conv1(X)
        # print(f'    conv1 {out.shape}')
        out = self.bn1(out)
        # print(f'    bn1 {out.shape}')
        out = self.relu(out)
        out = self.conv2(out)
        # print(f'    conv2 {out.shape}')
        out = self.bn2(out)
        # print(f'    bn2 {out.shape}')
        if self.conv3:
            X = self.conv3(X)
            # print(f'    conv3 {out.shape}')
        out += X
        out = self.relu(out)
        return out

# 查看输出形状是否正确
# blk = Residual(3, 3)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# print(Y.shape)



def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use1x1conv=True, strides=2)) # 注意这里strides给的是2
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = resnet_block(64, 64, 2, first_block=True)
        self.b3 = resnet_block(64, 128, 2)
        self.b4 = resnet_block(128, 256, 2)
        self.b5 = resnet_block(256, 512, 2)
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, 10)
    
    def forward(self, X):
        out = self.b1(X)
        # print(f'b1 {out.shape}')
        out = self.b2(out)
        # print(f'b2 {out.shape}')
        out = self.b3(out)
        # print(f'b3 {out.shape}')
        out = self.b4(out)
        # print(f'b4 {out.shape}')
        out = self.b5(out)
        # print(f'b5 {out.shape}')
        out = self.AvgPool(out)
        # print(f'avgpool {out.shape}')
        out = self.flat(out)
        # print(f'flat {out.shape}')
        out = self.fc(out)
        # print(f'fc {out.shape}')
        return out


Net = ResNet18()

X = torch.rand(1, 1, 224, 224)
for name, module in Net.named_children():
    print(name)
    X = module(X)
    print(X.shape)
