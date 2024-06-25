import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def calcScaleZeropoint(rmin, rmax, qmin=0, qmax=255):
    scale = (rmax - rmin) / (qmax - qmin)
    zero_point = torch.round(qmax - rmax / scale)
    return scale, zero_point

def quant(tensor, scale, zero_point, qmin=0, qmax=255):
    q = tensor / scale + zero_point
    q.round_().clamp_(qmin, qmax)
    return q

def dequant(q, scale, zero_point):
    r = (q - zero_point) * scale
    return r


class Qparam:
    def __init__(self, qmin, qmax):
        self.qmin = qmin
        self.qmax = qmax
        self.scale = None
        self.zero_point = None
        self.rmin = None
        self.rmax = None

    def update(self, tensor):
        self.rmin = tensor.min()
        self.rmax = tensor.max()
        self.scale, self.zero_point = calcScaleZeropoint(self.rmin, self.rmax, qmin=self.qmin, qmax=self.qmax)


    def quant(self, tensor):
        return quant(tensor, self.scale, self.zero_point, qmin=self.qmin, qmax=self.qmax)

    def dequant(self, q):
        return dequant(q, self.scale, self.zero_point)
    

class QModule(nn.Module):
    def __init__(self, qi, qo, qmin, qmax):
        super(QModule, self).__init__()
        if qi:
            self.qi = Qparam(qmin, qmax)
        if qo:
            self.qo = Qparam(qmin, qmax)

    def freeze(self):
        raise NotImplementedError('freeze should be implement')

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_infernence should be implement')
    

class QConv2d(QModule):
    def __init__(self, conv_module, qmin, qmax, qi=True, qo=True):
        """
        继承QModule的init,当传入根据传入的qi和qo来判断是否有创建的qi和qo
        实际上只有第一个qconv块需要qi,其余都是复用上一个块的qo
        注意这里只需要在初始化是给出需不需要qi和qo即可
        """
        super(QConv2d, self).__init__(qi=qi, qo=qo, qmin=qmin, qmax=qmax)
        self.qmin = qmin
        self.qmax = qmax
        self.conv_module = conv_module
        self.qw = Qparam(self.qmin, self.qmax)

    
    def freeze(self, qi=None, qo=None):
        """
         固定好每层和全部量化参数，并直接把参数量化好
         """
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        # 计算M
        self.M = self.qi.scale * self.qw.scale / self.qo.scale
        # 计算weight量化
        self.conv_module.weight.data = self.qw.quant(self.conv_module.weight.data)
        # print(self.conv_module.weight.data)
        # 计算bias量化
        self.conv_module.bias.data = quant(self.conv_module.bias.data, self.qi.scale*self.qw.scale, zero_point=128, qmin=self.qmin, qmax=self.qmax)

    def forward(self, x):
        """
        Net类的quantize_forward方法会调用
        使用fake quant,统计quant带来的误差时的rmin和rmax
        统计出来的参数在freeze时固定
        """
        # 如果有qi，则计算x的量化参数，并直接fake quant
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quant(x)
            x = self.qi.dequant(x)
        # 计算conv的量化参数w和b
        self.qw.update(self.conv_module.weight.data)
        self.conv_module.weight.data = self.qw.quant(self.conv_module.weight.data)
        self.conv_module.weight.data = self.qw.dequant(self.conv_module.weight.data)
        # fake quant后计算conv
        x = self.conv_module(x)

        # 如果有自己的qo，则计算qo的量化参数
        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.quant(x)
            x = self.qo.dequant(x)
        return x
    
    def quantize_inference(self, x):
        """量化推理时调用"""
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x + self.qo.zero_point
        return x
    
class QReLU(QModule):
    def __init__(self, relu_module, qi=False, qo=True, qmin=0, qmax=255):
        super(QReLU, self).__init__(qi=qi, qo=qo, qmin=qmin, qmax=qmax)
        self.qmin = qmin
        self.qmax = qmax
        self.relu_module = relu_module

    def freeze(self, qi=None, qo=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quant(x)
            x = self.qi.dequant(x)
        x = self.relu_module(x)
        return x

    def quantize_inference(self, x):
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x
    

class QLinear(QModule):
    def __init__(self, linear_module, qi=True, qo=True, qmin=0, qmax=255):
        super(QLinear, self).__init__(qi=qi, qo=qo, qmin=qmin, qmax=qmax)
        self.qmin = qmin
        self.qmax = qmax
        self.linear_module = linear_module
        self.qw = Qparam(self.qmin, self.qmax)

    def freeze(self, qi=None, qo=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M = self.qi.scale * self.qw.scale / self.qo.scale

        self.linear_module.weight.data = self.qw.quant(self.linear_module.weight.data)

        self.linear_module.bias.data = quant(self.linear_module.bias.data, self.qi.scale*self.qw.scale, zero_point=128, qmin=self.qmin, qmax=self.qmax)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quant(x)
            x = self.qi.dequant(x)
        self.qw.update(self.linear_module.weight.data)
        self.linear_module.weight.data = self.qw.quant(self.linear_module.weight.data)
        self.linear_module.weight.data = self.qw.dequant(self.linear_module.weight.data)
        x = self.linear_module(x)
        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.quant(x)
            x = self.qo.dequant(x)

        return x
    
    def quantize_inference(self, x):
        self.linear_module.weight.data = self.linear_module.weight.data - self.qw.zero_point
        x = x - self.qi.zero_point
        x = self.linear_module(x)
        x = self.M * x + self.qo.zero_point
        return x
        

class QMaxPool2d(QModule):
    def __init__(self, maxpool_module, qi=True, qo=False, qmin=0, qmax=255):
        super(QMaxPool2d, self).__init__(qi=qi, qo=qo, qmin=qmin, qmax=qmax)
        self.qmin = qmin
        self.qmax = qmax
        self.maxpool_module = maxpool_module

    def freeze(self, qi=None, qo=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
    
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quant(x)
            x = self.qi.dequant(x)
        x = self.maxpool_module(x)
        return x
    
    def quantize_inference(self, x):
        x = self.maxpool_module(x)
        return x

class Net(nn.Module):
    def __init__(self, num_channels=1):
        """实例化出模型包含的层"""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=40, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, groups=20)
        self.fc = nn.Linear(5*5*40, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """前向传播函数"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 5*5*40)
        x = self.fc(x)
        return x

    def quantize(self, qmin=0, qmax=255):
        """实例化量化层"""
        self.qconv1 = QConv2d(self.conv1, qmin=qmin, qmax=qmax, qi=True, qo=True)
        self.qrelu1 = QReLU(self.relu, qi=False, qo=False, qmin=qmin, qmax=qmax)
        self.qmaxpool2d1 = QMaxPool2d(self.maxpool, qi=False, qo=False, qmin=qmin, qmax=qmax)
        self.qconv2 =QConv2d(self.conv2, qmin=qmin, qmax=qmax, qi=False, qo=True)
        self.qrelu2 = QReLU(self.relu, qi=False, qo=False, qmin=qmin, qmax=qmax)
        self.qmaxpool2d2 = QMaxPool2d(self.maxpool, qi=False, qo=False, qmin=qmin, qmax=qmax)
        self.qfc = QLinear(self.fc, qi=False, qo=True, qmin=qmin, qmax=qmax)

    def quantize_forward(self, x):
        """
        调用每个层的forward函数,模拟量化误差,统计rmin,rmax
        """
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d2(x)
        x = x.view(-1, 5*5*40)
        x = self.qfc(x)
        return x
    
    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(qi=self.qconv1.qo)
        self.qmaxpool2d1.freeze(qi=self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(qi=self.qconv2.qo)
        self.qmaxpool2d2.freeze(qi=self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quant(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d2.quantize_inference(qx)
        qx = qx.view(-1, 5*5*40)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequant(qx)
        return out


def direct_quantize(model, test_loader):
    correct = 0
    count = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to('cuda'), target.to('cuda')
        output = model.quantize_forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += len(data)
        if i % 500 == 0:
            break
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / count))
    print('direct quantization finish')


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to('cuda'), target.to('cuda')
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


model = Net()
model.load_state_dict(torch.load('ckpt/mnist_cnn_groups20.pt'))
model = model.to('cuda')
model.quantize(qmin=0, qmax=255)

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True,pin_memory=True
    )

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])),
  batch_size=64, shuffle=True, pin_memory=True
)

direct_quantize(model, train_loader)
model.freeze()
quantize_inference(model, test_loader)

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")  # 只显示前两个值以避免过多输出
