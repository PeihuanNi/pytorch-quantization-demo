import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Qparam:
    def __init__(self, qmin, qmax):
        self.qmin = qmin
        self.qmax = qmax
        self.scale = None
        self.zero_point = None

    def update(self, tensor):
        self.rmin = tensor.min().item()
        self.rmax = tensor.max().item()
        self.scale = (self.rmax - self.rmin) / (self.qmax - self.qmin)
        self.zero_point = self.qmin - self.rmin / self.scale

    def quant(self, tensor):
        q = torch.round(tensor / self.scale + self.zero_point)
        q.clamp_(self.qmin, self.qmax)
        return q

    def dequant(self, q):
        return (q - self.zero_point) * self.scale


class QModule(nn.Module):
    def __init__(self, qmin=0, qmax=255):
        super(QModule, self).__init__()
        self.qmin = qmin
        self.qmax = qmax

    def quantize(self):
        raise NotImplementedError

    def dequantize(self):
        raise NotImplementedError


class QConv2d(QModule):
    def __init__(self, conv_module, qmin=0, qmax=255):
        super(QConv2d, self).__init__(qmin, qmax)
        self.conv = conv_module
        self.qw = Qparam(qmin, qmax)
        self.qi = Qparam(qmin, qmax)
        self.qo = Qparam(qmin, qmax)

    def forward(self, x):  
        if self.training:
            self.qi.update(x)
            self.qw.update(self.conv.weight.data)
            
            # x = self.qi.dequant(self.qi.quant(x))
            x = self.qi.quant(x)
            # self.conv.weight.data = self.qw.dequant(self.qw.quant(self.conv.weight.data))
            self.conv.weight.data = self.qw.quant(self.conv.weight.data)
            x = self.conv(x)
            x = x / self.qi.scale / self.qw.scale
            self.comv.weight.data = self.qw.dequant(self.conv.weight.data)
            self.qo.update(x)
            x = self.qo.dequant(self.qo.quant(x))
        else:
            # x = self.qi.dequant(self.qi.quant(x))
            x = self.qi.quant(x)
            # self.conv.weight.data = self.qw.dequant(self.qw.quant(self.conv.weight.data))
            self.conv.weight.data = self.qw.quant(self.conv.weight.data)
            x = self.conv(x)
            x = x / self.qi.scale / self.qw.quant
            # x = self.qo.dequant(self.qo.quant(x))
        return x


class QReLU(QModule):
    def __init__(self, relu_module, qmin=0, qmax=255):
        super(QReLU, self).__init__(qmin, qmax)
        self.relu = relu_module

    def forward(self, x):
        return self.relu(x)


class QLinear(QModule):
    def __init__(self, linear_module, qmin=0, qmax=255):
        super(QLinear, self).__init__(qmin, qmax)
        self.linear = linear_module
        self.qw = Qparam(qmin, qmax)
        self.qi = Qparam(qmin, qmax)
        self.qo = Qparam(qmin, qmax)

    def forward(self, x):
        if self.training:
            self.qi.update(x)
            self.qw.update(self.linear.weight.data)
            # x = self.qi.dequant(self.qi.quant(x))
            x = self.qi.quant(x)
            # self.linear.weight.data = self.qw.dequant(self.qw.quant(self.linear.weight.data))
            self.linear.weight.data = self.qw.quant(self.linear.weight.data)
            x = self.linear(x)
            x = x / self.qi.scale / self.qw.scale
            self.qo.update(x)
            x = self.qo.dequant(self.qo.quant(x))
        else:
            # x = self.qi.dequant(self.qi.quant(x))
            x = self.qi.quant(x)
            # self.linear.weight.data = self.qw.dequant(self.qw.quant(self.linear.weight.data))
            self.linear.weight.data = self.qw.quant(self.linear.weight.data)
            x = self.linear(x)
            x = x / self.qi.quant / self.qw.quant
            # x = self.qo.dequant(self.qo.quant(x))
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, groups=20, bias=False)
        self.fc = nn.Linear(5*5*40, 10, bias=False)
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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%\n')


if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 5
    lr = 0.01
    momentum = 0.5
    save_model = True

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(
        datasets.MNIST(root='D:/Dataset', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='D:/Dataset', train=False, download=False,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if save_model:
        torch.save(model.state_dict(), 'model_weights.pth')
        torch.save(model, 'model.pth')

