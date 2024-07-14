import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def quant(tensor, qmin, qmax):
    rmin = tensor.min()
    rmax = tensor.max()
    scale =  (qmax - qmin) / (rmax - rmin)
    tensor = torch.round(tensor * scale).clamp(qmin, qmax)
    return tensor, scale

class Net(nn.Module):
    def __init__(self, num_channels=1):
        """实例化出模型包含的层"""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=40, kernel_size=3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, groups=20, bias=False)
        self.fc = nn.Linear(5*5*40, 10, bias=False)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x, qmin, qmax, scale_conv1, scale_conv2, scale_fc):
        """前向传播函数"""
        x, scale_input = quant(x, qmin, qmax)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 5*5*40)
        x = self.fc(x)
        x = x / scale_conv1 / scale_conv2 / scale_fc
        return x

def test(model, device, test_loader, scale_conv1, scale_conv2, scale_fc, qmin, qmax):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data, qmin)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))

model = torch.load('model.pth')


params = model.state_dict()

for name, param in params.items():
    print(f'{name}: {param.shape}')
qmin = -128
qmax = 127
test_batch_size = 64
num_epochs = 5

params['conv1.weight'], scale_conv1 = quant(params['conv1.weight'], qmin, qmax)
params['conv2.weight'], scale_conv2 = quant(params['conv2.weight'], qmin, qmax)
params['fc.weight'], scale_fc = quant(params['fc.weight'], qmin, qmax)

# for name, param in params.items():
#     print(f'{name}: {param}')

test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, num_workers=32, pin_memory=True
    )

test(model, 'cuda', test_loader, scale_conv1, scale_conv2, scale_fc, qmin, qmax)