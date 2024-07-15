import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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



# X = torch.rand(1, 1, 224, 224)
# for name, module in Net.named_children():
#     print(name)
#     X = module(X)
#     print(X.shape)

batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Testing loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total} %')