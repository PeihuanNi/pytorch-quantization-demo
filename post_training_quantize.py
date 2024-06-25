from torch.serialization import load
from model import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


def direct_quantize(model, test_loader):
    correct = 0
    count = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to('cuda'), target.to('cuda')
        output = model.quantize_forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += len(data)
        if i % 500 == 0:
            break
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / count))
        
    print('direct quantization finish')


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to('cuda'), target.to('cuda')
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to('cuda'), target.to('cuda')
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 64
    using_bn = False
    load_quant_model_file = None
    # load_model_file = None

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    if using_bn:
        model = NetBN()
        model.load_state_dict(torch.load('./ckpt/mnist_cnnbn.pt', map_location='cpu'))
        save_file = "ckpt/mnist_cnnbn_ptq.pt"
    else:
        model = Net()
        model.load_state_dict(torch.load('ckpt/mnist_cnn_groups20.pt', map_location='cuda'))
        save_file = "ckpt/mnist_cnn_ptq.pt"
    model = model.to('cuda')
    model.eval()
    full_inference(model, test_loader)

    num_bits = 8
    model.quantize(num_bits=num_bits)
    model = model.to('cuda')
    model.eval()
    print('Quantization bit: %d' % num_bits)


    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print("Successfully load quantized model %s" % load_quant_model_file)
    

    direct_quantize(model, train_loader)

    torch.save(model.state_dict(), save_file)
    model.freeze()

    quantize_inference(model, test_loader)

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")  # 只显示前两个值以避免过多输出
    
    print(model.qconv1.qi.scale, model.qconv1.qi.zero_point)
    print(model.qconv1.qw.scale, model.qconv1.qw.zero_point)
    print(model.qconv1.qo.scale, model.qconv2.qo.zero_point)
    print('weight')
    print(model.qconv1.conv_module.weight.data)
    print('bias')
    print(model.qconv1.conv_module.bias.data)

    print(model.qconv2.qw.scale, model.qconv2.qw.zero_point)
    print(model.qconv2.qo.scale, model.qconv2.qo.zero_point)

    print(model.qfc.qw.scale, model.qfc.qw.zero_point)
    print(model.qfc.qo.scale, model.qfc.qo.zero_point)


    
