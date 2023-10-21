"""
Created on Fri Mar 19 10:02:50 2021

@author: ziqi
"""

import os
import torch
import torch.nn as nn 
import torch.utils.data as td
from torch import optim
from torch.autograd import Variable

from utils import AverageMeter
from network import Net
from dataset import prepare_dataset

# Network parameter
function = 'softmax'
conservative_a = 0.2
exp = 0
triangular = False

# Data parameters
num_classes = 10
train_all = True
train_index = [0, 1]
test_all = True
test_index = [0, 1]

# Train-Test parameters
num_epoch = 5
train_batch_size = 256
test_batch_size = 128
lr = 5e-3
weight_decay = 5e-6
print_freq = 1

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Device Configuration
device = torch.device("mps")

# Main Function
def main():
    net = Net(device, num_classes, function, conservative_a, triangular).to(device)

    trainset = prepare_dataset(train_all, train_index, test_all, test_index, 'train') 
    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test') 

    trainloader = td.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=1)   
    testloader = td.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_Acc = 0
    for epoch in range(1, num_epoch + 1):
        train_acc = train(trainloader, net, criterion, optimizer, epoch)
        test_acc = test(testloader, net)
        scheduler.step()
        with open(path + f'cifar_{function}_train_accuracy.txt', 'a') as f:
            f.write(f'[epoch {epoch}], train_accuracy is: {train_acc:.5f}\n')
        with open(path + f'cifar_{function}_test_accuracy.txt', 'a') as f:
            f.write(f'[epoch {epoch}], test_accuracy is: {test_acc:.5f}\n')
        if best_Acc < test_acc:
            best_Acc = test_acc
            torch.save(net.state_dict(), path + f'best_{function}_net_checkpoint.pt')
    return best_Acc

# Training Function
def train(train_loader, net, criterion, optimizer, epoch):
    net.train()
    train_loss = AverageMeter()
    Acc_v = 0
    nb = 0

    for i, data in enumerate(train_loader):
        X, Y = data
        X = Variable(X).to(device)
        Y = Variable(Y).squeeze().to(device)
        N = len(X)
        nb = nb + N

        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)
        loss = criterion(outputs, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data.item(), N)
        if epoch % print_freq == 0:
            print(f'[epoch {epoch}], [iter {i + 1} / {len(train_loader)}], [train loss {train_loss.avg:.5f}]')

    train_acc = (nb - Acc_v) / nb
    return train_acc

# Testing Function
def test(test_loader, net):
    net.eval()
    Acc_y = 0
    nb = 0
    for i, data in enumerate(test_loader):
        X, Y = data
        X = Variable(X).to(device)
        Y = Variable(Y.squeeze()).to(device)
        nb = nb + len(X)

        outputs = net(X)
        Acc_y = Acc_y + (outputs.argmax(1) - Y).nonzero().size(0)

    test_acc = (nb - Acc_y) / nb
    return test_acc

if __name__ == '__main__':
    path = 'runs'
    if not os.path.exists(path):
        os.makedirs(path)
    path += '/'
    with open(path + f'cifar_{function}_train_accuracy.txt', 'a') as f:
        f.write(f'function: {function}\n')
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'train_batch_size: {train_batch_size}\n')
        if train_all:
           f.write(f'train_all: {train_all}\n')
        else:
            f.write(f'train_index: {train_index}\n')
        f.write(f'num_epochs: {num_epoch}\n')
        f.write(f'learning_rate: {lr}\n')
        f.write(f'weight_decay: {weight_decay}\n')
        f.write(f'conservative_a: {conservative_a}\n')

    with open(path + f'cifar_{function}_test_accuracy.txt', 'a') as f:
        f.write(f'function: {function}\n')
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'train_batch_size: {train_batch_size}\n')