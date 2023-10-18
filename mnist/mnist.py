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

hps = {'function': 'softmax',
       'train_all': True,
       'train_index': [3,7],
       'test_all': True,
       'test_index': [3,7],
       'num_classes': 10,
       'train_batch_size': 32,
       'test_batch_size': 128,
       'epoch': 5,
       'lr': 1e-3,
       'weight_decay': 5e-6,
       'print_freq':1,
       'conservative_a': 0.2,
       'exp': 0,
       'triangular': False}

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

# When using apple silicon GPU:
device = torch.device("mps")

# When using other chip architecture:
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    net = Net(device, hps['num_classes'], hps['function'], hps['conservative_a']).to(device)

    trainset = prepare_dataset(hps['train_all'], hps['train_index'], hps['test_all'], hps['test_index'], 'train') 
    testset = prepare_dataset(hps['train_all'], hps['train_index'], hps['test_all'], hps['test_index'], 'test') 
        
    trainloader = td.DataLoader(trainset, batch_size=hps['train_batch_size'],
                                              shuffle=True, num_workers=1)   
    testloader = td.DataLoader(testset, batch_size=hps['test_batch_size'],
                                         shuffle=False, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=hps['lr'], weight_decay=hps['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_Acc = 0  
    for epoch in range(1, hps['epoch'] + 1):
        train_acc = train(trainloader, net, criterion, optimizer, epoch)
        test_acc = test(testloader, net)   
        scheduler.step()
        with open(path + f'mnist_{hps["function"]}_train_accuracy.txt', 'a') as f:
            f.write('[epoch %d], train_accuracy: %.5f \n' % (epoch, train_acc))
        with open(path + f'mnist_{hps["function"]}_test_accuracy.txt', 'a') as f:
            f.write('[epoch %d], test_accuracy: %.5f \n' % (epoch, test_acc))
        if best_Acc < test_acc:
            best_Acc = test_acc 
            torch.save(net.state_dict(), path + f'best_{hps["function"]}_net_checkpoint.pt')
    return best_Acc

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
        nb = nb+N

        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)
        loss = criterion(outputs, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data.item(), N)     
        if epoch % hps['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (epoch, i + 1, len(train_loader), train_loss.avg))

    train_acc = (nb - Acc_v)/nb
    return train_acc

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
  
    test_acc = (nb - Acc_y)/nb 
    return test_acc


if __name__ == '__main__':
    path = 'runs'
    if not os.path.exists(path):
        os.makedirs(path)
    path += '/'
    with open(path + f'mnist_{hps["function"]}_train_accuracy.txt', 'a') as f:
        f.write(f'function: {hps["function"]}\n')
        f.write(f'num_classes: {hps["num_classes"]}\n')
        f.write(f'train_batch_size: {hps["train_batch_size"]}\n')
        if hps['train_all']:
           f.write(f'train_all: {hps["train_all"]}\n')   
        else:
            f.write(f'train_index: {hps["train_index"]}\n') 
        f.write(f'num_epochs: {hps["epoch"]}\n')
        f.write(f'learning_rate: {hps["lr"]}\n')
        f.write(f'weight_decay: {hps["weight_decay"]}\n')
        f.write(f'conservative_a: {hps["conservative_a"]}\n')

    with open(path + f'mnist_{hps["function"]}_test_accuracy.txt', 'a') as f:
        f.write(f'function: {hps["function"]}\n')
        f.write(f'num_classes: {hps["num_classes"]}\n')
        f.write(f'train_batch_size: {hps["train_batch_size"]}\n')

        if hps['test_all']:
           f.write(f'test_all: {hps["test_all"]}\n')   
        else:
            f.write(f'test_index: {hps["test_index"]}\n') 

        f.write(f'test_batch_size: {hps["test_batch_size"]}\n')
        f.write(f'num_epochs: {hps["epoch"]}\n')
        f.write(f'learning_rate: {hps["lr"]}\n')

        if hps['function'] == "cons":
            f.write(f'conservative_a: {hps["conservative_a"]}\n')

    best_acc = main()

    with open(path + f'mnist_{hps["function"]}_test_accuracy.txt', 'a') as f:
        f.write(f'\nbest_accuracy: {best_acc}\n\n')
    with open(path + f'mnist_{hps["function"]}_train_accuracy.txt', 'a') as f:
        f.write('\n')