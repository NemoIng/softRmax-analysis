"""
@author: NemoIng (Nemo ingendaa)

Using/Inspiration code from:
- https://github.com/ziqiwangsilvia/attack 
"""
import os
import torch
import torch.nn as nn 
import torch.utils.data as td
from torch import optim
from torch.autograd import Variable

from utils import AverageMeter
from cifar_network import Net
from cifar_dataset import prepare_dataset
from cifar_bound import plot_decision_boundary

# Network parameter
function = 'softmax'

# Data parameters
num_classes = 10
train_all = True
train_index = [0, 1]
test_all = True
test_index = [0, 1]

# Train-Test parameters
num_epoch = 15
train_batch_size = 256
test_batch_size = 256
lr = 5e-3
weight_decay = 5e-6
print_freq = 1

plot_epochs = [] # for which epochs a decision boundary plot will be created

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Device Configuration
device = torch.device("mps")

# Main Function
def main():
    net = Net(device, num_classes, function).to(device)
    trainset = prepare_dataset(train_all, train_index, test_all, test_index, 'train') 
    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test') 

    trainloader = td.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=1)   
    testloader = td.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)

    if function == 'softRmax':
        criterion = nn.NLLLoss()
    else:   
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_Acc = 0
    for epoch in range(1, num_epoch + 1):
        scheduler.step()
        train_acc = train(trainloader, net, criterion, optimizer, epoch)
        test_acc = test(testloader, net)
        with open(path + f'cifar_{function}_train_accuracy.txt', 'a') as f:
            f.write(f'[epoch {epoch}], train_accuracy is: {train_acc:.5f}\n')
        with open(path + f'cifar_{function}_test_accuracy.txt', 'a') as f:
            f.write(f'[epoch {epoch}], test_accuracy is: {test_acc:.5f}\n')
        if best_Acc < test_acc:
            best_Acc = test_acc
            torch.save(net.state_dict(), path + f'best_{function}_net_checkpoint.pt')
        if epoch in plot_epochs:
            plot_decision_boundary(net, num_classes, epoch, function, index=test_index)
    per_class_acc(testloader, net)
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
        if function == 'softRmax':
            loss = criterion(torch.log(outputs), Y)
        else:
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
    for data in test_loader:
        X, Y = data
        X = Variable(X).to(device)
        Y = Variable(Y.squeeze()).to(device)
        nb = nb + len(X)

        outputs = net(X)
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        Acc_y = Acc_y + (outputs.argmax(1) - Y).nonzero().size(0)

    test_acc = (nb - Acc_y) / nb
    return test_acc

def per_class_acc(loader, net):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels.squeeze()).to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(images)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / (1e-8 + class_total[i])))

if __name__ == '__main__':
    if train_all:
        path = f'runs/{num_classes}_classes'
    else:
        path = f'runs/{train_index}_classes'
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

    with open(path + f'cifar_{function}_test_accuracy.txt', 'a') as f:
        f.write(f'function: {function}\n')
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'train_batch_size: {train_batch_size}\n')
    best_acc = main()