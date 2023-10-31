import os
import torch
import torch.nn as nn 
import torch.utils.data as td
from torch import optim
from torch.autograd import Variable

from utils import AverageMeter
from network import Net
from dataset import prepare_dataset

# Network parameters
function = 'softmax'
kernel_size = 3
conservative_a = 0.2

# Data parameters
num_classes = 10
train_all = True
train_index = [3, 7]
test_all = True
test_index = [3, 7]

# Train-Test parameters
num_epoch = 3
num_tries = 3
train_batch_size = 32
test_batch_size = 128
lr = 1e-3
weight_decay = 5e-6
print_freq = 1

# When using apple silicon GPU:
device = torch.device("mps")

# When using other chip architecture:
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    trainset = prepare_dataset(train_all, train_index, test_all, test_index, 'train') 
    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test') 
        
    trainloader = td.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=1)   
    testloader = td.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    
    best_overall_acc = 0
    for i in range(num_tries):
        with open(path + f'mnist_{function}_{num_epoch}_test_accuracy.txt', 'a') as f:
            f.write(f'\nrun_nr: {i+1}\n\n')
        # For each try we reinitialize the network
        net = Net(device, num_classes, function, conservative_a, kernel_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        best_curr_try_acc = 0
        for epoch in range(1, num_epoch + 1):
            train_acc = train(trainloader, net, criterion, optimizer, epoch)
            test_acc = test(testloader, net)   
            scheduler.step()
            with open(path + f'mnist_{function}_{num_epoch}_train_accuracy.txt', 'a') as f:
                f.write(f'[epoch {epoch}], train_accuracy: {train_acc:.5f}\n')
            with open(path + f'mnist_{function}_{num_epoch}_test_accuracy.txt', 'a') as f:
                f.write(f'[epoch {epoch}], test_accuracy: {test_acc:.5f}\n')
            if test_acc > best_curr_try_acc:
                best_curr_try_acc = test_acc 
        if best_curr_try_acc > best_overall_acc:
            best_overall_acc = best_curr_try_acc
            torch.save(net.state_dict(), path + f'best_{function}_{num_epoch}_net_checkpoint.pt')
    
    return best_overall_acc

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

    print(f'[epoch {epoch}], [train loss {train_loss.avg:.5f}]')

    train_acc = (nb - Acc_v) / nb
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
  
    test_acc = (nb - Acc_y) / nb 
    return test_acc

if __name__ == '__main__':
    path = 'runs'
    if not os.path.exists(path):
        os.makedirs(path)
    path += '/'
    with open(path + f'mnist_{function}_{num_epoch}_train_accuracy.txt', 'a') as f:
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
        f.write(f'kernel_size: {kernel_size}\n')

    with open(path + f'mnist_{function}_{num_epoch}_test_accuracy.txt', 'a') as f:
        f.write(f'function: {function}\n')
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'train_batch_size: {train_batch_size}\n')

        if test_all:
           f.write(f'test_all: {test_all}\n')   
        else:
            f.write(f'test_index: {test_index}\n') 

        f.write(f'test_batch_size: {test_batch_size}\n')
        f.write(f'num_epochs: {num_epoch}\n')
        f.write(f'learning_rate: {lr}\n')

        if function == "cons":
            f.write(f'conservative_a: {conservative_a}\n')

    best_acc = main()

    with open(path + f'mnist_{function}_{num_epoch}_test_accuracy.txt', 'a') as f:
        f.write(f'\nbest_accuracy: {best_acc}\n\n')
    with open(path + f'mnist_{function}_{num_epoch}_train_accuracy.txt', 'a') as f:
        f.write('\n')
