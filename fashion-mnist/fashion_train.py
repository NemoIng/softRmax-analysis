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
from fashion_network import Net
from fashion_dataset import prepare_dataset
from fashion_bound import plot_decision_boundary

# Network parameters
function = 'cons'

# Data parameters
num_classes = 10
train_all = True
train_index = [3,7]
test_all = True
test_index = [3,7]

# Train-Test parameters
num_epoch = 15
train_batch_size = 128
test_batch_size = 128
lr_list = [2e-3]
print_freq = 1

# When using apple silicon GPU:
device = torch.device("mps")

# When using other chip architecture:
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def main():
    trainset = prepare_dataset(train_all, train_index, test_all, test_index, 'train') 
    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test') 
        
    trainloader = td.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=1)   
    testloader = td.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    
    best_overall_acc = 0
    for i in range(len(lr_list)):
        lr = lr_list[i]
        with open(path + f'fashion_{function}_test_accuracy.txt', 'a') as f:
            f.write(f'\nrun_nr: {i+1}/{len(lr_list)}\n')
            f.write(f'learning_rate: {lr}\n\n')
        # For each try we reinitialize the network
        net = Net(device, num_classes, function).to(device)
        if function == 'softRmax' or function == 'cons':
            criterion = nn.NLLLoss()
        else:   
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        best_net = net
        for epoch in range(1, num_epoch + 1):
            scheduler.step()
            train_acc = train(trainloader, net, criterion, optimizer, epoch)
            test_acc = test(testloader, net)   
            with open(path + f'fashion_{function}_train_accuracy.txt', 'a') as f:
                f.write(f'[epoch {epoch}], train_accuracy: {train_acc:.5f}\n')
            with open(path + f'fashion_{function}_test_accuracy.txt', 'a') as f:
                f.write(f'[epoch {epoch}], test_accuracy: {test_acc:.5f}\n')
            if test_acc > best_overall_acc:
                best_overall_acc = test_acc
                torch.save(best_net.state_dict(), path + f'best_{function}_net_checkpoint.pt')

    class_acc(testloader, best_net)
    
    return best_overall_acc

def train(train_loader, net, criterion, optimizer, epoch):
    net.train()
    train_loss = AverageMeter()
    Acc_v = 0
    nb = 0

    for data in train_loader:
        X, Y = data 
        X = Variable(X).to(device)
        Y = Variable(Y).squeeze().to(device)
        N = len(X)
        nb = nb + N

        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)
        if function == 'softRmax' or function == 'cons':
            loss = criterion(torch.log(outputs), Y)
        else:
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

def class_acc(loader, net):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for data in loader:
            images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels.squeeze()).to(device) 
            output = net(images)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            total_correct += c.sum().item()
            total_samples += labels.size(0)
            for i in range(len(images)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]} : {round(100 * class_correct[i] / (1e-8 + class_total[i]),2)}%')
    
    total_accuracy = 100 * total_correct / total_samples
    print(f'Total Accuracy: {round(total_accuracy, 2)}%')

if __name__ == '__main__':
    if train_all:
        path = f'runs/{num_classes}_classes'
    else:
        path = f'runs/{train_index}_classes'

    if not os.path.exists(path):
        os.makedirs(path)
    path += '/'
    with open(path + f'fashion_{function}_train_accuracy.txt', 'a') as f:
        f.write(f'function: {function}\n')
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'train_batch_size: {train_batch_size}\n')
        if train_all:
           f.write(f'train_all: {train_all}\n')   
        else:
            f.write(f'train_index: {train_index}\n') 
        f.write(f'num_epochs: {num_epoch}\n')

    with open(path + f'fashion_{function}_test_accuracy.txt', 'a') as f:
        f.write(f'function: {function}\n')
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'train_batch_size: {train_batch_size}\n')

        if test_all:
           f.write(f'test_all: {test_all}\n')   
        else:
            f.write(f'test_index: {test_index}\n') 

        f.write(f'test_batch_size: {test_batch_size}\n')
        f.write(f'num_epochs: {num_epoch}\n')

    best_acc = main()

    with open(path + f'fashion_{function}_test_accuracy.txt', 'a') as f:
        f.write(f'\nbest_accuracy: {best_acc}\n\n')
    with open(path + f'fashion_{function}_train_accuracy.txt', 'a') as f:
        f.write('\n')
