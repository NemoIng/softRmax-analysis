"""
Created on Fri Mar 19 10:09:59 2021

@author: ziqi, edited by Nemo Ingendaa

"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class conservative_softmax(nn.Module): 
    def __init__(self, num_classes, a):
        super(conservative_softmax, self).__init__()
        self.num_classes = num_classes
        self.a = a
    def forward(self, input):
        nu = []
        pos = []
        for i in range(self.num_classes):
            nu.append(1/((self.a*input[:,i])**2 + 1e-20))
        for i in range(self.num_classes):
            pos.append(nu[i]/sum(nu))
        pos = torch.stack(pos, 1)
        return pos
    
class softRmax(nn.Module):
    def __init__(self, num_classes, device):
        super(softRmax, self).__init__()
        self.num_classes = num_classes
        self.e = torch.eye(num_classes).to(device)
    def forward(self, input):        
        nu = []
        pos = []
        for i in range(self.num_classes):
            nu.append(1/((input-self.e[i])**2).sum(1) + 1e-20)
        for i in range(self.num_classes):
            pos.append(nu[i]/sum(nu))
        pos = torch.stack(pos, 1)
        return pos

class Net(nn.Module):
    def __init__(self, device, num_classes, function, a, kernel_size):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=kernel_size)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 10)
        
        self.function = function
        if function == 'cons':
            self.softmax = conservative_softmax(num_classes, a)
        elif function == 'softRmax':
            self.softmax = softRmax(num_classes, device)
        elif function == 'softmax':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4_drop(self.conv4(x)))

        x = x.view(-1, 256)
        x = self.fc1(x)
        return self.softmax(x)