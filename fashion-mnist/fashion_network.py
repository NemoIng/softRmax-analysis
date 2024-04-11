"""
Created on Fri Mar 19 10:09:59 2021

@author: NemoIng (Nemo ingendaa)

Using/Inspiration code from:
- https://github.com/ziqiwangsilvia/attack 
- https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy (Network architecture)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
    
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
    
class conservative_softmax_monotone(nn.Module): 
    def __init__(self, num_classes, a):
        super(conservative_softmax_monotone, self).__init__()
        self.num_classes = num_classes
        self.a = a
    def forward(self, input):
        nu = []
        pos = []
        for i in range(self.num_classes):
            nu.append(input[:,i] + torch.sqrt(1 + (input[:,i])**2))
        for i in range(self.num_classes):
            pos.append(nu[i]/sum(nu))
        pos = torch.stack(pos, 1)
        return pos
    
class Net(nn.Module):
    def __init__(self, device, num_classes, function):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
        self.function = function
        if function == 'softRmax':
            self.softmax = softRmax(num_classes, device)
            self.conservative = True
        elif function == 'cons':
            self.softmax = conservative_softmax_monotone(num_classes, 0.1)
            self.conservative = True
        else:
            self.conservative = False

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if self.conservative: 
            return self.softmax(x)
        return x