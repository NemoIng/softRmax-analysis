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

class Net2(nn.Module):
    def __init__(self, device, num_classes, function, a, kernel_size):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64, num_classes)
        
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

        x = x.view(-1, 64)
        x = self.fc1(x)
        return self.softmax(x)
    
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