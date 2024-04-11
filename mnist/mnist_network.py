"""
@author: ziqi, edited by Nemo Ingendaa
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
    def __init__(self, device, num_classes, function, kernel_size):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=kernel_size)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64, num_classes)
        
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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4_drop(self.conv4(x)))

        x = x.view(-1, 64)
        x = self.fc1(x)

        if self.conservative: 
            return self.softmax(x)
        return x