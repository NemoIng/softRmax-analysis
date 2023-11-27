"""
Created on Fri Mar 19 10:09:59 2021

@author: ziqi, edited by Nemo Ingendaa

"""
import torch
import torch.nn as nn

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
    def __init__(self, device, num_classes, num_features, function, a):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.function = function
        if function == 'cons':
            self.softmax = conservative_softmax(num_classes, a)
        elif function == 'softRmax':
            self.softmax = softRmax(num_classes, device)
        elif function == 'softmax':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)