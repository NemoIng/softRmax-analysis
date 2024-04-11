"""
@author: ziqi, edited by Nemo Ingendaa
"""
import torch
import torchvision
import torch.nn as nn
    
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

        if num_classes == 100:
            self.net = torchvision.models.resnet50(pretrained=True)
            self.net.fc = nn.Linear(2048, num_classes)
        else:
            self.net = torchvision.models.vgg16(pretrained=True)
            self.net.classifier[6] = nn.Linear(4096,num_classes)
        self.function = function
        if function == 'softRmax':
            self.softmax = softRmax(num_classes, device)
            self.conservative = True
        elif function == 'cons':
            self.softmax = conservative_softmax_monotone(num_classes, 0.01)
            self.conservative = True
        else:
            self.conservative = False

    def forward(self, x):
        z = self.net(x)

        if self.conservative: 
            return self.softmax(z)
        return z
