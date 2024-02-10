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

class Net(nn.Module):
    def __init__(self, device, num_classes, function):
        super(Net, self).__init__()
        self.net = torchvision.models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(4096,num_classes)
        self.function = function
        if function == 'softRmax':
            self.softmax = softRmax(num_classes, device)
            self.conservative = True
        else:
            self.conservative = False

    def forward(self, x):
        z = self.net(x)

        if self.conservative: 
            return self.softmax(z)
        return z
