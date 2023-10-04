"""
Created on Fri Mar 19 10:09:59 2021

@author: ziqi, edited by Nemo Ingendaa

"""
import torch
import torchvision
import torch.nn as nn

class Triangular(nn.Module):
    def __init__(self):
         super(Triangular, self).__init__()
    def forward(self, input):
        out = nn.functional.relu(input + 1) - 2 * nn.functional.relu(input) + nn.functional.relu(input - 1)
        return out

def convert_relu_to_triangular(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Triangular())
        else:
            convert_relu_to_triangular(child)
            
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
    def __init__(self, device, num_classes, function, a, triangular):
        super(Net, self).__init__()

        self.net = torchvision.models.vgg16(pretrained=True)
        # self.net = torchvision.models.vgg19(pretrained=True)
        self.net.classifier[6] = nn.Linear(4096,num_classes)
        self.function = function
        if triangular:
            convert_relu_to_triangular(self.net)
        if function == 'cons_softmax':
            self.softmax = conservative_softmax(num_classes, a)
        elif function == 'softRmax':
            self.softmax = softRmax(num_classes, device)
        elif function == 'softmax':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.net(x)
        x = self.softmax(z)
        return x, z
