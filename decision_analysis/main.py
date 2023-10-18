import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
import matplotlib.pyplot as plt
import hickle as hkl
from torch import optim
from matplotlib.colors import LinearSegmentedColormap

from dataset import generate_train_test
from network import Net

num_classes = 2
num_class_samples = 1000
test_size = 0.3
mean = [1,2,3,4]
sigma = 0.4

hps = {'function': 'softmax',
       'train_all': True,
       'train_index': [0,1],
       'test_all': True,
       'test_index': [0,1],
       'train_batch_size': 16,
       'test_batch_size': 128,
       'epoch': 50,
       'lr': 5e-3,
       'weight_decay': 5e-6,
       'print_freq':1,
       'conservative_a': 0.2,
       'exp': 0}

# When using apple silicon GPU:
device = torch.device("cpu")

# When using other chip architecture:
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    net = Net(device, num_classes, hps['function'], hps['conservative_a'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    path = 'data/'
    if not os.path.exists(f'{path}{num_classes}_{num_class_samples}_{test_size}_{str(mean[:num_classes])}_{sigma}.hkl'):
        generate_train_test(num_classes, num_class_samples, test_size, mean, sigma)

    data = hkl.load(f'{path}{num_classes}_{num_class_samples}_{test_size}_{str(mean[:num_classes])}_{sigma}.hkl')
    x_train, x_test, y_train, y_test = data.values()
    
    X_train = torch.tensor(x_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = td.TensorDataset(X_train, y_train)
    train_loader = td.DataLoader(train_dataset, batch_size=64, shuffle=True)

    train(train_loader, net, criterion, optimizer)

    # plot_training_data(X_train, y_train)
    plot_classification(net, X_train, y_train)

    test(net, X_test, y_test)

def train(train_loader, net, criterion, optimizer):
    for epoch in range(hps['epoch']):
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"epoch {epoch+1}, loss: {running_loss/len(train_loader)}")

def test(net, X_test, y_test):
    net.eval()
    with torch.no_grad():
        test_outputs = net(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Test Accuracy: {accuracy}")

def plot_training_data(X, y, title="Data Distribution"):
    bins = int(np.sqrt(num_class_samples*num_classes))

    for class_id in range(num_classes):
        class_data = X[y == class_id]
        plt.hist(class_data, bins, alpha=0.5, label=f'Class {class_id}')

    plt.title(title)
    plt.legend()
    plt.show()

def plot_classification(net, X_train, y_train):
    _, axis = plt.subplots(2) 

    output = net(X_train)
    max_values, _ = output.max(1)
    confidence = max_values.cpu().detach().numpy()
    colorlist = ["red","violet","blue","green"]
    cmap = LinearSegmentedColormap.from_list("", colorlist[:num_classes])

    axis[0].scatter(X_train, confidence, c=y_train, cmap=cmap)
    bins = int(np.sqrt(num_class_samples*num_classes))

    for class_id in range(num_classes):
        class_data = X_train[y_train == class_id]
        axis[1].hist(class_data, bins, alpha=0.5, label=f'Class {class_id}'
                     , color = colorlist[class_id])

    plt.show()

if __name__ == '__main__':
    main()