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

# Network parameters
function = 'softRmax'
lr = 1e-2
conservative_a = 0.2

# Data parameters
num_classes = 5
num_class_samples = 10000
mean = [-4, -1, 2, 5, 8]
sigma = 0.6

# Train-Test parameters
num_epoch = [2,5,10,20,50]
num_tries = 5
train_batch_size = 512
test_batch_size = 512
test_size = 0.5

device = torch.device("cpu")

def main():
    path = 'data/'
    if not os.path.exists(f'{path}{num_classes}_{num_class_samples}_{sigma}_{str(mean[:num_classes])}.hkl'):
        generate_train_test(num_classes, num_class_samples, test_size, mean, sigma)

    data = hkl.load(f'{path}{num_classes}_{num_class_samples}_{sigma}_{str(mean[:num_classes])}.hkl')
    x_train, x_test, y_train, y_test = data.values()

    X_train = torch.tensor(x_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    # plot_training_data(X_train, y_train)

    train_dataset = td.TensorDataset(X_train, y_train)
    train_loader = td.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    for epoch in num_epoch:
        best_acc = 0
        print(f"Training with {epoch} epoch")
        for i in range(num_tries):
            # For each try, we reinitialize the network
            net = Net(device, num_classes, function, conservative_a)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)

            train(train_loader, net, criterion, optimizer, epoch)
            acc = test(net, X_test, y_test)

            if acc > best_acc:
                best_acc = acc
            plot_classification(net, X_test, y_test, epoch, acc, i)
        print(f"Best Accuracy: {best_acc}")


def train(train_loader, net, criterion, optimizer, epoch):
    for i in range(epoch):
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {i+1}, loss: {running_loss/len(train_loader)}")

def test(net, X_test, y_test):
    net.eval()
    with torch.no_grad():
        test_outputs = net(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        return accuracy

def plot_training_data(X, y, title="Data Distribution"):
    bins = int(np.sqrt(num_class_samples*num_classes))

    for class_id in range(num_classes):
        class_data = X[y == class_id]
        plt.hist(class_data, bins, alpha=0.5, label=f'Class {class_id}')

    plt.title(title)
    plt.legend()
    plt.show()

def plot_classification(net, X_test, y_test, epoch, acc, num):
    fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True) 
    ax1.grid()
    output = net(X_test)
    max_values, _ = output.max(1)
    confidence = max_values.cpu().detach().numpy()
    colorlist = ["red","violet","blue","green", "purple"]
    cmap = LinearSegmentedColormap.from_list("", colorlist[:num_classes])

    ax1.scatter(X_test, confidence, c=y_test, cmap=cmap, s=5)
    ax1.set_yticks(np.arange(0.2, 1.1, 0.1))
    ax1.set_title('Confidence of Model')
    ax1.set_ylabel('Confidence')

    bins = int(np.sqrt(num_class_samples*num_classes))

    for class_id in range(num_classes):
        class_data = X_test[y_test == class_id]
        ax2.hist(class_data, bins, alpha=0.5, label=f'Class {class_id}', color = colorlist[class_id])
        
    ax2.set_title('Training data distribution')
    ax2.set_xlabel('Input Data')
    ax2.set_ylabel('Frequency')

    fig.suptitle(f'{function}, acc: {round(acc,3)}')
    if not os.path.isdir(f'figures/{num_classes}_{num_class_samples}_{epoch}'):
        os.mkdir(f'figures/{num_classes}_{num_class_samples}_{epoch}')
    plt.savefig(f'figures/{num_classes}_{num_class_samples}_{epoch}/{num_classes}_{num_class_samples}_{sigma}_{str(mean[:num_classes])}_{function}_{num}.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    main()