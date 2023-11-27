import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td

from network import Net

# Network parameters
lr = 1e-2
conservative_a = 0.2

# Data parameters
num_classes = 2
num_class_samples = 500
train_batch_size = 256
device = torch.device("cpu")

num_epoch = [1,2,5,10,20,30]
num_tries = 5
test_size = 0.3
data_type = 'linear'

def main():
    # Generate synthetic data
    if data_type == 'blob':
        data = make_blobs(n_samples=num_class_samples*num_classes, cluster_std=0.7, centers=num_classes, center_box=(-5,5), random_state=6)
    elif data_type == 'moon':
        data = make_moons(n_samples=num_class_samples*num_classes, noise=0.2, random_state=None, shuffle=False)
    elif data_type == 'circle':
        data = make_circles(n_samples=num_class_samples*num_classes, noise=0.1, factor=0.4, random_state=1)
    elif data_type == 'linear':
        X, y = make_classification(
            n_samples=num_class_samples*num_classes, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, class_sep=1.7
        )
        data = (X, y)
    else:
        print('incorrect data type')
        return

    # Convert to PyTorch tensors
    X = torch.tensor(data[0], dtype=torch.float32)
    y = torch.tensor(data[1], dtype=torch.long)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    train_dataset = td.TensorDataset(X_train, y_train)
    train_loader = td.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    for epoch in num_epoch:
        best_acc_softmax, best_net_softmax = train_and_test_network('softmax', train_loader, X_test, y_test, epoch)
        best_acc_softRmax, best_net_softRmax = train_and_test_network('softRmax', train_loader, X_test, y_test, epoch)

        print(f"softmax acc: {best_acc_softmax}")
        print(f"softRmax acc: {best_acc_softRmax}")

        plot_decision_boundaries(best_net_softmax, best_net_softRmax, X_test, y_test, epoch, f"Softmax Decision Boundaries ({round(best_acc_softRmax,3)})", f"SoftRmax Decision Boundaries ({round(best_acc_softmax,3)})")

def train_and_test_network(function, train_loader, X_test, y_test, epoch):
    best_acc = 0
    best_net = None

    for i in range(num_tries):
        # Train and test the network
        net = Net(device, num_classes, 2, function, conservative_a)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train(train_loader, net, criterion, optimizer, epoch)
        acc = test(net, X_test, y_test)

        if acc > best_acc:
            best_acc = acc
            best_net = net

    return best_acc, best_net

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
        if i == epoch-1:
            print(f"epoch {i+1}, loss: {running_loss/len(train_loader)}")

def test(net, X_test, y_test):
    net.eval()
    with torch.no_grad():
        test_outputs = net(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        return accuracy

def plot_decision_boundaries(net1, net2, X, y, epoch, title1, title2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot decision boundary for net1
    ax = axs[0]
    ax.set_title(title1)
    plot_decision_boundary(ax, X, y, net1)

    # Plot decision boundary for net2
    ax = axs[1]
    ax.set_title(title2)
    plot_decision_boundary(ax, X, y, net2)

    path = f'figures/{data_type}_{num_classes}_{num_class_samples}'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + f'/decision_comparison_{epoch}.png', dpi=300)
    plt.close()

def plot_decision_boundary(ax, X, y, net, steps=1000, cmap_range=('red', 'blue')):
    cmap = LinearSegmentedColormap.from_list("", [cmap_range[0], 'white', cmap_range[1]])

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    Z = net(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).detach().numpy()
    y1 = Z.T[0].reshape(xx.shape[0],xx.shape[0])

    ax.contourf(xx, yy, y1, cmap=cmap, alpha=0.5)
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), cmap=cmap.reversed(), lw=0)

if __name__ == '__main__':
    main()
