import time
from matplotlib import pyplot as plt
import torch
from deepview import DeepView
import numpy as np
from torch.autograd import Variable
import torch.utils.data as td

from fashion_dataset import prepare_dataset
from fashion_network import Net

# When using apple silicon GPU:
device = torch.device("mps")

# For NVIDIA GPUs
# device = torch.device("cuda")

# Deepview parameters
batch_size = 200
max_samples = 500
data_shape = (1, 28, 28)
min_dist = 0.6
spread = 0.6
lam = 0.6

def plot_decision_boundary(net=None, num_classes=10, epoch=15, function='softmax', n_samples=200, 
                           data=None, fig_path=None, index=[3,7]):
    train_index = index
    test_index = index
    if num_classes == 10:
        train_all = True
        test_all = True
        classes = np.arange(num_classes)
    else: 
        train_all = False
        test_all = False
        classes = train_index

    if net == None:
        if train_all:
            path = f'runs/{num_classes}_classes/best_{function}_net_checkpoint.pt'
        else:
            path = f'runs/{train_index}_classes/best_{function}_net_checkpoint.pt'
        net = Net(device, num_classes, function).to(device)
        net.load_state_dict(torch.load(path))
    else:
        net = net.to(device)
    print(f'Loaded the {function}-{num_classes} network')
    net.eval()

    if data == None:
        data = prepare_dataset(train_all, train_index, test_all, test_index, 'test')
        sample_indices = np.arange(len(data))
        sample_ids = np.random.choice(sample_indices, size=200, replace=False)
        # sample_ids = sample_indices[:n_samples]
        X = np.array([ data[i][0].numpy() for i in sample_ids ])
        Y = np.array([ data[i][1] for i in sample_ids ])
    else:
        X = data[0]
        Y = data[1]

    if fig_path == None:
        if train_all:
            fig_path = f'figures/dec_bound/{function}/{epoch}_{n_samples}.png'
        else:
            fig_path = f'figures/dec_bound/{function}/{epoch}_{train_index}_{n_samples}.png'

    def pred_wrapper(x):
        with torch.no_grad():
            x = np.array(x, dtype=np.float32)
            tensor = torch.from_numpy(x).to(device)
            probabilities = net(tensor).cpu().numpy()
        return probabilities
    
    test_data = td.TensorDataset(torch.tensor(X), torch.tensor(Y))
    testloader = td.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=1)
    total_acc = class_acc(testloader, net, num_classes, classes)

    title = f'{function} - Fashion-MNIST ({total_acc})'
    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, 
                        data_shape, lam=lam, title=title, min_dist=min_dist, spread=spread, n=10,
                        brightness=0.7)

    # plt.imshow(X[0].transpose([1, 2, 0]))
    # plt.show()

    t0 = time.time()
    deepview.add_samples(X, Y)
    deepview.save(fig_path)
    deepview.close()

    print('Time to calculate visualization for %d samples: %.2f sec' % (n_samples, time.time() - t0))

def test(test_loader, net):
    net.eval()
    Acc_y = 0
    nb = 0
    for i, data in enumerate(test_loader):
        X, Y = data 
        X = Variable(X).to(device)
        Y = Variable(Y.squeeze()).to(device) 
        nb = nb + len(X)

        outputs = net(X)
        Acc_y = Acc_y + (outputs.argmax(1) - Y).nonzero().size(0)
  
    test_acc = (nb - Acc_y) / nb 
    return test_acc

def class_acc(loader, net, num_classes, classes):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for data in loader:
            images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels.squeeze()).to(device) 
            output = net(images)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            total_correct += c.sum().item()
            total_samples += labels.size(0)
            for i in range(len(images)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        print(f'Number of misclassified {classes[i]} samples: {int(class_total[i] - class_correct[i])}/{int(class_total[i])}')
    
    total_accuracy = 100 * total_correct / total_samples
    print(f'Total Accuracy: {round(total_accuracy, 2)}%')
    return total_accuracy

if __name__ == '__main__':
    plot_decision_boundary()
