from matplotlib.colors import LogNorm
import torch
import torch.utils.data as td
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sns

from dataset import prepare_dataset
from network import Net

# Network parameters
function = 'softmax'
num_epoch = 20
kernel_size = 3
conservative_a = 0.2

# Data parameters
num_classes = 10
train_all = True
train_index = [3, 7]
test_all = True
test_index = [3, 7]

# Test parameters
test_batch_size = 128

classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

# When using apple silicon GPU:
device = torch.device("cpu")

# When using other chip architecture:
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    if train_all:
        path = f'runs/{num_classes}_classes/best_{function}_{num_epoch}_net_checkpoint.pt'
    else:
        path = f'runs/{train_index}_classes/best_{function}_{num_epoch}_net_checkpoint.pt'

    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test') 
    testloader = td.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    
    net = Net(device, num_classes, function, conservative_a, kernel_size)
    net.load_state_dict(torch.load(path))
    print(f'Loaded the {function} network')

    class_acc(testloader, net)
    confusion_matrix(testloader, net)

def class_acc(loader, net):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for data in loader:
            images, labels = data
            output = net(images)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            total_correct += c.sum().item()
            total_samples += labels.size(0)
            for i in range(len(images)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]} : {round(100 * class_correct[i] / (1e-8 + class_total[i]),2)}%')
    
    total_accuracy = 100 * total_correct / total_samples
    print(f'Total Accuracy: {round(total_accuracy, 2)}%')

def confusion_matrix(loader, net):
    confusion_matrix = MulticlassConfusionMatrix(num_classes)

    with torch.no_grad():
        for data in loader:
            images, labels = data
            output = net(images)
            _, predicted = torch.max(output, 1)
            confusion_matrix.update(predicted, labels)

    confusion_matrix = confusion_matrix.compute().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Reds', norm=LogNorm())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'figures/conf_matrix/{num_classes}_{function}_{num_epoch}.png', dpi=200)

if __name__ == '__main__':
    main()