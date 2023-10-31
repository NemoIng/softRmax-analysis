import torch
import torch.nn as nn 
import torch.utils.data as td
from torch.autograd import Variable
from torchvision import transforms
from matplotlib import pyplot as plt
import seaborn as sns

from dataset import prepare_dataset
from network import Net

# FGSM Attack parameters
testing_eps = [0, 0.1, 0.3]
nr_plots_shown = 1

# Network parameters
function = 'softmax'
num_epoch = 3
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
    path = f'runs/best_{function}_{num_epoch}_net_checkpoint.pt'

    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test') 
    testloader = td.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    
    net = Net(device, num_classes, function, conservative_a, kernel_size)
    net.load_state_dict(torch.load(path))

    print(f'Loaded the {function}-{num_epoch} network')

    net.eval()

    for eps in testing_eps:
        plots = 0
        acc = 0
        nb = 0
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        for images, labels in testloader:
            images = Variable(images).to(device)
            labels = Variable(labels.squeeze()).to(device) 
            loss = nn.CrossEntropyLoss()

            adv_images = fgsm_attack(net, loss, images, labels, eps).to(device)
            perturbed_adv_images = transforms.Normalize((0.1307,), (0.3081,))(adv_images)
            if plots < nr_plots_shown:
                plot_fgsm_attack(images[0], perturbed_adv_images[0], eps)
                plots += 1

            nb = nb + len(images)
            outputs = net(perturbed_adv_images)
            
            _, predicted = torch.max(outputs, 1)
            acc = acc + (predicted - labels).nonzero().size(0)
            c = (predicted == labels).squeeze()
            for i in range(len(images)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        
        test_acc = (nb - acc)/nb 
        print(f'Accuracy for {eps} eps: {round(100 * float(test_acc), 3)}%')

        for i in range(num_classes):
            print(f'accuracy of {classes[i]} : {round(100 * (class_correct[i] / (1e-8 + class_total[i])),2)}%')

def denorm(batch, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def fgsm_attack(net, loss, images, labels, eps) :
    images.requires_grad = True
            
    outputs = net(images)
    loss = loss(outputs, labels)
    net.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    images_denorm = denorm(images)
    
    attack_images = images_denorm + eps*data_grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images

def plot_fgsm_attack(org_image, adv_image, eps):
    if eps == 0:
        return
    original_image_np = org_image.cpu().detach().numpy()
    adversarial_image_np = adv_image.cpu().detach().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np.transpose(1, 2, 0), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_image_np.transpose(1, 2, 0), cmap='gray')
    plt.title(f"Adversarial Image ({eps})")
    plt.axis('off')

    plt.savefig(f'figures/fgsm/{function}/{num_classes}_{eps}.png', dpi=200)

if __name__ == '__main__':
    main()

