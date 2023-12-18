"""
@author: NemoIng (Nemo ingendaa)

Using/Inspiration code from:
- https://github.com/ziqiwangsilvia/attack 
- https://pytorch.org/tutorials/beginner/fgsm_tutorial.html 
- https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html 
"""

import numpy as np
import torch
import torch.nn as nn 
import torch.utils.data as td
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from dataset import prepare_dataset
from network import Net

# Attack parameters
attack_type = 'bim'
# testing_eps = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
testing_eps = [0, 0.1, 0.2, 0.3]
class_to_plot = 'bird'
bim_iters = 10 # num of steps

# Network parameters
function = 'softmax'
triangular = False
conservative_a = 0.2

# Data parameters
normalized = False
num_classes = 10
train_all = True
train_index = [3, 7]
test_all = True
test_index = [3, 7]

# Test parameters
test_batch_size = 256

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# When using apple silicon GPU:
device = torch.device("mps")

# For NVIDIA GPUs
# device = torch.device("cuda")

# The CPU is used in the testing phase (performance reasons)
device2 = torch.device("cpu")

def main():
    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test', normalized) 
    testloader = td.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    net = Net(device, num_classes, function, conservative_a, triangular).to(device)
    net.load_state_dict(torch.load(path))

    print(f'Loaded the {function} network (norm:{normalized})')
    net.eval()

    adv_data = create_data(net, testloader, testing_eps)

    # We move the network to the CPU for the testing phase 
    net = Net(device2, num_classes, function, conservative_a, triangular).to(device2)
    net.load_state_dict(torch.load(path))
    acc_per_eps = []
    acc_class_per_eps = []

    for i, eps in enumerate(testing_eps):
        print(f'Testing eps: {eps}')
        acc, classes_acc = test_model_accuracy(net, adv_data[i], eps)
        acc_per_eps.append(acc)
        acc_class_per_eps.append(classes_acc)

    plot_acc_graph(acc_per_eps)
    plot_acc_class_per_eps(acc_class_per_eps)

def create_data(net, testloader, testing_eps):
    data = []
    for eps in testing_eps:
        print(f'\nCreating adversarial ({attack_type}) data for eps: {eps}')
        adversarial_data = []

        for images, labels in testloader:
            loss_func = nn.CrossEntropyLoss()
            if eps == 0:
                adv_images, noise = images, None
            elif attack_type == 'fgsm':
                adv_images, noise = fgsm_attack(net, loss_func, images, labels, eps)
            else:
                adv_images, noise = bim_attack(net, loss_func, images, labels, eps)
            adversarial_data.append((images, adv_images, labels, noise))

        data.append(adversarial_data)
    return data

def fgsm_attack(net, loss_func, images, labels, eps) :
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
            
    outputs = net(images)
    loss = loss_func(outputs, labels)
    net.zero_grad()

    loss.backward()
    data_grad = images.grad.data

    noise = eps*data_grad.sign()
    attack_images = images + noise

    # If the model is trained on non normalized data
    # Clamp the images between 0 and 1, to make sure they have the same range as the orginals
    if not normalized:
         attack_images = torch.clamp(attack_images, 0, 1)
    noise = noise.clamp(0, 1).cpu().detach().numpy()
    return attack_images, noise

def bim_attack(net, loss_func, images, labels, eps):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    org_images = images.clone().detach()
    noise = None
    alpha = eps / bim_iters
    for i in range(bim_iters):    
        images.requires_grad = True
        outputs = net(images)

        loss = loss_func(outputs, labels)
        net.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        noise = alpha*data_grad.sign() 

        attack_images = images + alpha*data_grad.sign()

        a = torch.clamp(org_images - eps, min=0)
        b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
        c = (b > attack_images+eps).float()*(attack_images+eps) + (attack_images+eps >= b).float()*b

        images = torch.clamp(c, max=255).detach()

    noise = torch.clamp(images - org_images, 0, 1)
    noise = noise.cpu().detach().numpy()
    return images, noise

def test_model_accuracy(net, adv_data, eps):
    if not train_all:
        # class_label = train_index.index(class_to_plot)
        class_label = 'car'        
    else:
        class_label = classes.index(class_to_plot)
    printed = False

    acc = 0
    nb = 0
    class_correct = [0.0 for _ in range(num_classes)]
    class_total = [0.0 for _ in range(num_classes)]

    for images, adv_images, labels, noise in adv_data:
        # We move the data to the CPU for the testing phase 
        images = images.to(device2)
        adv_images = adv_images.to(device2)
        labels = labels.to(device2)
        if not printed and eps != 0:
            for i, label in enumerate(labels.cpu().detach().numpy()):
                if label == class_label:
                    plot_attack(images[i], adv_images[i], noise[i,0], eps, class_to_plot)
                    printed = True
                    break
            # imshow(torchvision.utils.make_grid(attack_images[:16].cpu().data, normalize=True), eps)
            # printed = True

        nb = nb + len(adv_images)
        outputs = net(adv_images)
        _, predicted = torch.max(outputs, 1)
        acc = acc + (predicted - labels).nonzero().size(0)
        c = (predicted == labels).squeeze()
        for i in range(len(adv_images)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    classes_acc = []
    for i in range(num_classes):
        classes_acc.append(round(class_correct[i] / (1e-8 + class_total[i]), 2))
        print(f'Accuracy of {classes[i]}: {round(100 * (class_correct[i] / (1e-8 + class_total[i])), 2)}%')

    test_acc = (nb - acc) / nb
    print(f'Total accuracy: {round(100 * float(test_acc), 3)}%\n')
    return test_acc, classes_acc

def plot_attack(org_images, adv_images, noise, eps, label):
    if eps == 0:
        return
    
    if normalized:
        # Denormalize the images for plotting
        denormalize = transforms.Compose([
            transforms.Normalize(mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], std=[1/0.2023, 1/0.1994, 1/0.2010])
        ])
        org_images = denormalize(org_images)
        adv_images = denormalize(adv_images)
    
    org_images = org_images.detach().numpy()
    adv_images = adv_images.detach().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(org_images.transpose(1, 2, 0))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adv_images.transpose(1, 2, 0))
    plt.title(f"Adversarial Image ({eps})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(noise)
    plt.title(f"Added Noise")
    plt.axis('off')

    if normalized: 
        plt.savefig(f'figures/{attack_type}/{function}/{num_classes}_{eps}_{label}.png', dpi=200)
    else: 
        plt.savefig(f'figures/no_norm/{attack_type}/{function}/{num_classes}_{eps}_{label}.png', dpi=200)
    plt.close()

def plot_acc_graph(acc_per_eps):
    plt.figure(figsize=(8,5))
    plt.plot(testing_eps, acc_per_eps, '*-')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    step = testing_eps[1] - testing_eps[0]
    plt.xticks(np.arange(0, max(testing_eps)+step, step=step))
    plt.title("Accuracy vs Epsilon")
    plt.suptitle(f'{function} (CIFAR {num_classes})')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    if normalized:
        plt.savefig(f'figures/{attack_type}/{function}/acc_eps_{num_classes}.png', dpi=200)
    else: 
        plt.savefig(f'figures/no_norm/{attack_type}/{function}/acc_eps_{num_classes}.png', dpi=200)
    plt.close()

def plot_acc_class_per_eps(acc_class_per_eps):
    num_classes = len(acc_class_per_eps[0])  # Assuming acc_per_class_per_eps is a 2D array
    plt.figure(figsize=(10, 6))

    for class_id in range(num_classes):
        class_accuracies = [eps[class_id] for eps in acc_class_per_eps]
        plt.plot(testing_eps, class_accuracies, label=f'{classes[class_id]}')

    plt.yticks(np.arange(0, 1.1, step=0.1))
    step = testing_eps[1] - testing_eps[0]
    plt.xticks(np.arange(0, max(testing_eps)+step, step=step))
    plt.title("Accuracy per Class vs Epsilon")
    plt.suptitle(f'{function} (CIFAR {num_classes})')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)
    if normalized:
        plt.savefig(f'figures/{attack_type}/{function}/acc_class_per_eps_{num_classes}.png', dpi=200)
    else: 
        plt.savefig(f'figures/no_norm/{attack_type}/{function}/acc_class_per_eps_{num_classes}.png', dpi=200)
    plt.close()

def imshow(img, eps):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'figures/{attack_type}/{function}/{num_classes}_{eps}.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    if normalized:
        if train_all:
            path = f'runs/{num_classes}_classes/best_{function}_net_checkpoint.pt'
        else:
            path = f'runs/{train_index}_classes/best_{function}_net_checkpoint.pt'
    else: 
        if train_all:
            path = f'runs/no_norm/{num_classes}_classes/best_{function}_net_checkpoint.pt'
        else:
            path = f'runs/no_norm/{train_index}_classes/best_{function}_net_checkpoint.pt'
    main()

