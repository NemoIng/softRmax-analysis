"""
@author: NemoIng (Nemo ingendaa)

Using/Inspiration code from:
- https://github.com/ziqiwangsilvia/attack 
- https://pytorch.org/tutorials/beginner/fgsm_tutorial.html 
- https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html 
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn 
import torch.utils.data as td
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

from dataset import prepare_dataset
from network import Net
from mnist_bound import plot_decision_boundary

# All attack parameters
test_function = 'softRmax'
attack_type = 'average'
# testing_eps = [0.0, 0.8]
testing_eps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# testing_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
plot_eps = [0.0, 0.8]
class_to_plot = 7

# Average attack parameters
target_class = 0
attack_all_classes = True
adv_class = 1 # if all_classes_attack is False, specify which class
dec_bound = False # plot decision boundary yes/no

# FGSM/BIM attack parameters
adv_function = 'softRmax'
bim_iters = 10 # number of fgsm steps for BIM

# Network parameters
kernel_size = 3
conservative_a = 0.2 # parameter for conservative softmax

# Data parameters
normalized = True
num_classes = 10
train_all = True
train_index = [3, 7]
test_all = True
test_index = [3, 7]
test_batch_size = 300
 
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# When using apple silicon GPU:
device = torch.device("mps")

# For NVIDIA GPUs
# device = torch.device("cuda")

# The CPU is used in the testing phase (performance reasons)
device2 = torch.device("cpu")

def main():
    testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test', normalized) 
    testloader = td.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    net = Net(device, num_classes, adv_function, conservative_a, kernel_size).to(device)
    net.load_state_dict(torch.load(path))

    if not attack_type == "average":
        print(f'Loaded the {adv_function}-{num_classes} network')
    net.eval()

    adv_data = create_data(net, testloader, testing_eps)

    # We create the testing network and move it to the CPU along with the data
    net2 = Net(device2, num_classes, test_function, conservative_a, kernel_size)
    net2.load_state_dict(torch.load(path2))
    print(f'Loaded the {test_function}-{num_classes} network')
    acc_per_eps = []
    acc_class_per_eps = []

    for i, eps in enumerate(testing_eps):
        if dec_bound and eps in plot_eps:
            plot_dec_bound(eps, adv_data[i], 200)

        print(f'Testing eps: {eps}')
        acc, classes_acc = test_model_accuracy(net2, adv_data[i], eps)
        acc_per_eps.append(acc)
        acc_class_per_eps.append(classes_acc)

    if attack_type == "average" and not attack_all_classes:
        plot_acc_graph(acc_class_per_eps, testing_eps)
    else:
        plot_acc_graph(acc_per_eps, testing_eps)
        plot_acc_class_per_eps(acc_class_per_eps, testing_eps)

def create_data(net, testloader, testing_eps):
    data = []
    for eps in testing_eps:
        print(f'\nCreating adversarial ({attack_type}) data for eps: {eps}')
        adversarial_data = []

        for images, labels in testloader:
            loss_func = nn.CrossEntropyLoss()
            if eps == 0:
                adversarial_data.append((images, images, labels, None))
            elif attack_type == 'fgsm':
                adv_images, noise = fgsm_attack(net, loss_func, images, labels, eps)
                adversarial_data.append((images, adv_images, labels, noise))
            elif attack_type == 'bim':
                adv_images, noise = bim_attack(net, loss_func, images, labels, eps)
                adversarial_data.append((images, adv_images, labels, noise))
            else: 
                if not attack_all_classes:
                    images, adv_images, noise, labels = average_attack(images, labels, eps, adv_class)
                    adversarial_data.append((images, adv_images, labels, noise))
                else:
                    adv_data = []
                    for i in range(num_classes):   
                        new_images, adv_images, noise, new_labels = average_attack(images, labels, eps, classes[i])
                        adv_data.append((new_images, adv_images, new_labels, noise))
                    for i in adv_data:
                        adversarial_data.append(i)

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
        data_grad = images.grad.data # type: ignore

        noise = alpha*data_grad.sign() 

        attack_images = images + alpha*data_grad.sign()

        a = torch.clamp(org_images - eps, min=0)
        b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
        c = (b > attack_images+eps).float()*(attack_images+eps) + (attack_images+eps >= b).float()*b

        images = torch.clamp(c, max=255).detach()

    noise = torch.clamp(images - org_images, 0, 1)
    noise = noise.cpu().detach().numpy()
    return images, noise

def average_attack(images, labels, eps, adv_class):
    target_class_images = []
    adv_class_images = []
    other_class_images = []
    target_labels = []
    adv_labels = []
    other_labels = []

    for img, lab in zip(images, labels):
        if lab == target_class:
            target_class_images.append(img)
            target_labels.append(lab)
        if lab == adv_class:
            adv_class_images.append(img)
            adv_labels.append(lab)
        else: 
            other_class_images.append(img)
            other_labels.append(lab)

    # Change the lists to tensors
    adv_class_images = torch.stack(adv_class_images, dim=0)
    target_class_images = torch.stack(target_class_images, dim=0)
    other_class_images = torch.stack(other_class_images, dim=0)

    adv_labels = torch.stack(adv_labels, dim=0)
    target_labels = torch.stack(target_labels, dim=0)
    other_labels = torch.stack(other_labels, dim=0)

    # create the average image samples
    avg_adv_class_image = torch.mean(adv_class_images, dim=0)
    avg_target_class_image = torch.mean(target_class_images, dim=0)

    addition = eps*(avg_target_class_image - avg_adv_class_image).sign()

    attack_images = adv_class_images + addition

    # If the model is trained on non normalized data
    # Clamp the images between 0 and 1, to make sure they have the same range as the orginals
    if not normalized:
        attack_images = torch.clamp(attack_images, 0, 1)

    # concatenate the tensors back together
    if not attack_all_classes:
        attack_images = torch.cat((attack_images, target_class_images, other_class_images), dim=0)
        org_images = torch.cat((adv_class_images, target_class_images, other_class_images), dim=0)
        updated_labels = torch.cat((adv_labels, target_labels, other_labels), dim=0)
        return org_images, attack_images, addition, updated_labels
    
    return images, attack_images, addition, adv_labels

def test_model_accuracy(net, adv_data, eps):
    net = net.to(device2)
    if not train_all:
        class_label = train_index.index(class_to_plot)        
    else:
        class_label = class_to_plot
    printed = False

    acc = 0
    nb = 0
    class_correct = [0.0 for _ in range(num_classes)]
    class_total = [0.0 for _ in range(num_classes)]

    for images, attack_images, labels, noise in adv_data:
        images = images.to(device2)
        attack_images = attack_images.to(device2)
        if not printed and eps != 0:
            for i, label in enumerate(labels.cpu().detach().numpy()):
                if label == class_label:
                    if attack_type == "average":
                        plot_attack(images[i], attack_images[i], noise[0], eps, class_to_plot)
                    else:
                        plot_attack(images[i], attack_images[i], noise[i,0], eps, class_to_plot)
                    printed = True
                    break

        nb = nb + len(attack_images)
        outputs = net(attack_images)
        _, predicted = torch.max(outputs, 1)

        acc = acc + (predicted - labels).nonzero().size(0)
        c = (predicted == labels).squeeze()
        for i in range(len(attack_images)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    classes_acc = []
    for i in range(num_classes):
        classes_acc.append(round(class_correct[i] / (1e-8 + class_total[i]), 2))
        print(f'Accuracy of {classes[i]}: {round(100 * (class_correct[i] / (1e-8 + class_total[i])), 2)}%')

    test_acc = (nb - acc) / nb
    print(f'Total accuracy: {round(100 * float(test_acc), 3)}%\n')
    if attack_type == "average" and not attack_all_classes:
        return test_acc, classes_acc[adv_class]
    return test_acc, classes_acc

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def plot_dec_bound(eps, data, n_samples):
    net = Net(device, num_classes, test_function, conservative_a, kernel_size).to(device)
    net.load_state_dict(torch.load(path2))

    if attack_type == "average":   
        if attack_all_classes:
            attack = f"{num_classes}_all_to_{target_class}_{test_function}"
        else: 
            attack = f"{num_classes}_{adv_class}_to_{target_class}_{test_function}"
    else:
        attack = f"{adv_function}_{test_function}"

    if normalized:
        fig_path = f'figures/{attack_type}/dec_bound/{attack}'
        create_path(fig_path)
        if train_all:
            fig_path += f'/{n_samples}_{eps}.png'
        else:
            fig_path += f'/{train_index}_{n_samples}_{eps}.png'
    else:
        fig_path = f'figures/no_norm/{attack_type}/dec_bound/{attack}'
        create_path(fig_path)
        if train_all:
            fig_path += f'/{n_samples}_{eps}.png'
        else:
            fig_path += f'/{train_index}_{n_samples}_{eps}.png'
    
    if attack_all_classes:
        data_X = torch.tensor([])
        data_Y = torch.tensor([])
        for i in range(num_classes):
            X = data[i][1].cpu().detach()[:int(n_samples/num_classes)]
            Y = data[i][2][:int(n_samples/num_classes)]
            data_X = torch.cat((data_X, X))
            data_Y = torch.cat((data_Y, Y)).int()
        data = [np.array(data_X),np.array(data_Y)]
    else:
        data = [np.array(data[0][1]),np.array(data[0][2])]
    plot_decision_boundary(net, num_classes, 15, normalized, test_function, n_samples, data, fig_path, test_index)

def plot_attack(org_images, adv_images, noise, eps, label):
    if eps == 0:
        return
    
    if normalized:
        # Denormalize the images for plotting
        denormalize = transforms.Compose([
            transforms.Normalize(mean=[-0.1307/0.3081], std=[1/0.3081])
        ])

        org_images = denormalize(org_images)
        adv_images = denormalize(adv_images)
    
    org_images = org_images.detach().numpy()
    adv_images = adv_images.detach().numpy()
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(org_images.transpose(1, 2, 0), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adv_images.transpose(1, 2, 0), cmap='gray')
    plt.title(f"Adversarial Image ({eps})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(noise,cmap='gray')
    plt.title(f"Added Noise")
    plt.axis('off')

    if attack_type == "average":   
        if attack_all_classes:
            attack = f"{num_classes}_all_to_{target_class}"
        else: 
            attack = f"{num_classes}_{adv_class}_to_{target_class}"
    else:
        attack = f"{num_classes}_{adv_function}_{test_function}"

    if normalized: 
        fig_path = f'figures/{attack_type}/{attack}'
    else: 
        fig_path = f'figures/no_norm/{attack_type}/{attack}'
    create_path(fig_path)
    fig_path += f'/{eps}_{label}.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()

def plot_acc_graph(acc_per_eps, testing_eps):
    plt.figure(figsize=(8,5))
    plt.plot(testing_eps, acc_per_eps, '*-')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    step = testing_eps[1] - testing_eps[0]
    plt.xticks(np.arange(0, max(testing_eps)+step, step=step))
    if attack_type == "average":
        plt.title(f"Accuracy vs Epsilon, {adv_class} class")
    else:
        plt.title("Accuracy vs Epsilon")
        plt.suptitle(f'{adv_function} (MNIST {num_classes})')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)

    if attack_type == "average":   
        if attack_all_classes:
            attack = f"{num_classes}_all_to_{target_class}"
        else: 
            attack = f"{num_classes}_{adv_class}_to_{target_class}"
    else:
        attack = f"{num_classes}_{adv_function}_{test_function}"

    if attack_type == "average":
        if attack_all_classes:
            if normalized:                 
                fig_path = f'figures/{attack_type}/{attack}/{test_function}_acc_eps.png'
            else: 
                fig_path = f'figures/no_norm/{attack_type}/{attack}/{test_function}_acc_eps.png'
        else:
            if normalized:                 
                fig_path = f'figures/{attack_type}/{attack}/{test_function}_acc_eps_{adv_class}.png'
            else: 
                fig_path = f'figures/no_norm/{attack_type}/{attack}/{test_function}_acc_eps_{adv_class}.png'
    else:
        if normalized:
            fig_path = f'figures/{attack_type}/{attack}/acc_eps.png'
        else: 
            fig_path = f'figures/no_norm/{attack_type}/{attack}/acc_eps.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()

def plot_acc_class_per_eps(acc_class_per_eps, testing_eps):
    num_classes = len(acc_class_per_eps[0])
    plt.figure(figsize=(10, 6))

    for class_id in range(num_classes):
        class_accuracies = [eps[class_id] for eps in acc_class_per_eps]
        if test_all:
            plt.plot(testing_eps, class_accuracies, label=f'{class_id}')
        else:
            plt.plot(testing_eps, class_accuracies, label=f'{test_index[class_id]}')

    plt.yticks(np.arange(0, 1.1, step=0.1))
    step = testing_eps[1] - testing_eps[0]
    plt.xticks(np.arange(0, max(testing_eps)+step, step=step))
    plt.title("Accuracy per Class vs Epsilon")
    plt.suptitle(f'{adv_function} (MNIST {num_classes})')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)

    if attack_type == "average":   
        if attack_all_classes:
            attack = f"{num_classes}_all_to_{target_class}"
        else: 
            attack = f"{num_classes}_{adv_class}_to_{target_class}"
    else:
        attack = f"{num_classes}_{adv_function}_{test_function}"

    if attack_type == "average":
        if normalized: 
            fig_path = f'figures/{attack_type}/{attack}/{test_function}_acc_class_per_eps.png'
        else: 
            fig_path = f'figures/no_norm/{attack_type}/{attack}/{test_function}_acc_class_per_eps.png'
    else:
        if normalized:
            fig_path = f'figures/{attack_type}/{attack}/acc_class_per_eps.png'
        else: 
            fig_path = f'figures/no_norm/{attack_type}/{attack}/acc_class_per_eps.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    if normalized:
        if train_all:
            path = f'runs/{num_classes}_classes/best_{adv_function}_net_checkpoint.pt'
        else:
            path = f'runs/{train_index}_classes/best_{adv_function}_net_checkpoint.pt'
    else:
        if train_all:
            path = f'runs/no_norm/{num_classes}_classes/best_{adv_function}_net_checkpoint.pt'
        else:
            path = f'runs/no_norm/{train_index}_classes/best_{adv_function}_net_checkpoint.pt'
    if normalized:
        if train_all:
            path2 = f'runs/{num_classes}_classes/best_{test_function}_net_checkpoint.pt'
        else:
            path2 = f'runs/{train_index}_classes/best_{test_function}_net_checkpoint.pt'
    else:
        if train_all:
            path2 = f'runs/no_norm/{num_classes}_classes/best_{test_function}_net_checkpoint.pt'
        else:
            path2 = f'runs/no_norm/{train_index}_classes/best_{test_function}_net_checkpoint.pt'
    main()

