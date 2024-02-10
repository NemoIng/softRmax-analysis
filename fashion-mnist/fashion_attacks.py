"""
@author: NemoIng (Nemo ingendaa)

Using/Inspiration code from:
- https://github.com/ziqiwangsilvia/attack 
- https://pytorch.org/tutorials/beginner/fgsm_tutorial.html 
- https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html 
- https://github.com/LTS4/DeepFool
- https://github.com/bethgelab/foolbox/blob/bb56af1d215572d8d468c4759e8d3f5b3acfeb65/foolbox/attacks/boundary_attack.py 
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn 
import torch.utils.data as td
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from foolbox.attacks import BoundaryAttack
from foolbox.models import PyTorchModel

from fashion_dataset import prepare_dataset
from fashion_network import Net
from deepfool import get_clip_bounds, deepfool, display_attack, compute_robustness

# All attack parameters
test_function = 'softRmax'
attack_type = 'boundary'
if attack_type == 'average':
    testing_eps = [0.8]
    plot_eps = []
    test_batch_size = 400
elif attack_type == 'bim':
    testing_eps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plot_eps = [0.1, 0.3]
    test_batch_size = 400
elif attack_type == 'fgsm':
    testing_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    plot_eps = [0.1, 0.3]
    test_batch_size = 400
elif attack_type == 'deepfool':
    testing_eps = []
    plot_eps = []
    test_batch_size = 100
    max_num_images = 1000
elif attack_type == 'boundary':
    testing_eps = []
    plot_eps = []
    test_batch_size = 100
    max_num_images = 100
else:
    print('attack doesnt exist')
    sys.exit(0)
class_to_plot = 'Bag'
dec_bound = False # plot decision boundary yes/no

# Average attack parameters
adv_class = 'Bag' # if all_classes_attack is False, specify which class
if attack_type == 'average':
    class_to_plot = adv_class

# FGSM/BIM attack parameters
adv_function = 'softmax' # function for which the adverarial examples will created
bim_iters = 10 # number of fgsm steps for BIM

# DeepFool variables
compare_fgsm = True # show visual comparison with fgsm

# Data parameters
num_classes = 10
train_all = True
train_index = [3, 7]
test_all = True
test_index = [3, 7]

denormalize = transforms.Compose([
    transforms.Normalize(mean=[-0.2860/0.3205], std=[1/0.3205])
])
data_min, data_max = get_clip_bounds(0.2860,0.3205,28)

classes = ['T-shirt&top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# When using apple silicon GPU:
device = torch.device("mps")

# For NVIDIA GPUs
# device = torch.device("cuda")

# The CPU is used in the testing phase (performance reasons)
device2 = torch.device("cpu")


def main():
    if attack_type == 'boundary':
        testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test', False)
    else:
        testset = prepare_dataset(train_all, train_index, test_all, test_index, 'test') 
    testloader = td.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    if attack_type == 'average':
        accuracies = []
        net = Net(device2, num_classes, test_function).to(device2)
        net.load_state_dict(torch.load(path2))
        net.eval()
        print(f'Loaded the {test_function}-{num_classes} network (testing)')
        for i in range(num_classes):
            if i == classes.index(adv_class):
                continue

            adv_data = create_data(net, testloader, testing_eps, i)
            for j, eps in enumerate(testing_eps):
                acc, classes_acc = test_model_accuracy(net, adv_data[j], 0, target_class=i)
                accuracies.append(classes_acc[classes.index(adv_class)])
        print(f'Average accuracy for class {adv_class}: {round(100 * sum(accuracies) / (num_classes-1), 2)}%')
       
    elif attack_type == 'boundary':
        accuracies = []
        net = Net(device2, num_classes, test_function).to(device2)
        net.load_state_dict(torch.load(path2))
        net.eval()
        preprocessing = dict(mean=0.2860, std=0.3205)
        net = PyTorchModel(net, (0,1), device2, preprocessing)
        print(f'Loaded the {test_function}-{num_classes} network (testing)')

        adv_data = create_data(net, testloader, 0)
        test_model_accuracy(net, adv_data, 0)

    elif attack_type == 'deepfool':
        net = Net(device, num_classes, test_function).to(device)
        net.load_state_dict(torch.load(path2))
        net.eval()
        print(f'Loaded the {test_function}-{num_classes} network (testing/creation)')
        if compare_fgsm:
            deepfool_vs_fgsm_visualize(net, testset, 0.3)
            
        adv_data = create_data(net, testloader, testing_eps, 0)

        # We create the testing network and move it to the CPU along with the data
        net = Net(device2, num_classes, test_function)
        net.load_state_dict(torch.load(path2))
        net.eval()

        test_model_accuracy(net, adv_data, 0)

    else:
        net = Net(device, num_classes, adv_function).to(device)
        net.load_state_dict(torch.load(path))
        net.eval()
        print(f'Loaded the {adv_function}-{num_classes} network (creation)')
        adv_data = create_data(net, testloader, testing_eps)

        # We create the testing network and move it to the CPU along with the data
        net2 = Net(device2, num_classes, test_function)
        net2.load_state_dict(torch.load(path2))
        net2.eval()
        print(f'Loaded the {test_function}-{num_classes} network')
        acc_per_eps = []
        acc_class_per_eps = []

        for i, eps in enumerate(testing_eps):
            print(f'Testing eps: {eps}')
            acc, classes_acc = test_model_accuracy(net2, adv_data[i], eps)
            acc_per_eps.append(acc)
            acc_class_per_eps.append(classes_acc)

        plot_acc_graph(acc_per_eps, testing_eps)
        plot_acc_class_per_eps(acc_class_per_eps, testing_eps)

def create_data(net, testloader, testing_eps, target_class=0):
    data = []
    if attack_type == 'deepfool' or attack_type == 'boundary':
        iterations = 0
        num_imgs = 0
        robustness1 = []
        robustness2 = []
        correctly_predicted_images = []
        start = time.time()
        for images, labels in testloader:
            _, predicted_labels = torch.max(net(images), 1)

            for i in range(len(images)):
                if predicted_labels[i] == labels[i]:
                    correctly_predicted_images.append(images[i].clone().detach())
            new_images = torch.stack(correctly_predicted_images)
            _, new_labels = torch.max(net(new_images), 1)

            if num_imgs >= max_num_images: break
            if attack_type == 'deepfool':
                images = images.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                images, adv_images, noise, iterations, robust1, robust2 = deepfool_attack(net, new_images, new_labels, iterations)
            else:
                adv_images, noise, robust1, robust2 = boundary_attack(net, new_images, new_labels)
            num_imgs += len(adv_images)
            robustness1.append(robust1)
            robustness2.append(robust2.to(device2))
            data.append((new_images, adv_images, new_labels, noise))

        end = time.time()
        if attack_type == 'deepfool':
            print(f'Average iterations per image: {round(iterations / num_imgs, 3)}') 
        print(f'Average robustness per image: {round(float(np.mean(robustness1)), 5)} (deepfool method)')
        print(f'Average robustness per image: {round(float(np.mean(robustness2)), 7)} (squared error)')
        print(f'Average time spend per image: {round((end - start) / num_imgs, 3)} sec')
    else:
        for eps in testing_eps:
            print(f'\nCreating adversarial ({attack_type}) data for eps: {eps}')
            adversarial_data = []
            for images, labels in testloader:
                if eps == 0:
                    adversarial_data.append((images, images, labels, None))
                elif attack_type == 'fgsm':
                    adv_images, noise = fgsm_attack(net, images, labels, eps)
                    adversarial_data.append((images, adv_images, labels, noise))
                elif attack_type == 'bim':
                    adv_images, noise = bim_attack(net, images, labels, eps)
                    adversarial_data.append((images, adv_images, labels, noise))
                else: 
                    images, adv_images, noise, labels = average_attack(images, labels, eps, adv_class, target_class)
                    adversarial_data.append((images, adv_images, labels, noise))

            data.append(adversarial_data)
    return data

def fgsm_attack(net, images, labels, eps) :
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
            
    outputs = net(images)
    net.zero_grad()
    if adv_function == 'softmax':
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(outputs, labels)
    else:
        loss_func = nn.NLLLoss()
        loss = loss_func(torch.log(outputs), labels)

    loss.backward()
    data_grad = images.grad.data

    noise = eps*data_grad.sign()
    attack_images = images + noise

    noise = noise.clamp(0, 1).cpu().detach().numpy()
    return attack_images, noise

def bim_attack(net, images, labels, eps):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    org_images = images.clone().detach()
    alpha = eps / bim_iters
    for i in range(bim_iters):    
        images.requires_grad = True

        outputs = net(images)
        net.zero_grad()
        if adv_function == 'softmax':
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(outputs, labels)
        else:
            loss_func = nn.NLLLoss()
            loss = loss_func(torch.log(outputs), labels)
        loss.backward()
        data_grad = images.grad.data # type: ignore

        attack_images = images + alpha*data_grad.sign()

        a = torch.clamp(org_images - eps, min=0)
        b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
        c = (b > attack_images+eps).float()*(attack_images+eps) + (attack_images+eps >= b).float()*b

        images = torch.clamp(c, max=255).detach()

    noise = torch.clamp(images - org_images, 0, 1)
    noise = noise.cpu().detach().numpy()
    return images, noise

def average_attack(images, labels, eps, adv_class, target_class):
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
        if lab == classes.index(adv_class):
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

    noise = eps*(avg_target_class_image - avg_adv_class_image).sign()

    attack_images = adv_class_images + noise
    return adv_class_images, attack_images, noise, adv_labels

def deepfool_attack(net, images, labels, iterations):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    if test_function == "softmax":
        attack_images, noise, _, _, iters = deepfool(net, data_min, data_max, images, device, False, labels)
    else:
        attack_images, noise, _, _, iters = deepfool(net, data_min, data_max, images, device, True, labels)
    iterations += sum(iters)
    robustness1 = compute_robustness(images, noise)
    robustness2 = (attack_images - images) ** 2
    return images, attack_images, noise, iterations, np.mean(robustness1), robustness2

def deepfool_vs_fgsm_visualize(net, testset, eps):
    args = [128, 10, 0.02, 50]
    if test_function == 'softmax':
        display_attack(device, net, testset, denormalize, data_min, data_max, eps, args, False, label_map=classes)
    else:
        display_attack(device, net, testset, denormalize, data_min, data_max, eps, args, True, label_map=classes)

def boundary_attack(net, images, labels):
    images = images.clone().detach().to(device2)
    labels = labels.clone().detach().to(device2)
    attack = BoundaryAttack(steps=100000)
    attack_images = attack.run(net, images, labels)

    noise = attack_images - images
    robustness1 = compute_robustness(images, noise)
    robustness2 = (attack_images - images) ** 2
    return attack_images, torch.clamp(noise, 0, 1), np.mean(robustness1), robustness2

def test_model_accuracy(net, adv_data, eps, target_class=0):
    if not train_all:
        class_label = train_index.index(classes.index(class_to_plot))        
    else:
        class_label = classes.index(class_to_plot)
    printed = False
    acc = 0
    nb = 0
    class_correct = [0.0 for _ in range(num_classes)]
    class_total = [0.0 for _ in range(num_classes)]

    for images, attack_images, labels, noise in adv_data:
        images = images.to(device2)
        attack_images = attack_images.to(device2)
        labels = labels.to(device2)
        _, predicted_labels = torch.max(net(attack_images), 1)
        if attack_type == 'deepfool' or attack_type == 'boundary':
            for i, label in enumerate(labels.cpu().detach().numpy()):
                if predicted_labels[i] != label:
                    plot_attack(images[i], attack_images[i], noise[i], eps, label, predicted_labels[i])
        if not printed and eps != 0:
            for i, label in enumerate(labels.cpu().detach().numpy()):
                if label == class_label and predicted_labels[i] != label:
                    if attack_type == "average":
                        plot_attack(images[i], attack_images[i], noise[0], eps, label, predicted_labels[i], target_class)
                    else:
                        plot_attack(images[i], attack_images[i], noise[i,0], eps, label, predicted_labels[i])
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
    return test_acc, classes_acc

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def plot_attack(org_image, adv_image, noise, eps, label, pred_label, target_class=0):
    # Denormalize the images before plotting
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.2860/0.3205], std=[1/0.3205])
    ])
    org_image = denormalize(org_image).detach().numpy()
    adv_image = denormalize(adv_image).detach().numpy()
    
    if attack_type == 'deepfool' or attack_type == 'boundary':
        noise = denormalize(noise.to(device2)).detach().numpy().transpose(1, 2, 0)
        attack = f"{test_function}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        fig_path += f'/{classes[label]}.png'
    elif attack_type == "average":   
        attack = f"{num_classes}_{adv_class}_to_{classes[target_class]}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        fig_path += f'/{eps}_{classes[label]}.png'
    else:
        attack = f"{num_classes}_{adv_function}_{test_function}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        fig_path += f'/{eps}_{classes[label]}.png'
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(org_image.transpose(1, 2, 0), cmap='gray')
    plt.title(f"Original Image ({classes[label]})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adv_image.transpose(1, 2, 0), cmap='gray')
    plt.title(f"Adversarial Image ({classes[pred_label]})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(noise,cmap='gray')
    plt.title(f"Added Noise")
    plt.axis('off')

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

    attack = f"{num_classes}_{adv_function}_{test_function}"

    if attack_type == "average":              
        fig_path = f'figures/{attack_type}/{attack}/{test_function}_acc_eps_{adv_class}.png'
    else:
        fig_path = f'figures/{attack_type}/{attack}/acc_eps.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()

def plot_acc_class_per_eps(acc_class_per_eps, testing_eps):
    num_classes = len(acc_class_per_eps[0])
    plt.figure(figsize=(10, 6))

    for class_id in range(num_classes):
        class_accuracies = [eps[class_id] for eps in acc_class_per_eps]
        if test_all:
            plt.plot(testing_eps, class_accuracies, label=f'{classes[class_id]}')
        else:
            plt.plot(testing_eps, class_accuracies, label=f'{classes[test_index[class_id]]}')

    plt.yticks(np.arange(0, 1.1, step=0.1))
    step = testing_eps[1] - testing_eps[0]
    plt.xticks(np.arange(0, max(testing_eps)+step, step=step))
    plt.title("Accuracy per Class vs Epsilon")
    plt.suptitle(f'{adv_function} (MNIST {num_classes})')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)

    attack = f"{num_classes}_{adv_function}_{test_function}"

    if attack_type == "average":
        fig_path = f'figures/{attack_type}/{attack}/{test_function}_acc_class_per_eps.png'
    else:
        fig_path = f'figures/{attack_type}/{attack}/acc_class_per_eps.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    if train_all:
        path = f'runs/{num_classes}_classes/best_{adv_function}_net_checkpoint.pt'
    else:
        path = f'runs/{train_index}_classes/best_{adv_function}_net_checkpoint.pt'

    if train_all:
        path2 = f'runs/{num_classes}_classes/best_{test_function}_net_checkpoint.pt'
    else:
        path2 = f'runs/{train_index}_classes/best_{test_function}_net_checkpoint.pt'
    main()

