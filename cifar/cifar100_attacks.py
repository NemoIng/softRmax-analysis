"""
@author: NemoIng (Nemo Ingendaa)

Using/Inspiration code from:
- https://github.com/ziqiwangsilvia/attack 
- https://pytorch.org/tutorials/beginner/fgsm_tutorial.html 
- https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html  
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

from cifar_dataset import prepare_dataset_cifar100
from cifar_network import Net
from deepfool import get_clip_bounds, deepfool, display_attack, compute_robustness

# All attack parameters
test_function = 'softmax'
attack_type = 'fgsm'
if attack_type == 'average':
    testing_eps = 0.3
    plot_eps = []
    test_batch_size = 3000
elif attack_type == 'bim':
    testing_eps = [0.0, 0.01, 0.05, 0.1, 0.3]
    test_batch_size = 512
elif attack_type == 'fgsm':
    testing_eps = [0.0, 0.01, 0.05, 0.1, 0.3]
    test_batch_size = 512
elif attack_type == 'fgsm-target':
    testing_eps = [0.3]
    plot_eps = []
    max_num_images = 1000
    test_batch_size = max_num_images
elif attack_type == 'deepfool':
    testing_eps = []
    plot_eps = []
    max_num_images = 1000
    test_batch_size = max_num_images
else:
    print('attack doesnt exist')
    sys.exit(0)
class_to_plot = 'chimpanzee'
plot_image_comparison = False # plot comparison between og and adv image

# FGSM/BIM attack parameters
adv_function = 'softmax' # function for which the adverarial examples will created
bim_iters = 10 # number of fgsm steps for BIM

# DeepFool variables
compare_fgsm = False # show visual comparison with fgsm

# Data parameters
num_classes = 100
train_all = True
train_index = [3, 7]
test_all = True
test_index = [3, 7]

denormalize = transforms.Compose([
    transforms.Normalize(mean=[-0.5074/0.2011, -0.4867/0.1987, -0.4411/0.2025], std=[1/0.2011, 1/0.1987, 1/0.2025])
])
cifar_mean = [0.5074, 0.4867, 0.4411]
cifar_std = [0.2011, 0.1987, 0.2025]
data_min, data_max = get_clip_bounds(cifar_mean,cifar_std,32)
 
classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 
    'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 
    'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 
    'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 
    'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 
    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 
    'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')

# When using apple silicon GPU:
device = torch.device("mps")

# For NVIDIA GPUs
# device = torch.device("cuda")

# The CPU is used in the testing phase (performance reasons)
device2 = torch.device("cpu")

def main():
    testset = prepare_dataset_cifar100(train_all, train_index, test_all, test_index, 'test') 
    testloader = td.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    if attack_type == 'average':
        net = Net(device2, num_classes, test_function).to(device2)
        net.load_state_dict(torch.load(path2))
        net.eval()
        acc = []
        print(f'Loaded the {test_function}-{num_classes} network (testing)')
        for adv in range(num_classes):
            accuracies = []
            for target in range(num_classes):
                if target == adv:
                    continue

                adv_data = create_data(net, testloader, testing_eps, target, adv)
                for i, eps in enumerate(testing_eps):
                    _, classes_acc = test_model_accuracy(net, adv_data[i], eps, target, adv)
                    accuracies.append(classes_acc[adv])
            acc.append(round(100 * sum(accuracies) / (num_classes-1), 2))
            print(f'Average accuracy for class {adv}: {round(100 * sum(accuracies) / (num_classes-1), 2)}%')
        for i in range(num_classes): 
            print(f'{acc[i]}%')

    elif attack_type == 'deepfool':
        # We input 'softmax' as the testfunction because than softRmax is not used before output
        # deepfool only works well without the final layer activation function
        net = Net(device, num_classes, 'softmax').to(device)
        net.load_state_dict(torch.load(path2))
        net.eval()
        print(f'Loaded the {test_function}-{num_classes} network (testing/creation)')
        if compare_fgsm:
            deepfool_vs_fgsm_visualize(net, testset, 0.2)
            
        adv_data = create_data(net, testloader, 0)

        # We create the testing network and move it to the CPU
        net = Net(device2, num_classes, test_function)
        net.load_state_dict(torch.load(path2))
        net.eval()

        test_model_accuracy(net, adv_data, 0)

    elif attack_type == 'fgsm-target':
        if test_function != adv_function:
            print(f'black box testing {test_function} with cons network loss')
        else:
            print(f'white box testing {test_function}')

        net = Net(device, num_classes, adv_function).to(device)
        net.load_state_dict(torch.load(path))
        net.eval()

        net2 = Net(device2, num_classes, test_function).to(device2)
        net2.load_state_dict(torch.load(path2))
        net2.eval()

        accuracies = [0 for i in range(num_classes)]

        for target in range(num_classes):
            adv_data = create_data(net, testloader, testing_eps, target)
            _, classes_acc = test_model_accuracy(net2, adv_data[0], testing_eps[0], target)

            for i in range(num_classes):
                if i != target:
                    accuracies[i] += classes_acc[i]

        for i in range(num_classes):
            avg_acc = round(100 * accuracies[i] / (num_classes-1), 2)
            print(f'{avg_acc}%')

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

        for i, eps in enumerate(testing_eps):
            print(f'Testing eps: {eps}')
            acc, classes_acc = test_model_accuracy(net2, adv_data[i], eps)
            acc_per_eps.append(acc)

        plot_acc_graph(acc_per_eps, testing_eps)

def create_data(net, testloader, testing_eps, target_class=None, adv_class=None):
    data = []
    if attack_type == 'deepfool':
        num_imgs = 0
        robustness1 = []
        robustness2 = []
        start = time.time()
        for images, labels in testloader:
            correctly_predicted_images = []
            images = images.clone().detach()
            labels = labels.clone().detach()
                
            _, predicted_labels = torch.max(net(images), 1)
            for i in range(len(images)):
                if predicted_labels[i] == labels[i]:
                    correctly_predicted_images.append(images[i].clone().detach())
            new_images = torch.stack(correctly_predicted_images)
            _, new_labels = torch.max(net(new_images), 1)
            num_imgs = len(new_images)
            print(f"Number of images: {num_imgs}")  
        
            adv_images, noise, iterations, robust1, robust2 = deepfool_attack(net, new_images, new_labels)
            robustness1.append(robust1)
            robustness2.append(robust2.to(device2))
            data.append((new_images, adv_images, new_labels, noise))
            break

        end = time.time()
        print(f'Average iterations per image: {round(iterations / len(new_images), 3)}') 
        print(f'Average robustness per image: {round(float(np.mean(robustness1)), 5)} (deepfool method)')
        print(f'Average robustness per image: {round(float(np.mean(robustness2)), 7)} (squared error)')
        print(f'Average time spend per image: {round((end - start) / len(new_images), 3)} sec')
    else:
        for eps in testing_eps:
            print(f'\nCreating adversarial ({attack_type}) data for eps: {eps}')
            adversarial_data = []
            for images, labels in testloader:
                if eps == 0:
                    adversarial_data.append((images, images, labels, None))
                elif attack_type == 'fgsm' or attack_type == 'fgsm-target':
                    adv_images, noise = fgsm_attack(net, images, labels, eps, target_class)
                    adversarial_data.append((images, adv_images, labels, noise))
                    if attack_type == 'fgsm-target':
                        break
                elif attack_type == 'bim':
                    adv_images, noise = bim_attack(net, images, labels, eps)
                    adversarial_data.append((images, adv_images, labels, noise))
                else: 
                    images, adv_images, noise, labels = average_attack(images, labels, eps, target_class, adv_class)
                    adversarial_data.append((images, adv_images, labels, noise))

            data.append(adversarial_data)
    return data

def fgsm_attack(net, images, labels, eps, target_class=None):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
            
    outputs = net(images)
    net.zero_grad()

    if target_class == None:
        if adv_function == 'softRmax' or adv_function == 'cons':
            loss_func = nn.NLLLoss()
            loss = loss_func(torch.log(outputs), labels)
        else:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(outputs, labels) 
    else:
        if adv_function == 'softRmax' or adv_function == 'cons':
            loss_func = nn.NLLLoss()
            loss = loss_func(torch.log(outputs), 
                    torch.tensor([target_class for i in range(len(outputs))], dtype=torch.long).to(device))
        else:
            loss_func = nn.CrossEntropyLoss()
            print(torch.tensor([target_class for i in range(len(outputs))], dtype=torch.long).to(device).shape)
            loss = loss_func(outputs, 
                    torch.tensor([target_class for i in range(len(outputs))], dtype=torch.long).to(device))
            
    loss.backward()
    data_grad = images.grad.data

    noise = eps*data_grad.sign()
    if target_class == None:
        attack_images = images + noise
    else: 
        attack_images = images - noise
    
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

        images = torch.clamp(c, data_min.to(device), data_max.to(device)).detach()

    noise = torch.clamp(images - org_images, 0, 1)
    noise = noise.cpu().detach().numpy()
    return images, noise

def average_attack(images, labels, eps, target_class, adv_class):
    target_class_images = []
    adv_class_images = []
    target_labels = []
    adv_labels = []
    for img, lab in zip(images, labels):
        if lab == target_class:
            target_class_images.append(img)
            target_labels.append(lab)
        if lab == adv_class:
            adv_class_images.append(img)
            adv_labels.append(lab)
    # Change the lists to tensors
    adv_class_images = torch.stack(adv_class_images, dim=0)
    target_class_images = torch.stack(target_class_images, dim=0)

    adv_labels = torch.stack(adv_labels, dim=0)
    target_labels = torch.stack(target_labels, dim=0)

    # create the average image samples
    avg_adv_class_image = torch.mean(adv_class_images, dim=0)
    avg_target_class_image = torch.mean(target_class_images, dim=0)

    noise = eps*(avg_target_class_image - avg_adv_class_image).sign()

    attack_images = adv_class_images + noise
    return adv_class_images, attack_images, noise, adv_labels

def deepfool_attack(net, images, labels):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    if test_function == "softmax":
        attack_images, noise, _, _, iters = deepfool(net, data_min, data_max, images, device, 
                                            False, labels, num_classes=num_classes, batch_size=num_classes)
    else:
        attack_images, noise, _, _, iters = deepfool(net, data_min, data_max, images, device,
                                            True, labels, num_classes=num_classes, batch_size=num_classes)
    robustness1 = compute_robustness(images, noise)
    robustness2 = (images - attack_images) ** 2
    return attack_images, noise, sum(iters), np.mean(robustness1), robustness2, images, labels

def deepfool_vs_fgsm_visualize(net, testset, eps):
    args = [128, 10, 0.02, 50] 
    if test_function == 'softmax':
        display_attack(device, net, testset, denormalize, data_min, data_max, eps, args, False, label_map=classes)
    else:
        display_attack(device, net, testset, denormalize, data_min, data_max, eps, args, True, label_map=classes)

def test_model_accuracy(net, adv_data, eps, target_class=None, adv_class=None):
    acc = 0
    nb = 0
    printed = [False for i in range(num_classes)]
    class_correct = [0.0 for _ in range(num_classes)]
    class_total = [0.0 for _ in range(num_classes)]

    for images, attack_images, labels, noise in adv_data:
        images = images.to(device2)
        attack_images = attack_images.to(device2)
        labels = labels.to(device2)
        _, predicted_labels = torch.max(net(attack_images), 1)

        if plot_image_comparison:
            if attack_type == 'deepfool' or attack_type == 'boundary':
                for i, label in enumerate(labels.cpu().detach().numpy()):
                    if not printed[label] and predicted_labels[i] != label:
                        plot_attack(images[i], attack_images[i], noise[i], eps, label, predicted_labels[i])
                        printed[label] = True
            else:
                if eps != 0:
                    for i, label in enumerate(labels.cpu().detach().numpy()):
                        if not printed[label] and predicted_labels[i] != label:
                            if attack_type == "average":
                                plot_attack(images[i], attack_images[i], noise[0], eps, label, predicted_labels[i], target_class, adv_class)
                            else:
                                if label == adv_class or target_class == None:
                                    plot_attack(images[i], attack_images[i], noise[i,0], eps, label, predicted_labels[i], target_class, adv_class)
                            printed[label] = True

        nb = nb + len(attack_images)
        acc = acc + (predicted_labels - labels).nonzero().size(0)
        c = (predicted_labels == labels).squeeze()
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

def plot_attack(org_image, adv_image, noise, eps, label, pred_label, target_class=None, adv_class=None):
    # Denormalize the images before plotting
    org_image = denormalize(org_image).detach().numpy()
    adv_image = denormalize(adv_image).detach().numpy()
    
    if attack_type == 'deepfool' or attack_type == 'boundary':
        noise = denormalize(noise.to(device2)).detach().numpy().transpose(1, 2, 0)
        attack = f"{test_function}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        fig_path += f'/{classes[label]}.png'
    elif attack_type == "average":   
        attack = f"{num_classes}_{classes[adv_class]}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        fig_path += f'/{eps}_{classes[adv_class]}_to_{classes[target_class]}.png'
    else:
        attack = f"{num_classes}_{adv_function}_{test_function}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        if target_class == None:
            fig_path += f'/{eps}_{classes[label]}.png'
        else:
            fig_path += f'/{eps}_{classes[adv_class]}_to_{classes[target_class]}.png'

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(org_image.transpose(1, 2, 0))
    plt.title(f"Original Image ({classes[label]})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adv_image.transpose(1, 2, 0))
    plt.title(f"Adversarial Image ({classes[pred_label]})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(noise)
    plt.title(f"Added Noise")
    plt.axis('off')

    plt.savefig(fig_path, dpi=200)
    plt.close()

def plot_acc_graph(acc_per_eps, testing_eps, adv_class=None):
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