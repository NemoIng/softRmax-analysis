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
import numpy as np
import torch
import time
import torch.nn as nn 
import torch.utils.data as td
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from foolbox.attacks import BoundaryAttack
from foolbox.models import PyTorchModel

from mnist_dataset import prepare_dataset
from mnist_network import Net
from mnist_bound import plot_decision_boundary
from deepfool import get_clip_bounds, deepfool, display_attack, compute_robustness

# All attack parameters
test_function = 'softmax'
attack_type = 'bim'
if attack_type == 'average':
    testing_eps = [0.5]
    plot_eps = []
    test_batch_size = 200
elif attack_type == 'bim':
    testing_eps = [0.1, 0.3]
    plot_eps = [0.1, 0.3]
    test_batch_size = 400
elif attack_type == 'fgsm':
    testing_eps = [0.1, 0.3]
    plot_eps = [0.1, 0.3]
    test_batch_size = 200
elif attack_type == 'fgsm-target':
    testing_eps = [0.5]
    plot_eps = []
    test_batch_size = 200
elif attack_type == 'deepfool':
    testing_eps = []
    plot_eps = []
    max_num_images = 5000
    test_batch_size = max_num_images
elif attack_type == 'boundary':
    testing_eps = []
    plot_eps = []
    max_num_images = 2000
    test_batch_size = max_num_images
else:
    print('attack doesnt exist')  
    sys.exit(0)

dec_bound = False # plot decision boundary yes/no
if dec_bound: test_batch_size = 200 # if selected set batch_size to 200 (matching the nr. samples deepview uses)

plot_image_comparison = False # plot comparison between og and adv image

# FGSM/BIM attack parameters
adv_function = 'softmax' # function for which the adverarial examples will created
bim_iters = 10 # number of fgsm steps for BIM

# DeepFool variables
compare_fgsm = False # show visual comparison with fgsm

# Network parameters
kernel_size = 3

# Data parameters
num_classes = 10
train_all = True
train_index = [3, 7]
test_all = True
test_index = [3, 7]

denormalize = transforms.Compose([
    transforms.Normalize(mean=[-0.1307/0.3081], std=[1/0.3081])
])
data_min, data_max = get_clip_bounds(0.1307,0.3081,28)
 
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# When using apple silicon GPU:
device = torch.device("mps")

# For NVIDIA GPUs
# device = torch.device("cuda")

# The CPU is used in the testing phase (performance reasons)
device2 = torch.device("cpu")

def main():
    if attack_type == 'boundary':
        testset = prepare_dataset(True, train_index, test_all, test_index, 'test', False)
    else:
        testset = prepare_dataset(True, train_index, test_all, test_index, 'test') 
    testloader = td.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    if attack_type == 'average':
        net = Net(device2, num_classes, test_function, kernel_size).to(device2)
        net.load_state_dict(torch.load(path2))
        net.eval()
        acc = []
        print(f'Loaded the {test_function}-{num_classes} network (testing)')

        for adv in range(num_classes):
            accuracies = []
            for i in range(num_classes):
                if i == adv:
                    continue

                adv_data = create_data(net, testloader, testing_eps, i, adv)
                for i, eps in enumerate(testing_eps):
                    _, classes_acc = test_model_accuracy(net, adv_data[i], eps, i, adv)
                    accuracies.append(classes_acc[adv])
            acc.append(round(100 * sum(accuracies) / (num_classes-1), 2))
            print(f'Average accuracy for class {adv}: {round(100 * sum(accuracies) / (num_classes-1), 2)}%')
        for i in range(num_classes): 
            print(f'{acc[i]}%')

    elif attack_type == 'boundary':
        accuracies = []
        net = Net(device2, num_classes, test_function, kernel_size).to(device2)
        net.load_state_dict(torch.load(path2))
        net.eval()
        preprocessing = dict(mean=0.1307, std=0.3081)
        net = PyTorchModel(net, (0,1), device2, preprocessing)
        print(f'Loaded the {test_function}-{num_classes} network (testing)')

        adv_data = create_data(net, testloader, 0)

        if dec_bound:
            fig_path = f'figures/{attack_type}/dec_bound/{test_function}.png'
            plot_dec_bound(adv_data, 200, fig_path=fig_path)

        test_model_accuracy(net, adv_data, 0)
    
    elif attack_type == 'deepfool':
        # We input 'softmax' as the testfunction because than softRmax is not used before output
        # deepfool only works well without the final layer activation function
        net = Net(device, num_classes, 'softmax', kernel_size).to(device)
        net.load_state_dict(torch.load(path2))
        net.eval()
        print(f'Loaded the {test_function}-{num_classes} network (testing/creation)')
        if compare_fgsm:
            deepfool_vs_fgsm_visualize(net, testset, 0.5)
            
        adv_data = create_data(net, testloader, testing_eps, 0)

        if dec_bound:
            fig_path = f'figures/{attack_type}/dec_bound/{test_function}.png'
            plot_dec_bound(adv_data, 200, fig_path=fig_path)

        # We create the testing network and move it to the CPU along with the data
        net = Net(device2, num_classes, test_function, kernel_size).to(device2)
        net.load_state_dict(torch.load(path2))
        net.eval()

        test_model_accuracy(net, adv_data, 0)

    elif attack_type == 'fgsm-target':
        if test_function != adv_function:
            print(f'black box testing {test_function} with {adv_function} network loss')
        else:
            print(f'white box testing {test_function}')

        net = Net(device, num_classes, adv_function, kernel_size).to(device)
        net.load_state_dict(torch.load(path))
        net.eval()

        net2 = Net(device2, num_classes, test_function, kernel_size).to(device2)
        net2.load_state_dict(torch.load(path2))
        net2.eval()

        accuracies = [0 for i in range(num_classes)]

        for i in range(num_classes):
            adv_data = create_data(net, testloader, testing_eps, i)

            if dec_bound:
                fig_path = f'figures/{attack_type}/dec_bound/{adv_function}_{test_function}/to_{i}.png'
                plot_dec_bound(adv_data[0], 200, testing_eps[0], fig_path)

            _, classes_acc = test_model_accuracy(net2, adv_data[0], testing_eps[0], i)

            for i in range(num_classes):
                if i != i:
                    accuracies[i] += classes_acc[i]

        for i in range(num_classes):
            avg_acc = round(100 * accuracies[i] / (num_classes-1), 2)
            print(f'{avg_acc}%')

    else:
        net = Net(device, num_classes, adv_function, kernel_size).to(device)
        net.load_state_dict(torch.load(path))
        net.eval()
        print(f'Loaded the {adv_function}-{num_classes} network (creation)')
        adv_data = create_data(net, testloader, testing_eps)

        # We create the testing network and move it to the CPU along with the data
        net2 = Net(device2, num_classes, test_function, kernel_size).to(device2)
        net2.load_state_dict(torch.load(path2))
        net2.eval()
        print(f'Loaded the {test_function}-{num_classes} network (testing)')
        acc_per_eps = []
        acc_class_per_eps = []

        for i, eps in enumerate(testing_eps):
            if dec_bound and eps in plot_eps:
                plot_dec_bound(adv_data[i], 200, eps)

            print(f'Testing eps: {eps}')
            acc, classes_acc = test_model_accuracy(net2, adv_data[i], eps)
            acc_per_eps.append(acc)
            acc_class_per_eps.append(classes_acc)

        # plot_acc_graph(acc_per_eps, testing_eps)
        # plot_acc_class_per_eps(acc_class_per_eps, testing_eps)

def create_data(net, testloader, testing_eps, target_class=None, adv_class=None):
    data = []
    if attack_type == 'deepfool' or attack_type == 'boundary':
        if attack_type == 'deepfool':
            dev = torch.device('mps')
        else:
            dev = torch.device('cpu')
        num_imgs = 0
        robustness1 = []
        robustness2 = []
        start = time.time()
        for images, labels in testloader:
            correctly_predicted_images = []
            images = images.clone().detach().to(dev)
            
            if not test_all:
                new_labels = []
                for lab in labels:
                    new_labels.append(torch.tensor(test_index.index(lab)))
                labels = torch.stack(new_labels, dim=0)
            labels = labels.clone().detach().to(dev)
            _, predicted_labels = torch.max(net(images), 1)

            for i in range(len(images)):
                if predicted_labels[i] == labels[i]:
                    correctly_predicted_images.append(images[i].clone().detach())

            new_images = torch.stack(correctly_predicted_images)
            _, new_labels = torch.max(net(new_images), 1)
            num_imgs = len(new_images)
            print(f"Number of images: {num_imgs}")  

            if attack_type == 'deepfool':
                adv_images, noise, iterations, robust1, robust2 = deepfool_attack(net, new_images, new_labels)
            else:
                adv_images, noise, robust1, robust2 = boundary_attack(net, new_images, new_labels)

            num_imgs = len(new_images)
            print(f"Number of images: {num_imgs}")  

            robustness1.append(robust1)
            robustness2.append(np.mean([robust2.to(device2)]))
            data.append((new_images, adv_images, new_labels, noise))
            break

        end = time.time()
        print(f'Average iterations per image: {round(iterations / num_imgs, 3)}') 
        print(f'Average robustness per image: {round(float(np.mean(robustness1)), 5)} (deepfool method)')
        print(f'Average robustness per image: {round(float(np.mean(robustness2)), 7)} (squared error)')
        print(f'Average time spend per image: {round((end - start) / num_imgs, 3)} sec')

    else:
        for eps in testing_eps:
            adversarial_data = []
            for images, labels in testloader:
                if eps == 0:
                    adversarial_data.append((images, images, labels, None))
                elif attack_type == 'fgsm' or attack_type == 'fgsm-target':
                    adv_images, noise = fgsm_attack(net, images, labels, eps, target_class)
                    adversarial_data.append((images, adv_images, labels, noise))
                elif attack_type == 'bim':
                    adv_images, noise = bim_attack(net, images, labels, eps)
                    adversarial_data.append((images, adv_images, labels, noise))
                else:
                    images, adv_images, noise, labels = average_attack(images, labels, eps, target_class, adv_class)
                    adversarial_data.append((images, adv_images, labels, noise))

            data.append(adversarial_data)
    return data

def fgsm_attack(net, images, labels, eps, target_class) :
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
            
    outputs = net(images)
    net.zero_grad()

    if not test_all and target_class != None:
        target_class = test_index[target_class]

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
        if not test_all:
            lab = torch.tensor(train_index.index(lab))

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

    noise = noise.clamp(data_min, data_max).cpu().detach().numpy()
    return adv_class_images, attack_images, noise, adv_labels

def deepfool_attack(net, images, labels):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    if test_function == "softmax":
        attack_images, noise, _, _, iters = deepfool(net, data_min, data_max, images, device, False, labels)
    else:
        attack_images, noise, _, _, iters = deepfool(net, data_min, data_max, images, device, True, labels)
    robustness1 = compute_robustness(images, noise)
    robustness2 = (images - attack_images) ** 2
    return attack_images, noise, sum(iters), np.mean(robustness1), robustness2

def deepfool_vs_fgsm_visualize(net, testset, eps):
    args = [128, 10, 0.02, 50]
    if test_function == 'softmax':
        display_attack(device, net, testset, denormalize, data_min, data_max, eps, args, False)
    else:
        display_attack(device, net, testset, denormalize, data_min, data_max, eps, args, True)

def boundary_attack(net, images, labels):
    images = images.clone().detach().to(device2)
    labels = labels.clone().detach().to(device2)
    attack = BoundaryAttack(steps=100000)
    attack_images = attack.run(net, images, labels)

    noise = attack_images - images
    robustness1 = compute_robustness(images, noise)
    robustness2 = (attack_images - images) ** 2
    return attack_images, torch.clamp(noise, 0, 1), np.mean(robustness1), robustness2

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

        # for mnist-3&7
        if not test_all and (attack_type == 'fgsm' or attack_type == 'fgsm-target'):
            new_labels = []
            for lab in labels:
                new_labels.append(torch.tensor(test_index.index(lab)))
            labels = torch.stack(new_labels, dim=0)

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
                            elif attack_type == 'fgsm-target':
                                plot_attack(images[i], attack_images[i], noise[i,0], eps, label, predicted_labels[i], target_class, label)
                            else:
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
        classes_acc.append(round(class_correct[i] / (1e-8 + class_total[i]), 4))
        print(f'Accuracy of {classes[i]}: {round(100 * (class_correct[i] / (1e-8 + class_total[i])), 2)}%')

    test_acc = (nb - acc) / nb
    print(f'Total accuracy: {round(100 * float(test_acc), 3)}%\n')
    return test_acc, classes_acc

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def plot_dec_bound(data, n_samples, eps=0.0, fig_path=None):
    net = Net(device, num_classes, test_function, kernel_size).to(device)
    net.load_state_dict(torch.load(path2))
    if fig_path == None:
        if attack_type == 'bim' or attack_type == 'fgsm':
            attack = f"{adv_function}_{test_function}"

        fig_path = f'figures/{attack_type}/dec_bound/{attack}'
        create_path(fig_path)
        if attack_type == 'bim' or attack_type == 'fgsm' or attack_type == 'average': 
            if test_all:
                fig_path += f'/{n_samples}_{eps}.png'
            else:
                fig_path += f'/{test_index}_{n_samples}_{eps}.png'
        else:
            fig_path += f'/{n_samples}.png'
    data = [np.array(data[0][1].cpu().detach()),np.array(data[0][2].cpu().detach().int())]
    plot_decision_boundary(net, num_classes, 10, test_function, n_samples, data, fig_path, test_index)

def plot_attack(org_image, adv_image, noise, eps, label, pred_label, target_class=None, adv_class=None):
    # Denormalize the images before plotting
    org_image = denormalize(org_image).detach().numpy()
    adv_image = denormalize(adv_image).detach().numpy()

    if attack_type == 'deepfool' or attack_type == 'boundary':
        noise = denormalize(noise.to(device2)).detach().numpy().transpose(1, 2, 0)
        attack = f"{test_function}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        fig_path += f'/{label}.png'
    elif attack_type == "average":   
        attack = f"{num_classes}_{adv_class}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        fig_path += f'/{eps}_{adv_class}_to_{target_class}.png'
    else:
        attack = f"{num_classes}_{adv_function}_{test_function}"
        fig_path = f'figures/{attack_type}/{attack}'
        create_path(fig_path)
        if target_class == None:
            fig_path += f'/{eps}_{label}.png'
        else:
            fig_path += f'/{num_classes}_{adv_class}'
            create_path(fig_path)
            fig_path += f'/{eps}_{adv_class}_to_{target_class}.png'

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(org_image.transpose(1, 2, 0), cmap='gray')
    plt.title(f"Original Image ({label})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adv_image.transpose(1, 2, 0), cmap='gray')
    plt.title(f"Adversarial Image ({pred_label})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(noise,cmap='gray')
    plt.title(f"Added Noise")
    plt.axis('off')

    plt.savefig(fig_path, dpi=200)
    plt.close()

def plot_acc_graph(acc_per_eps, testing_eps, adv_class=None):
    plt.figure(figsize=(10,6))
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
    if test_all:
        path = f'runs/{num_classes}_classes/best_{adv_function}_net_checkpoint.pt'
    else:
        path = f'runs/{test_index}_classes/best_{adv_function}_net_checkpoint.pt'

    if test_all:
        path2 = f'runs/{num_classes}_classes/best_{test_function}_net_checkpoint.pt'
    else:
        path2 = f'runs/{test_index}_classes/best_{test_function}_net_checkpoint.pt'
    main()

