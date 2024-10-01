# softRmax Analysis
This repository contains an analysis and comparison between the traditional softmax function and softRmax, a more robust alternative. We evaluate the performance of these methods on three well-known datasets: MNIST, CIFAR10, and Fashion-MNIST, particularly under adversarial attacks, including FGSM, BIM, and Average-sample.

## Bachelor Paper
The accompanying [bachelor's thesis](https://www.cs.ru.nl/bachelors-theses/2024/Nemo_Ingendaa___1063653___Analysis_and_exploration_of_polynomial_activation_function_-_softRmax.pdf) details the testing methods and provides a comprehensive discussion of the experimental results obtained from the code in this repository.

## Installation & Workings 
**Installation**
```
#  Install neccessary libaries
pip install -r requirements.txt

# Train model
# 1. Set prefered model/data parameters
# 2. Run training script
python cifar-train.py
python mnist-train.py

# Attack
# 1. Set prefered model/data parameters
# 2. Run attack script
python cifar-attack.py
python mnist-attack.py
```
