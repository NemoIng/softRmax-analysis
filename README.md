# softRmax-analysis
Analysis and comparison between regular softmax and softRmax, a more robust alternative.

Testing its robustness on three well known datasets: MNIST, CIFAR10, and Fashion-MNIST using adversarial attacks (FGSM, BIM and Average-sample).

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
