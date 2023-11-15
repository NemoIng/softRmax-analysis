# softRmax-analysis
Analysis and comparison between regular softmax and softRmax, a more robust alternative.

Testing its robustness on two well known datasets: MNIST and CIFAR10 using adversarial attacks (FGSM and BIM).

## Comparison
### FGSM Performance of softmax and softRmax for MNIST (non normalized)
<img src="https://github.com/NemoIng/softRmax-analysis/assets/82096802/56dc14d7-9716-4e53-b58b-ea29cc29f8d9" width=500>
<img src="https://github.com/NemoIng/softRmax-analysis/assets/82096802/0d0e6a0a-1dae-4862-b051-fba45276c97e" width=500>

### FGSM Performance of softmax and softRmax for CIFAR10 (normalized)
<img src="https://github.com/NemoIng/softRmax-analysis/assets/82096802/87a5dd31-5427-4538-ba0b-d758933326e6" width=500>
<img src="https://github.com/NemoIng/softRmax-analysis/assets/82096802/f589f263-7774-420c-816e-69baeac1a603" width=500>

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
