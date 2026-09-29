# Practical Deep Learning — Fall 2026

Coursework and learning notes for the Yandex Data School Practical Deep Learning course.

Official course repository:
https://github.com/yandexdataschool/Practical_DL

Course branch:
fall26

## Progress

| Week | Topic | Homework | Status |
| --- | --- | --- | --- |
| 01 | Backpropagation & Adaptive Optimization | Backprop + SGD variants | Completed ✅ |
| 02 | Autodiff & PyTorch | PyTorch homework | Completed ✅ |
| 03 | Convolutional Neural Networks | CIFAR-10 CNN | Completed ✅ |

## Week 01

Directory:

    week01_backprop/

Assignments:

- `backprop.ipynb`
  - Implement backpropagation for an arbitrary number of layers
  - 5 points

- `adaptive_sgd.ipynb`
  - Implement and compare SGD modifications
  - 5 points

### Completed

- Implemented ReLU forward/backward from scratch with NumPy
- Implemented Dense forward/backward and parameter updates
- Implemented arbitrary-depth forward propagation and backpropagation
- Used numerical gradient checks to verify layer gradients
- Trained a multilayer perceptron on MNIST
- Implemented logistic regression with quadratic feature expansion
- Implemented and tested vanilla SGD, Momentum, and RMSProp
- Bonus: implemented Adam with first/second moment estimates and bias correction
- Restarted kernels and verified both notebooks with `Run All`

### Main topics

- Forward propagation
- Chain rule
- Backpropagation
- Dense layer gradients
- Activation gradients
- Gradient checking
- Mini-batch SGD
- Momentum
- RMSProp
- Adam

## Week 02

Directory:

    week02_autodiff/

### Completed

- Practiced PyTorch tensors and vectorized operations
- Used autograd for automatic differentiation
- Trained models with `torch.optim`
- Implemented binary cross-entropy training on notMNIST
- Implemented tensor-only polar-coordinate operations
- Implemented Conway's Game of Life with PyTorch `conv2d`
- Built a nonlinear MLP with two linear layers for 10-class notMNIST classification
- Achieved 91.01% test accuracy without convolutional layers

### Main topics

- PyTorch tensors
- Automatic differentiation
- Computational graphs
- `nn.Module` and `nn.Sequential`
- Loss functions
- Optimizers
- Mini-batch training
- `conv2d` and tensor dimensions
- Multiclass classification
- Cross-entropy
- Train/evaluation modes

## Week 03

Directory:

    week03_convnets/

### Completed

- Built dense and convolutional CIFAR-10 baselines
- Added BatchNorm, dropout, pooling, and data augmentation
- Iteratively increased CNN depth from one to three convolutional blocks
- Used validation-based checkpoint selection and early stopping
- Used Apple MPS acceleration for training
- Achieved 85.61% best validation accuracy
- Achieved 84.70% final CIFAR-10 test accuracy without pretrained models


## Goal

The goal of this repository is not only to complete the assignments, but also
to understand the mathematical and implementation details behind the methods.
