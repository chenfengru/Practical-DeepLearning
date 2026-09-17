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
| 02 | Autodiff & PyTorch | PyTorch homework | Not Started |
| 03 | Convolutional Neural Networks | TBD | Not Started |

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

## Goal

The goal of this repository is not only to complete the assignments, but also
to understand the mathematical and implementation details behind the methods.
