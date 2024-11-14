# **Fashion-MNIST Classification using Custom MLP and CNN Models**

This project implements image classification on the Fashion-MNIST dataset using a custom Multilayer Perceptron (MLP) and a Convolutional Neural Network (CNN). The MLP is implemented from scratch, including forward and backward passes, while the CNN utilizes PyTorch's built-in modules. The goal is to understand the fundamentals of neural networks and explore the integration of custom components with standard deep learning frameworks.

---

## **Table of Contents**

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [Custom MLP Classifier](#1-custom-mlp-classifier)
  - [CNN Backbone Model](#2-cnn-backbone-model)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## **Overview**

- **Objective:** Classify images in the Fashion-MNIST dataset using both a custom-implemented MLP and a CNN model.
- **Key Features:**
  - Implemented MLP from scratch, including custom forward and backward passes.
  - Utilized PyTorch's modules for the CNN backbone.
  - Integrated the custom MLP classifier with the CNN model.
  - Explored different hyperparameters and weight initialization methods.
  - Evaluated models on training, validation, and test datasets.

---

## **Dataset**

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is a collection of 70,000 grayscale images of 28x28 pixels, categorized into 10 fashion classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

**Data Split:**

- **Training Set:** 48,000 images (80% of training data)
- **Validation Set:** 12,000 images (20% of training data)
- **Test Set:** 10,000 images

---

## **Model Architecture**

### **1. Custom MLP Classifier**

- **Implemented from Scratch:**
  - **Linear Layers:** Custom implementation of linear transformations.
  - **Activation Functions:** Custom ReLU activation functions.
  - **Forward Pass:** Manually coded to process inputs through the network layers.
  - **Backward Pass:** Manually computed gradients for backpropagation.
- **Architecture:**
  - **Input Layer:** Accepts flattened 28x28 images (784 inputs).
  - **Hidden Layers:**
    - First hidden layer with 256 neurons.
    - Second hidden layer with 128 neurons.
  - **Output Layer:** 10 neurons corresponding to the 10 classes.
- **Loss Function:** Custom implementation of cross-entropy loss.

### **2. CNN Backbone Model**

- **Convolutional Layers:**
  - **Layer 1:** Conv2d (1 input channel, 32 output channels) → ReLU → MaxPool2d
  - **Layer 2:** Conv2d (32 → 64) → ReLU → MaxPool2d
  - **Layer 3:** Conv2d (64 → 128) → ReLU
  - **Layer 4:** Conv2d (128 → 256) → ReLU
  - **Layer 5:** Conv2d (256 → 512) → ReLU → MaxPool2d
- **Feature Extraction:** Extracted features are flattened and fed into the custom MLP classifier.
- **Weight Initialization:** He (Kaiming) initialization for convolutional layers.

---

## **Requirements**

- **Python 3.6+**
- **Libraries:**
  - `torch`
  - `torchvision`
  - `numpy`
  - `matplotlib` (optional, for plotting)
- **Hardware:**
  - CPU or GPU (optional, for faster training)

---

## **Installation**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/fashion-mnist-mlp-cnn.git
   cd fashion-mnist-mlp-cnn
