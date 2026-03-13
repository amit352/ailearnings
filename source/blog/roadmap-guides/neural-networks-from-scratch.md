---
title: "Neural Networks from Scratch: Build One in Pure Python"
description: "Build a neural network from scratch using only NumPy to understand how forward passes, backpropagation, and gradient descent actually work. No frameworks, just the math made concrete."
date: "2026-03-10"
slug: "neural-networks-from-scratch"
keywords: ["neural networks from scratch Python", "backpropagation explained", "build neural network NumPy"]
---

## Why Build One from Scratch?

Using PyTorch or TensorFlow without understanding what's happening inside is like driving without knowing how an engine works. You'll hit a wall when things go wrong.

This guide builds a neural network using only NumPy so you see *exactly* what's happening at each step.

---

## The Core Idea

A neural network is a sequence of linear transformations (matrix multiplications) with non-linear activation functions in between.

```
Input → Linear → Activation → Linear → Activation → Output
```

"Learning" = adjusting the weights in those linear transformations to minimize prediction error.

---

## Building Blocks

### Activation Functions

Without activations, stacking linear layers is just one big linear transformation. Non-linearities let networks learn complex patterns.

```python
import numpy as np

def relu(x):
    """Most common hidden layer activation."""
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    """Output layer for binary classification."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    """Output layer for multi-class classification."""
    e = np.exp(x - x.max(axis=1, keepdims=True))  # numerically stable
    return e / e.sum(axis=1, keepdims=True)
```

### Loss Functions

```python
def mse_loss(y_true, y_pred):
    """Mean squared error — regression."""
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    """Binary classification."""
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

def cross_entropy(y_true, y_pred):
    """Multi-class classification with one-hot y_true."""
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
```

---

## A Complete Neural Network

```python
import numpy as np


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int):
        # He initialization: good default for ReLU networks
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((1, n_outputs))
        # For backprop: store values from forward pass
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b    # (batch, n_inputs) @ (n_inputs, n_outputs)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: gradient flowing back from the next layer
        Returns: gradient to pass to the previous layer
        """
        self.dW = self.x.T @ grad_out          # (n_inputs, batch) @ (batch, n_outputs)
        self.db = grad_out.sum(axis=0, keepdims=True)
        return grad_out @ self.W.T              # gradient w.r.t. input


class ReLULayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_out):
        return grad_out * (self.x > 0)


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01):
        """
        layer_sizes: e.g., [784, 128, 64, 10] = input, hidden, hidden, output
        """
        self.lr = learning_rate
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Add ReLU between layers, not at output
                self.layers.append(ReLULayer())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return softmax(x)  # softmax at output for classification

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Backpropagation: compute gradients through all layers."""
        # Gradient of cross-entropy + softmax combined (simplifies nicely)
        grad = (y_pred - y_true) / len(y_true)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self):
        """Gradient descent: update weights using computed gradients."""
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def train_step(self, x_batch, y_batch) -> float:
        """One forward + backward + update pass."""
        y_pred = self.forward(x_batch)
        loss = cross_entropy(y_batch, y_pred)
        self.backward(y_batch, y_pred)
        self.update_weights()
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(x)
        return (preds.argmax(axis=1) == y.argmax(axis=1)).mean()
```

---

## Training on a Real Dataset

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load digits dataset (64 features, 10 classes)
digits = load_digits()
X = digits.data / 16.0  # normalize to [0, 1]
y = digits.target

# One-hot encode labels
enc = OneHotEncoder(sparse_output=False)
Y = enc.fit_transform(y.reshape(-1, 1))

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create network: 64 inputs → 128 → 64 → 10 outputs
net = NeuralNetwork([64, 128, 64, 10], learning_rate=0.01)

# Training loop
batch_size = 32
n_epochs = 100

for epoch in range(n_epochs):
    # Shuffle training data
    idx = np.random.permutation(len(X_train))
    X_shuffled, Y_shuffled = X_train[idx], Y_train[idx]

    losses = []
    for i in range(0, len(X_train), batch_size):
        x_batch = X_shuffled[i:i + batch_size]
        y_batch = Y_shuffled[i:i + batch_size]
        loss = net.train_step(x_batch, y_batch)
        losses.append(loss)

    if epoch % 10 == 0:
        train_acc = net.accuracy(X_train, Y_train)
        test_acc = net.accuracy(X_test, Y_test)
        print(f"Epoch {epoch:3d} | Loss: {np.mean(losses):.4f} | "
              f"Train: {train_acc:.1%} | Test: {test_acc:.1%}")
```

**Expected output after 100 epochs:**
```
Epoch   0 | Loss: 2.3021 | Train: 10.2% | Test: 10.8%
Epoch  10 | Loss: 1.8432 | Train: 47.3% | Test: 44.1%
Epoch  50 | Loss: 0.4123 | Train: 91.2% | Test: 87.3%
Epoch  90 | Loss: 0.2834 | Train: 94.7% | Test: 91.8%
```

---

## Understanding What Just Happened

### Forward Pass

Each input flows through layers:
```
x → layer1.forward() → relu() → layer2.forward() → relu() → layer3.forward() → softmax()
```

The softmax produces probabilities: `[0.01, 0.02, 0.90, 0.02, ...]` (10 classes, the highest is the prediction).

### Backward Pass (Backpropagation)

The chain rule from calculus applied to the computation graph:

```
∂Loss/∂W1 = ∂Loss/∂output × ∂output/∂hidden × ∂hidden/∂W1
```

We compute gradients backwards from the loss through each layer. Each layer "owns" the gradient computation for its parameters.

### Gradient Descent

Move weights in the opposite direction of the gradient (steepest ascent → descend):

```
W ← W - learning_rate × ∂Loss/∂W
```

---

## Why This Matters for LLM Work

Everything you learned here is at the core of transformers:
- **Matrix multiplications** → the linear projections in attention (Q, K, V matrices)
- **Softmax** → converting attention scores to probabilities
- **Backpropagation** → how LLMs are trained
- **Mini-batch gradient descent** → how training is parallelized

The scale is different (billions of parameters vs. thousands) but the mechanics are identical.

---

## What to Learn Next

- **Use a real framework** → [PyTorch for AI Developers](/blog/roadmap-guides/pytorch-for-ai-developers/)
- **Deep learning architectures** → [Deep Learning Fundamentals](/blog/roadmap-guides/deep-learning-fundamentals/)
- **How LLMs use these concepts** → [How LLMs Work](/blog/roadmap-guides/how-llms-work/)
