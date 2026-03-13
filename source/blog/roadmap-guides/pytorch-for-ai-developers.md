---
title: "PyTorch for AI Developers: Practical Guide from Tensors to Training"
description: "A hands-on PyTorch guide covering tensors, autograd, model building with nn.Module, training loops, GPU acceleration, and loading pre-trained models from HuggingFace."
date: "2026-03-10"
slug: "pytorch-for-ai-developers"
keywords: ["PyTorch tutorial for developers", "PyTorch nn.Module training", "PyTorch beginner guide"]
---

## Why PyTorch?

PyTorch is the dominant framework for AI research and is rapidly taking over production ML too. HuggingFace, most LLM research, and a growing share of production systems run on PyTorch.

```bash
pip install torch torchvision transformers datasets
```

---

## Tensors: NumPy with Superpowers

Tensors are like NumPy arrays but can run on GPU and track gradients.

```python
import torch
import numpy as np

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)               # 3x4 tensor of zeros
z = torch.randn(2, 3, 4)            # random normal, shape (2, 3, 4)

# Shape, dtype, device
print(x.shape)    # torch.Size([3])
print(z.shape)    # torch.Size([2, 3, 4])
print(x.dtype)    # torch.float32

# Move to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
x_gpu = x.to(device)

# Operations (same as NumPy)
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
print(a + b)          # element-wise
print(a @ b)          # matrix multiply
print(a.T)            # transpose
print(a.sum())        # 10.0
print(a.mean(dim=0))  # mean along first dimension

# Convert to/from NumPy
arr = x.numpy()         # tensor → numpy (CPU only)
t = torch.from_numpy(arr)  # numpy → tensor
```

---

## Autograd: Automatic Differentiation

PyTorch tracks computations and computes gradients automatically:

```python
# requires_grad=True tells PyTorch to track this tensor
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

x = torch.tensor(2.0)

# Forward pass: compute output
y_pred = w * x + b   # y = 3*2 + 1 = 7
loss = (y_pred - 5.0) ** 2  # (7 - 5)^2 = 4

# Backward pass: compute gradients
loss.backward()

print(w.grad)  # d(loss)/dw = 2*(y_pred - 5) * x = 2*2*2 = 8
print(b.grad)  # d(loss)/db = 2*(y_pred - 5) * 1 = 4

# Update weights manually
with torch.no_grad():  # don't track gradient for this operation
    w -= 0.1 * w.grad   # gradient descent step
    b -= 0.1 * b.grad
    w.grad.zero_()       # clear gradients (they accumulate otherwise!)
    b.grad.zero_()
```

---

## Building Models with nn.Module

Every PyTorch model is an `nn.Module`:

```python
import torch.nn as nn

class MLP(nn.Module):
    """Multi-layer perceptron."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(input_size=128, hidden_sizes=[256, 128], output_size=10)
print(model)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")   # 66,826

# Sample forward pass
x = torch.randn(32, 128)   # batch of 32
out = model(x)
print(out.shape)   # (32, 10)
```

---

## The Training Loop

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────
        model.train()   # enable dropout + batch norm training mode
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()           # clear previous gradients
            output = model(x_batch)         # forward pass
            loss = criterion(output, y_batch)  # compute loss
            loss.backward()                 # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimizer.step()                # update weights
            total_loss += loss.item()

        # ── Validation ────────────────────────────────────────────
        model.eval()    # disable dropout
        correct = total = 0
        with torch.no_grad():   # no gradient tracking needed
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val).argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += len(y_val)

        val_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1%}")

    return history


# Example usage
X = torch.randn(1000, 128)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
train_set, val_set = torch.utils.data.random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

model = MLP(128, [256, 128], 10).to(device)
history = train(model, train_loader, val_loader, epochs=20)
```

---

## Saving and Loading Models

```python
# Save: model weights only (recommended)
torch.save(model.state_dict(), "model.pt")

# Load
loaded_model = MLP(128, [256, 128], 10)
loaded_model.load_state_dict(torch.load("model.pt", map_location=device))
loaded_model.eval()

# Save: full model (architecture + weights, less portable)
torch.save(model, "full_model.pt")
loaded = torch.load("full_model.pt")
```

---

## GPU Acceleration

```python
# Check available hardware
print(torch.cuda.is_available())          # NVIDIA GPU
print(torch.backends.mps.is_available())  # Apple Silicon

# Recommended device selection
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Move model and data to device
model = model.to(device)
x = x.to(device)

# With NVIDIA: multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## Working with HuggingFace

HuggingFace Transformers is built on top of PyTorch:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Inference
texts = ["This movie was fantastic!", "I hated every minute of it."]
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.softmax(dim=-1)

labels = ["NEGATIVE", "POSITIVE"]
for text, pred in zip(texts, predictions):
    label = labels[pred.argmax()]
    confidence = pred.max().item()
    print(f"{label} ({confidence:.1%}): {text}")
```

---

## Debugging Tips

```python
# 1. Check for NaN in outputs
assert not torch.isnan(output).any(), "NaN in model output!"

# 2. Print shapes at each layer
class DebugNet(nn.Module):
    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.layer1(x)
        print(f"After layer1: {x.shape}")
        return x

# 3. Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")

# 4. Memory usage on GPU
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
torch.cuda.empty_cache()  # free cached memory
```

---

## What to Learn Next

- **How transformers work** → [How LLMs Work](/blog/roadmap-guides/how-llms-work/)
- **Fine-tuning LLMs with HuggingFace** → [Fine-Tuning LLMs Guide](/blog/fine-tuning-llms-guide/)
- **Build a project** → [RAG Document Assistant](/projects/rag-document-assistant/)
