---
title: "Deep Learning Fundamentals: CNNs, RNNs, and the Road to Transformers"
description: "Understand the deep learning architectures that shaped modern AI — convolutional networks for images, recurrent networks for sequences, and how their limitations led to the transformer revolution."
date: "2026-03-10"
slug: "deep-learning-fundamentals"
keywords: ["deep learning fundamentals", "CNN RNN transformer explained", "deep learning architectures"]
---

## What Makes Deep Learning "Deep"?

Depth = number of layers. A "shallow" network has 1-2 hidden layers. Deep networks have dozens to hundreds. The key insight from the 2010s: *depth enables hierarchical feature learning*.

```
Shallow:  Input → Features → Output
Deep:     Input → Low-level → Mid-level → High-level → Output
          (pixels) (edges) (shapes) (objects) (faces)
```

Each layer learns increasingly abstract representations. The output of each layer becomes the input to the next.

---

## Convolutional Neural Networks (CNNs)

CNNs are specialized for grid-like data (images, time series). Instead of connecting every neuron to every input (too expensive for images), they slide a small filter across the input.

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 channels → 32 feature maps, 3x3 filter
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (B, 3, H, W) → (B, 32, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # halves spatial dims

            # Block 2: 32 → 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → (B, 128, 4, 4) regardless of input size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (B, 128*4*4) = (B, 2048)
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN(n_classes=10)
x = torch.randn(8, 3, 32, 32)  # batch of 8 RGB 32x32 images
output = model(x)
print(output.shape)  # (8, 10)
```

**Key CNN concepts:**
- **Convolution**: scan with a small filter to detect local patterns
- **Pooling (MaxPool)**: downsample — keep only the strongest activation
- **Feature maps**: the outputs of each layer, detecting different patterns
- **Receptive field**: how much of the input each neuron "sees"

**When to use CNNs**: images, spectrograms, spatial data.

**Famous architectures**: LeNet (1998), AlexNet (2012), VGG, ResNet (2015).

---

## Residual Connections (ResNets)

Training very deep networks (100+ layers) was difficult — gradients vanish during backpropagation. ResNets solved this with skip connections:

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Skip connection: add input directly to output
        return self.relu(self.block(x) + x)
        # Even if block(x) → 0 (vanishing gradient), gradients still flow through x
```

This allowed training of 152-layer networks. The intuition: it's easier to learn a "residual" (small change) than a full transformation.

---

## Recurrent Neural Networks (RNNs)

For sequences (text, time series, audio), order matters. RNNs maintain a hidden state that carries information from previous timesteps:

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # hidden_state = tanh(W_hh * prev_hidden + W_ih * input + bias)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, hidden = self.rnn(x)
        # output: (batch, seq_len, hidden_size) — all timestep outputs
        # hidden: (1, batch, hidden_size) — final hidden state

        # Use final output for classification
        return self.fc(output[:, -1, :])


# For text: embed tokens first
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, token_ids):
        x = self.embedding(token_ids)        # (B, seq_len, embed_dim)
        out, _ = self.rnn(x)                  # (B, seq_len, hidden*2)
        return self.fc(out[:, -1, :])         # classify from final state
```

### LSTM: The Gated Improvement

Vanilla RNNs suffer from vanishing gradients on long sequences. LSTMs add "gates" to selectively remember or forget:

```python
# LSTM has 4 gates, maintains both hidden state and cell state
lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)

# GRU: simpler alternative to LSTM, often works as well
gru = nn.GRU(input_size=128, hidden_size=256, batch_first=True)
```

**RNN/LSTM limitations** (that transformers solved):
1. Sequential processing — can't parallelize
2. Long-range dependencies — even LSTMs struggle with very long sequences
3. Fixed-size bottleneck — sequence → single vector loses information

---

## The Transformer Revolution

Transformers (2017) replaced RNNs for sequence tasks by using *attention* instead of recurrence:

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape

        # Project and split into heads
        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        # Feed-forward with residual connection
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
```

**Why transformers won:**
- **Parallelizable**: all positions processed simultaneously (unlike RNNs)
- **Long-range attention**: any token can directly attend to any other token
- **Scalable**: more data + more compute = better performance (scaling laws)

---

## Batch Normalization and Layer Normalization

Normalization layers stabilize training:

```python
# BatchNorm: normalize across the batch dimension (used in CNNs)
bn = nn.BatchNorm2d(64)  # for feature maps

# LayerNorm: normalize across the feature dimension (used in transformers)
ln = nn.LayerNorm(512)  # for sequences

# Why LayerNorm in transformers?
# - Works at any batch size (including batch=1)
# - Works consistently for sequences of different lengths
```

---

## Training Deep Networks: Practical Tips

```python
# 1. Optimizer choice
# Adam: usually best default
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# AdamW: Adam with weight decay (preferred for transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 2. Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 3. Gradient clipping (prevents exploding gradients in RNNs/transformers)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Mixed precision training (2x faster on modern GPUs)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 5. Dropout for regularization
dropout = nn.Dropout(p=0.3)  # set p=0 during eval
model.train()   # enables dropout
model.eval()    # disables dropout (and BatchNorm running stats)
```

---

## Pre-trained Models: The Modern Approach

In practice, you rarely train CNNs or transformers from scratch. You use pre-trained models:

```python
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer

# Pre-trained ResNet for images (transfer learning)
resnet = models.resnet50(pretrained=True)
# Freeze base, replace classifier
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(2048, 5)  # 5-class problem
# Now only train the new final layer

# Pre-trained BERT for text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")
```

**Transfer learning** — take a model trained on millions of examples, fine-tune the last few layers on your smaller dataset. Often achieves 90%+ of training-from-scratch performance with 1% of the data.

---

## What to Learn Next

- **PyTorch hands-on** → [PyTorch for AI Developers](/blog/roadmap-guides/pytorch-for-ai-developers/)
- **How LLMs are built on transformers** → [How LLMs Work](/blog/roadmap-guides/how-llms-work/)
- **Fine-tuning pre-trained models** → [Fine-Tuning LLMs Guide](/blog/fine-tuning-llms-guide/)
