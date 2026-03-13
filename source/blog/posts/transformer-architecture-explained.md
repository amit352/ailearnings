---
title: "Transformer Architecture Explained for Developers"
description: "Learn how transformer models power modern AI systems."
date: "2026-03-13"
slug: "transformer-architecture-explained"
keywords: ["transformer architecture", "how transformers work", "self-attention explained", "LLM architecture", "transformer explained"]
---

## Learning Objectives

- Understand why transformers replaced RNNs for language tasks
- Explain self-attention and why it matters
- Walk through the full transformer architecture layer by layer
- Distinguish between encoder-only, decoder-only, and encoder-decoder models
- Know how pre-training and fine-tuning work at a high level

---

## Why Transformers?

Before transformers (2017), sequence modeling used RNNs and LSTMs. These had two critical limitations:

1. **Sequential processing** — each token depends on the previous, making parallelization impossible during training
2. **Vanishing gradients** — long-range dependencies were hard to learn; the model "forgot" tokens from early in the sequence

The transformer architecture, introduced in ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), solved both problems:

- **Parallel processing** — all tokens are processed simultaneously
- **Direct attention** — any token can attend to any other token in a single step, regardless of distance

---

## The Big Picture

A transformer processes a sequence of tokens. Each token is first converted to a vector (embedding), then processed through multiple transformer blocks, and finally decoded into a prediction.

```
Input Tokens → Embeddings + Positional Encoding
     ↓
[Transformer Block] × N layers
     ↓
Output (next token logits, classification, etc.)
```

Each **Transformer Block** contains:
1. Multi-Head Self-Attention
2. Add & Norm (residual connection + layer normalization)
3. Feed-Forward Network
4. Add & Norm

---

## Token Embeddings

Tokens (words or subwords) are mapped to dense vectors via an embedding matrix.

```python
import torch
import torch.nn as nn

vocab_size = 50000
d_model    = 512  # embedding dimension

embedding = nn.Embedding(vocab_size, d_model)
tokens    = torch.tensor([[1, 542, 23, 9, 1002]])  # batch of 1 sequence
x         = embedding(tokens)  # shape: (1, 5, 512)
```

The embedding dimension `d_model` is a key hyperparameter. GPT-2 uses 768; GPT-3 uses 12288.

---

## Positional Encoding

Self-attention has no inherent sense of order. Positional encoding adds position information to each token embedding.

**Sinusoidal positional encoding (original paper):**

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # even dims
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
    return pe  # shape: (seq_len, d_model)
```

Modern LLMs often use **Rotary Positional Embedding (RoPE)** or **ALiBi** instead, which better handle longer sequences.

---

## Self-Attention: The Core Mechanism

Self-attention lets each token look at all other tokens and decide which ones to focus on.

### Queries, Keys, and Values

Each token embedding is projected into three vectors via learned weight matrices:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I pass on?"

```python
d_k = 64  # dimension of queries and keys

# Learned projections
W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_k, bias=False)

Q = W_q(x)  # shape: (batch, seq_len, d_k)
K = W_k(x)
V = W_v(x)
```

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```

```python
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

**The intuition:** `QK^T` computes a similarity score between every pair of tokens. Softmax converts scores to probabilities (attention weights). The output is a weighted average of the value vectors — tokens attend more to semantically related tokens.

The `1/sqrt(d_k)` scaling prevents softmax from saturating when d_k is large (which would push gradients toward zero).

---

## Multi-Head Attention

Instead of one attention head, transformers run H attention heads in parallel, each with its own Q/K/V projections. Each head can learn to attend to different types of relationships.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.size()
        h = self.num_heads

        # Project and split into heads
        Q = self.W_q(x).view(batch, seq_len, h, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, h, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, h, self.d_k).transpose(1, 2)

        # Attention per head
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_o(attn_out)
```

GPT-3 uses 96 attention heads. GPT-2 (small) uses 12 heads.

---

## Feed-Forward Network

After attention, each token passes through a two-layer fully connected network independently:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),            # modern models use GELU over ReLU
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)
```

The FFN dimension `d_ff` is typically 4× the model dimension.

---

## Layer Normalization and Residual Connections

Residual connections (skip connections) allow gradients to flow directly through the network:

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

**Pre-norm** (shown above, used in GPT models) is more stable during training than the original **post-norm** formulation.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ffn   = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
```

---

## Model Variants

### Encoder-Only (BERT family)
- Bidirectional: each token sees all other tokens
- Used for: text classification, NER, question answering
- Pre-trained with masked language modeling (MLM)
- Examples: BERT, RoBERTa, DeBERTa

### Decoder-Only (GPT family)
- Causal/autoregressive: each token sees only previous tokens (causal mask)
- Used for: text generation, chat, completion
- Pre-trained with next-token prediction (CLM)
- Examples: GPT-2, GPT-4, LLaMA, Mistral, Gemma

### Encoder-Decoder (T5 / BART family)
- Encoder processes input, decoder generates output
- Used for: translation, summarization, structured generation
- Examples: T5, BART, mT5

---

## Pre-training vs Fine-tuning

**Pre-training:** Train on massive text corpus (terabytes) to predict next tokens. This teaches general language understanding. Very expensive (millions of dollars in compute).

**Fine-tuning:** Take a pre-trained model and continue training on a smaller task-specific dataset. Adapts the model to a specific task or domain. Affordable.

**PEFT (Parameter-Efficient Fine-Tuning):** Instead of updating all parameters, methods like **LoRA** add small trainable adapter matrices. Reduces memory and compute by 10–100×.

---

## Troubleshooting

**Out-of-memory errors during training**
- Reduce batch size and increase gradient accumulation steps
- Use gradient checkpointing: `model.gradient_checkpointing_enable()`
- Use mixed precision: `torch.cuda.amp.autocast()`

**Model generates repetitive text**
- Adjust temperature (higher = more diverse)
- Use repetition penalty
- Try top-p (nucleus) sampling instead of greedy decoding

**Attention weights don't look meaningful**
- This is normal for lower layers. Deeper layers develop more interpretable attention patterns.

---

## FAQ

**How many parameters does a transformer have?**
Depends on depth and width. GPT-2: 117M–1.5B. GPT-3: 175B. Llama 3: 8B–70B. Most parameters live in the FFN and attention projection matrices.

**What is the context window?**
The maximum sequence length the model can process at once. Determined by positional encoding design and attention mask. GPT-4 supports 128K tokens.

**What is a token?**
Typically a word piece (subword unit). "unbelievable" might tokenize to ["un", "believ", "able"]. Most LLMs use BPE (Byte Pair Encoding) tokenization.

---

## What to Learn Next

- **How LLMs work overall** → [How Large Language Models Work](/blog/how-llms-work/)
- **The attention mechanism in depth** → [Attention Mechanism Explained](/blog/attention-mechanism-explained/)
- **Fine-tuning LLMs** → [LLM Fine-Tuning Guide](/blog/llm-fine-tuning-guide/)
