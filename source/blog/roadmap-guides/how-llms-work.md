---
title: "How LLMs Work: From Pretraining to ChatGPT — A Developer's Guide"
description: "A technical deep dive into how large language models actually work — tokenization, pretraining, RLHF, inference, and the architectural decisions that define modern LLMs like GPT-4 and Claude."
date: "2026-03-10"
slug: "how-llms-work"
keywords: ["how LLMs work explained", "large language models explained", "GPT architecture explained"]
---

## The Core Mechanism: Next Token Prediction

Everything in LLMs starts here. Given a sequence of tokens, predict the next one:

```
"The capital of France is" → "Paris"
"def fibonacci(n):" → "\n    if n <= 1:"
```

That's it. A model trained to do this *extremely well* across trillions of text examples learns to encode:
- Grammar and syntax
- Facts about the world
- Reasoning patterns
- Code logic
- Conversational patterns

The "intelligence" is an emergent property of doing next-token prediction at scale.

---

## Tokenization

Text must be converted to integers before the model can process it.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, world! How are you?"
tokens = enc.encode(text)
print(tokens)  # [9906, 11, 1917, 0, 2650, 527, 499, 30]
print(len(tokens))  # 8 tokens

# Decode back
decoded = enc.decode(tokens)
print(decoded)  # "Hello, world! How are you?"

# Tokens are NOT words
tokens = enc.encode("unbelievable")
print(enc.decode_single_token_bytes(t) for t in tokens)
# [b'un', b'bel', b'iev', b'able']
```

**Byte Pair Encoding (BPE)** — the algorithm most LLMs use:
1. Start with individual characters/bytes as vocabulary
2. Find most frequent pair (e.g., 't' + 'h' → 'th')
3. Merge it into the vocabulary
4. Repeat until vocabulary size is reached (GPT-4: ~100k tokens)

**Why it matters:**
- GPT-4 context window = 128k *tokens*, not words
- Non-English text uses more tokens per word → more expensive
- Rare words split into many tokens

---

## Architecture: The GPT Transformer

Modern LLMs use a **decoder-only transformer** (as opposed to encoder-decoder models like the original Transformer or T5).

```python
import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """Self-attention that can only attend to past tokens (causal/autoregressive)."""
    def __init__(self, d_model, n_heads, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask: prevent attending to future positions
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # pre-norm residual
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, max_seq=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)    # learned positional encoding
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share embedding and unembedding weights
        self.token_emb.weight = self.head.weight

    def forward(self, token_ids):
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))    # (B, T, vocab_size)
        return logits

    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        """Autoregressive generation: sample one token at a time."""
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            context = ids[:, -1024:]  # truncate to context window
            logits = self(context)[:, -1, :] / temperature

            # Top-k sampling: sample from top k most likely tokens
            top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_vals, dim=-1)
            sampled = torch.multinomial(probs, 1)
            next_token = top_idx.gather(-1, sampled)

            ids = torch.cat([ids, next_token], dim=1)
        return ids
```

**Scale of real LLMs:**
| Model | d_model | n_heads | n_layers | Parameters |
|-------|---------|---------|----------|------------|
| GPT-2 small | 768 | 12 | 12 | 117M |
| GPT-2 XL | 1600 | 25 | 48 | 1.5B |
| GPT-3 | 12288 | 96 | 96 | 175B |
| GPT-4 (est.) | ~25600 | 128 | 120 | ~1.8T |

---

## Pretraining: Learning from the Web

```python
# Simplified pretraining loop (conceptually)
model = MiniGPT(vocab_size=50257)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training data: internet text, tokenized
# Process: predict each token from context
for batch in data_loader:
    # batch: (B, seq_len) token IDs
    input_ids = batch[:, :-1]    # all tokens except last
    target_ids = batch[:, 1:]    # shifted by one — the "next" token

    logits = model(input_ids)    # (B, seq_len, vocab_size)
    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, 50257),   # flatten batch × seq
        target_ids.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

GPT-3 was trained on ~300B tokens. GPT-4 on ~13T tokens. This takes thousands of H100 GPUs for months. The resulting model is called the "base model" — it completes text but isn't yet useful as an assistant.

---

## Instruction Tuning and RLHF

**The problem:** A base model asked "What is the capital of France?" might respond with:
```
"What is the capital of Germany? What is the capital of Spain? ..."
```
(continuing the "list of trivia questions" pattern from pretraining data)

**Instruction tuning** (SFT — Supervised Fine-Tuning): Fine-tune on examples of (instruction, good response) pairs. Teaches the model to *follow instructions*.

**RLHF (Reinforcement Learning from Human Feedback)**:
1. Collect human preferences: "Response A is better than B"
2. Train a **reward model** to predict human preferences
3. Use PPO (reinforcement learning) to optimize the LLM to maximize reward

This is what turns "base GPT-4" → "ChatGPT." The model learns to be helpful, harmless, and honest per human evaluators' preferences.

---

## Inference: How Generation Actually Works

```python
# What happens when you call the API:
from openai import OpenAI
client = OpenAI()

# Each token is generated ONE AT A TIME
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,  # see each token as it's generated
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
# Output appears token by token: "1... 2... 3... 4... 5..."
```

**Inference is autoregressive:** generate token → append to context → generate next token → repeat until `<|endoftext|>` or max_tokens.

**KV Cache:** During inference, key-value pairs are computed once and cached. Without KV cache, each token generation would be O(n²) in sequence length.

**Latency comes from:**
- Time to first token (TTFT): loading prompt into memory
- Tokens per second: how fast the model generates

---

## The Context Window

```python
# Context window = how much text the model can "see" at once
# Everything outside the window is forgotten

# GPT-4o: 128k tokens ≈ 95,000 words ≈ a medium-length novel

# Managing long contexts:
# 1. Summarize older messages
# 2. Use RAG — retrieve relevant chunks instead of loading everything
# 3. Use models with larger windows (Gemini 1.5: 1M tokens)

# Check how many tokens you're using
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

messages = [{"role": "user", "content": "Some very long text..."}]
total_tokens = sum(len(enc.encode(m["content"])) for m in messages)
print(f"Using {total_tokens} / 128,000 tokens")
```

---

## Temperature and Sampling

The model outputs logits (raw scores) for each possible next token. How you convert these to a choice determines output character:

```python
import torch

logits = torch.tensor([2.0, 1.5, 0.5, -1.0, ...])  # one score per vocab token

# Temperature scaling
temperature = 0.7  # < 1.0 = sharper distribution (more deterministic)
scaled = logits / temperature

# Softmax: convert to probabilities
probs = torch.softmax(scaled, dim=0)

# Strategies:
# 1. Greedy (temperature=0): always take argmax
next_token = probs.argmax()

# 2. Sampling (temperature>0): sample from distribution
next_token = torch.multinomial(probs, 1)

# 3. Top-p (nucleus) sampling: only sample from tokens summing to p
sorted_probs, sorted_idx = torch.sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=0)
remove = cumsum - sorted_probs > 0.9  # top-p = 0.9
sorted_probs[remove] = 0
next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
```

---

## What to Learn Next

- **Run LLMs locally** → [Working with Local LLMs](/blog/roadmap-guides/working-with-local-llms/)
- **Fine-tune your own LLM** → [Fine-Tuning LLMs Guide](/blog/fine-tuning-llms-guide/)
- **Build with LLMs** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
