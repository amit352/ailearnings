---
title: "Open Source LLMs Guide: Llama, Mistral, Gemma, and How to Run Them Locally"
description: "Compare the top open source LLMs — Llama 3, Mistral, Gemma, Phi — and learn how to run them locally with Ollama, llama.cpp, and Hugging Face Transformers."
date: "2026-03-10"
slug: "open-source-llms-guide"
keywords: ["open source LLMs", "run LLM locally", "Llama 3 guide", "Ollama tutorial", "Mistral guide"]
---

## Learning Objectives

- Know the major open-source LLM families and their strengths
- Run LLMs locally with Ollama (zero-config)
- Use `llama.cpp` for quantized inference on CPU
- Load models via Hugging Face for custom pipelines
- Choose the right model for your hardware and use case

---

## Why Open Source LLMs?

| Reason | Details |
|--------|---------|
| Privacy | Data never leaves your machine |
| Cost | No per-token API charges at inference time |
| Control | Fine-tune, customize, modify weights freely |
| No rate limits | Run as many requests as your hardware allows |
| Offline use | Works without internet |

The trade-off: you manage infrastructure, and the largest open models (70B+) still lag behind frontier models like GPT-4o on complex reasoning.

---

## Major Model Families (2025)

### Llama 3.x (Meta)
- Sizes: 1B, 3B, 8B, 70B, 405B
- Llama 3.2 3B/1B: multimodal vision models (compact)
- Llama 3.1 8B: best-in-class small model
- Llama 3.1 70B: near-GPT-4 quality
- License: Meta Llama 3 License (free for most commercial use)

### Mistral / Mixtral (Mistral AI)
- Mistral 7B: punches above its weight class; excellent instruction following
- Mixtral 8×7B: Mixture of Experts (MoE) — 47B total params, 13B active per token
- Mistral Nemo 12B: strong multilingual support
- License: Apache 2.0

### Gemma 2 (Google)
- Sizes: 2B, 9B, 27B
- Excellent for research and fine-tuning
- Strong on coding and reasoning for its size
- License: Custom Gemma License (free for research and commercial use)

### Phi-3 / Phi-4 (Microsoft)
- Phi-3 Mini (3.8B), Phi-3 Small (7B), Phi-3 Medium (14B)
- Phi-4 (14B): state-of-the-art at 14B size for reasoning
- Optimized for efficiency — excellent on CPU/mobile
- License: MIT

### Qwen2.5 (Alibaba)
- Sizes: 0.5B to 72B
- Coder variant excels at code generation
- Strong multilingual (especially Chinese)
- License: Apache 2.0

---

## Choosing a Model for Your Hardware

| GPU VRAM | Recommended Model | Notes |
|---------|------------------|-------|
| 4–8 GB | Phi-4 (14B Q4), Gemma 2 2B | Use 4-bit quantization |
| 8–16 GB | Llama 3.1 8B, Mistral 7B | Comfortable at 4-bit |
| 16–24 GB | Llama 3.1 8B (full), Mixtral 8×7B Q4 | Good quality |
| 40–80 GB | Llama 3.1 70B Q4 | Near-frontier quality |
| CPU only | Phi-3 Mini, Gemma 2 2B, Llama 3.2 1B | Slow but works |

---

## Method 1: Ollama (Easiest)

Ollama is a tool that makes running LLMs locally as simple as `ollama run llama3`.

### Install

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com
```

### Pull and Run a Model

```bash
# Pull a model (downloaded once)
ollama pull llama3.2           # 3B — fast, 2GB download
ollama pull mistral            # 7B — great quality
ollama pull phi4               # 14B — excellent reasoning

# Interactive chat
ollama run llama3.2
# >>> Hello! What is RAG in AI?
```

### Use the Ollama API

Ollama runs a local server at `http://localhost:11434` with an OpenAI-compatible API.

```python
from openai import OpenAI

# Point the client at your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required but ignored
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Explain transformer attention in 100 words."}
    ],
)
print(response.choices[0].message.content)
```

This means any code written for the OpenAI API works with Ollama by changing `base_url`.

### Manage Models

```bash
ollama list                   # list downloaded models
ollama rm mistral             # delete a model
ollama show llama3.2          # model info (params, context, etc.)
```

---

## Method 2: llama.cpp (CPU + Quantized Inference)

`llama.cpp` is a C++ implementation of LLaMA inference, optimized for CPU. It supports GGUF quantized model files.

### Install (Python bindings)

```bash
pip install llama-cpp-python
```

For GPU acceleration on macOS (Metal):
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

### Download a GGUF Model

Find quantized models on Hugging Face — search for "GGUF" + model name. Example: `bartowski/Llama-3.2-3B-Instruct-GGUF`.

```bash
pip install huggingface_hub
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    --local-dir ./models
```

### Run Inference

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,       # context window
    n_gpu_layers=-1,  # -1 = offload all layers to GPU (if available)
)

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What are the key differences between SQL and NoSQL?"},
    ],
    max_tokens=512,
    temperature=0.7,
)

print(output["choices"][0]["message"]["content"])
```

### Quantization Formats (GGUF)

| Format | Quality | Size Reduction |
|--------|---------|----------------|
| Q8_0 | Excellent | 45% |
| Q5_K_M | Very Good | 55% |
| Q4_K_M | Good (recommended) | 60% |
| Q3_K_M | Acceptable | 70% |
| Q2_K | Moderate | 75% |

**Q4_K_M** is the sweet spot for most use cases — minimal quality loss, fits in less VRAM.

---

## Method 3: Hugging Face Transformers

For custom pipelines, fine-tuning, and research.

```bash
pip install transformers accelerate torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "google/gemma-2-2b-it"  # instruction-tuned Gemma 2 2B

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain gradient descent in simple terms."}
]

input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

---

## Running a Local API Server

Expose your local model as an OpenAI-compatible API:

```bash
# With vLLM (high-performance, GPU)
pip install vllm
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000

# With LiteLLM proxy (wraps any model)
pip install litellm
litellm --model ollama/llama3.2 --port 8000
```

Then connect any OpenAI-compatible client to `http://localhost:8000`.

---

## Troubleshooting

**Model is very slow on CPU**
- Use a smaller model (1B–3B) or a more aggressively quantized file (Q4 or Q3)
- Increase `n_threads` in llama_cpp: `Llama(model_path=..., n_threads=8)`

**Out of memory**
- Use a smaller quantization (Q4_K_M instead of Q8_0)
- Reduce context window: `n_ctx=2048`

**Gibberish output**
- Ensure you're applying the correct chat template for the model
- Check that you're using the instruction-tuned variant (look for `-Instruct` or `-it` in the model name)

---

## FAQ

**Are open-source LLMs as good as GPT-4?**
For many tasks: yes. Llama 3.1 70B is competitive with GPT-4 on coding and general reasoning. On complex multi-step reasoning and instruction following, frontier models still lead.

**Can I use these for commercial projects?**
Most open models (Llama 3, Mistral, Gemma, Phi) allow commercial use. Always check the specific license. Meta's Llama license has restrictions for large companies (>700M MAU).

**What's the difference between base and instruct models?**
Base models predict the next token (raw language modeling). Instruct models are fine-tuned to follow instructions and have a conversation. Always use instruct models for chat applications.

---

## What to Learn Next

- **Fine-tuning these models** → [Fine-Tuning LLMs Guide](/blog/fine-tuning-llms-guide/)
- **RAG with local LLMs** → [RAG Tutorial](/blog/rag-tutorial-step-by-step/)
- **Deploying AI apps** → deploying-ai-applications
