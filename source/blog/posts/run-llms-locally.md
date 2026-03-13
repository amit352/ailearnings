---
title: "How to Run LLMs Locally Using Ollama"
description: "Learn how to run large language models locally using Ollama."
date: "2026-03-13"
slug: "run-llms-locally"
keywords: ["run LLMs locally", "Ollama tutorial", "local LLM", "offline LLM", "run GPT locally", "Ollama guide"]
---

# How to Run LLMs Locally Using Ollama

Running LLMs locally means no API costs, no data leaving your machine, and no rate limits. With Ollama, setting up a local language model takes less than five minutes. This guide covers installation, running models, integrating with Python, and building local AI applications.

---

## What is Running LLMs Locally

Running LLMs locally means executing language model inference on your own hardware instead of sending requests to a cloud API. The model weights run on your CPU or GPU, and all data stays on your machine.

**Ollama** is the most popular tool for local LLM deployment. It handles model downloading, GGUF quantization, hardware detection, and provides a REST API that mirrors the OpenAI API format. This compatibility means you can swap a local Ollama model for OpenAI with a single line change.

---

## Why Running LLMs Locally Matters for Developers

Local LLMs solve specific problems:

**Privacy** — Sensitive data (medical records, legal documents, proprietary code) never leaves your machine.

**Cost** — No per-token API fees. Once the model is downloaded, inference is free.

**Offline access** — Develop and test AI features without internet connectivity.

**Latency** — For simple tasks on modern hardware, local inference can be faster than an API round-trip.

**Experimentation** — Try open-source models (Llama, Mistral, Gemma, Qwen) without billing.

The tradeoff: local hardware is less powerful than cloud inference. A 4090 GPU runs 70B models slowly compared to a cluster. For production at scale, cloud APIs are still the standard.

---

## How Ollama Works

Ollama downloads quantized GGUF model files, loads them into memory, and serves them through a local REST API on `http://localhost:11434`. It automatically uses GPU acceleration when available (NVIDIA CUDA, AMD ROCm, Apple Metal).

Models are stored at `~/.ollama/models`. Quantized models are smaller than the full-precision originals — a 7B model in Q4 format is about 4GB; in Q8 format about 8GB.

---

## Practical Examples

### Installation

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from ollama.com
```

### Download and Run a Model

```bash
# Pull a model (downloads ~4GB for 7B Q4)
ollama pull llama3.2

# Run interactively
ollama run llama3.2

# Other popular models
ollama pull mistral          # Mistral 7B — fast, capable
ollama pull gemma3           # Google Gemma 3
ollama pull qwen2.5          # Alibaba Qwen 2.5
ollama pull phi4             # Microsoft Phi-4 (small but capable)
ollama pull codellama        # Code-specialized model
ollama pull nomic-embed-text # Embedding model for RAG
```

### REST API

```bash
# Chat endpoint
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "What is gradient descent?"}],
  "stream": false
}'
```

### Python with the Ollama Library

```python
import ollama

# Simple completion
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to check if a number is prime."}
    ]
)
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain recursion."}],
    stream=True
):
    print(chunk["message"]["content"], end="", flush=True)
print()
```

### OpenAI-Compatible Interface

```python
from openai import OpenAI

# Point the OpenAI client at local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Not used, but required by the client
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Summarize the transformer architecture."}],
    temperature=0,
)
print(response.choices[0].message.content)
```

This is the key advantage of Ollama's OpenAI-compatible API: you can build against the OpenAI interface and swap to local models with a single configuration change.

### Local Embeddings for RAG

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Generate embeddings with a local model
response = client.embeddings.create(
    model="nomic-embed-text",
    input=["What is retrieval augmented generation?"]
)
embedding = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding)}")
```

### LangChain + Ollama

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

llm = Ollama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Everything local — no API calls
vectorstore = Chroma.from_documents(chunks, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
result = qa_chain.invoke({"query": "What is our refund policy?"})
```

---

## Tools and Frameworks

**Ollama** — Local model server with OpenAI-compatible API. The simplest way to run open-source models.

**LM Studio** — GUI application for downloading and running models locally. Good for non-technical users.

**llama.cpp** — The underlying inference engine that Ollama uses. Can be used directly for more control.

**GPT4All** — Another local model runner with a built-in chat UI.

**text-generation-webui** — Feature-rich web UI for local model inference with support for many model formats.

---

## Common Mistakes

**Not checking hardware requirements** — A 7B model in Q4 format needs at least 8GB RAM. A 13B model needs 16GB. Check your machine's RAM before downloading.

**Running large models on CPU only** — CPU inference is 5–20x slower than GPU. If you have an NVIDIA GPU, ensure CUDA drivers are installed so Ollama uses it.

**Using the same embedding model as OpenAI RAG** — If you indexed documents with `text-embedding-3-small`, you cannot retrieve them with `nomic-embed-text`. Embedding models are not interchangeable.

**Expecting cloud-level quality from small models** — A 7B local model is not as capable as GPT-4o. Match model size to task complexity.

**Not using quantized models** — Full-precision models require more memory than most hardware has. Always use quantized (GGUF Q4 or Q8) models for local inference.

---

## Best Practices

- **Use `phi4` or `gemma3` for lightweight tasks** — Small, efficient models are often sufficient for classification, summarization, and code generation on simple inputs.
- **Keep the OpenAI interface** — Build against `openai.OpenAI` with a configurable `base_url`. Switch between local and cloud by changing one environment variable.
- **Use `nomic-embed-text` for local RAG** — It is specifically designed for embedding and retrieval, performs well, and is much faster than general-purpose models for this task.
- **Benchmark before committing** — Test a local model on your specific tasks before building an entire pipeline around it. Quality varies significantly by task type.
- **Pull models once, reuse many times** — Downloaded models persist in `~/.ollama/models`. You only need to pull once.

---

## Summary

Ollama makes running open-source LLMs locally as simple as `ollama pull llama3.2` and `ollama run llama3.2`. The OpenAI-compatible API means existing code works with minimal changes.

Local LLMs are ideal for privacy-sensitive applications, offline development, and cost-conscious experimentation. For production at scale, cloud APIs remain the practical choice.

For how LLMs work under the hood, see [how large language models work](/blog/how-llms-work/). For fine-tuning local models on custom data, see [LLM fine-tuning guide](/blog/llm-fine-tuning-guide/).
