---
title: "LLM Inference and Serving: vLLM, Ollama, and Production Optimization"
description: "Guide to LLM inference optimization — KV cache, continuous batching, quantization, vLLM serving, Ollama local deployment, and benchmarking throughput and latency."
date: "2026-03-10"
slug: "llm-inference-and-serving"
keywords: ["LLM inference", "LLM serving", "vLLM guide", "LLM optimization", "inference optimization"]
---

## Learning Objectives

- Understand the bottlenecks in LLM inference
- Use vLLM for high-throughput production serving
- Deploy local models with Ollama
- Apply quantization to reduce memory requirements
- Benchmark and optimize inference performance

---

## LLM Inference Fundamentals

### The Two Phases of Inference

**Prefill (prompt processing):** Process all input tokens in parallel. GPU-bound. Fast.

**Decode (token generation):** Generate one token at a time. Memory bandwidth-bound. Slow.

The decode phase is the primary bottleneck. Most optimization techniques target it.

### KV Cache

During decoding, the model recomputes attention keys and values for every previous token on each step — unless you cache them.

**KV cache** stores (Key, Value) pairs from previous decoding steps. This reduces redundant computation from `O(n²)` to `O(n)` per step.

The KV cache grows with sequence length and batch size. A 70B model with 4096-token context uses ~80GB of KV cache memory at full batch.

### Memory Layout

```
GPU Memory = Model Weights + KV Cache + Activations
           ~  70GB         +  ~20GB   +  ~5GB     (70B model, BF16)
```

Running out of KV cache space is the primary cause of OOM errors during inference.

---

## vLLM: High-Throughput Serving

vLLM introduces **PagedAttention** — manages KV cache in fixed-size blocks (like OS memory pages), enabling efficient sharing and dramatically higher throughput.

```bash
pip install vllm
```

### Start a vLLM Server

```bash
# Serve Llama 3.1 8B with OpenAI-compatible API
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1   # use 2+ for multi-GPU

# With quantization (half the memory)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization awq \
    --dtype half
```

### Call the vLLM Server

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain attention mechanisms."}],
    max_tokens=512,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "List 5 Python tips."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### vLLM Async Python Client

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate(prompt: str) -> str:
    sampling = SamplingParams(temperature=0.7, max_tokens=256)
    results = engine.generate(prompt, sampling, request_id="req1")

    full_output = ""
    async for output in results:
        full_output = output.outputs[0].text

    return full_output

asyncio.run(generate("What is PagedAttention?"))
```

---

## Ollama: Zero-Config Local Serving

Ollama is the easiest way to run open-source LLMs locally. Perfect for development.

```bash
# Install (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run
ollama pull llama3.2           # 2GB download
ollama run llama3.2            # interactive chat

# Run as server (starts automatically on install)
# Server at http://localhost:11434
```

### Custom Modelfile

Customize system prompts and parameters:

```modelfile
# Create file: Modelfile
FROM llama3.2

SYSTEM """You are a concise coding assistant. Always include working code examples."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

```bash
ollama create coding-assistant -f Modelfile
ollama run coding-assistant
```

### Ollama Python API

```python
import requests

def ollama_chat(prompt: str, model: str = "llama3.2") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
    )
    return response.json()["response"]


# Or use the OpenAI-compatible endpoint
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain RLHF in 100 words."}],
)
```

---

## Quantization

Quantization reduces model precision from 32-bit floats to 8-bit or 4-bit integers. Roughly halves memory per bit reduction with modest quality loss.

### GGUF Quantization (llama.cpp)

```python
from llama_cpp import Llama

# Q4_K_M: good balance of quality and size
llm = Llama(
    model_path="./models/Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # offload all layers to GPU
    n_threads=8,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is quantization?"}],
    max_tokens=256,
)
print(output["choices"][0]["message"]["content"])
```

### AWQ Quantization (for vLLM)

```bash
pip install autoawq

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
model.quantize(tokenizer, quant_config={'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'})
model.save_quantized('./Llama-3.1-8B-Instruct-AWQ')
"
```

### Memory Savings by Quantization Format

| Format | Bits | 8B Model Size | Quality Loss |
|--------|------|---------------|-------------|
| BF16 | 16 | 16 GB | None (baseline) |
| INT8 | 8 | 8 GB | Minimal |
| Q4_K_M | ~4.5 | 5 GB | Low |
| Q3_K_M | ~3.5 | 4 GB | Moderate |
| Q2_K | ~2.5 | 3 GB | Noticeable |

---

## Benchmarking Inference

```python
import time
import statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm")

def benchmark_latency(
    prompt: str,
    model: str,
    n_runs: int = 20,
    max_tokens: int = 100,
) -> dict:
    latencies = []
    tokens_per_sec = []

    for _ in range(n_runs):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - start

        total_tokens = response.usage.completion_tokens
        latencies.append(elapsed)
        tokens_per_sec.append(total_tokens / elapsed)

    return {
        "p50_latency_ms": round(statistics.median(latencies) * 1000, 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * n_runs)] * 1000, 1),
        "avg_tokens_per_sec": round(statistics.mean(tokens_per_sec), 1),
        "n_runs": n_runs,
    }


results = benchmark_latency(
    prompt="Explain the transformer architecture in 3 sentences.",
    model="meta-llama/Llama-3.1-8B-Instruct",
    n_runs=20,
)
print(results)
# {'p50_latency_ms': 1420.3, 'p95_latency_ms': 1680.1, 'avg_tokens_per_sec': 67.2, 'n_runs': 20}
```

---

## Performance Optimization Tips

| Optimization | Impact | When to Use |
|-------------|--------|-------------|
| Quantization (INT4) | 2× memory reduction | Always, unless quality is critical |
| Continuous batching | 10–50× throughput | High-concurrency serving (vLLM does this automatically) |
| Prompt caching | 5–10× latency reduction | Long, repeated system prompts |
| Speculative decoding | 2–3× latency reduction | High-volume serving with a draft model |
| Tensor parallelism | Linear GPU scaling | Multi-GPU setups |
| Flash Attention 2 | 2–4× memory reduction | All transformers (automatic in vLLM) |

---

## Troubleshooting

**Out of memory (CUDA OOM)**
- Reduce `--max-model-len` (shorter context window)
- Lower `--gpu-memory-utilization` (default 0.90 → try 0.80)
- Use quantization (AWQ or GGUF Q4)
- Use tensor parallelism across multiple GPUs

**Slow throughput (< 20 tokens/sec on 8B model)**
- Enable Flash Attention (automatic in recent vLLM)
- Increase batch size for offline processing
- Check GPU utilization — should be > 80%

**Responses are truncated**
- Increase `--max-model-len` (but this increases memory)
- Increase `max_tokens` in your API call

---

## FAQ

**vLLM vs llama.cpp — which should I use?**
vLLM: production serving with high concurrency on GPU. llama.cpp: local development, CPU inference, no server needed.

**What GPU do I need for 8B models?**
At INT4 quantization: 8B model fits on an RTX 3080 (10GB) or RTX 4090 (24GB). For full BF16 precision: RTX 4090 (24GB) is the minimum.

---

## What to Learn Next

- **Open-source LLMs** → [Open Source LLMs Guide](/blog/open-source-llms-guide/)
- **Fine-tuning** → [Fine-Tuning LLMs Guide](/blog/fine-tuning-llms-guide/)
- **Deploying AI apps** → [Deploying AI Applications](/blog/deploying-ai-applications/)
