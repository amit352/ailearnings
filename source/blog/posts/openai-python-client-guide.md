---
title: "OpenAI Python Client Guide – Chat, Embeddings, Tools"
description: "Learn how to use the OpenAI Python client to build AI apps."
date: "2026-03-13"
slug: "openai-python-client-guide"
keywords: ["OpenAI Python client", "openai Python library", "OpenAI SDK Python", "Python GPT API", "OpenAI Python guide"]
---

# OpenAI Python Client Guide – Chat, Embeddings, Tools

The `openai` Python package is the standard way to interact with OpenAI's models in Python applications. It provides a clean interface for chat completions, embeddings, function calling, streaming, and batch processing. This guide covers the full range of features developers use in production.

---

## What is the OpenAI Python Client

The `openai` Python client is the official SDK for OpenAI's API. It handles authentication, HTTP requests, response parsing, streaming, and retry logic. The v1.x API (released late 2023) introduced a fully typed, resource-based interface that replaced the older function-based style.

Key features:
- Synchronous and async interfaces
- Automatic retry with exponential backoff
- Streaming support with typed event models
- Structured output via function calling
- Full type annotations compatible with mypy and pyright

---

## Why the OpenAI Python Client Matters for Developers

The Python client is the fastest path to integrating LLM capabilities into Python applications. It abstracts HTTP, auth, and error handling so you can focus on application logic. The typed interface catches configuration errors at development time rather than runtime.

For the API fundamentals and parameter reference, see [OpenAI API tutorial](/blog/openai-api-tutorial/).

---

## How the OpenAI Python Client Works

### Installation and Setup

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  # Default: reads OPENAI_API_KEY env var
    timeout=30.0,                          # Request timeout in seconds
    max_retries=3,                         # Auto-retry on transient errors
)
```

### Resource Structure

The client is organized into resources that mirror the API:

```python
client.chat.completions.create(...)    # Chat completions
client.embeddings.create(...)          # Embeddings
client.images.generate(...)            # Image generation
client.audio.transcriptions.create(...)# Audio transcription
client.files.create(...)               # File upload
client.fine_tuning.jobs.create(...)    # Fine-tuning
client.moderations.create(...)         # Content moderation
```

---

## Practical Examples

### Chat Completions with Type Safety

```python
from openai import OpenAI
from openai.types.chat import ChatCompletion

client = OpenAI()

def ask(prompt: str, system: str = "You are a helpful assistant.") -> str:
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# Usage tracking
result = ask("What is the capital of France?")
print(result)
```

### Structured Outputs with Pydantic

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class CodeReview(BaseModel):
    has_bugs: bool
    bugs: list[str]
    suggestions: list[str]
    overall_quality: str  # "poor" | "fair" | "good" | "excellent"

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": f"Review this:\ndef div(a, b):\n    return a / b"}
    ],
    response_format=CodeReview,
)

review: CodeReview = response.choices[0].message.parsed
print(f"Has bugs: {review.has_bugs}")
print(f"Bugs: {review.bugs}")
```

### Streaming

```python
def stream_response(prompt: str):
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta

    print()  # newline at end
    return full_response

stream_response("Explain how neural networks learn in simple terms.")
```

### Async Client for Web Applications

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def ask_async(prompt: str) -> str:
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

async def batch_questions(questions: list[str]) -> list[str]:
    """Process multiple questions concurrently."""
    tasks = [ask_async(q) for q in questions]
    return await asyncio.gather(*tasks)

# Run
results = asyncio.run(batch_questions([
    "What is RAG?",
    "What is LoRA?",
    "What is an embedding?",
]))
for q, a in zip(["RAG", "LoRA", "Embedding"], results):
    print(f"{q}: {a[:100]}...")
```

### Batch Embeddings

```python
def embed_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed multiple texts in a single API call."""
    response = client.embeddings.create(model=model, input=texts)
    # Response preserves input order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

texts = ["What is machine learning?", "How do neural networks work?", "Explain backpropagation."]
embeddings = embed_batch(texts)
print(f"Embedded {len(embeddings)} texts, each with {len(embeddings[0])} dimensions")
```

### Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    # Prices in $ per million tokens (as of early 2026)
    prices = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    p = prices.get(model, prices["gpt-4o-mini"])
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

tokens = count_tokens("Explain the transformer architecture in detail.")
print(f"Tokens: {tokens}, Estimated cost: ${estimate_cost(tokens, 500):.6f}")
```

---

## Tools and Frameworks

**tiktoken** — OpenAI's tokenizer library. Count tokens before making API calls to estimate cost and avoid context overflow.

**Instructor** — Wrapper around the OpenAI client for reliable structured outputs using Pydantic. Handles validation and automatic retry on parse failures.

**LiteLLM** — Unified client that exposes an OpenAI-compatible interface for dozens of providers. Useful for model-agnostic applications.

**LangChain** — `ChatOpenAI` provides a LangChain-compatible wrapper. See [LangChain tutorial](/blog/langchain-tutorial/) for details.

For the underlying API concepts, see [OpenAI API tutorial](/blog/openai-api-tutorial/). For embeddings use in search and RAG, see [embeddings explained](/blog/embeddings-explained/).

---

## Common Mistakes

**Not using environment variables for API keys** — Always use `os.environ["OPENAI_API_KEY"]` or a `.env` file with `python-dotenv`. Never hardcode keys.

**Ignoring the response structure** — `response.choices[0].message.content` can be `None` when function calling is used. Check `tool_calls` on the message first.

**Not handling `APIError` exceptions** — Network issues, rate limits, and invalid requests all raise specific exception types. Catch them explicitly.

```python
from openai import APIError, RateLimitError, APIConnectionError

try:
    response = client.chat.completions.create(...)
except RateLimitError:
    # Wait and retry
    pass
except APIConnectionError:
    # Network issue
    pass
except APIError as e:
    print(f"API error {e.status_code}: {e.message}")
```

**Using the synchronous client in async frameworks** — In FastAPI or other async frameworks, use `AsyncOpenAI` to avoid blocking the event loop.

---

## Best Practices

- **Reuse the client instance** — Create one `OpenAI()` client at module level and reuse it. Each instantiation opens a new connection pool.
- **Use `response_format` for structured output** — More reliable than asking for JSON in plain text.
- **Set `max_tokens` explicitly** — Prevents runaway generation and controls cost.
- **Use the async client in production web apps** — Concurrent requests are significantly faster than sequential ones when using `AsyncOpenAI` with `asyncio.gather`.
- **Count tokens before large requests** — Use `tiktoken` to verify your context does not exceed the model's limit.

---

## Summary

The OpenAI Python client is a fully typed, feature-rich SDK for building AI applications. It handles authentication, retries, streaming, structured outputs, and async concurrency.

Use the synchronous client for scripts and simple applications. Use `AsyncOpenAI` for web services. Use Pydantic structured outputs for any application that needs to parse model responses reliably.

For a broader look at the API features, see [OpenAI API tutorial](/blog/openai-api-tutorial/). For how embeddings work in AI applications, see [embeddings explained](/blog/embeddings-explained/).
