---
title: "OpenAI API Tutorial for Developers"
description: "Learn how to use the OpenAI API to build AI-powered applications."
date: "2026-03-13"
slug: "openai-api-tutorial"
keywords: ["OpenAI API tutorial", "OpenAI API guide", "how to use OpenAI API", "GPT API", "ChatGPT API developer"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
---

# OpenAI API Tutorial for Developers

You can go from zero API knowledge to a working AI feature in about 30 minutes with the OpenAI API. No infrastructure to manage, no models to host, and a Python SDK that makes the most common patterns feel natural. But the gap between a demo that works once and an API integration that behaves reliably in production is significant — and it shows up in specifics: how you handle tokens, structure prompts, parse output, manage conversation history, and handle the inevitable rate limit errors. This tutorial covers the whole picture.

---

## What the OpenAI API Provides

The OpenAI API is a REST API (with official Python and Node.js SDKs) that gives you access to:

- **Chat completions** — GPT-4o and GPT-4o-mini for text generation, reasoning, and code
- **Embeddings** — `text-embedding-3-small` and `text-embedding-3-large` for semantic search and similarity
- **Images** — DALL-E 3 for image generation from text prompts
- **Audio** — Whisper for speech-to-text transcription, TTS for text-to-speech synthesis
- **Moderation** — Content classification for safety filtering
- **Structured outputs** — JSON mode and function calling for reliable structured data extraction

For most AI application development, you will spend 90% of your time with chat completions and embeddings. Vision, audio, and image generation are additive capabilities you reach for when the product requires them.

---

## Setup

```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

```python
from openai import OpenAI

# Reads OPENAI_API_KEY from environment automatically
client = OpenAI()
```

Never hardcode your API key in source code. Use environment variables, a `.env` file loaded with `python-dotenv`, or a secrets manager in production. The OpenAI SDK will raise a clear error if the key is missing.

---

## Chat Completions: The Core API

The chat completions endpoint takes a list of messages with roles and returns the model's response. Understanding the message structure is the foundation of everything else.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user",   "content": "Write a Python function to reverse a string."}
    ],
    temperature=0,
    max_tokens=500
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

**Message roles:**
- `system` — Sets the assistant's persona and behavior. Applied to every turn. Put your instructions, constraints, and context here.
- `user` — The human's input.
- `assistant` — Previous model responses. Include these to give the model context in multi-turn conversations.

**Key parameters:**

| Parameter | Purpose | When to Change |
|-----------|---------|----------------|
| `model` | Which model to use | Use `gpt-4o` for complex reasoning, `gpt-4o-mini` for most tasks |
| `temperature` | Output randomness (0 = deterministic) | Set to 0 for structured/extraction tasks, 0.7 for creative generation |
| `max_tokens` | Maximum response length | Always set a limit to control costs |
| `stream` | Stream tokens as generated | Set `True` for chat UIs |

---

## Streaming Responses

For user-facing applications, streaming is not optional — it is expected. A non-streamed response forces the user to wait 5–10 seconds for the full text to appear at once. Streamed responses start appearing in under a second.

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain recursion in 3 paragraphs."}],
    stream=True
)

# Print tokens as they arrive
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # newline at the end
```

In a FastAPI backend, use `StreamingResponse` with Server-Sent Events (SSE) to stream to the browser. In LangChain, use `.stream()` instead of `.invoke()`.

---

## Function Calling and Structured Output

Asking an LLM to "respond in JSON" is unreliable — it might include preamble, commentary, or malformed JSON. Function calling gives you a schema-enforced structured output with automatic retries on the model side.

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_contact",
            "description": "Extract contact information from free text",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":  {"type": "string", "description": "Full name"},
                    "email": {"type": "string", "description": "Email address"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "company": {"type": "string", "description": "Company or organization"}
                },
                "required": ["name"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content":
         "Contact: Jane Smith, Principal Engineer at Acme Corp. jane@acme.com, +1-555-0100"}
    ],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_contact"}}
)

tool_call = response.choices[0].message.tool_calls[0]
contact = json.loads(tool_call.function.arguments)
print(contact)
# {"name": "Jane Smith", "email": "jane@acme.com", "phone": "+1-555-0100", "company": "Acme Corp"}
```

For simpler cases, use the newer `response_format` parameter with `json_schema` — it is cleaner and handles the parsing automatically.

---

## Embeddings: Semantic Search

Embeddings convert text into dense vectors where semantically similar text maps to nearby vectors. This powers semantic search, clustering, and the retrieval step in RAG applications.

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "How does RAG work?",
        "Retrieval augmented generation explained",
        "Python list comprehension syntax"
    ]
)

vectors = [item.embedding for item in response.data]
print(f"Embedding dimensions: {len(vectors[0])}")  # 1536 for text-embedding-3-small

# Compute cosine similarity manually
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# These two should be more similar to each other than to the third
print(cosine_similarity(vectors[0], vectors[1]))  # ~0.92
print(cosine_similarity(vectors[0], vectors[2]))  # ~0.45
```

`text-embedding-3-small` is the right choice for most applications — it is fast, cheap, and produces 1536-dimensional vectors that capture semantic meaning well. Use `text-embedding-3-large` (3072 dimensions) when retrieval precision is critical.

---

## Multi-Turn Conversations

Building a conversational application requires maintaining message history. The OpenAI API is stateless — you must send the full conversation history with every request.

```python
conversation_history = [
    {"role": "system", "content": "You are an expert Python tutor. Be concise."}
]

def chat(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        temperature=0.7,
        max_tokens=500
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message

print(chat("What is a list comprehension?"))
print(chat("Can you show me a real example with filtering?"))
# Second call — model has context from first turn
print(chat("How would I do the same thing with a dict?"))
```

**Important:** conversation history grows unbounded. After 10–20 turns in a complex conversation, you will approach the model's context limit (128K tokens for GPT-4o). Implement a truncation strategy: keep the system message and last N turns, or use a summarization step to compress older history.

---

## Vision: Analyzing Images

GPT-4o supports image inputs alongside text. Pass image URLs or base64-encoded image data directly in the message content.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what is in this chart and what trend it shows."},
                {"type": "image_url", "image_url": {"url": "https://example.com/sales-chart.png"}}
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

For local images, encode them as base64:

```python
import base64

with open("./chart.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

content = [
    {"type": "text", "text": "What does this error message say?"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
]
```

---

## Error Handling and Retry Logic

OpenAI API calls fail for predictable reasons: rate limits, network timeouts, and occasional server errors. Production code must handle these gracefully.

```python
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
import time

client = OpenAI()

def call_with_retry(messages, model="gpt-4o-mini", max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30,
            )
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except APITimeoutError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
        except APIConnectionError as e:
            raise RuntimeError(f"Could not connect to OpenAI API: {e}")
```

For more robust retry handling, use the `tenacity` library, which provides decorators for retry logic with configurable backoff strategies.

---

## Common Mistakes

**Hardcoding API keys** — Never put your API key in source code. If it ends up in a git repository, even briefly, rotate it immediately. Use environment variables or a secrets manager.

**Not handling rate limits** — OpenAI enforces token-per-minute (TPM) and request-per-minute (RPM) limits per model. Without retry logic, rate limit errors will surface as 429 responses to your users. Implement exponential backoff.

**Unbounded conversation history** — Every token in the conversation history costs money and counts against the context window. Implement truncation after a reasonable number of turns.

**Using `gpt-4o` for everything** — `gpt-4o-mini` handles the majority of tasks at 10–20x lower cost. Default to `gpt-4o-mini` and only upgrade specific high-complexity calls to `gpt-4o`.

**Not setting `max_tokens`** — Without a limit, a single request can generate thousands of tokens unexpectedly (especially if you ask the model to write code or explain complex topics). Always set `max_tokens` appropriate to your use case.

**Parsing JSON from plain text responses** — Asking "respond in JSON" in the system prompt produces inconsistent output. Use function calling or `response_format={"type": "json_object"}` for reliable structured output.

---

## Best Practices

1. **Always log API calls in production** — Log model, input token count, output token count, latency, and any errors. You cannot optimize costs or debug failures without this data.
2. **Use `temperature=0` for deterministic tasks** — Extraction, classification, code generation, and structured output should be consistent across calls.
3. **Set spending limits** — Configure a monthly spending limit in the OpenAI dashboard. An infinite loop or unexpected traffic spike can generate unexpected charges.
4. **Test with real inputs** — Edge cases from real user inputs reveal prompt failures that synthetic tests miss. Collect a sample of real queries early and use them for regression testing.
5. **Version your prompts** — Treat prompts as code. Store them in version-controlled constants, not inline strings. When you change a prompt, run your test suite to catch regressions.

---

## What to Learn Next

The OpenAI API is the foundation. Once you understand it, the natural next steps add retrieval, orchestration, and deployment:

- **Build a full AI application using the API** → [How to Build Your First AI App](/blog/build-ai-app/)
- **Add retrieval to your LLM calls (RAG)** → [How to Build a RAG Application](/blog/build-rag-app/)
- **Use LangChain to orchestrate complex pipelines** → [LangChain Tutorial](/blog/langchain-tutorial/)
