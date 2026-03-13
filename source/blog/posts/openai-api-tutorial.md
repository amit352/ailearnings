---
title: "OpenAI API Tutorial for Developers"
description: "Learn how to use the OpenAI API to build AI-powered applications."
date: "2026-03-13"
slug: "openai-api-tutorial"
keywords: ["OpenAI API tutorial", "OpenAI API guide", "how to use OpenAI API", "GPT API", "ChatGPT API developer"]
---

# OpenAI API Tutorial for Developers

The OpenAI API gives developers access to GPT-4o, embedding models, image generation, and speech-to-text. It is the most widely used AI API in production applications. This tutorial covers everything you need to start building: authentication, chat completions, streaming, function calling, and embeddings.

---

## What is the OpenAI API

The OpenAI API is a REST API that provides access to OpenAI's models:
- **Chat completions** — GPT-4o, GPT-4o-mini for text generation and reasoning
- **Embeddings** — text-embedding-3-small, text-embedding-3-large for semantic search
- **Images** — DALL-E 3 for image generation
- **Audio** — Whisper for transcription, TTS for text-to-speech
- **Moderation** — Content classification for safety filtering

For most AI application development, the chat completions and embeddings APIs are the primary interfaces. The Python SDK wraps the REST API with a clean object-oriented interface.

---

## Why the OpenAI API Matters for Developers

The OpenAI API is the fastest path from idea to working AI feature:
- No infrastructure to manage
- Pay per token — no upfront cost
- Regular model updates automatically improve your application
- Extensive documentation and community support
- Structured output, function calling, and vision built in

For building applications with the Python client, see [OpenAI Python client guide](/blog/openai-python-client-guide/).

---

## How the OpenAI API Works

### Setup

```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

```python
from openai import OpenAI
client = OpenAI()  # Reads OPENAI_API_KEY from environment
```

### Chat Completions

The chat completions endpoint takes a list of messages and returns a response.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to reverse a string."}
    ],
    temperature=0,
    max_tokens=500
)
print(response.choices[0].message.content)
```

**Message roles:**
- `system` — Sets the assistant's behavior and persona
- `user` — The human's message
- `assistant` — The model's previous responses (for multi-turn conversations)

### Key Parameters

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `model` | Which model to use | `gpt-4o-mini` |
| `temperature` | Randomness (0=deterministic) | 0 for structured tasks, 0.7 for creative |
| `max_tokens` | Maximum output length | 500–2000 |
| `top_p` | Nucleus sampling | Leave default unless tuning |
| `stream` | Stream tokens as generated | `True` for chat UIs |

---

## Practical Examples

### Streaming Responses

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain recursion in 3 paragraphs."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Function Calling (Structured Output)

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_contact",
            "description": "Extract contact information from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"}
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
         "Contact: Jane Smith, jane@example.com, 555-0100"}
    ],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_contact"}}
)

tool_call = response.choices[0].message.tool_calls[0]
contact = json.loads(tool_call.function.arguments)
print(contact)
# {"name": "Jane Smith", "email": "jane@example.com", "phone": "555-0100"}
```

### Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["How does RAG work?", "Retrieval augmented generation explained"]
)

vectors = [item.embedding for item in response.data]
print(f"Embedding dimensions: {len(vectors[0])}")  # 1536
```

### Vision (Image Understanding)

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

### Multi-Turn Conversation

```python
conversation_history = [
    {"role": "system", "content": "You are a Python tutor."}
]

def chat(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        temperature=0.7
    )
    assistant_message = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message

print(chat("What is a list comprehension?"))
print(chat("Can you show me an example?"))
```

---

## Tools and Frameworks

**LangChain OpenAI integration** — `ChatOpenAI` and `OpenAIEmbeddings` wrap the API with LangChain's chain and retrieval interfaces. See [LangChain tutorial](/blog/langchain-tutorial/).

**OpenAI Cookbook** — Official repository of example notebooks and guides for common use cases.

**Instructor** — Library for reliable structured outputs using Pydantic models. Simplifies function calling with automatic retries and validation.

**LiteLLM** — Proxy that provides a unified interface across OpenAI, Anthropic, Gemini, and 100+ other providers.

---

## Common Mistakes

**Hardcoding API keys** — Never put your API key in source code. Use environment variables or a secrets manager.

**Not handling rate limits** — OpenAI enforces token-per-minute and request-per-minute limits. Use exponential backoff for retries in production.

**Ignoring token costs** — Every token in and out costs money. Set `max_tokens` limits and monitor usage in the OpenAI dashboard.

**Not streaming for user-facing apps** — Users perceive streamed responses as faster even when total generation time is the same. Always stream in chat interfaces.

**Storing full conversation history unbounded** — Conversation history grows with each turn. Implement truncation or summarization to stay within the context window.

---

## Best Practices

- **Use `gpt-4o-mini` for development** — It is fast and cheap. Switch to `gpt-4o` only for tasks that require higher capability.
- **Set `temperature=0` for deterministic tasks** — Extraction, classification, and structured output should be consistent. Reserve higher temperatures for creative generation.
- **Log requests and responses** — API calls are billable and hard to debug without logs. Log model, tokens used, and response for every production call.
- **Use structured outputs for parsing** — Function calling with a schema is more reliable than asking for JSON in plain text.
- **Implement retry logic with backoff** — Network errors and rate limits happen. Use `tenacity` or similar for robust retries.

---

## Summary

The OpenAI API provides access to GPT models through a simple REST interface. Chat completions, streaming, function calling, and embeddings cover the vast majority of production AI application patterns.

Start with `gpt-4o-mini` for cost efficiency, use `temperature=0` for structured tasks, and implement streaming for user-facing interfaces. Function calling is the most reliable way to get structured output from the model.

For the Python client details, see [OpenAI Python client guide](/blog/openai-python-client-guide/). For building a complete application, see [how to build your first AI app](/blog/build-ai-app/).
