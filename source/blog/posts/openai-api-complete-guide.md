---
title: "OpenAI API Complete Guide: Chat, Embeddings, Functions, and Streaming"
description: "Master the OpenAI API — chat completions, function calling, embeddings, streaming responses, error handling, cost optimization, and building production-ready integrations."
date: "2026-03-10"
slug: "openai-api-complete-guide"
keywords: ["OpenAI API guide", "OpenAI API tutorial", "GPT API", "OpenAI function calling"]
---

## Learning Objectives

- Make your first API call and understand the response structure
- Build multi-turn conversations with the chat API
- Use function calling to integrate LLMs with external tools
- Generate embeddings for semantic search
- Handle errors, retries, and rate limits in production
- Optimize for cost and latency

---

## Setup

```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment
```

---

## Chat Completions

### Basic Request

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user",   "content": "What is RAG in AI?"},
    ],
    max_tokens=256,
    temperature=0.7,
)

print(response.choices[0].message.content)
print(f"Tokens: {response.usage.total_tokens}")
```

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `model` | Model ID | `gpt-4o`, `gpt-4o-mini` |
| `temperature` | Randomness (0=deterministic, 2=chaotic) | 0.0–1.0 |
| `max_tokens` | Max output tokens | 256–4096 |
| `top_p` | Nucleus sampling | 0.9 (don't set both temperature and top_p) |
| `frequency_penalty` | Reduce repetition | 0.0–1.0 |
| `presence_penalty` | Encourage new topics | 0.0–1.0 |

### Multi-Turn Conversations

```python
class ChatSession:
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=1024,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message


session = ChatSession("You are a Python tutor.")
print(session.chat("What is a list comprehension?"))
print(session.chat("Show me 3 examples."))
```

---

## Streaming Responses

Stream tokens as they're generated — better UX for long responses.

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain Docker in 200 words."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()  # newline at end
```

---

## Function Calling (Tool Use)

Function calling lets the model decide when to call external tools — search, databases, APIs — and formats the call for you.

### Define Tools

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]
```

### Handle Tool Calls

```python
def get_weather(city: str, unit: str = "celsius") -> dict:
    # In reality: call a weather API
    return {"city": city, "temperature": 22, "unit": unit, "condition": "sunny"}

def run_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    # First call — model decides whether to use a tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )

    msg = response.choices[0].message

    # Check if model wants to call a tool
    if msg.tool_calls:
        messages.append(msg)  # add assistant's tool-call message

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Execute the actual function
            if name == "get_weather":
                result = get_weather(**args)
            else:
                result = {"error": f"Unknown tool: {name}"}

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        # Second call — model generates final response using tool results
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return final_response.choices[0].message.content

    return msg.content


print(run_with_tools("What's the weather like in Tokyo?"))
```

---

## Structured Output

Force the model to return valid JSON matching a schema:

```python
from pydantic import BaseModel
from typing import List

class ProductReview(BaseModel):
    sentiment: str          # "positive" | "negative" | "neutral"
    score: int              # 1-5
    key_points: List[str]
    summary: str

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract review information."},
        {"role": "user",   "content": "Great product! Fast shipping and exactly as described. 5 stars."},
    ],
    response_format=ProductReview,
)

review = response.choices[0].message.parsed
print(review.sentiment)   # "positive"
print(review.score)       # 5
print(review.key_points)  # ["Fast shipping", "Exactly as described"]
```

---

## Embeddings

Convert text to dense vectors for semantic search, clustering, and similarity matching.

```python
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# Compute cosine similarity
import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


e1 = get_embedding("machine learning")
e2 = get_embedding("deep learning")
e3 = get_embedding("cooking recipes")

print(cosine_similarity(e1, e2))  # ~0.92 (similar)
print(cosine_similarity(e1, e3))  # ~0.70 (dissimilar)
```

**Embedding models:**
- `text-embedding-3-small` — 1536 dims, cheapest
- `text-embedding-3-large` — 3072 dims, highest quality

---

## Error Handling and Retries

```python
import time
from openai import RateLimitError, APIConnectionError, APIStatusError

def chat_with_retry(messages: list, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=512,
            )
            return response.choices[0].message.content

        except RateLimitError:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"Rate limited. Retrying in {wait}s...")
            time.sleep(wait)

        except APIConnectionError as e:
            print(f"Connection error: {e}")
            time.sleep(1)

        except APIStatusError as e:
            if e.status_code == 500:
                time.sleep(1)  # transient server error
            else:
                raise  # don't retry 400/401/404 errors

    raise Exception("Max retries exceeded")
```

---

## Cost Optimization

### Choose the Right Model

| Task | Recommended Model | Why |
|------|------------------|-----|
| Simple Q&A, extraction | `gpt-4o-mini` | 15× cheaper than gpt-4o |
| Complex reasoning | `gpt-4o` | Best capability |
| High-volume classification | `gpt-4o-mini` | Fast + cheap |
| Long document analysis | `gpt-4o` with large context | 128K context |

### Reduce Token Usage

```python
# Bad: entire document in every message
messages = [{"role": "user", "content": full_10k_document + "\n\nSummarize this."}]

# Better: chunk and summarize
# Or: use RAG to retrieve only relevant chunks
```

### Cache Repetitive Requests

```python
import hashlib
import functools

@functools.lru_cache(maxsize=1000)
def cached_embedding(text: str) -> tuple:
    return tuple(get_embedding(text))
```

---

## Troubleshooting

**`AuthenticationError`**
→ Check `OPENAI_API_KEY` is set and valid. No extra spaces or quotes.

**`RateLimitError`**
→ Implement exponential backoff (shown above). Consider upgrading your rate tier.

**Truncated responses**
→ Increase `max_tokens`. Check `finish_reason` — if it's `"length"`, the output was cut off.

**Model ignores system prompt**
→ Move key instructions to the beginning and end of the system prompt. Use explicit constraint language: "Always respond in JSON. Never break character."

---

## FAQ

**Which model should I start with?**
Start with `gpt-4o-mini` — it handles most tasks well at low cost. Upgrade to `gpt-4o` only when you need more complex reasoning.

**How do I count tokens?**
Use the `tiktoken` library: `pip install tiktoken`. `len(tiktoken.encoding_for_model("gpt-4o").encode(text))` gives the token count.

**Can I use the API for commercial projects?**
Yes. See OpenAI's usage policies. By default your data is not used for training (you can opt in).

---

## What to Learn Next

- **Function calling advanced patterns** → ai-agent-frameworks-comparison
- **RAG with OpenAI embeddings** → [RAG Tutorial](/blog/rag-tutorial-step-by-step/)
- **LangChain with OpenAI** → langchain-tutorial-complete
