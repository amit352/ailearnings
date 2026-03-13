---
title: "AI Application Architecture: Designing Production-Ready LLM Systems"
description: "Architectural patterns for production AI applications — prompt management, context windows, fallback strategies, observability, cost control, and testing LLM systems."
date: "2026-03-10"
slug: "ai-application-architecture"
keywords: ["AI application architecture", "LLM application design", "production AI systems", "LLM architecture patterns"]
---

## Learning Objectives

- Apply proven architectural patterns for LLM applications
- Design for reliability with fallbacks and retries
- Manage prompts as versioned, testable artifacts
- Implement observability and cost controls
- Test LLM applications effectively

---

## The Four Layers of an AI Application

```
┌─────────────────────────────────────────┐
│  4. Application Layer                   │
│     (FastAPI, Next.js, CLI)             │
├─────────────────────────────────────────┤
│  3. Orchestration Layer                 │
│     (LangChain, LangGraph, custom)      │
├─────────────────────────────────────────┤
│  2. AI Services Layer                   │
│     (LLM, Embeddings, Vector DB)        │
├─────────────────────────────────────────┤
│  1. Data Layer                          │
│     (Documents, DB, Cache, Storage)     │
└─────────────────────────────────────────┘
```

Each layer should be independently testable and replaceable.

---

## Prompt Management

Prompts are code. Treat them as versioned, testable artifacts.

```python
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path

@dataclass
class PromptTemplate:
    name: str
    version: str
    system: str
    user_template: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024

    def format(self, **kwargs) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user",   "content": self.user_template.format(**kwargs)},
        ]

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_json(cls, path: str) -> "PromptTemplate":
        return cls(**json.loads(Path(path).read_text()))


# prompts/summarize_v2.json
SUMMARIZE_PROMPT = PromptTemplate(
    name="summarize",
    version="2.0",
    system=(
        "You are a technical documentation specialist. "
        "Produce concise summaries that preserve key technical details. "
        "Format: 3-5 bullet points followed by a one-sentence TL;DR."
    ),
    user_template="Summarize the following {content_type}:\n\n{content}",
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=512,
)

# Usage
messages = SUMMARIZE_PROMPT.format(
    content_type="technical article",
    content="Long article text here..."
)
```

### Prompt Registry

```python
class PromptRegistry:
    def __init__(self, prompts_dir: str = "./prompts"):
        self.prompts: dict[str, PromptTemplate] = {}
        self._load_from_dir(prompts_dir)

    def _load_from_dir(self, dir_path: str):
        for path in Path(dir_path).glob("*.json"):
            pt = PromptTemplate.from_json(str(path))
            self.prompts[pt.name] = pt

    def get(self, name: str) -> PromptTemplate:
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found")
        return self.prompts[name]

    def list_prompts(self) -> list[str]:
        return list(self.prompts.keys())


registry = PromptRegistry()
prompt = registry.get("summarize")
```

---

## Resilient LLM Client

```python
import time
import logging
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-4o-mini"  # could be a different provider
    max_retries: int = 3
    timeout: float = 30.0
    base_delay: float = 1.0


class ResilientLLMClient:
    def __init__(self, config: LLMConfig = LLMConfig()):
        self.client = OpenAI(timeout=config.timeout)
        self.config = config

    def complete(self, messages: list, **kwargs) -> str:
        model = kwargs.pop("model", self.config.primary_model)
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                delay = self.config.base_delay * (2 ** attempt)
                logger.warning(f"Rate limit (attempt {attempt+1}). Waiting {delay}s...")
                time.sleep(delay)
                last_error = e

            except APIConnectionError as e:
                delay = self.config.base_delay
                logger.warning(f"Connection error (attempt {attempt+1}). Waiting {delay}s...")
                time.sleep(delay)
                last_error = e

            except APIStatusError as e:
                if e.status_code >= 500:  # server error — retry
                    time.sleep(self.config.base_delay)
                    last_error = e
                else:
                    raise  # 4xx errors — don't retry

        # Try fallback model
        if model != self.config.fallback_model:
            logger.warning(f"Primary model failed. Trying fallback: {self.config.fallback_model}")
            try:
                response = self.client.chat.completions.create(
                    model=self.config.fallback_model,
                    messages=messages,
                    **kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e

        raise Exception(f"All retries exhausted: {last_error}")


llm = ResilientLLMClient()
```

---

## Context Window Management

LLMs have token limits. Manage context carefully:

```python
import tiktoken

class ContextManager:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 4096):
        self.enc = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        self.safety_margin = 512  # reserve for output

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def count_messages_tokens(self, messages: list) -> int:
        total = 3  # message overhead
        for msg in messages:
            total += 4 + self.count_tokens(msg.get("content", ""))
        return total

    def truncate_to_fit(self, text: str, max_chars: int) -> str:
        tokens = self.enc.encode(text)
        if len(tokens) <= max_chars:
            return text
        truncated = self.enc.decode(tokens[:max_chars])
        return truncated + "\n[...truncated...]"

    def trim_history(self, messages: list, system_prompt: str) -> list:
        """Remove oldest messages until history fits in context window."""
        budget = self.max_tokens - self.safety_margin
        system_tokens = self.count_tokens(system_prompt) + 10

        result = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        # Add messages from most recent, working backward
        current_tokens = system_tokens
        kept_messages = []
        for msg in reversed(non_system):
            msg_tokens = self.count_tokens(msg["content"]) + 4
            if current_tokens + msg_tokens > budget:
                break
            kept_messages.insert(0, msg)
            current_tokens += msg_tokens

        return result + kept_messages


ctx_mgr = ContextManager(max_tokens=8192)
```

---

## Caching Strategy

```python
import hashlib
import json
import time
from functools import wraps

class LLMCache:
    def __init__(self, ttl: int = 3600):
        self._store: dict = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def _key(self, messages: list, model: str, **kwargs) -> str:
        payload = json.dumps({"messages": messages, "model": model, **kwargs}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, messages: list, model: str, **kwargs) -> str | None:
        key = self._key(messages, model, **kwargs)
        entry = self._store.get(key)
        if entry and time.time() - entry["ts"] < self.ttl:
            self.hits += 1
            return entry["value"]
        self.misses += 1
        return None

    def set(self, messages: list, model: str, value: str, **kwargs):
        key = self._key(messages, model, **kwargs)
        self._store[key] = {"value": value, "ts": time.time()}

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


cache = LLMCache(ttl=3600)

def cached_complete(messages: list, model: str = "gpt-4o-mini", **kwargs) -> str:
    # Check cache first
    cached = cache.get(messages, model, **kwargs)
    if cached:
        return cached

    # Call LLM
    response = llm.complete(messages, model=model, **kwargs)
    cache.set(messages, model, response, **kwargs)
    return response
```

---

## Observability

```python
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LLMCallLog:
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool = False
    error: str = None

COSTS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o":      {"input": 0.0025,  "output": 0.01},
}

class LLMObserver:
    def __init__(self):
        self.calls: list[LLMCallLog] = []
        self.logger = logging.getLogger("llm_observer")

    @contextmanager
    def track(self, model: str):
        start = time.perf_counter()
        log = LLMCallLog(
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
            cost_usd=0,
        )
        try:
            yield log
        finally:
            log.latency_ms = round((time.perf_counter() - start) * 1000, 1)
            costs = COSTS.get(model, {"input": 0, "output": 0})
            log.cost_usd = round(
                log.prompt_tokens / 1000 * costs["input"]
                + log.completion_tokens / 1000 * costs["output"],
                6,
            )
            self.calls.append(log)
            self.logger.info(f"LLM call: {log}")

    def summary(self) -> dict:
        if not self.calls:
            return {}
        return {
            "total_calls": len(self.calls),
            "total_cost_usd": round(sum(c.cost_usd for c in self.calls), 4),
            "avg_latency_ms": round(sum(c.latency_ms for c in self.calls) / len(self.calls), 1),
            "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in self.calls),
        }


observer = LLMObserver()

def observed_complete(messages: list, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI()

    with observer.track(model) as log:
        response = client.chat.completions.create(model=model, messages=messages)
        log.prompt_tokens = response.usage.prompt_tokens
        log.completion_tokens = response.usage.completion_tokens
        return response.choices[0].message.content
```

---

## Testing LLM Applications

```python
import pytest

class TestSummarizer:
    def setup_method(self):
        self.summarizer = YourSummarizer()

    def test_returns_non_empty_string(self):
        result = self.summarizer.summarize("The quick brown fox...")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_output_shorter_than_input(self):
        long_text = "This is a long document. " * 100
        summary = self.summarizer.summarize(long_text)
        assert len(summary) < len(long_text)

    def test_contains_key_concepts(self):
        text = "LangChain is a framework for building LLM-powered applications."
        summary = self.summarizer.summarize(text)
        # Check that key concepts are preserved
        assert any(kw in summary.lower() for kw in ["langchain", "llm", "framework"])

    def test_handles_empty_input(self):
        with pytest.raises(ValueError):
            self.summarizer.summarize("")

    def test_format_compliance(self):
        result = self.summarizer.summarize("Some text...")
        lines = result.strip().split('\n')
        # Check expected format: bullet points
        bullet_lines = [l for l in lines if l.strip().startswith('-') or l.strip().startswith('•')]
        assert len(bullet_lines) >= 2
```

---

## FAQ

**Should I use LangChain or build from scratch?**
LangChain for rapid prototyping. Custom for production (simpler dependency tree, easier debugging, full control). Many teams start with LangChain and gradually replace components.

**How do I prevent prompt injection?**
Separate system prompts from user input clearly. Sanitize user input. Use structured outputs instead of free-form responses where possible. Never execute code from LLM output without sandboxing.

---

## What to Learn Next

- **Deploying AI applications** → [Deploying AI Applications](/blog/deploying-ai-applications/)
- **LangChain patterns** → [LangChain Complete Tutorial](/blog/langchain-tutorial-complete/)
- **Building chatbots** → [Building AI Chatbots](/blog/building-ai-chatbots/)
