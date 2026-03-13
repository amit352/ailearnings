---
title: "Python for AI Developers: Complete Setup and Essential Libraries"
description: "Set up a production-ready Python AI development environment. Covers virtual environments, essential libraries (NumPy, pandas, OpenAI SDK), Jupyter notebooks, and common patterns used in AI projects."
date: "2026-03-10"
slug: "python-for-ai-complete-guide"
keywords: ["Python for AI development", "Python AI setup", "NumPy pandas OpenAI tutorial"]
---

## Prerequisites

Basic Python knowledge (variables, functions, loops, classes). This guide covers the *AI-specific* Python ecosystem, not Python fundamentals.

---

## Environment Setup

### Python Version

Use Python 3.11+. Many AI libraries require 3.10+.

```bash
# Check version
python --version

# Install Python 3.11 via pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.11.9
pyenv global 3.11.9
```

### Virtual Environments

Always use virtual environments. AI projects have many dependencies that conflict across projects.

```bash
# Create a virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Deactivate
deactivate
```

Alternatively, use `uv` (much faster than pip):

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install openai numpy pandas  # 10x faster than pip
```

---

## Essential Libraries for AI Development

### Core Data Stack

```bash
pip install numpy pandas matplotlib scikit-learn
```

**NumPy** — numerical computing, arrays, matrix operations. Foundation of everything else.

```python
import numpy as np

# Arrays are the fundamental unit
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Vectorized operations (no loops needed)
result = arr * 2 + 1       # [3, 5, 7, 9, 11]
dot = np.dot(matrix, matrix)  # matrix multiplication

# Useful for working with embeddings
embedding = np.array([0.23, -0.45, 0.87, ...])
similarity = np.dot(embedding1, embedding2)  # cosine similarity component
```

**pandas** — tabular data, CSV processing, data manipulation.

```python
import pandas as pd

# Load CSV
df = pd.read_csv("data.csv")

# Explore
print(df.head())
print(df.describe())
print(df.dtypes)

# Filter and transform
filtered = df[df["score"] > 0.8]
df["category"] = df["text"].apply(lambda x: classify(x))

# Group and aggregate
summary = df.groupby("category")["score"].mean()
```

**matplotlib** — plotting and visualization.

```python
import matplotlib.pyplot as plt

# Line chart
plt.plot(history["loss"], label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()
```

### AI/LLM Stack

```bash
pip install openai anthropic langchain tiktoken
```

**OpenAI SDK** — the most-used LLM API:

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

# Chat completion
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in one sentence."},
    ],
    temperature=0.7,
    max_tokens=200,
)
text = response.choices[0].message.content

# Embeddings
embedding_resp = client.embeddings.create(
    model="text-embedding-3-small",
    input="Some text to embed",
)
vector = embedding_resp.data[0].embedding  # list of 1536 floats
```

**tiktoken** — count tokens before sending to API:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = enc.encode("Hello, world!")
print(len(tokens))  # 4

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
```

### Vector DB

```bash
pip install chromadb  # local, no server needed
# or
pip install pinecone-client  # managed cloud
```

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")

# Add documents with embeddings
collection.add(
    ids=["doc1", "doc2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Text of doc 1", "Text of doc 2"],
)

# Query by similarity
results = collection.query(
    query_embeddings=[[0.15, 0.25, ...]],
    n_results=3,
)
```

### PDF / Document Processing

```bash
pip install pypdf requests beautifulsoup4
```

```python
from pypdf import PdfReader

reader = PdfReader("paper.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)

# Web scraping
import requests
from bs4 import BeautifulSoup

resp = requests.get("https://example.com", headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")
content = soup.find("article").get_text(separator="\n")
```

---

## Managing API Keys Securely

**Never hardcode API keys.** Use environment variables.

```bash
# .env file (add to .gitignore!)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

```python
# Load .env file
pip install python-dotenv

from dotenv import load_dotenv
import os

load_dotenv()  # reads .env file
api_key = os.getenv("OPENAI_API_KEY")
```

Or set in your shell:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Jupyter Notebooks for Exploration

Notebooks are ideal for prototyping AI experiments:

```bash
pip install jupyter
jupyter notebook   # opens browser
# or
pip install jupyterlab
jupyter lab
```

Useful Jupyter patterns for AI:

```python
# Cell 1: Setup
from openai import OpenAI
client = OpenAI()

# Cell 2: Helper function
def ask(prompt, model="gpt-4o-mini", temp=0.7):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return resp.choices[0].message.content

# Cell 3: Experiment
result = ask("What are the 5 key differences between RAG and fine-tuning?")
print(result)

# Cell 4: Adjust and iterate
```

---

## Python Patterns Common in AI Projects

### Retry with exponential backoff

API rate limits are common. Handle them gracefully:

```python
import time
import random
from openai import RateLimitError

def call_with_retry(fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.random()
            print(f"Rate limited. Waiting {wait:.1f}s...")
            time.sleep(wait)
```

### Async for parallel API calls

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def analyze_batch(texts: list[str]) -> list[str]:
    tasks = [
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize: {t}"}],
        )
        for t in texts
    ]
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in responses]

# Run 10 API calls concurrently instead of sequentially
results = asyncio.run(analyze_batch(my_texts))
```

### Caching expensive operations

```python
import json
import hashlib
from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def cached_embed(text: str) -> list[float]:
    key = hashlib.md5(text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    embedding = response.data[0].embedding
    cache_file.write_text(json.dumps(embedding))
    return embedding
```

---

## Project Structure for AI Apps

```
my-ai-project/
├── .env                    # API keys (in .gitignore)
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py           # Settings, env vars
│   ├── llm.py              # LLM client + helpers
│   ├── embeddings.py       # Embedding utilities
│   └── utils.py            # Shared utilities
├── data/
│   └── ...                 # Input data
├── notebooks/
│   └── exploration.ipynb   # Prototyping
└── tests/
    └── test_*.py
```

---

## Essential `requirements.txt`

```txt
openai>=1.0.0
anthropic>=0.20.0
langchain>=0.2.0
chromadb>=0.5.0
tiktoken>=0.7.0
pypdf>=4.0.0
numpy>=1.26.0
pandas>=2.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
streamlit>=1.35.0
fastapi>=0.110.0
uvicorn>=0.27.0
```

---

## What to Learn Next

- **Build your first project** → [Build an AI Chatbot](/projects/ai-chatbot-python/)
- **Master the OpenAI API** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
- **Machine learning foundations** → [Neural Networks from Scratch](/blog/roadmap-guides/neural-networks-from-scratch/)
