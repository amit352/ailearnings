---
title: "Working with Local LLMs: Run AI Models Privately with Ollama and llama.cpp"
description: "Run powerful LLMs completely offline on your own hardware. Set up Ollama, run Llama 3, Mistral, and Code Llama locally, build Python integrations, and understand when to use local vs API models."
date: "2026-03-10"
slug: "working-with-local-llms"
keywords: ["run LLM locally Python", "Ollama tutorial", "local AI models privacy"]
---

## Why Run Models Locally?

- **Privacy**: sensitive data never leaves your machine
- **No API costs**: run unlimited queries for free (after hardware cost)
- **Offline capability**: works without internet
- **Latency**: no network round-trip for short prompts
- **Customization**: fine-tune on your own data without vendor lock-in

**When to use cloud APIs instead**: GPT-4 class capability, very large context windows (> 128k), latest features, no GPU available.

---

## Hardware Requirements

| Model Size | RAM/VRAM Needed | Hardware Example |
|-----------|----------------|-----------------|
| 7B params | 8 GB | M2/M3 MacBook Pro, RTX 3060 |
| 13B params | 16 GB | M2 Pro/Max, RTX 4080 |
| 30B params | 24–32 GB | M2 Ultra, RTX 4090 |
| 70B params | 40–48 GB | Mac Studio M2 Ultra |

Most models are **quantized** (compressed) to fit in less memory. A 4-bit quantized 7B model runs in ~4GB.

---

## Ollama: The Easiest Way to Run Local LLMs

### Installation

```bash
# macOS (also installs as a menu bar app)
curl -fsSL https://ollama.ai/install.sh | sh

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: download from https://ollama.ai
```

### Running Models

```bash
# Pull and run Llama 3 (8B) — great general purpose
ollama run llama3

# Pull and run Mistral (7B) — excellent, fast
ollama run mistral

# Code-specialized model
ollama run codellama

# Small but capable (3.8B) — works on any modern laptop
ollama run phi3

# List downloaded models
ollama list

# Delete a model
ollama rm llama3

# Show model info
ollama show llama3
```

### Ollama Server API

Ollama runs as a local HTTP server on port 11434:

```bash
# Generate
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Explain recursion in one sentence",
  "stream": false
}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "mistral",
  "messages": [{"role": "user", "content": "What is RAG?"}],
  "stream": false
}'
```

---

## Python Integration with Ollama

### Option 1: OpenAI-Compatible API

Ollama's API is OpenAI-compatible, so you can use the OpenAI SDK:

```python
from openai import OpenAI

# Point to local Ollama server instead of OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required but ignored by Ollama
)

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

This means you can **switch between local and cloud** models with one variable:

```python
import os

if os.getenv("USE_LOCAL_LLM") == "1":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL = "llama3"
else:
    client = OpenAI()
    MODEL = "gpt-4o-mini"

# Rest of code is identical
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Summarize AI trends in 2026"}],
)
```

### Option 2: Official Ollama Python Library

```bash
pip install ollama
```

```python
import ollama

# Simple generation
response = ollama.generate(model="mistral", prompt="Why is the sky blue?")
print(response["response"])

# Chat with history
messages = [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "What is a decorator?"},
]

response = ollama.chat(model="codellama", messages=messages)
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(model="llama3", messages=messages, stream=True):
    print(chunk["message"]["content"], end="", flush=True)

# Embeddings
embedding = ollama.embeddings(model="nomic-embed-text", prompt="Some text")
vector = embedding["embedding"]  # list of floats
```

---

## Local RAG with Ollama

Build a completely private RAG system:

```python
# private_rag.py
import ollama
import chromadb
from pathlib import Path

# Setup
chroma = chromadb.PersistentClient(path="./local_chroma")
collection = chroma.get_or_create_collection("local_docs")


def embed_local(text: str) -> list[float]:
    """Use Ollama's embedding model (runs locally)."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def ingest(file_path: str):
    """Ingest a document into local vector store."""
    text = Path(file_path).read_text(encoding="utf-8")
    chunks = [text[i:i+500] for i in range(0, len(text), 400)]  # 500-char chunks, 100 overlap

    for i, chunk in enumerate(chunks):
        embedding = embed_local(chunk)
        collection.add(
            ids=[f"{Path(file_path).stem}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": Path(file_path).name}],
        )
    print(f"Ingested {len(chunks)} chunks from {file_path}")


def query(question: str, n_results: int = 4) -> str:
    """Query local RAG system — 100% private."""
    q_embedding = embed_local(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=n_results)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Answer the question based only on this context:

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.generate(model="llama3", prompt=prompt)
    return response["response"]


# Usage
ingest("company_docs.txt")
answer = query("What is our refund policy?")
print(answer)
```

---

## Custom Modelfiles: Fine-tune Personality

Ollama's `Modelfile` lets you customize system prompts, parameters, and base models:

```dockerfile
# Modelfile
FROM llama3

# Set system prompt
SYSTEM """You are a senior Python developer assistant. You:
- Write production-quality, well-documented Python code
- Follow PEP 8 and modern Python best practices
- Always include error handling and type hints
- Suggest tests when appropriate"""

# Adjust parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

```bash
# Build and run your custom model
ollama create my-python-expert -f Modelfile
ollama run my-python-expert
```

---

## LangChain with Ollama

```python
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# LLM
llm = OllamaLLM(model="llama3", temperature=0.7)
result = llm.invoke("Explain gradient descent in simple terms")
print(result)

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Full RAG chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("my_docs.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./langchain_chroma")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
result = qa_chain.invoke("What does the document say about pricing?")
print(result["result"])
```

---

## Choosing the Right Local Model

| Model | Size | Best For |
|-------|------|----------|
| `phi3` | 3.8B | Fast inference, low RAM, general tasks |
| `mistral` | 7B | Best quality/speed ratio for general use |
| `llama3` | 8B | Strong instruction following |
| `codellama` | 7B–34B | Code generation and analysis |
| `llama3:70b` | 70B | GPT-4-level quality (needs 40GB+ RAM) |
| `nomic-embed-text` | - | Text embeddings for RAG |

---

## Performance Tuning

```bash
# Control GPU layers (more = faster, uses more VRAM)
ollama run llama3 --num-gpu 35  # move 35 layers to GPU

# Parallel model instances
OLLAMA_NUM_PARALLEL=4 ollama serve  # run 4 simultaneous requests

# Keep model loaded in memory (reduce reload time)
OLLAMA_KEEP_ALIVE=60m ollama serve
```

---

## What to Learn Next

- **Build a local RAG app** → [RAG Document Assistant](/projects/rag-document-assistant/)
- **LangChain in depth** → [LangChain Complete Tutorial](/blog/langchain-tutorial-complete/)
- **Semantic search** → [Semantic Search Explained](/blog/roadmap-guides/semantic-search-explained/)
