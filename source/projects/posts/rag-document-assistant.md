---
title: "Build a RAG Document Assistant: Chat with Your Documents"
description: "Build a Retrieval-Augmented Generation system that lets you upload documents and ask questions. Uses embeddings, vector search, and GPT-4o-mini for accurate, grounded answers."
date: "2026-03-10"
slug: "rag-document-assistant"
level: "Intermediate"
time: "4–6 hours"
stack: "Python, OpenAI API, ChromaDB, Streamlit"
keywords: ["RAG Python tutorial", "document Q&A AI", "retrieval augmented generation project"]
---

## Project Overview

A document Q&A assistant that ingests PDFs and text files, stores them in a vector database, and answers questions using only the content in your documents. Prevents hallucinations by grounding every answer in retrieved context.

---

## Learning Goals

- Build a complete RAG pipeline from scratch
- Use OpenAI embeddings to convert text to vectors
- Query ChromaDB for semantic similarity search
- Design prompts that cite sources and stay grounded

---

## Architecture

```
Documents (PDF/txt)
        ↓
Text extraction + chunking
        ↓
Embed chunks → ChromaDB vector store
        ↓
User question → embed → similarity search → top-k chunks
        ↓
LLM with retrieved context → grounded answer
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai chromadb pypdf streamlit tiktoken
```

### Step 2: Document Ingestion

```python
# ingest.py
import os
import hashlib
import tiktoken
import chromadb
from pathlib import Path
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("documents")
enc = tiktoken.encoding_for_model("text-embedding-3-small")

CHUNK_SIZE = 500   # tokens
CHUNK_OVERLAP = 50


def extract_text(path: str) -> str:
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return Path(path).read_text(encoding="utf-8")


def chunk_text(text: str, source: str) -> list[dict]:
    tokens = enc.encode(text)
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunk_text = enc.decode(chunk_tokens)
        chunk_id = hashlib.md5(f"{source}:{chunk_idx}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": source,
            "chunk_index": chunk_idx,
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_idx += 1
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_document(path: str) -> int:
    print(f"Processing: {path}")
    text = extract_text(path)
    chunks = chunk_text(text, os.path.basename(path))

    batch_size = 100
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch],
        )
        total += len(batch)
        print(f"  Indexed {total}/{len(chunks)} chunks...")

    print(f"Done. Indexed {total} chunks from {path}")
    return total
```

### Step 3: Query Engine

```python
# query.py
from openai import OpenAI
from ingest import client, collection, embed_texts

RAG_PROMPT = """Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't have information about that in the provided documents."
Always cite which document(s) you used.

Context:
{context}

Question: {question}

Answer (with source citations):"""


def retrieve(question: str, n_results: int = 5) -> list[dict]:
    embedding = embed_texts([question])[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
    )
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return chunks


def answer_question(question: str, n_results: int = 5) -> dict:
    chunks = retrieve(question, n_results)

    if not chunks:
        return {"answer": "No documents indexed yet.", "sources": []}

    context_parts = []
    sources = set()
    for chunk in chunks:
        context_parts.append(f"[From: {chunk['source']}]\n{chunk['text']}")
        sources.add(chunk["source"])

    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": RAG_PROMPT.format(
            context=context, question=question
        )}],
        max_tokens=800,
        temperature=0.1,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": list(sources),
        "chunks_used": len(chunks),
    }
```

### Step 4: Streamlit App

```python
# app.py
import streamlit as st
from ingest import ingest_document
from query import answer_question
import tempfile, os

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")
st.caption("Upload documents and ask questions — answers grounded in your content")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("PDF or text files", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Index Documents", type="primary"):
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            with st.spinner(f"Indexing {f.name}..."):
                count = ingest_document(tmp_path)
            os.unlink(tmp_path)
            st.success(f"Indexed {f.name} ({count} chunks)")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = answer_question(prompt)
        st.write(result["answer"])
        if result.get("sources"):
            st.caption(f"Sources: {', '.join(result['sources'])} | Chunks retrieved: {result['chunks_used']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })
```

### Step 5: Run

```bash
# Index documents via CLI
python -c "from ingest import ingest_document; ingest_document('report.pdf')"

# Launch the app
streamlit run app.py
```

---

## Extension Ideas

1. **Multi-collection support** — separate namespaces per project or user
2. **Conversation memory** — maintain chat history for follow-up questions
3. **Hybrid search** — combine BM25 keyword search with vector search
4. **Confidence scores** — show retrieval distance scores to user
5. **Document management** — list/delete indexed documents from the UI

---

## What to Learn Next

- **RAG architecture** → [RAG System Architecture](/blog/rag-system-architecture/)
- **Vector databases** → [Vector Database Guide](/blog/vector-database-guide/)
- **Chunking strategies** → [Document Chunking Strategies](/blog/document-chunking-strategies/)
