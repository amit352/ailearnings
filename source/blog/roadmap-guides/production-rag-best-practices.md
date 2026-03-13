---
title: "Production RAG Best Practices: Chunking, Reranking, and Evaluation"
description: "Go beyond basic RAG with advanced techniques: recursive chunking strategies, metadata filtering, reranking retrieved documents, evaluating RAG pipelines, and building production-ready retrieval systems."
date: "2026-03-10"
slug: "production-rag-best-practices"
keywords: ["RAG best practices production", "RAG reranking chunking", "RAG evaluation pipeline"]
---

## Why Basic RAG Fails in Production

A naive RAG implementation (chunk text → embed → retrieve → generate) works in demos but breaks in production because of:

1. **Bad chunking**: splitting mid-sentence loses context
2. **Missing metadata**: can't filter by date, source, or category
3. **Poor retrieval**: top-k doesn't mean most relevant
4. **No evaluation**: no way to know if it's working
5. **Context stuffing**: just concatenating chunks degrades quality

This guide covers the production patterns that address each failure mode.

---

## Advanced Chunking Strategies

### 1. Recursive Character Splitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    # Try to split on paragraph → sentence → word → character
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

chunks = splitter.split_text(long_document)
```

### 2. Semantic Chunking

Split by meaning, not character count:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()


def embed_sentences(sentences: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences,
    )
    return np.array([e.embedding for e in response.data])


def semantic_chunk(text: str, breakpoint_percentile: float = 0.85) -> list[str]:
    """Split text at semantic breakpoints (where meaning changes significantly)."""
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 3:
        return [text]

    # Embed all sentences
    embeddings = embed_sentences(sentences)

    # Compute cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        similarities.append(float(sim))

    # Find breakpoints (where similarity drops significantly)
    threshold = np.percentile(similarities, (1 - breakpoint_percentile) * 100)
    breakpoints = [i + 1 for i, s in enumerate(similarities) if s < threshold]

    # Build chunks
    chunks, prev = [], 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[prev:bp]))
        prev = bp
    chunks.append(" ".join(sentences[prev:]))
    return [c for c in chunks if c.strip()]
```

### 3. Parent Document Retriever

Store small chunks for retrieval, return large parent chunks for context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Small chunks for search
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
# Large chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
store = InMemoryByteStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
# Searches small chunks, returns parent documents → best of both worlds
```

---

## Metadata Filtering

Always enrich chunks with metadata for precision filtering:

```python
import chromadb
from datetime import datetime

collection = chromadb.PersistentClient("./db").get_or_create_collection("docs")


def ingest_with_metadata(file_path: str, doc_type: str, department: str):
    import hashlib
    from pathlib import Path

    content = Path(file_path).read_text()
    chunks = chunk_text(content)  # your chunking function

    for i, chunk in enumerate(chunks):
        embedding = embed(chunk)
        doc_id = hashlib.md5(f"{file_path}:{i}".encode()).hexdigest()

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": Path(file_path).name,
                "doc_type": doc_type,           # "policy", "faq", "manual"
                "department": department,        # "hr", "legal", "product"
                "date_indexed": datetime.now().isoformat()[:10],
                "chunk_index": i,
            }],
        )


# Filter at query time
results = collection.query(
    query_embeddings=[embed("vacation policy")],
    n_results=5,
    where={
        "$and": [
            {"doc_type": {"$eq": "policy"}},
            {"department": {"$in": ["hr", "legal"]}},
        ]
    },
)
```

---

## Reranking: Improve Retrieval Quality

Vector search returns the `top_k` most similar chunks, but similarity ≠ relevance. A **reranker** scores each result for actual relevance to the specific query.

### Cross-Encoder Reranker

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_and_rerank(query: str, collection, n_retrieve: int = 20, n_final: int = 5) -> list[dict]:
    """
    1. Retrieve many candidates via vector search (fast)
    2. Rerank using cross-encoder (slow but accurate)
    3. Return top-n after reranking
    """
    # Step 1: Retrieve candidates
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=n_retrieve,
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Step 2: Rerank
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    # Step 3: Sort by reranker score
    ranked = sorted(
        zip(docs, metas, scores),
        key=lambda x: x[2],
        reverse=True,
    )[:n_final]

    return [
        {"text": doc, "metadata": meta, "score": float(score)}
        for doc, meta, score in ranked
    ]
```

### LLM-Based Reranker (more accurate, slower)

```python
from openai import OpenAI
import json

client = OpenAI()


def llm_rerank(query: str, candidates: list[str], top_k: int = 3) -> list[int]:
    """Use GPT to rank candidates by relevance."""
    numbered = "\n".join(f"{i+1}. {c[:200]}" for i, c in enumerate(candidates))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Rank these text passages by relevance to the query.
Return a JSON array of indices (1-based) in order from most to least relevant.
Only return the top {top_k}.

Query: {query}

Passages:
{numbered}

Return JSON: {{"ranking": [3, 1, 5, ...]}}"""}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    data = json.loads(response.choices[0].message.content)
    return [i - 1 for i in data.get("ranking", [])[:top_k]]
```

---

## Evaluating RAG Pipelines

You can't improve what you don't measure. Here's a practical evaluation framework:

```python
from openai import OpenAI
import json

client = OpenAI()

EVAL_PROMPT = """Evaluate this RAG system response.

Question: {question}
Retrieved Context: {context}
System Answer: {answer}
Ground Truth Answer: {ground_truth}

Rate each dimension 1-5:
- Faithfulness: Is the answer supported by the context? (no hallucination)
- Relevance: Does the answer address the question?
- Completeness: Does the answer cover all key points?
- Context Quality: Is the retrieved context relevant to the question?

Return JSON:
{{
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "context_quality": <1-5>,
  "faithfulness_reason": "brief explanation",
  "issues": ["list any problems found"]
}}"""


def evaluate_response(question: str, context: str, answer: str, ground_truth: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EVAL_PROMPT.format(
            question=question, context=context[:2000],
            answer=answer, ground_truth=ground_truth,
        )}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def run_evaluation(test_cases: list[dict], rag_fn) -> dict:
    """
    test_cases: [{"question": ..., "ground_truth": ...}]
    rag_fn: function that takes question and returns (answer, context)
    """
    scores = {"faithfulness": [], "relevance": [], "completeness": [], "context_quality": []}
    failures = []

    for tc in test_cases:
        answer, context = rag_fn(tc["question"])
        result = evaluate_response(tc["question"], context, answer, tc["ground_truth"])

        for metric in scores:
            scores[metric].append(result.get(metric, 0))

        if result.get("issues"):
            failures.append({
                "question": tc["question"],
                "issues": result["issues"],
            })

    return {
        "avg_faithfulness":   sum(scores["faithfulness"]) / len(scores["faithfulness"]),
        "avg_relevance":      sum(scores["relevance"]) / len(scores["relevance"]),
        "avg_completeness":   sum(scores["completeness"]) / len(scores["completeness"]),
        "avg_context_quality":sum(scores["context_quality"]) / len(scores["context_quality"]),
        "n_failures": len(failures),
        "failure_examples": failures[:3],
    }
```

---

## Context Assembly: What to Put in the Prompt

How you assemble retrieved chunks matters as much as what you retrieve:

```python
RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Important:
- Cite specific sections when possible: "According to [source]..."
- If context is contradictory, note the discrepancy
- Be concise but complete"""


def build_context(retrieved_chunks: list[dict], max_tokens: int = 3000) -> str:
    """Assemble context with source attribution and token budget."""
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    parts = []
    total_tokens = 0

    for chunk in retrieved_chunks:
        source = chunk.get("metadata", {}).get("source", "unknown")
        text = f"[Source: {source}]\n{chunk['text']}"
        chunk_tokens = len(enc.encode(text))

        if total_tokens + chunk_tokens > max_tokens:
            break

        parts.append(text)
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(parts)
```

---

## RAG Anti-Patterns to Avoid

**1. Top-k too small**: returning only 3 chunks misses relevant content. Retrieve 15–20, rerank to 5.

**2. Ignoring chunk boundaries**: splitting at fixed character counts breaks sentences. Use semantic or recursive splitters.

**3. No query transformation**: complex questions often need reformulation. Use HyDE (Hypothetical Document Embeddings) or query decomposition.

**4. Single retrieval pass**: for multi-hop questions, do retrieval → LLM → retrieval again.

**5. No fallback**: when retrieval fails, the LLM should say "I don't know" — not hallucinate.

---

## What to Learn Next

- **Build a RAG project** → [RAG Document Assistant](/projects/rag-document-assistant/)
- **Vector databases** → [Vector Database Guide](/blog/vector-database-guide/)
- **LangChain RAG patterns** → [LangChain RAG Tutorial](/blog/langchain-rag-tutorial/)
