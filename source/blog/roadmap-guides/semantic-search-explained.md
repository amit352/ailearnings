---
title: "Semantic Search Explained: How Embeddings Enable Meaning-Based Search"
description: "Understand how semantic search works — from text embeddings and vector spaces to HNSW indexes and approximate nearest neighbor search. Build a production-quality semantic search engine in Python."
date: "2026-03-10"
slug: "semantic-search-explained"
keywords: ["semantic search Python tutorial", "embeddings vector search", "HNSW approximate nearest neighbor"]
---

## Keyword Search vs. Semantic Search

**Keyword search** matches exact words:
```
Query: "car repair"
Finds: "car repair shops near me" ✓
Misses: "automobile maintenance" ✗  (same meaning, different words)
```

**Semantic search** matches meaning:
```
Query: "car repair"
Finds: "automobile maintenance" ✓
Finds: "vehicle service center" ✓
Finds: "fix my broken engine" ✓
```

This is possible because embeddings encode *meaning* as vectors, and similar meanings produce similar vectors.

---

## How Embeddings Work

An embedding model converts text → a dense vector of floats:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return np.array(response.data[0].embedding)

# These phrases have different words but similar meanings
cat_emb    = embed("cat")
feline_emb = embed("feline")
dog_emb    = embed("dog")
code_emb   = embed("Python programming language")

# Cosine similarity: how alike are two vectors?
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"cat ↔ feline: {cosine_sim(cat_emb, feline_emb):.3f}")   # 0.91 (very similar)
print(f"cat ↔ dog:    {cosine_sim(cat_emb, dog_emb):.3f}")       # 0.82 (related)
print(f"cat ↔ code:   {cosine_sim(cat_emb, code_emb):.3f}")      # 0.17 (unrelated)
```

The 1536-dimensional embedding space geometrically encodes semantic relationships.

---

## Building a Basic Semantic Search Engine

```python
import numpy as np
from openai import OpenAI
from dataclasses import dataclass

client = OpenAI()


@dataclass
class Document:
    id: str
    text: str
    metadata: dict


class SemanticSearchEngine:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts efficiently in one API call."""
        response = client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data])

    def add_documents(self, docs: list[Document], batch_size: int = 100):
        """Add documents and compute their embeddings."""
        self.documents.extend(docs)
        texts = [doc.text for doc in docs]

        new_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = self.embed_batch(batch)
            new_embeddings.append(embs)
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

        new_embs = np.vstack(new_embeddings)

        # Normalize for cosine similarity (dot product of normalized = cosine)
        norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
        new_embs = new_embs / norms

        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> list[dict]:
        """Find documents most semantically similar to query."""
        if self.embeddings is None:
            return []

        # Embed and normalize query
        q_emb = self.embed_batch([query])[0]
        q_emb = q_emb / np.linalg.norm(q_emb)

        # Cosine similarities (dot product of normalized vectors)
        similarities = self.embeddings @ q_emb

        # Sort and filter
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append({
                    "document": self.documents[idx],
                    "score": round(sim, 4),
                })
        return results


# Example usage
engine = SemanticSearchEngine()

docs = [
    Document("1", "How to train a machine learning model with scikit-learn", {"category": "ml"}),
    Document("2", "Introduction to neural networks and deep learning", {"category": "dl"}),
    Document("3", "Building REST APIs with FastAPI and Python", {"category": "backend"}),
    Document("4", "Kubernetes deployment strategies for microservices", {"category": "devops"}),
    Document("5", "Fine-tuning LLMs with LoRA for custom tasks", {"category": "llm"}),
    Document("6", "Vector databases and approximate nearest neighbor search", {"category": "rag"}),
    Document("7", "GPT-4 API integration in Python applications", {"category": "llm"}),
    Document("8", "Gradient descent optimization algorithms explained", {"category": "ml"}),
]

engine.add_documents(docs)

# Search — even if query uses different words from documents
results = engine.search("teaching computers to learn from data")
for r in results:
    print(f"{r['score']:.3f}: {r['document'].text}")

# Expected top results: ml/dl articles, not kubernetes or FastAPI
```

---

## Production: Approximate Nearest Neighbor (ANN) Search

The naive approach (`O(n * d)` dot products) works for thousands of documents. For millions, you need **Approximate Nearest Neighbor (ANN)** indexes.

### FAISS (Facebook AI Similarity Search)

```bash
pip install faiss-cpu  # or faiss-gpu for NVIDIA GPU
```

```python
import faiss
import numpy as np

# Build a FAISS index
d = 1536  # embedding dimension

# Flat index: exact search (good up to ~100k)
index_flat = faiss.IndexFlatIP(d)  # Inner Product = cosine similarity (with normalized vectors)

# IVF index: approximate search (millions of vectors)
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, 100)  # 100 clusters

# HNSW: best recall/speed tradeoff (recommended for most use cases)
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 = M parameter (connections per node)
index_hnsw.hnsw.efConstruction = 40  # build-time parameter

# Add vectors
embeddings = np.random.randn(100000, d).astype(np.float32)
faiss.normalize_L2(embeddings)  # normalize for cosine similarity

index_hnsw.add(embeddings)
print(f"Index contains {index_hnsw.ntotal} vectors")

# Search
query = np.random.randn(1, d).astype(np.float32)
faiss.normalize_L2(query)

k = 10  # return top 10
distances, indices = index_hnsw.search(query, k)
print(f"Top match index: {indices[0][0]}, similarity: {distances[0][0]:.4f}")

# Save and load
faiss.write_index(index_hnsw, "search_index.bin")
loaded_index = faiss.read_index("search_index.bin")
```

### HNSW Explained

HNSW (Hierarchical Navigable Small World) is a graph-based index:

```
Layer 2 (sparse):  A -------- F
                   |          |
Layer 1:           A --- C -- F --- H
                   |    |    |     |
Layer 0 (dense):   A-B-C-D-E-F-G-H-I-J
```

**Search**: start at top layer, greedily navigate to nearest neighbor, descend to lower layers. O(log n) instead of O(n).

**Key parameters:**
- `M` (16–64): connections per node. Higher = better recall, more memory.
- `efConstruction` (100–800): build-time search width. Higher = better quality, slower build.
- `ef` (50–500): search-time width. Higher = better recall, slower search.

---

## ChromaDB: Full-Featured Vector Database

```bash
pip install chromadb
```

```python
import chromadb
from chromadb.utils import embedding_functions

# Use OpenAI embeddings automatically
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small",
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    "articles",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # use cosine similarity
)

# Add documents (auto-embeds)
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=["Text of doc 1", "Text of doc 2", "Text of doc 3"],
    metadatas=[{"category": "ml"}, {"category": "dl"}, {"category": "rag"}],
)

# Query
results = collection.query(
    query_texts=["machine learning tutorial"],
    n_results=3,
    where={"category": "ml"},  # metadata filtering
    include=["documents", "distances", "metadatas"],
)

for doc, dist, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print(f"{1 - dist:.3f}: [{meta['category']}] {doc[:80]}")
```

---

## Hybrid Search: Best of Both Worlds

Pure semantic search sometimes misses exact keyword matches. **Hybrid search** combines BM25 (keyword) + vector (semantic):

```python
from rank_bm25 import BM25Okapi
import numpy as np


class HybridSearchEngine:
    def __init__(self, semantic_engine: SemanticSearchEngine):
        self.semantic = semantic_engine
        self.bm25 = None

    def build_bm25(self):
        """Build BM25 index from document texts."""
        tokenized = [doc.text.lower().split() for doc in self.semantic.documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> list[dict]:
        """
        alpha: weight for semantic (1 - alpha for BM25)
        alpha=1.0 = pure semantic, alpha=0.0 = pure keyword
        """
        n = len(self.semantic.documents)

        # Semantic scores
        sem_results = self.semantic.search(query, top_k=n, threshold=0)
        sem_scores = np.zeros(n)
        for r in sem_results:
            idx = self.semantic.documents.index(r["document"])
            sem_scores[idx] = r["score"]

        # BM25 scores (normalize to [0, 1])
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # Combine
        combined = alpha * sem_scores + (1 - alpha) * bm25_scores
        top_indices = np.argsort(combined)[::-1][:top_k]

        return [
            {"document": self.semantic.documents[i], "score": round(float(combined[i]), 4)}
            for i in top_indices if combined[i] > 0
        ]
```

**When to use hybrid:**
- Technical documentation (needs exact term matching)
- Code search (identifiers must match exactly)
- Medical/legal (precise terminology matters)

**When pure semantic is fine:**
- General knowledge Q&A
- Customer support
- FAQ matching

---

## What to Learn Next

- **RAG systems** → [RAG System Architecture](/blog/rag-system-architecture/)
- **Vector databases deep dive** → [Vector Database Guide](/blog/vector-database-guide/)
- **Build a search app** → [RAG Document Assistant](/projects/rag-document-assistant/)
