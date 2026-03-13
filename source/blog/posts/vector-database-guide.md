---
title: "Vector Database Guide: Embeddings, Similarity Search, and Choosing the Right DB"
description: "Complete guide to vector databases — how embeddings work, similarity search algorithms, HNSW indexing, and a comparison of Pinecone, Weaviate, Qdrant, Chroma, and pgvector."
date: "2026-03-10"
slug: "vector-database-guide"
keywords: ["vector database", "vector database guide", "embeddings similarity search", "Pinecone Weaviate Qdrant"]
---

## Learning Objectives

- Understand what vector databases are and why they exist
- Know how similarity search works (cosine similarity, HNSW)
- Set up and query Chroma locally
- Compare the major vector database options
- Design an efficient vector search pipeline

---

## What Is a Vector Database?

A vector database stores high-dimensional numerical vectors and enables fast similarity search. Instead of asking "find rows where name = 'John'", you ask "find the 10 vectors most similar to this query vector."

This is the foundation of:
- **Semantic search** — find documents by meaning, not keywords
- **RAG (Retrieval-Augmented Generation)** — find relevant context before calling an LLM
- **Recommendation systems** — find similar products or content
- **Duplicate detection** — find near-duplicate documents

---

## How Embeddings Work

An **embedding** is a dense vector (list of floats) that encodes the meaning of text (or images, audio) in a high-dimensional space. Semantically similar content has vectors that point in similar directions.

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    ).data[0].embedding

e_ml  = embed("machine learning")
e_ai  = embed("artificial intelligence")
e_dog = embed("golden retriever puppy")

# e_ml and e_ai are close in vector space
# e_dog is far from both
```

**Embedding dimensions:**
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-3-large`: 3072 dimensions
- Open source (BAAI/bge-small-en): 384 dimensions

---

## Similarity Metrics

### Cosine Similarity
Measures the angle between two vectors. Range: [-1, 1]. Most common for text embeddings.

```
similarity = (A · B) / (|A| × |B|)
```

```python
import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### Dot Product
Faster than cosine (no normalization). Use when embeddings are normalized (same as cosine for unit vectors). Used by `text-embedding-3` models internally.

### Euclidean Distance (L2)
Physical distance in vector space. Useful when magnitude matters, not just direction.

---

## Indexing: How Fast Search Works

Brute-force similarity search (`O(n)`) doesn't scale. Vector databases use approximate nearest neighbor (ANN) algorithms.

### HNSW (Hierarchical Navigable Small World)
The dominant indexing algorithm. Builds a multi-layer graph. Search complexity: `O(log n)`.

Key parameters:
- `M` (max connections per node): higher = more accurate but slower and more memory
- `ef_construction` (search depth during build): higher = better quality index
- `ef_search` (search depth at query time): higher = better recall

```python
# HNSW in practice (hnswlib)
import hnswlib
import numpy as np

dim = 1536
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=100000, ef_construction=200, M=16)

# Add vectors
vectors = np.random.randn(10000, dim).astype(np.float32)
ids = np.arange(10000)
index.add_items(vectors, ids)

# Search
index.set_ef(50)  # search quality (higher = better recall, slower)
query = np.random.randn(1, dim).astype(np.float32)
labels, distances = index.knn_query(query, k=10)
print(f"Top-10 IDs: {labels[0]}")
```

---

## Chroma: Local Vector Database

Chroma is the easiest vector database to get started with — runs in-memory or persisted locally, no server needed.

```bash
pip install chromadb
```

### Create and Populate a Collection

```python
import chromadb
from chromadb.utils import embedding_functions

# Persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Use OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-small",
)

collection = client.get_or_create_collection(
    name="docs",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)

# Add documents
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "RAG combines retrieval with generation for better LLM outputs.",
        "Vector databases store embeddings for fast similarity search.",
        "Transformers use attention mechanisms to process sequences.",
    ],
    metadatas=[
        {"source": "rag_article", "date": "2026-03-10"},
        {"source": "vector_db_article", "date": "2026-03-10"},
        {"source": "transformer_article", "date": "2026-03-10"},
    ],
)
```

### Query

```python
results = collection.query(
    query_texts=["How do vector stores work?"],
    n_results=3,
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(f"[{dist:.3f}] {doc[:80]}...")
    print(f"  Source: {meta['source']}\n")
```

### Filter by Metadata

```python
results = collection.query(
    query_texts=["attention mechanism"],
    n_results=5,
    where={"source": "transformer_article"},
)
```

---

## Qdrant: Production-Grade Local or Cloud

Qdrant is a high-performance vector database with rich filtering. Runs as a Docker container.

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

```bash
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Insert points
points = [
    PointStruct(id=1, vector=embed("RAG architecture"),    payload={"topic": "rag", "level": "intermediate"}),
    PointStruct(id=2, vector=embed("neural network"),      payload={"topic": "ml",  "level": "beginner"}),
    PointStruct(id=3, vector=embed("transformer model"),   payload={"topic": "llm", "level": "advanced"}),
]
client.upsert(collection_name="articles", points=points)

# Search with filter
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="articles",
    query_vector=embed("how do LLMs work?"),
    limit=3,
    query_filter=Filter(
        must=[FieldCondition(key="level", match=MatchValue(value="advanced"))]
    ),
)

for r in results:
    print(f"[{r.score:.3f}] id={r.id}, topic={r.payload['topic']}")
```

---

## Comparison of Vector Databases

| Database | Best For | Hosting | Highlights |
|----------|---------|---------|------------|
| **Chroma** | Local dev, prototyping | Local only | Zero setup, Python-native |
| **Qdrant** | Production self-hosted | Local + Cloud | Rich filtering, high perf |
| **Weaviate** | Hybrid search + GraphQL | Local + Cloud | BM25 + vector hybrid |
| **Pinecone** | Serverless cloud | Cloud only | Zero ops, managed |
| **Milvus** | High-scale production | Local + Cloud | Billion-vector scale |
| **pgvector** | Existing PostgreSQL users | Local + Cloud | SQL + vector in one DB |

### pgvector (Vector Search in PostgreSQL)

```sql
-- Install extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- Insert
INSERT INTO documents (content, embedding)
VALUES ('RAG overview', '[0.1, 0.2, ...]');

-- Search (cosine similarity)
SELECT content, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

---

## Chunking Strategy Matters

The quality of your vector search depends heavily on how you split documents before embedding.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,      # overlap prevents context loss at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_text(long_document)
```

**Chunking strategies:**
- Fixed size (512–1024 chars): simple baseline
- Sentence-level: preserves semantic units
- Paragraph-level: good for narrative text
- Semantic chunking: split where meaning changes (advanced, needs embedding)

---

## Troubleshooting

**Search returns irrelevant results**
- Check embedding model quality — use a specialized model for your domain
- Reduce chunk size — overly long chunks dilute meaning
- Increase `n_results` and re-rank with a cross-encoder

**Very slow queries**
- Ensure HNSW index is built (not brute-force)
- Reduce `ef_search` (lower recall but faster)
- For Qdrant/Milvus: verify collection is indexed

**High memory usage**
- Use dimensionality reduction (MRL with `text-embedding-3`)
- Use 4-bit quantized embeddings (supported by Qdrant)
- Archive old vectors to cold storage

---

## FAQ

**Should I use a vector DB or just store embeddings in PostgreSQL?**
For < 1 million vectors, pgvector works well. For larger scale, filtering-heavy, or high-throughput workloads, a dedicated vector database (Qdrant, Weaviate) is better.

**Which embedding model should I use?**
For English text: `text-embedding-3-small` (OpenAI) or `BAAI/bge-large-en-v1.5` (open-source, free). For multilingual: `multilingual-e5-large`.

---

## What to Learn Next

- **Build a RAG pipeline** → [RAG Tutorial Step by Step](/blog/rag-tutorial-step-by-step/)
- **Document chunking strategies** → document-chunking-strategies
- **LangChain with vector stores** → langchain-rag-tutorial
