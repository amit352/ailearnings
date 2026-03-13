---
title: "Embeddings Explained – How AI Understands Meaning"
description: "Understand embeddings and how they power semantic search and AI applications."
date: "2026-03-13"
slug: "embeddings-explained"
keywords: ["embeddings explained", "text embeddings", "word embeddings", "semantic search embeddings", "what are embeddings"]
---

## Learning Objectives

- Understand what embeddings are and why they matter for AI
- Generate text embeddings using OpenAI and open-source models
- Measure semantic similarity between texts
- Build a semantic search engine from scratch
- Choose the right embedding model for your use case

---

## What Are Embeddings?

An **embedding** is a list of numbers (a vector) that represents the meaning of a piece of text. Two pieces of text with similar meanings produce vectors that point in similar directions in high-dimensional space.

```python
# "king" and "queen" will have similar embeddings
# "king" and "bicycle" will have very different embeddings

king   = embed("king")    # [0.23, -0.14, 0.87, 0.03, ...]  (1536 numbers)
queen  = embed("queen")   # [0.21, -0.12, 0.89, 0.05, ...]  (similar!)
bicycle = embed("bicycle") # [0.91, 0.67, -0.22, 0.44, ...]  (different)
```

The famous example: `embed("king") - embed("man") + embed("woman") ≈ embed("queen")`.

---

## Why Embeddings Matter for AI

| Use Case | How Embeddings Help |
|----------|-------------------|
| Semantic search | Find documents by meaning, not keyword |
| RAG | Find relevant context for an LLM prompt |
| Recommendation | Find similar items to what a user liked |
| Clustering | Group similar documents together |
| Classification | Use vector distance as a feature |
| Duplicate detection | Find near-duplicate content |
| Anomaly detection | Flag items far from all clusters |

---

## Generating Embeddings

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

# Single text
vector = embed("The quick brown fox")
print(f"Dimensions: {len(vector)}")  # 1536

# Batch (more efficient — one API call)
texts = ["machine learning", "deep learning", "cooking pasta"]
response = client.embeddings.create(model="text-embedding-3-small", input=texts)
vectors = [item.embedding for item in response.data]
```

**OpenAI embedding models:**
- `text-embedding-3-small` — 1536 dims, $0.02/M tokens — best value
- `text-embedding-3-large` — 3072 dims, $0.13/M tokens — highest quality
- `text-embedding-ada-002` — legacy, use 3-small instead

### Open-Source Embeddings (Free, Local)

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # 33M params, fast, 384 dims

texts = ["machine learning", "artificial intelligence", "cooking pasta"]
embeddings = model.encode(texts, normalize_embeddings=True)
print(embeddings.shape)  # (3, 384)
```

**Top open-source embedding models:**

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good |
| `BAAI/bge-large-en-v1.5` | 1024 | Medium | Very good |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Very fast | Good |
| `intfloat/multilingual-e5-large` | 1024 | Medium | Best multilingual |

---

## Measuring Similarity

### Cosine Similarity

```python
import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

e1 = embed("I love machine learning")
e2 = embed("Deep learning is fascinating")
e3 = embed("I enjoy cooking Italian food")

print(cosine_similarity(e1, e2))  # ~0.85 (similar topics)
print(cosine_similarity(e1, e3))  # ~0.72 (different topics)
```

Range: -1 to 1. Above 0.85 is usually very similar. Below 0.70 is usually different.

### Batch Similarity with NumPy

```python
def batch_cosine_similarity(query: list, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query and all corpus vectors."""
    q = np.array(query)
    q_norm = q / np.linalg.norm(q)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    return corpus_norm @ q_norm

query_vec = np.array(embed("how do neural networks learn?"))
corpus_vecs = np.array([embed(t) for t in texts])  # (N, 1536)

similarities = batch_cosine_similarity(query_vec, corpus_vecs)
top_idx = np.argsort(similarities)[::-1][:5]
for i in top_idx:
    print(f"[{similarities[i]:.3f}] {texts[i]}")
```

---

## Building a Semantic Search Engine

```python
import numpy as np
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class Document:
    id: str
    content: str
    metadata: dict

class SemanticSearchEngine:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, docs: list[Document]):
        self.documents.extend(docs)
        texts = [d.content for d in docs]

        # Batch embed all documents
        response = self.client.embeddings.create(model=self.model, input=texts)
        new_vecs = np.array([item.embedding for item in response.data])

        if self.embeddings is None:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Document]]:
        q_response = self.client.embeddings.create(model=self.model, input=[query])
        q_vec = np.array(q_response.data[0].embedding)

        # Normalize
        q_norm = q_vec / np.linalg.norm(q_vec)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        similarities = doc_norms @ q_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(float(similarities[i]), self.documents[i]) for i in top_indices]


# Usage
engine = SemanticSearchEngine()
engine.add_documents([
    Document("1", "Python is a high-level programming language", {"lang": "en"}),
    Document("2", "Machine learning uses algorithms to learn from data", {"lang": "en"}),
    Document("3", "Neural networks are inspired by the human brain", {"lang": "en"}),
    Document("4", "Docker containers package applications with their dependencies", {"lang": "en"}),
    Document("5", "Gradient descent optimizes model parameters iteratively", {"lang": "en"}),
])

results = engine.search("how do AI models learn?", top_k=3)
for score, doc in results:
    print(f"[{score:.3f}] {doc.content}")
```

---

## Dimensionality Reduction and Visualization

Embeddings are hard to visualize in 1536 dimensions. Use UMAP or t-SNE to reduce to 2D:

```bash
pip install umap-learn matplotlib
```

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

# Sample topics
topics = [
    "machine learning", "deep learning", "neural networks",
    "Python programming", "JavaScript web dev", "React components",
    "cooking recipes", "Italian pasta", "French cuisine",
]
labels = ["ML", "ML", "ML", "Dev", "Dev", "Dev", "Food", "Food", "Food"]
colors = {"ML": "blue", "Dev": "green", "Food": "orange"}

embeddings = np.array([embed(t) for t in topics])

# Reduce to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
coords = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 7))
for i, (text, label) in enumerate(zip(topics, labels)):
    plt.scatter(coords[i, 0], coords[i, 1], c=colors[label], s=100)
    plt.annotate(text, (coords[i, 0], coords[i, 1]), fontsize=9)

plt.title("Topic Clusters in Embedding Space")
plt.tight_layout()
plt.savefig("embedding_clusters.png")
plt.show()
```

Similar topics cluster together in the 2D projection.

---

## Fine-Tuned Embeddings

Generic embeddings work well for most use cases. For domain-specific search (medical, legal, code), fine-tuning embeddings on your data improves retrieval significantly.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Training pairs: (query, relevant_doc, irrelevant_doc)
train_examples = [
    InputExample(texts=["What is RAG?", "RAG combines retrieval with generation", "Italian pasta recipe"]),
    InputExample(texts=["Python syntax", "Variables and data types in Python", "Weather forecast"]),
]

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./fine-tuned-embeddings",
)
```

---

## Troubleshooting

**Embeddings give wrong similarity scores**
- Make sure you're normalizing vectors before cosine similarity
- Check if the embedding model matches your domain — generic models may struggle with specialized jargon

**API costs are high**
- Cache embeddings — store them in a database once, never re-embed the same text
- Batch requests — embed many texts in one API call
- Use a smaller/local model for bulk processing

**Different lengths of text embed differently**
- Very short texts (1–2 words) embed poorly — try to embed phrases or sentences
- Very long texts (> model token limit) get truncated — chunk them first

---

## FAQ

**Are embeddings the same as word2vec?**
Related concept, but different. Word2vec creates per-word embeddings. Modern sentence transformers create per-sentence/passage embeddings that capture full meaning in context.

**How do I store embeddings persistently?**
Use a vector database (Chroma, Qdrant, Pinecone) or store as numpy arrays with `np.save`. For PostgreSQL users, pgvector adds native vector storage and search.

---

## What to Learn Next

- **Vector databases** → [Vector Databases Explained](/blog/vector-database-explained/)
- **RAG pipeline** → [RAG Explained](/blog/rag-explained/)
- **Using embeddings via OpenAI Python** → [OpenAI Python Client Guide](/blog/openai-python-client-guide/)
