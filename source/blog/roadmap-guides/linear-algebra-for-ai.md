---
title: "Linear Algebra for AI Developers: The Concepts That Actually Matter"
description: "A practical guide to the linear algebra used in AI and machine learning — vectors, matrices, dot products, and matrix multiplication explained with Python code and real AI use cases."
date: "2026-03-10"
slug: "linear-algebra-for-ai"
keywords: ["linear algebra for machine learning", "vectors matrices AI", "linear algebra Python"]
---

## Why Bother With Linear Algebra?

You don't need to prove theorems. But these concepts appear *everywhere* in AI:

- **Embeddings** are vectors. Semantic similarity is a dot product.
- **Neural network layers** are matrix multiplications.
- **Attention in transformers** is matrix operations on query/key/value vectors.
- **PCA, SVD** (dimensionality reduction) are linear algebra operations.

Understanding these at a conceptual level will make you a better AI engineer, even if you never implement them from scratch.

---

## Vectors: The Foundation

A vector is an ordered list of numbers. In AI, vectors represent:
- Word/sentence meanings (embeddings)
- Image features (pixel intensities)
- User preferences (recommendation systems)

```python
import numpy as np

# A 3-dimensional vector
v = np.array([1.0, 2.0, 3.0])

# A 1536-dimensional embedding (real OpenAI embeddings are this size)
embedding = np.array([0.023, -0.145, 0.087, ...])  # 1536 numbers

# Vectors have magnitude (length) and direction
magnitude = np.linalg.norm(v)  # √(1² + 2² + 3²) = √14 ≈ 3.74
print(f"Magnitude: {magnitude:.2f}")
```

### Vector Operations

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Addition: combine two vectors
print(a + b)  # [5, 7, 9]

# Scalar multiplication: scale a vector
print(2 * a)  # [2, 4, 6]

# Dot product: measures similarity
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot}")
```

---

## The Dot Product: Semantic Similarity in AI

The dot product is arguably the most important operation in modern AI. It's how:
- Embedding similarity is measured
- Attention weights are computed in transformers
- Neural network outputs are calculated

```python
# Cosine similarity: normalize dot product by magnitudes
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Real example: are these embeddings similar?
from openai import OpenAI
client = OpenAI()

def embed(text):
    return np.array(client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding)

cat_emb = embed("cat")
dog_emb = embed("dog")
car_emb = embed("automobile")
code_emb = embed("Python programming")

print(f"cat ↔ dog:  {cosine_similarity(cat_emb, dog_emb):.3f}")   # ~0.85 (similar)
print(f"cat ↔ car:  {cosine_similarity(cat_emb, car_emb):.3f}")   # ~0.40 (different)
print(f"dog ↔ code: {cosine_similarity(dog_emb, code_emb):.3f}")  # ~0.20 (very different)
```

**Key insight:** Cosine similarity of 1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite meaning.

---

## Matrices: Transformations

A matrix is a 2D array of numbers. In neural networks, each layer is a matrix multiplication that *transforms* an input vector into an output vector.

```python
# A weight matrix (3 inputs → 2 outputs)
W = np.array([
    [0.1, 0.2, 0.3],   # weights for output neuron 1
    [0.4, 0.5, 0.6],   # weights for output neuron 2
])  # shape: (2, 3)

x = np.array([1.0, 2.0, 3.0])  # input vector, shape: (3,)

# Matrix-vector multiplication: apply the transformation
output = W @ x   # or np.matmul(W, x)
# = [0.1*1 + 0.2*2 + 0.3*3,   → [1.4]
#    0.4*1 + 0.5*2 + 0.6*3]   → [3.2]
print(output)  # [1.4, 3.2]
```

A neural network layer is literally:
```
output = activation(W @ input + bias)
```

### Matrix Multiplication Rules

```python
A = np.random.randn(3, 4)   # shape (3, 4)
B = np.random.randn(4, 5)   # shape (4, 5)

# Valid: inner dimensions match (4 == 4)
C = A @ B   # result shape: (3, 5)

# Invalid: inner dimensions don't match
D = np.random.randn(3, 5)
# D @ A would fail: (3,5) @ (3,4) — inner 5 ≠ 3

# Shape rule: (m, n) @ (n, p) → (m, p)
```

---

## Batch Operations: Why Shape Matters

In practice, you process many examples at once (a "batch"). This is where tensor shapes become critical.

```python
# Single example
x = np.array([1.0, 2.0, 3.0])    # shape: (3,)

# Batch of 32 examples
X = np.random.randn(32, 3)       # shape: (32, 3)

W = np.random.randn(3, 128)      # weight matrix, shape: (3, 128)
bias = np.random.randn(128)

# Apply same transformation to all 32 examples at once
output = X @ W + bias            # shape: (32, 128) — 32 examples, 128 features each
```

This is why PyTorch and NumPy always report shapes — you need to track dimensions to ensure operations are valid.

---

## Attention: Where Linear Algebra Gets Magical

The attention mechanism in transformers is fundamentally linear algebra. Here's the simplified version:

```python
# Scaled Dot-Product Attention (simplified)
import numpy as np

def attention(Q, K, V):
    """
    Q (queries), K (keys), V (values) — all matrices
    Returns weighted sum of values based on query-key similarity
    """
    d_k = Q.shape[-1]          # key dimension for scaling

    # Compute similarity scores: how much does each query "attend to" each key?
    scores = Q @ K.T / np.sqrt(d_k)   # shape: (seq_len, seq_len)

    # Softmax: convert scores to probabilities
    scores -= scores.max(axis=-1, keepdims=True)  # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    return weights @ V   # shape: (seq_len, d_v)

# Example: 5 tokens, 64 dimensions
seq_len, d_model = 5, 64
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output = attention(Q, K, V)
print(f"Output shape: {output.shape}")  # (5, 64)
```

Each token "queries" all other tokens (dot products), converts to attention weights (softmax), and produces a weighted combination of values. This is how "the cat sat on the mat" knows that "sat" relates to "cat."

---

## Practical: Semantic Search with Embeddings

Here's a complete example combining everything:

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embeddings(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([item.embedding for item in response.data])


def semantic_search(query: str, documents: list[str], top_k: int = 3) -> list[tuple]:
    """Find the most semantically similar documents to the query."""
    # Embed query and documents
    query_emb = get_embeddings([query])[0]    # shape: (1536,)
    doc_embs = get_embeddings(documents)       # shape: (n_docs, 1536)

    # Normalize (for cosine similarity via dot product)
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

    # Dot product gives cosine similarity for normalized vectors
    similarities = doc_norms @ query_norm    # shape: (n_docs,)

    # Sort by similarity, descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(documents[i], similarities[i]) for i in top_indices]


docs = [
    "Machine learning uses algorithms to learn from data",
    "Python is a popular programming language",
    "Neural networks are inspired by the human brain",
    "Deep learning requires large amounts of training data",
    "JavaScript runs in web browsers",
]

results = semantic_search("How do computers learn from examples?", docs)
for doc, score in results:
    print(f"{score:.3f}: {doc}")
```

---

## The Concepts Worth Going Deeper On

If you want to understand more of the ML math:

**Eigenvalues/eigenvectors** — used in PCA for dimensionality reduction, understanding covariance matrices.

**Singular Value Decomposition (SVD)** — matrix factorization used in recommendation systems, data compression, and understanding LLM weight matrices.

**Gradient vectors** — in backpropagation, gradients are vectors pointing in the direction of steepest loss increase. Gradient descent moves *opposite* to the gradient.

---

## What to Learn Next

- **Statistics for ML** → [Statistics for Machine Learning](/blog/roadmap-guides/statistics-for-machine-learning/)
- **Neural networks** → [Neural Networks from Scratch](/blog/roadmap-guides/neural-networks-from-scratch/)
- **Embeddings in depth** → [Embeddings Explained](/blog/embeddings-explained/)
