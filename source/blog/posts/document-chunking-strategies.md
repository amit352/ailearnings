---
title: "Document Chunking Strategies for RAG: How to Split Text for Better Retrieval"
description: "Learn the best document chunking strategies for RAG — fixed size, sentence, semantic, and hierarchical chunking. Includes benchmarks and code for each approach."
date: "2026-03-10"
slug: "document-chunking-strategies"
keywords: ["document chunking RAG", "text chunking strategies", "RAG chunking", "chunk size for RAG"]
---

## Learning Objectives

- Understand why chunking matters for RAG retrieval quality
- Implement and compare five chunking strategies
- Choose the right chunk size for your use case
- Handle edge cases: tables, code, PDFs, and multilingual text
- Evaluate chunking quality

---

## Why Chunking Matters

In RAG, you embed chunks and retrieve the top-k most similar to the query. If chunks are:
- **Too large** → diluted meaning, retrieves irrelevant content alongside relevant
- **Too small** → loses context, answer may span multiple chunks
- **Cut across sentence boundaries** → broken semantic units reduce embedding quality

The goal: **chunks that are semantically self-contained and as small as possible while still answering a question.**

---

## Strategy 1: Fixed-Size Chunking

Split by character or token count, with optional overlap.

```python
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Basic character splitting
splitter = CharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separator="\n",
)
chunks = splitter.split_text(text)

# Recursive (tries \n\n → \n → . → space — better for natural text)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)
chunks = splitter.split_text(text)
```

**When to use:** Simple baseline. Works well for well-structured prose.
**Limitations:** Ignores semantic boundaries; can cut mid-sentence.

**Recommended sizes:**
- Dense technical docs: 256–512 chars
- General text / articles: 512–1024 chars
- Long narratives: 1024–2048 chars

---

## Strategy 2: Token-Based Chunking

LLMs count tokens, not characters. Use `tiktoken` to chunk by tokens for precise control.

```bash
pip install tiktoken
```

```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_by_tokens(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    enc = tiktoken.encoding_for_model("gpt-4o")

    def token_length(s: str) -> int:
        return len(enc.encode(s))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


chunks = chunk_by_tokens(long_document, chunk_size=256, overlap=32)
print(f"Chunks: {len(chunks)}, avg tokens: {sum(len(enc.encode(c)) for c in chunks) // len(chunks)}")
```

**When to use:** When working with models with strict token limits. More precise than character-based.

---

## Strategy 3: Sentence-Level Chunking

Group complete sentences. Preserves semantic units better than fixed-size.

```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
```

```python
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")

def chunk_by_sentences(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)

    return chunks


chunks = chunk_by_sentences(text, sentences_per_chunk=5, overlap=1)
```

**When to use:** News articles, FAQs, support documentation. Anything where sentences are meaningful units.

---

## Strategy 4: Semantic Chunking

Split where meaning changes significantly — the most intelligent approach but requires an embedding call per sentence.

```bash
pip install langchain-experimental
```

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# percentile: split when cosine distance between adjacent sentences > the 95th percentile
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

chunks = chunker.split_text(text)
print(f"Semantic chunks: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(c)} chars) ---\n{c[:200]}...")
```

**When to use:** Long documents with clearly distinct topics (annual reports, textbooks, documentation). When quality matters more than speed.
**Limitations:** More expensive (embedding each sentence), slower.

---

## Strategy 5: Hierarchical / Parent-Child Chunking

Index small chunks for precise retrieval, but return their parent (larger) chunk to the LLM for context.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Child splitter: small, precise
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# Parent splitter: large, context-rich
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Vector store (stores child embeddings)
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

# Document store (stores parent docs)
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)

# Retrieves small child for matching, returns large parent for context
results = retriever.invoke("What is the authentication timeout?")
print(f"Retrieved {len(results)} parent chunks")
```

**When to use:** When you need high precision in retrieval but rich context in answers. Great for technical documentation.

---

## Handling Special Content

### Code Blocks

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)

markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=512,
    chunk_overlap=50,
)
```

### PDF Documents

PDFs often have poor text extraction. Use preprocessing:

```python
from langchain_community.document_loaders import PyPDFLoader
import re

def clean_pdf_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)        # collapse multiple blank lines
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)   # rejoin hyphenated line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)    # join single line breaks
    return text.strip()

loader = PyPDFLoader("document.pdf")
pages = loader.load()
for page in pages:
    page.page_content = clean_pdf_text(page.page_content)
```

### Tables

Tables lose structure when extracted as text. Strategies:
1. Use a table-aware extractor (Unstructured, pdfplumber)
2. Convert table to Markdown before chunking
3. Store table cells with rich metadata

```python
import pdfplumber

def extract_tables_as_markdown(pdf_path: str) -> list[str]:
    tables_md = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                # Convert to Markdown
                header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
                divider = "| " + " | ".join("---" for _ in table[0]) + " |"
                rows = [
                    "| " + " | ".join(str(c or "") for c in row) + " |"
                    for row in table[1:]
                ]
                tables_md.append("\n".join([header, divider] + rows))
    return tables_md
```

---

## Choosing Chunk Size: Decision Tree

```
Does your text have clear paragraph structure?
├── Yes → RecursiveCharacterTextSplitter, 512 chars
└── No  → Continue:
    Does meaning shift gradually (academic paper, report)?
    ├── Yes → SemanticChunker
    └── No  → Continue:
        Is precise token count important (LLM context limits)?
        ├── Yes → Token-based chunking
        └── No  → Sentence-based, 5 sentences/chunk
```

---

## Benchmarking Chunking Strategies

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_chunking(chunks: list[str], test_queries: list[tuple]) -> dict:
    """
    test_queries: [(query, expected_answer_substring), ...]
    """
    # Embed all chunks
    from openai import OpenAI
    client = OpenAI()
    chunk_embeddings = np.array([
        client.embeddings.create(model="text-embedding-3-small", input=c).data[0].embedding
        for c in chunks
    ])

    hits = 0
    for query, expected in test_queries:
        q_emb = np.array(client.embeddings.create(
            model="text-embedding-3-small", input=query
        ).data[0].embedding)

        sims = cosine_similarity([q_emb], chunk_embeddings)[0]
        top_chunks = [chunks[i] for i in np.argsort(sims)[::-1][:3]]

        if any(expected.lower() in c.lower() for c in top_chunks):
            hits += 1

    return {
        "recall@3": hits / len(test_queries),
        "num_chunks": len(chunks),
        "avg_chunk_len": int(np.mean([len(c) for c in chunks])),
    }
```

---

## FAQ

**What chunk size should I start with?**
Start with 512 chars, `RecursiveCharacterTextSplitter`. Benchmark against your test queries. Adjust based on what you find.

**Should I always use overlap?**
Yes for fixed/token-based chunking. Use 10–20% of chunk size. Overlap ensures that information near chunk boundaries isn't lost. Semantic chunking handles this naturally.

---

## What to Learn Next

- **Vector databases** → [Vector Database Guide](/blog/vector-database-guide/)
- **Full RAG pipeline** → [LangChain RAG Tutorial](/blog/langchain-rag-tutorial/)
- **Advanced RAG** → [RAG System Architecture](/blog/rag-system-architecture/)
