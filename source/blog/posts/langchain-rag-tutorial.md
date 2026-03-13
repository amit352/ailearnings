---
title: "LangChain RAG Tutorial: Build a Document Q&A System Step by Step"
description: "Build a complete RAG pipeline with LangChain — document loading, text splitting, vector store indexing, retrieval chains, and conversational Q&A with memory."
date: "2026-03-10"
slug: "langchain-rag-tutorial"
keywords: ["LangChain RAG", "LangChain tutorial", "RAG with LangChain", "document Q&A LangChain"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
---

# LangChain RAG Tutorial: Build a Document Q&A System Step by Step

Most AI tutorials stop at "send a message, get a reply." Real applications need more: the ability to answer questions about *your* documents — internal policies, product manuals, research papers — data the model has never seen. Retrieval-Augmented Generation (RAG) is how you do that, and LangChain provides every building block you need to implement it correctly. This tutorial builds a complete, working document Q&A system from raw PDF files to a conversational interface that remembers context across turns.

---

## What You Will Build

By the end of this guide you will have:

- A document ingestion pipeline that loads, splits, and indexes any PDF or text file
- A vector store (Chroma) containing semantic embeddings of your document chunks
- A retrieval QA chain that retrieves relevant passages and generates grounded answers
- A conversational wrapper that handles follow-up questions with context awareness
- Evaluation tooling to measure and improve retrieval quality

---

## Setup

```bash
pip install langchain langchain-openai langchain-chroma langchain-community
pip install pypdf docx2txt unstructured
export OPENAI_API_KEY="sk-..."
```

---

## Architecture Overview

Understanding the data flow before writing code saves you significant debugging time later.

```
Documents (PDF, web, text)
        ↓
  Document Loaders       ← convert files to LangChain Document objects
        ↓
  Text Splitters         ← break documents into overlapping chunks
        ↓
  Embedding Model        ← convert each chunk to a dense vector
        ↓
  Vector Store (Chroma)  ← store vectors for similarity search
        ↓  ← Query embedding at runtime
  Retriever (top-k)      ← find the most relevant chunks
        ↓
  LLM (GPT-4o-mini)      ← synthesize an answer from retrieved chunks
        ↓
  Answer + Sources
```

There are two distinct phases: **indexing** (run once, offline) and **querying** (run on every user request). Separating these clearly in your codebase prevents the most common performance mistake: re-embedding documents on every startup.

---

## Step 1: Load Documents

LangChain's document loaders normalize different file formats into a consistent `Document` object with `.page_content` (the text) and `.metadata` (source, page number, etc.).

```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader, DirectoryLoader
)

# Load a single PDF — each page becomes one Document
loader = PyPDFLoader("./docs/manual.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages")

# Load a live web page
loader = WebBaseLoader("https://docs.python.org/3/library/functions.html")
docs = loader.load()

# Load all PDFs from a directory recursively
loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Plain text or markdown
loader = TextLoader("./README.md")
docs = loader.load()
```

Inspect your documents before chunking — print `docs[0].page_content[:500]` to verify the text extracted correctly. Scanned PDFs require OCR; use `UnstructuredPDFLoader` for those.

---

## Step 2: Split Into Chunks

Raw pages are too large to embed usefully. You need to split them into smaller, focused chunks that contain one idea each.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")
print(chunks[0].page_content[:200])
```

`RecursiveCharacterTextSplitter` tries each separator in order, preferring to split on paragraph breaks before sentence breaks before words. The `chunk_overlap` ensures that sentences spanning a split boundary appear in both adjacent chunks — preventing information loss at boundaries.

**Chunk size guidelines:**

- Short, precise Q&A (policy docs, FAQs): 256–512 characters
- Technical documentation (API docs, code): 512–1024 characters
- Long-form narrative (books, articles): 1024–2048 characters

Start with 512 and 64 overlap. Measure retrieval quality, then tune.

---

## Step 3: Create a Vector Store

Embeddings convert text chunks into dense vectors. Similar text maps to nearby vectors in high-dimensional space — this is what enables semantic search.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Index once — this calls the OpenAI embeddings API for every chunk
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print(f"Indexed {vectorstore._collection.count()} chunks")
```

### Load an Existing Vector Store

On subsequent runs, load the persisted store. Do not re-index unless documents have changed.

```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

If you want free embeddings without an OpenAI key, use a local model:

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# Runs locally on CPU — no API key required
```

---

## Step 4: Build a Retrieval Chain

The retrieval chain connects the vector store to the LLM. At query time, it embeds the user's question, finds the most similar chunks, and asks the LLM to synthesize an answer from them.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Return the 4 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# The system prompt grounds the model in the retrieved context
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question based only on the context provided.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}"""),
    ("human", "{input}"),
])

# Combine retrieved docs + prompt → LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Query
result = rag_chain.invoke({"input": "What are the main product features?"})
print(result["answer"])

# Always show sources — this is what makes RAG trustworthy
print("\nSources:")
for doc in result["context"]:
    print(f"  - {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})")
```

The key constraint in the prompt is "answer based only on the context provided." Without this, the model will confidently invent answers from its training data when the retrieved chunks are insufficient.

---

## Step 5: Add Conversation Memory

Single-turn Q&A is useful, but users ask follow-up questions. "Tell me more about that" or "What are the limitations?" require the system to understand what "that" refers to. `create_history_aware_retriever` handles this by reformulating follow-up questions as standalone questions before retrieval.

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Reformulate ambiguous follow-up questions using chat history
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and a follow-up question,
rephrase it as a standalone question that can be understood without the history.
Return only the rephrased question."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# QA prompt now includes chat history for context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based only on this context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_rag = create_retrieval_chain(history_aware_retriever, qa_chain)

# Maintain history as a list of message objects
chat_history = []

def chat(question: str) -> str:
    result = conversational_rag.invoke({
        "input": question,
        "chat_history": chat_history,
    })
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result["answer"]),
    ])
    return result["answer"]

print(chat("What is the main topic of this document?"))
print(chat("Tell me more about the limitations."))  # "limitations" understood in context
print(chat("Can you give a specific example?"))      # "example" understood in context
```

---

## Advanced Retrieval Techniques

The default similarity search retrieves the top-k most similar chunks. In practice, the top-k chunks often contain redundant content (multiple chunks from the same paragraph). These techniques improve retrieval diversity and precision.

### MMR (Maximal Marginal Relevance)

Returns diverse results instead of the top-k most similar. Balances relevance with novelty.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)
# fetch_k=20: fetch 20 candidates, then select 6 diverse ones
# lambda_mult=0.5: balance between relevance (1.0) and diversity (0.0)
```

### Metadata Filtering

When you know which document to search, filter by source to avoid noise from other documents.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"source": "manual.pdf"},
    }
)
```

### Self-Query Retriever

Automatically parses filters from natural language questions — users can say "find information about authentication in the API docs" and the retriever extracts the filter automatically.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

metadata_info = [
    AttributeInfo(name="source", description="The source filename", type="string"),
    AttributeInfo(name="page", description="Page number in the document", type="integer"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Technical product documentation",
    metadata_field_info=metadata_info,
)
# Query: "Find information about rate limits on page 3 of api_docs.pdf"
# → automatically applies {"source": "api_docs.pdf", "page": 3}
```

---

## Evaluating Your RAG Pipeline

Measuring retrieval quality is the most important — and most skipped — step in RAG development. If retrieval fails, no amount of prompt engineering will fix the answers.

### Measure Retrieval Hit Rate

```python
# Build a test set of (question, expected_source) pairs
test_cases = [
    ("What is the rate limit?",   "api_docs.pdf"),
    ("How do I authenticate?",    "auth_guide.pdf"),
    ("What are the pricing tiers?", "pricing.pdf"),
]

hits = 0
for question, expected_source in test_cases:
    retrieved_docs = retriever.invoke(question)
    sources = [d.metadata.get("source", "") for d in retrieved_docs]
    if expected_source in sources:
        hits += 1

print(f"Retrieval hit rate: {hits / len(test_cases):.0%}")
```

### Use RAGAs for Automated Evaluation

RAGAs provides standardized metrics: faithfulness (does the answer stay grounded in context?), answer relevancy (does the answer address the question?), and context recall (are the right chunks retrieved?).

```bash
pip install ragas datasets
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

eval_data = {
    "question":   ["What is the vacation policy?"],
    "answer":     ["Employees receive 20 days of paid vacation annually."],
    "contexts":   [["The company provides 20 days PTO per year for full-time employees..."]],
    "ground_truth": ["The vacation policy grants 20 days of paid time off per year."],
}

result = evaluate(Dataset.from_dict(eval_data), metrics=[faithfulness, answer_relevancy])
print(result)
```

---

## Common Pitfalls

**Re-embedding documents on every server start** — This is the single most common performance mistake. Embedding 500 chunks costs money and takes 30+ seconds. Index once, persist to disk, and reload on startup.

**Chunk size too large** — Chunks of 2,000+ characters dilute the embedding signal. The retrieved chunk is "about" many things, so it matches too broadly and the LLM has to wade through irrelevant sentences to find the answer.

**No chunk overlap** — Sentences spanning a chunk boundary are split and the semantic meaning is lost. Always use 10–15% overlap of the chunk size.

**Ignoring the grounding instruction in the prompt** — Without explicit instructions to stay within the provided context, the model will supplement retrieved passages with training-data knowledge. This produces confident hallucinations that appear plausible.

**Using `k=3` for all applications** — The right value depends on your content type and query complexity. Test with k=4, k=6, and k=8, and measure answer quality for each.

---

## What to Learn Next

The patterns in this tutorial apply across a wide range of document-based applications. Once you have a working retrieval pipeline, the natural next steps are:

- **Understand the full RAG architecture in depth** → [RAG Explained](/blog/rag-explained/)
- **Learn the LangChain fundamentals powering this pipeline** → [LangChain Tutorial](/blog/langchain-tutorial/)
- **Add a vector database for production scale** → [Vector Database Explained](/blog/vector-database-explained/)
