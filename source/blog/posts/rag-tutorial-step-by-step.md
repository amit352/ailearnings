---
title: "RAG Tutorial 2026: Build a Retrieval-Augmented Generation Pipeline from Scratch"
description: "Step-by-step RAG tutorial for developers in 2026. Build a complete Retrieval-Augmented Generation pipeline — document loading, chunking, vector embeddings, semantic search, and LLM generation with LangChain and ChromaDB."
date: "2026-03-09"
slug: "rag-tutorial-step-by-step"
keywords: ["RAG tutorial", "retrieval-augmented generation", "build RAG pipeline", "LangChain RAG", "vector database tutorial"]
---

# RAG Tutorial 2026: Build a Retrieval-Augmented Generation Pipeline from Scratch

Retrieval-Augmented Generation (RAG) is the most important AI architecture for building real-world applications. This tutorial walks you through a complete implementation — from loading your first document to evaluating output quality.

## What is RAG and Why Does It Matter?

LLMs have a fundamental limitation: their knowledge is frozen at training time. Ask GPT-4 about your company's internal documentation, a PDF you wrote last week, or news from yesterday — it doesn't know.

RAG solves this by retrieving relevant information at query time and injecting it into the LLM's context window. The model then generates an answer grounded in your actual data.

**The RAG pipeline in plain English:**
1. User asks a question
2. Embed the question as a vector
3. Search a vector database for similar document chunks
4. Stuff the top results into the LLM prompt
5. LLM generates a grounded answer

---

## Prerequisites

```bash
pip install langchain langchain-openai langchain-anthropic chromadb pypdf ragas
```

You'll also need an API key from OpenAI or Anthropic. Both have free tiers.

---

## Step 1: Load Your Documents

LangChain provides document loaders for every common format: PDFs, web pages, Notion, Google Docs, YouTube transcripts, and more.

```python
from langchain.document_loaders import PyPDFLoader, WebBaseLoader

# Load a PDF
loader = PyPDFLoader("your-document.pdf")
documents = loader.load()

# Or load a web page
loader = WebBaseLoader("https://example.com/article")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"First doc preview: {documents[0].page_content[:200]}")
```

Each document is a Python object with `.page_content` (the text) and `.metadata` (source, page number, etc.).

---

## Step 2: Split Documents into Chunks

The chunk size is one of the most important decisions in RAG. Too small = lack of context. Too large = irrelevant content dilutes retrieval.

**Rule of thumb:** Start with 512 tokens and 64 token overlap. Adjust based on your evaluation results.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
```

`RecursiveCharacterTextSplitter` tries to split on paragraph breaks first, then newlines, then sentences — preserving logical structure where possible.

---

## Step 3: Generate Embeddings and Store in a Vector Database

Embeddings convert text into numerical vectors. Similar text → similar vectors → retrievable by similarity.

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Use OpenAI's embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store from chunks (this embeds and stores all chunks)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # Save to disk
)

print(f"Stored {vectorstore._collection.count()} vectors")
```

**Free embedding alternatives:**
- `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (runs locally, no API key)
- Cohere embed-english-v3.0 (free tier available)

---

## Step 4: Build the Retriever

The retriever takes a query and returns the most semantically similar chunks.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 chunks
)

# Test it
query = "What are the main conclusions of the document?"
relevant_docs = retriever.invoke(query)

for doc in relevant_docs:
    print(f"Score: {doc.metadata.get('score', 'N/A')}")
    print(f"Content: {doc.page_content[:150]}\n")
```

**Search type options:**
- `"similarity"` — standard cosine similarity (default)
- `"mmr"` — Maximal Marginal Relevance, reduces redundancy in results
- `"similarity_score_threshold"` — only return chunks above a minimum score

---

## Step 5: Build the RAG Chain

Now connect the retriever to an LLM to generate grounded answers.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# The LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# The prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chain using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run a query
answer = rag_chain.invoke("What are the key findings?")
print(answer)
```

---

## Step 6: Add Streaming

For production UIs, streaming the response improves perceived latency.

```python
for chunk in rag_chain.stream("What are the key findings?"):
    print(chunk, end="", flush=True)
print()  # newline after stream
```

---

## Step 7: Evaluate RAG Quality

**The #1 mistake developers make:** deploying RAG without evaluating it. Use RAGAS to measure:

- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are the retrieved chunks actually relevant?

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Build evaluation dataset
eval_data = {
    "question": ["What are the key findings?", "What methodology was used?"],
    "answer": [answer1, answer2],       # from your RAG chain
    "contexts": [contexts1, contexts2], # retrieved chunks per question
    "ground_truth": ["Expected answer 1", "Expected answer 2"],  # optional
}

dataset = Dataset.from_dict(eval_data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
```

**Target scores:** Faithfulness > 0.85, Answer Relevancy > 0.80, Context Precision > 0.75.

---

## Common Problems and Fixes

### Problem: Retrieval returns irrelevant chunks

**Causes:** Chunk size too large, no overlap, embedding model doesn't match your domain.

**Fix:** Try smaller chunks (256 tokens), add metadata filtering, or switch to a domain-specific embedding model.

### Problem: LLM ignores the context

**Cause:** Prompt doesn't strongly enough instruct the model to use the context.

**Fix:** Make the constraint explicit: *"Answer ONLY based on the provided context. Do not use prior knowledge."*

### Problem: High latency

**Cause:** Embedding queries, vector search, and LLM generation are all sequential.

**Fix:** Cache embeddings, use a faster LLM (Groq with Llama 3 is very fast and free-tier), reduce `k` from 5 to 3.

---

## Production Checklist

Before going live with a RAG application:

- [ ] Evaluate on 20+ diverse questions with RAGAS
- [ ] Add error handling for empty retrieval results
- [ ] Implement streaming for better UX
- [ ] Log all queries and retrieved contexts (LangSmith is free)
- [ ] Set up alerts for failed retrievals
- [ ] Test with adversarial prompts (prompt injection)

---

## What's Next

Once you have a working RAG pipeline, explore:

- **Hybrid search** — combine semantic search with BM25 keyword search for better recall
- **Reranking** — use a cross-encoder model to rerank retrieved chunks by relevance
- **Parent document retrieval** — store small chunks but retrieve their larger parent documents
- **Multi-vector retrieval** — generate multiple embeddings per chunk for better coverage

See Phase 4 of the [AI roadmap](/ai-roadmap/) for curated resources on advanced RAG patterns. The [AI projects guide](/ai-projects/) has a full RAG chatbot project idea with more context.
