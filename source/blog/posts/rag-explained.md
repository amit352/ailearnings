---
title: "RAG Explained – How Retrieval-Augmented Generation Works"
description: "Learn how Retrieval-Augmented Generation works and how developers build RAG systems."
date: "2026-03-13"
slug: "rag-explained"
keywords: ["RAG explained", "retrieval augmented generation", "RAG system", "how RAG works", "LLM with retrieval"]
---

# RAG Explained – How Retrieval-Augmented Generation Works

Language models know what they were trained on. They do not know what happened after training, they do not know your company's internal documents, and they hallucinate when asked to recall specific facts. Retrieval-Augmented Generation (RAG) solves this by giving the model access to external knowledge at query time. It is one of the most important patterns in production AI development.

---

## What is Retrieval-Augmented Generation

RAG is a technique that enhances LLM responses by retrieving relevant documents from an external knowledge base and injecting them into the prompt as context before asking the model to answer.

Instead of relying on the model's training data alone, a RAG system:
1. Takes the user's question
2. Retrieves the most relevant documents from a vector database
3. Injects those documents into the prompt
4. Asks the model to answer based on the provided context

The model's job becomes synthesis and reasoning over the retrieved text — not memorization. This grounds answers in real, up-to-date, verifiable source material.

---

## Why RAG Matters for Developers

RAG solves three core problems that developers face when building AI applications:

**Knowledge cutoff** — LLMs are trained on data up to a certain date. RAG lets you query real-time or recent data without retraining the model.

**Private data** — Models do not know your internal documentation, product data, or knowledge base. RAG injects this context at query time without exposing training data.

**Hallucination reduction** — When instructed to answer only from provided context, models hallucinate far less. The retrieved documents anchor the answer.

**Citation and auditability** — RAG systems can return the source documents alongside the answer, making responses verifiable and auditable.

For building a full RAG application, see [how to build a RAG app](/blog/build-rag-app/). For the underlying retrieval mechanism, see [vector databases explained](/blog/vector-database-explained/).

---

## How RAG Works

RAG has two main phases: **indexing** (done once, offline) and **retrieval + generation** (done at query time).

### Phase 1: Indexing

```
Documents → Chunking → Embedding → Vector Store
```

1. **Load documents** — PDFs, web pages, markdown files, database records
2. **Chunk** — Split into segments of 200–500 tokens with overlap
3. **Embed** — Convert each chunk to a dense vector using an embedding model
4. **Store** — Save vectors and text in a vector database

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed and store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)
```

### Phase 2: Retrieval and Generation

```
User Query → Embed Query → Vector Search → Top-K Chunks → Prompt → LLM → Answer
```

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is our refund policy?"})
print(result["result"])
print([doc.metadata for doc in result["source_documents"]])
```

### The RAG Prompt

The prompt template is the core of the RAG pattern:

```
Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer,
say "I don't have enough information to answer this."

Context:
{retrieved_chunks}

Question: {user_question}

Answer:
```

The instruction "based ONLY on the provided context" is critical. Without it, the model may mix retrieved information with its prior knowledge, reducing grounding.

---

## Practical Examples

### Document Q&A System

```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load and index
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
).split_documents(docs)

vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

# Query
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
)

answer = qa.invoke({"query": "How many vacation days do employees get?"})
print(answer["result"])
```

---

## Tools and Frameworks

**LangChain** — The most common framework for building RAG pipelines. Provides document loaders, text splitters, vector store integrations, and retrieval chains.

**LlamaIndex** — Specializes in data-connected LLM applications. Excellent for complex document hierarchies and multi-document reasoning.

**Chroma** — Open-source, embedded vector database. Easy to set up locally, no server required.

**FAISS** — Facebook's library for efficient similarity search. Fast and battle-tested but does not persist to disk natively.

**Pinecone / Weaviate / Qdrant** — Managed cloud vector databases with production features: scaling, filtering, and metadata search.

For a deep dive on vector databases, see [vector databases explained](/blog/vector-database-explained/). For embeddings, see [embeddings explained](/blog/embeddings-explained/).

---

## Common Mistakes

**Chunks too large or too small** — Chunks that are too large dilute the relevant signal. Too small and they lose context. The sweet spot is usually 200–500 tokens with 10–15% overlap.

**No metadata filtering** — When your knowledge base spans multiple domains or document types, filter by metadata at retrieval time. Otherwise you retrieve irrelevant chunks from the wrong context.

**Retrieving too few or too many chunks** — Three to five chunks is typical. Too few misses relevant information. Too many fills the context window with noise and dilutes attention.

**Not grounding the model in the context** — Without explicit instructions to use only the provided context, models blend retrieved facts with prior knowledge and hallucinate. Always include grounding instructions.

**Ignoring retrieval quality** — If retrieval returns irrelevant chunks, the generation step has nothing to work with. Measure retrieval precision and recall independently of generation quality.

---

## Best Practices

- **Evaluate retrieval and generation separately** — A good RAG system requires both good retrieval (right chunks returned) and good generation (right answer from those chunks). Measure each independently.
- **Add metadata to your documents** — Source, date, document type, section. Good metadata enables filtered retrieval and source citation.
- **Test with adversarial queries** — Queries that should return "I don't know" are as important to test as queries that should return an answer.
- **Use a re-ranker for high-stakes retrieval** — A cross-encoder re-ranker (like Cohere Rerank) improves precision by scoring retrieved chunks against the query before passing them to the model.
- **Monitor chunk quality** — Chunks split in the middle of a sentence or table lose meaning. Review a sample of chunks manually after indexing.

---

## Summary

RAG is the primary pattern for connecting LLMs to external knowledge. It solves the knowledge cutoff problem, enables private data integration, and reduces hallucinations by grounding answers in retrieved documents.

The architecture is straightforward: index your documents as embedded chunks, retrieve the most relevant chunks at query time, inject them as context, and ask the model to answer based only on that context.

To build a complete RAG application end-to-end, see [how to build a RAG app](/blog/build-rag-app/). For the vector database infrastructure that powers retrieval, see [vector databases explained](/blog/vector-database-explained/). For the embedding models that make semantic search possible, see [embeddings explained](/blog/embeddings-explained/).
