---
title: "How to Build a RAG Application Step-by-Step"
description: "Learn how to build a Retrieval-Augmented Generation app using vector databases and LLMs."
date: "2026-03-13"
slug: "build-rag-app"
keywords: ["build RAG app", "RAG application tutorial", "retrieval augmented generation tutorial", "LangChain RAG", "vector database app"]
---

# How to Build a RAG Application Step-by-Step

A RAG application lets users ask questions against your own documents — product manuals, internal knowledge bases, research papers, code documentation. The model cannot answer these from its training data alone. RAG gives it the context it needs. This guide walks through building a complete RAG app from scratch.

---

## What is a RAG Application

A RAG application combines a retrieval system (vector database + semantic search) with a language model to answer questions grounded in your specific documents. The retrieval system finds the most relevant passages, and the language model synthesizes an answer from them.

Unlike a standard chatbot, a RAG app can answer questions about private or recent data, cite its sources, and refuse to answer when the relevant information is not in the knowledge base.

For an explanation of the underlying concepts, see [RAG explained](/blog/rag-explained/).

---

## Why Building RAG Apps Matters for Developers

RAG is the most common architecture for production AI assistants. Internal Q&A tools, customer support bots, documentation assistants, and research tools all use this pattern. Understanding how to build one from scratch gives you:

- Full control over document processing and chunking
- Ability to tune retrieval parameters for your content type
- Flexibility to swap embedding models, vector stores, and LLMs
- Insight into failure modes and how to fix them

---

## How a RAG App Works

A RAG application has two distinct phases:

**Indexing (offline):** Load documents → split into chunks → embed each chunk → store in vector database

**Query (online):** Embed the user's question → search vector database → retrieve top-K chunks → build prompt → LLM generates answer

---

## Practical Examples

### Step 1: Install Dependencies

```bash
pip install langchain langchain-openai langchain-community \
            chromadb pypdf sentence-transformers
```

### Step 2: Load and Chunk Documents

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load a single PDF
loader = PyPDFLoader("docs/handbook.pdf")
documents = loader.load()

# Or load all PDFs from a directory
# loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks from {len(documents)} documents")
```

### Step 3: Create the Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_documents"
)
print("Vector store created and persisted.")
```

### Step 4: Build the Retrieval Chain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Custom prompt that grounds the model in context
PROMPT_TEMPLATE = """
Answer the question based ONLY on the following context.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
```

### Step 5: Query the App

```python
def ask(question: str) -> dict:
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "?"),
                "snippet": doc.page_content[:200]
            }
            for doc in result["source_documents"]
        ]
    }

response = ask("What is the company's vacation policy?")
print(response["answer"])
for src in response["sources"]:
    print(f"  Source: {src['source']}, Page {src['page']}")
```

### Step 6: Load an Existing Vector Store

```python
# In production, load the persisted store — don't re-index every time
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my_documents"
)
```

---

## Tools and Frameworks

**Chroma** — Embedded vector database, ideal for local development and small to medium deployments. No server required.

**FAISS** — Facebook's similarity search library. Very fast for in-memory search. Use when you do not need persistence.

**Pinecone** — Managed cloud vector database. Good for production at scale with filtering, namespaces, and automatic scaling.

**Weaviate / Qdrant** — Open-source vector databases with rich filtering and hybrid search (keyword + semantic).

**LangChain** — Provides the glue between document loaders, text splitters, vector stores, and LLMs. See [LangChain tutorial](/blog/langchain-tutorial/) for a full guide.

For a deep dive on vector database options, see [vector databases explained](/blog/vector-database-explained/). For how embeddings power the retrieval step, see [embeddings explained](/blog/embeddings-explained/).

---

## Common Mistakes

**Re-embedding documents on every run** — Index once, persist, and reload. Re-embedding is expensive and unnecessary.

**Wrong chunk size** — Too large: retrieval returns big blocks with irrelevant sentences. Too small: individual chunks lose context. Start with 400–500 tokens and tune from there.

**No source attribution** — Always return the source document metadata with the answer. This makes RAG apps auditable and lets users verify answers.

**Only testing happy-path queries** — Test queries that should return "I don't know." If the model answers anyway, your grounding prompt needs strengthening.

**Skipping chunk overlap** — Overlapping chunks by 10–15% prevents important information from being split across chunks and lost during retrieval.

---

## Best Practices

- **Separate indexing from serving** — Run the indexing pipeline as a background job. The query API should load the persisted vector store, not re-index on startup.
- **Store metadata with chunks** — Document name, page number, section title, date. This enables filtering and citation.
- **Use hybrid search for better recall** — Combine semantic search with keyword (BM25) search. The semantic search finds conceptually related content; keyword search finds exact terms.
- **Monitor retrieval quality** — Log which chunks are retrieved for each query. If the wrong chunks come back, the problem is in retrieval — not generation.
- **Set a similarity threshold** — Reject retrieved chunks below a minimum similarity score. This prevents the model from answering from weakly related content.

---

## Summary

Building a RAG application involves four steps: load and chunk documents, embed and index chunks into a vector store, retrieve relevant chunks at query time, and generate an answer grounded in those chunks.

The architecture is straightforward, but production quality requires attention to chunk size, retrieval parameters, prompt grounding, and source attribution.

Start with Chroma locally, use LangChain's RetrievalQA chain, and tune chunk size and top-K based on your specific documents. Once basic retrieval works, add metadata filtering and hybrid search for production robustness.

For the underlying concepts, see [RAG explained](/blog/rag-explained/). For the vector database infrastructure, see [vector databases explained](/blog/vector-database-explained/). For a complete LangChain walkthrough, see [LangChain tutorial](/blog/langchain-tutorial/).
