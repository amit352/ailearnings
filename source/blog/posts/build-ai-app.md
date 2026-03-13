---
title: "How to Build Your First AI Application"
description: "A practical guide for developers building AI-powered applications."
date: "2026-03-13"
slug: "build-ai-app"
keywords: ["build AI app", "how to build AI application", "AI app tutorial", "first AI application", "LLM application development"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
---

# How to Build Your First AI Application

The hardest part of building your first AI application is not the AI — it is figuring out what to actually build and how the pieces connect. The APIs are mature, the frameworks are stable, and the patterns repeat across nearly every production AI product. What is missing for most developers is a concrete example of the full stack working together: document loading, embedding, retrieval, LLM generation, a REST API, and basic error handling. This guide builds all of it, step by step, explaining the decision at each layer.

---

## What You Will Build

A document Q&A assistant with a FastAPI backend. Users can ask questions about a collection of PDFs and receive grounded answers with source citations. This is the canonical first AI application because it touches every important layer of the stack:

- **Data layer** — Documents ingested, chunked, and embedded into a vector store
- **Retrieval layer** — Semantic search to find relevant passages at query time
- **LLM layer** — Language model that synthesizes answers from retrieved passages
- **Application layer** — FastAPI REST API with request validation, error handling, and logging
- **Memory layer** — Conversation history for multi-turn interactions

---

## The Core Architecture

Every AI application has the same three fundamental layers:

1. **LLM layer** — The model that generates responses (OpenAI GPT-4o-mini, Anthropic Claude, or a local model via Ollama)
2. **Data layer** — The documents, databases, or APIs the model can access at query time
3. **Application layer** — Interface, routing, memory, validation, and business logic

For document-based applications, add a fourth layer: the retrieval system (vector database + embedding model). This is the layer that converts your private data into something the model can reason about.

---

## Step 1: Set Up the Environment

```bash
pip install openai langchain langchain-openai langchain-community \
            chromadb pypdf python-dotenv fastapi uvicorn httpx
```

Create a `.env` file in your project root. Never commit this file.

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

```python
# config.py — central configuration
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_STORE_DIR = "./chroma_db"
DOCS_DIR = "./docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4
```

Centralizing configuration means you change model names, chunk sizes, and paths in one place.

---

## Step 2: Build the Document Indexing Pipeline

The indexing pipeline runs once (or whenever documents change). It loads documents, splits them into chunks, embeds each chunk, and persists the vectors to disk.

```python
# indexer.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import EMBEDDING_MODEL, VECTOR_STORE_DIR, DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def index_documents(docs_dir: str = DOCS_DIR, persist_dir: str = VECTOR_STORE_DIR) -> int:
    """Load, chunk, embed, and persist all PDF documents. Returns number of chunks indexed."""

    # Load all PDFs from the directory
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {docs_dir}")

    if not documents:
        raise ValueError(f"No PDF files found in {docs_dir}")

    # Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Embed and index
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="documents"
    )

    count = vectorstore._collection.count()
    print(f"Indexed {count} chunks to {persist_dir}")
    return count


if __name__ == "__main__":
    index_documents()
```

Run this script once after placing your PDFs in the `./docs/` directory:

```bash
mkdir docs
cp your-documents/*.pdf docs/
python indexer.py
# Loaded 42 pages
# Created 187 chunks
# Indexed 187 chunks to ./chroma_db
```

---

## Step 3: Build the Query Interface

The query module loads the persisted vector store and builds a retrieval chain. It is separate from the indexer — the application server should never re-index at startup.

```python
# qa_engine.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from config import CHAT_MODEL, EMBEDDING_MODEL, VECTOR_STORE_DIR, RETRIEVAL_K

GROUNDING_PROMPT = PromptTemplate(
    template="""Answer the question using ONLY the information in the context below.
If the context does not contain the answer, respond with:
"I don't have that information in the available documents."

Never invent facts or use knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)


def build_qa_chain():
    """Load the persisted vector store and build the retrieval QA chain."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=CHAT_MODEL, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": GROUNDING_PROMPT},
        return_source_documents=True
    )
```

---

## Step 4: Build the FastAPI Backend

```python
# main.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qa_engine import build_qa_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Q&A API", version="1.0.0")

# Build once at startup — not on every request
try:
    qa_chain = build_qa_chain()
    logger.info("QA chain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize QA chain: {e}")
    qa_chain = None


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          description="The question to ask about your documents")


class SourceReference(BaseModel):
    source: str
    page: int | str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    question: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if qa_chain is not None else "degraded",
        "qa_chain": "ready" if qa_chain is not None else "not initialized"
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA service not available")

    logger.info(f"Question received: {request.question[:100]}")

    try:
        result = qa_chain.invoke({"query": request.question})
    except Exception as e:
        logger.error(f"QA chain error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process question")

    sources = [
        SourceReference(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", "?")
        )
        for doc in result.get("source_documents", [])
    ]

    logger.info(f"Answer generated. Sources: {len(sources)}")
    return AnswerResponse(
        answer=result["result"],
        sources=sources,
        question=request.question
    )
```

### Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Test with a Request

```python
import httpx

# Test the API
response = httpx.post(
    "http://localhost:8000/ask",
    json={"question": "What is the company's vacation policy?"}
)
data = response.json()
print(data["answer"])
print("\nSources:")
for src in data["sources"]:
    print(f"  {src['source']} — page {src['page']}")
```

---

## Step 5: Add Conversation Memory

A stateless Q&A endpoint is useful, but users naturally ask follow-up questions. Multi-turn conversations require memory — the ability to understand "What are its limitations?" when the previous question was "What is the main topic?".

```python
# In main.py — add a conversational endpoint
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional

# Maintain one memory object per conversation session
conversation_sessions: dict[str, list] = {}


class ConversationRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: str = Field(default="default", max_length=100)


@app.post("/chat")
def chat(request: ConversationRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA service not available")

    # Get or create history for this session
    history = conversation_sessions.setdefault(request.session_id, [])

    # Build a fresh conversational chain with the session's history
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model=CHAT_MODEL, temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        return_source_documents=True
    )

    result = conv_chain.invoke({
        "question": request.question,
        "chat_history": history
    })

    # Append this turn to history
    history.append(HumanMessage(content=request.question))
    history.append(AIMessage(content=result["answer"]))

    # Keep last 10 turns to control context size
    if len(history) > 20:
        conversation_sessions[request.session_id] = history[-20:]

    return {"answer": result["answer"], "session_id": request.session_id}
```

---

## Common Mistakes

**Re-indexing on every server startup** — The vector store is built once and persisted. Loading it from disk takes under a second. Re-indexing takes 30–60 seconds and costs API money. Never do this on startup.

**No error handling on LLM calls** — OpenAI API calls fail for rate limits, timeouts, and server errors. Every LLM call must be wrapped in a try/except block with meaningful error propagation to the caller.

**No input validation** — Without length limits and format checks, a single malformed request can cause an error that is hard to debug. Pydantic validation in FastAPI handles this automatically — use `Field(..., min_length=1, max_length=2000)`.

**Hardcoding configuration** — Model names, chunk sizes, retrieval K, and API keys should be in config or environment variables. You will tune these parameters as you improve the application; hardcoded values make this painful.

**No logging** — Log every question received, the number of sources retrieved, and any errors. Without logs, you cannot debug production failures or understand usage patterns.

**Not testing the failure modes** — Test that questions outside your document scope return "I don't have that information." Test that the API returns proper 4xx/5xx responses for invalid inputs. Test with a vector store that does not exist yet.

---

## Best Practices

1. **Build end-to-end before optimizing** — Get a working pipeline from document to answer before tuning chunk size, retrieval K, or prompt wording. Optimization without a working baseline is premature.
2. **Separate indexing from serving** — The indexer script and the API server are separate processes. The API never re-indexes; the indexer never starts a server.
3. **Test with real documents and real questions** — Synthetic test data hides the messiness of real PDFs: inconsistent formatting, tables, headers, and footnotes. Get real documents early.
4. **Version your prompts** — The grounding prompt is business logic. Store it as a named constant, not an inline string. Track changes in git. Run regression tests when you modify it.
5. **Add observability from day one** — Log at minimum: question text (or a hash if sensitive), number of sources retrieved, answer length, and any errors. These logs become invaluable for debugging and improving the application.

---

## What to Learn Next

This application gives you a working foundation. The natural next steps are deeper understanding of each layer and more advanced application patterns:

- **Go deeper on RAG architecture and retrieval quality** → [How to Build a RAG Application](/blog/build-rag-app/)
- **Master the OpenAI API powering this application** → [OpenAI API Tutorial](/blog/openai-api-tutorial/)
- **Build more complex agent workflows** → [Build AI Agents](/blog/build-ai-agents/)
