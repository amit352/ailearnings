---
title: "How to Build a RAG Application Step-by-Step"
description: "Learn how to build a Retrieval-Augmented Generation app using vector databases and LLMs."
date: "2026-03-13"
slug: "build-rag-app"
keywords: ["build RAG app", "RAG application tutorial", "retrieval augmented generation tutorial", "LangChain RAG", "vector database app"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
---

# How to Build a RAG Application Step-by-Step

Every organization has a knowledge base: product documentation, internal policies, support articles, research papers, code documentation. GPT-4o knows nothing about any of it. You could fine-tune a model on your data — but fine-tuning cannot inject specific passages into answers, cannot cite sources, and cannot be updated without retraining. Retrieval-Augmented Generation solves all of this. It fetches the relevant passages at query time and feeds them to the model as context. This guide builds a complete, production-ready RAG application from scratch.

---

## What You Are Building

A document Q&A assistant that:
- Loads and indexes PDF documents into a vector store
- Accepts natural language questions from users
- Retrieves the most relevant document passages
- Generates grounded answers that cite sources
- Returns "I don't have that information" when the answer is not in the documents

This is the most common first production AI application. The pattern generalizes to customer support bots, internal knowledge bases, documentation assistants, and research tools.

---

## How RAG Works

RAG has two clearly separated phases:

**Indexing phase (offline):**
1. Load documents (PDF, web pages, text files)
2. Split into overlapping chunks (500–1000 characters each)
3. Embed each chunk using an embedding model
4. Store embedding vectors in a vector database

**Query phase (online, per user request):**
1. Embed the user's question
2. Search the vector database for the most similar chunks (top-K by cosine similarity)
3. Insert retrieved chunks into the LLM prompt as context
4. Generate an answer grounded in that context

The retrieval step is what makes RAG different from standard chat: the model only has access to what you retrieve, not its full training-data knowledge about the topic.

---

## Step 1: Install Dependencies

```bash
pip install langchain langchain-openai langchain-community \
            chromadb pypdf sentence-transformers
export OPENAI_API_KEY="sk-..."
```

---

## Step 2: Load and Chunk Documents

Document loading converts different file formats into LangChain's `Document` objects. Chunking breaks those documents into segments small enough for an embedding model to represent meaningfully.

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load a single PDF
loader = PyPDFLoader("docs/employee-handbook.pdf")
documents = loader.load()

# Or load all PDFs from a directory
# loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()

print(f"Loaded {len(documents)} pages")
print(f"First page preview: {documents[0].page_content[:300]}")
print(f"Metadata: {documents[0].metadata}")

# Split into chunks — each chunk gets embedded and stored independently
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,         # overlap prevents information loss at boundaries
    separators=["\n\n", "\n", " ", ""]  # prefer paragraph breaks
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks from {len(documents)} pages")
```

The chunk size is one of the most impactful parameters in RAG. Small chunks (256–512 chars) produce precise retrieval but lose surrounding context. Large chunks (1024–2048 chars) preserve context but dilute the embedding signal. For most document types, 400–600 characters is a good starting point.

---

## Step 3: Create the Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Index all chunks — this calls the embeddings API for each chunk
# Run this once; the result is persisted to disk
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_documents"
)

print(f"Indexed {vectorstore._collection.count()} chunks")
print("Vector store persisted to ./chroma_db")
```

Embedding 500 chunks with `text-embedding-3-small` costs roughly $0.002 and takes about 10 seconds. Once persisted, you never need to re-index unless documents change.

To reload the persisted vector store on subsequent runs:

```python
# In production: load from disk, never re-index on startup
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my_documents"
)
print(f"Loaded {vectorstore._collection.count()} indexed chunks")
```

---

## Step 4: Build the Retrieval QA Chain

The retrieval chain connects your vector store to an LLM. At query time it: embeds the question, retrieves the top-K most relevant chunks, formats them into a prompt, and calls the LLM.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# The grounding prompt is the most critical piece
# Without explicit constraints, the model uses training-data knowledge
PROMPT_TEMPLATE = """Answer the question based ONLY on the context provided below.
If the context does not contain the information needed to answer the question,
say exactly: "I don't have enough information to answer that."

Do not invent or infer information not present in the context.

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
    search_kwargs={"k": 4}   # retrieve top 4 most relevant chunks
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",         # "stuff" = insert all chunks into one prompt
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
```

---

## Step 5: Query the Application

```python
def ask(question: str) -> dict:
    """Ask a question and return the answer with source citations."""
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200] + "..."
            }
            for doc in result["source_documents"]
        ]
    }

# Test it
response = ask("What is the company's vacation policy?")
print(response["answer"])
print("\nSources:")
for src in response["sources"]:
    print(f"  {src['source']}, page {src['page']}")
    print(f"  \"{src['snippet']}\"")

# Test the "I don't know" path
response = ask("What is the square root of 144?")
print(response["answer"])
# Should say: "I don't have enough information to answer that."
```

Testing the "I don't know" path is as important as testing happy-path queries. If your model answers questions that are not in the documents, your grounding prompt needs strengthening.

---

## Step 6: Add a FastAPI Backend

Expose the RAG chain as a REST API for integration with any front-end or service.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Document Q&A API")

# Load vector store once at startup — not on every request
_vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my_documents"
)
_qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    retriever=_vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)


class QuestionRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    source: str
    page: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = _qa_chain.invoke({"query": request.question})

    sources = [
        SourceDoc(
            source=doc.metadata.get("source", "unknown"),
            page=str(doc.metadata.get("page", "?"))
        )
        for doc in result["source_documents"]
    ]

    return QuestionResponse(answer=result["result"], sources=sources)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": _vectorstore._collection.count()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Improving Retrieval Quality

The default similarity search is a good start. These techniques improve precision and recall for production applications.

### MMR (Maximal Marginal Relevance)

Standard top-K retrieval often returns multiple near-duplicate chunks (adjacent paragraphs from the same section). MMR reranks results to maximize diversity while maintaining relevance.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # return 5 diverse results
        "fetch_k": 20,    # from 20 initial candidates
        "lambda_mult": 0.5  # balance between relevance (1.0) and diversity (0.0)
    }
)
```

### Metadata Filtering

Filter by document source when the user's query context implies a specific document.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"source": "employee-handbook.pdf"}  # only search this document
    }
)
```

### Similarity Score Threshold

Reject chunks that are only weakly related to the query. This prevents the model from generating answers from tangentially related content.

```python
# Return only chunks with similarity score above 0.7
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 4}
)
```

---

## Common Mistakes

**Re-indexing documents on every startup** — Embedding 500 documents takes 30 seconds and costs money. Index once, persist to disk, and load the persisted store on startup. This is the most common performance mistake in RAG applications.

**Chunk size too large** — A 2,000-character chunk contains many ideas. The embedding averages over all of them, making the chunk match many different queries weakly rather than one query strongly. Start with 400–500 characters.

**No chunk overlap** — Information at a chunk boundary is split in half. With `chunk_overlap=0`, a sentence that spans the boundary is lost. Use 10–15% overlap relative to chunk size.

**Weak grounding prompt** — The system prompt must explicitly constrain the model to the retrieved context. Without this instruction, GPT-4o will supplement retrieved passages with its training data, producing confident answers that are not in your documents.

**No source attribution** — Always return the source document metadata with the answer. Users need to be able to verify answers, and source citations dramatically increase trust in the application.

**Only testing with questions the documents can answer** — You must test with out-of-scope questions to verify the model correctly says "I don't know." If it does not, strengthen the grounding constraint in the prompt.

---

## What to Learn Next

A working RAG application is the foundation for more complex AI systems. The natural next steps deepen your understanding of each component:

- **Understand why RAG works** → [RAG Explained](/blog/rag-explained/)
- **Go deeper on vector databases** → [Vector Database Explained](/blog/vector-database-explained/)
- **LangChain for complex retrieval pipelines** → [LangChain Tutorial](/blog/langchain-tutorial/)
