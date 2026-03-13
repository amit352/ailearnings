---
title: "LangChain RAG Tutorial: Build a Document Q&A System Step by Step"
description: "Build a complete RAG pipeline with LangChain — document loading, text splitting, vector store indexing, retrieval chains, and conversational Q&A with memory."
date: "2026-03-10"
slug: "langchain-rag-tutorial"
keywords: ["LangChain RAG", "LangChain tutorial", "RAG with LangChain", "document Q&A LangChain"]
---

## Learning Objectives

- Load and split documents using LangChain loaders and splitters
- Create a vector store from documents using Chroma
- Build a simple retrieval QA chain
- Add conversation memory for follow-up questions
- Evaluate and improve retrieval quality

---

## Setup

```bash
pip install langchain langchain-openai langchain-chroma langchain-community
pip install pypdf docx2txt unstructured
export OPENAI_API_KEY="sk-..."
```

---

## Architecture Overview

```
Documents (PDF, web, text)
        ↓
  Document Loaders
        ↓
  Text Splitters (chunks)
        ↓
  Embedding Model
        ↓
  Vector Store (Chroma)
        ↓  ← Query
  Retriever (top-k chunks)
        ↓
  LLM (GPT-4o-mini)
        ↓
  Answer
```

---

## Step 1: Load Documents

```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader, DirectoryLoader
)

# PDF
loader = PyPDFLoader("./docs/manual.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages")

# Web page
loader = WebBaseLoader("https://docs.python.org/3/library/functions.html")
docs = loader.load()

# All PDFs in a directory
loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Plain text
loader = TextLoader("./README.md")
docs = loader.load()
```

Each document has `.page_content` (text) and `.metadata` (source, page, etc.).

---

## Step 2: Split into Chunks

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

**Chunk size guidelines:**
- Short, precise Q&A: 256–512 chars
- Technical documentation: 512–1024 chars
- Long-form narrative: 1024–2048 chars

---

## Step 3: Create a Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create and persist vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print(f"Indexed {vectorstore._collection.count()} chunks")
```

### Load an Existing Vector Store

```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

---

## Step 4: Build a Retrieval Chain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retriever: return top 4 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question based only on the context provided.
If the answer isn't in the context, say "I don't have information about that."

Context:
{context}"""),
    ("human", "{input}"),
])

# Chain: retrieve → format prompt → LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Run a query
result = rag_chain.invoke({"input": "What are the main features?"})
print(result["answer"])
print("\nSources:")
for doc in result["context"]:
    print(f"  - {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})")
```

---

## Step 5: Add Conversation Memory

For multi-turn Q&A where follow-up questions reference previous context:

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Reformulate question in context of chat history
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and a follow-up question,
rephrase it as a standalone question. Return only the rephrased question."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# QA prompt with history
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based only on this context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_rag = create_retrieval_chain(history_aware_retriever, qa_chain)

# Chat loop
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
print(chat("Tell me more about it."))   # "it" refers to previous answer
print(chat("What are the limitations?"))
```

---

## Advanced Retrieval Techniques

### MMR (Maximal Marginal Relevance)
Returns diverse results instead of the top-k most similar (which may all be nearly identical).

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)
```

### Metadata Filtering

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"source": "manual.pdf"},  # only search this source
    }
)
```

### Self-Query Retriever
Automatically extracts filters from the question:

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

metadata_info = [
    AttributeInfo(name="source", description="The source filename", type="string"),
    AttributeInfo(name="page", description="Page number", type="integer"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Technical documentation",
    metadata_field_info=metadata_info,
)
# "Find information about authentication on page 3 of auth.pdf"
# → automatically applies {"source": "auth.pdf", "page": 3} filter
```

---

## Evaluating Your RAG Pipeline

### Measure Retrieval Quality

```python
# Build a test set of (question, expected_source) pairs
test_cases = [
    ("What is the rate limit?",    "api_docs.pdf"),
    ("How to authenticate?",       "auth.pdf"),
]

hits = 0
for question, expected_source in test_cases:
    docs = retriever.invoke(question)
    sources = [d.metadata.get("source") for d in docs]
    if expected_source in sources:
        hits += 1

print(f"Retrieval accuracy: {hits / len(test_cases):.0%}")
```

### Use RAGAs for Automated Evaluation

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

eval_data = {
    "question":   ["What is RAG?"],
    "answer":     ["RAG combines retrieval with generation."],
    "contexts":   [["RAG stands for Retrieval-Augmented Generation..."]],
    "ground_truth": ["RAG is a technique that retrieves relevant documents before generating an answer."],
}

result = evaluate(Dataset.from_dict(eval_data), metrics=[faithfulness, answer_relevancy])
print(result)
```

---

## Troubleshooting

**Irrelevant documents retrieved**
- Check chunk size — try smaller chunks for better precision
- Try MMR retrieval for diversity
- Add a metadata filter if you know which source to query
- Inspect embedded queries: `embeddings.embed_query("your question")`

**Answer says "I don't have information"**
- Increase `k` to retrieve more chunks
- Check if the answer exists in your source documents
- Try different query phrasings

**Context is too long (token limit exceeded)**
- Reduce `k` or chunk size
- Use a model with larger context (GPT-4o supports 128K tokens)
- Apply document compression before passing to LLM

---

## FAQ

**How many documents can I index?**
Chroma handles millions of chunks comfortably. For high-scale production use Qdrant or Weaviate.

**Do I need OpenAI for embeddings?**
No. Use free alternatives: `langchain-huggingface` with `BAAI/bge-small-en-v1.5` runs locally.

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

---

## What to Learn Next

- **Full RAG architecture** → [RAG Tutorial Step by Step](/blog/rag-tutorial-step-by-step/)
- **Vector databases in depth** → [Vector Database Guide](/blog/vector-database-guide/)
- **AI agents with LangChain** → langchain-agents-tutorial
