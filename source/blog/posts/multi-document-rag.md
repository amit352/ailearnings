---
title: "Multi-Document Retrieval in RAG Pipelines"
description: "Learn how to build multi-document RAG pipelines that retrieve and synthesize information across multiple sources, with routing, filtering, and fusion strategies."
date: "2026-03-15"
slug: "multi-document-rag"
keywords: ["multi document rag", "rag multiple documents", "multi source retrieval", "document routing rag", "rag pipeline multi document", "langchain multi document"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-15"
---

# Multi-Document Retrieval in RAG Pipelines

The jump from "RAG over one PDF" to "RAG over 200 documents from five different sources" is where most early RAG systems hit a wall. The naive approach — dump everything into a single flat vector store and retrieve globally — works until it doesn't. At some point, a user asks a question that spans two regulatory documents, and the retriever surfaces chunks from the wrong one. Or a question about Product A returns chunks that mention Product A only in a comparison table for Product B.

Multi-document RAG is a family of techniques that bring structure to this problem. Instead of treating your knowledge base as one undifferentiated blob of vectors, you add routing logic, per-document namespaces, metadata filtering, and cross-document synthesis. The result is a system that can answer "Compare the privacy policies of our EU and US products" — a question that requires content from two different documents, understood in relation to each other.

This guide covers the practical patterns for building RAG pipelines that operate across multiple document sources. For the foundational RAG concepts, see the [RAG Architecture Guide](/blog/rag-architecture-guide).

---

## Concept Overview

Multi-document RAG adds a layer of orchestration on top of basic retrieval. The core idea is that different documents serve different intents, and the retrieval strategy should reflect that structure.

There are three primary patterns:
- **Namespace-based retrieval** — segment the vector store by document or document type, retrieve within or across namespaces based on query intent
- **Metadata-filtered retrieval** — add rich metadata to chunks and filter during retrieval (by date, document type, author, version, etc.)
- **Routing retrieval** — classify the query first, then retrieve only from the relevant document set

In practice, these patterns are often combined. A system might route a query to a namespace, then filter by date within that namespace, then synthesize across the retrieved chunks.

---

## How It Works

![Architecture diagram](/assets/diagrams/multi-document-rag-diagram-1.png)

---

## Implementation Example

### Strategy 1: Metadata Filtering

The simplest approach — store all documents in one collection but add rich metadata to every chunk. Filter at query time.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ── Index documents with rich metadata ───────────────────────────────
def index_documents_with_metadata(docs_config: list[dict], persist_dir: str) -> Chroma:
    """
    docs_config: list of {
        "path": str,
        "doc_type": str,       # e.g. "legal", "product", "support"
        "doc_name": str,
        "version": str,
        "effective_date": str
    }
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    for doc_config in docs_config:
        loader = PyPDFLoader(doc_config["path"])
        pages = loader.load()

        # Add document-level metadata to every chunk
        for page in pages:
            page.metadata.update({
                "doc_type": doc_config["doc_type"],
                "doc_name": doc_config["doc_name"],
                "version": doc_config.get("version", "1.0"),
                "effective_date": doc_config.get("effective_date", "")
            })

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"Indexed {len(chunks)} chunks from {doc_config['doc_name']}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="multi_doc"
    )
    print(f"Total: {vs._collection.count()} vectors")
    return vs


# Example usage
docs_config = [
    {"path": "./docs/privacy-policy-eu.pdf", "doc_type": "legal", "doc_name": "EU Privacy Policy", "version": "2.1", "effective_date": "2025-01-01"},
    {"path": "./docs/privacy-policy-us.pdf", "doc_type": "legal", "doc_name": "US Privacy Policy", "version": "1.8", "effective_date": "2024-06-01"},
    {"path": "./docs/product-guide-pro.pdf", "doc_type": "product", "doc_name": "Pro Product Guide", "version": "3.0"},
    {"path": "./docs/support-faq.pdf", "doc_type": "support", "doc_name": "Support FAQ"},
]


# ── Filtered retrieval ────────────────────────────────────────────────
def retrieve_with_filter(
    vectorstore: Chroma,
    query: str,
    doc_type: str = None,
    doc_name: str = None,
    k: int = 5
) -> list:
    """Retrieve chunks with optional metadata filters."""
    filter_dict = {}
    if doc_type:
        filter_dict["doc_type"] = doc_type
    if doc_name:
        filter_dict["doc_name"] = doc_name

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": filter_dict if filter_dict else None
        }
    )
    return retriever.invoke(query)


# Retrieve only from legal docs
# chunks = retrieve_with_filter(vs, "data retention requirements", doc_type="legal")

# Retrieve from a specific document
# chunks = retrieve_with_filter(vs, "cancellation policy", doc_name="EU Privacy Policy")
```

### Strategy 2: Query Routing

Route queries to the right document set before retrieving:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

ROUTER_PROMPT = PromptTemplate(
    template="""Classify the user's question into one of these categories based on what type of document would answer it best:

Categories:
- "legal": questions about policies, compliance, terms of service, privacy, regulations
- "product": questions about features, pricing, how to use the product, technical specs
- "support": questions about troubleshooting, error messages, account issues
- "multi": questions that require information from more than one document type

Respond with ONLY the category name, nothing else.

Question: {question}

Category:""",
    input_variables=["question"]
)

class QueryRouter:
    VALID_CATEGORIES = {"legal", "product", "support", "multi"}

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=10)
        self.chain = ROUTER_PROMPT | self.llm

    def route(self, question: str) -> str:
        response = self.chain.invoke({"question": question})
        category = response.content.strip().lower()
        return category if category in self.VALID_CATEGORIES else "multi"


class RoutedRAGPipeline:
    def __init__(self, vectorstore: Chroma):
        self.vs = vectorstore
        self.router = QueryRouter()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _get_retriever(self, category: str):
        base_kwargs = {"k": 5}
        if category != "multi":
            base_kwargs["filter"] = {"doc_type": category}

        return self.vs.as_retriever(
            search_type="mmr",
            search_kwargs=base_kwargs
        )

    def _format_chunks_with_sources(self, chunks: list) -> str:
        parts = []
        for chunk in chunks:
            name = chunk.metadata.get("doc_name", "Unknown")
            page = chunk.metadata.get("page", "?")
            version = chunk.metadata.get("version", "")
            header = f"[{name} v{version}, p.{page}]" if version else f"[{name}, p.{page}]"
            parts.append(f"{header}\n{chunk.page_content}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, verbose: bool = False) -> dict:
        # Step 1: Route
        category = self.router.route(question)

        # Step 2: Retrieve from appropriate namespace
        retriever = self._get_retriever(category)
        chunks = retriever.invoke(question)

        # Step 3: Format context with source attribution
        context = self._format_chunks_with_sources(chunks)

        # Step 4: Generate
        prompt = f"""Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)

        sources = list({
            chunk.metadata.get("doc_name", "unknown") for chunk in chunks
        })

        if verbose:
            print(f"Routed to: {category}")
            print(f"Retrieved from: {sources}")

        return {
            "answer": response.content,
            "sources": sources,
            "category": category
        }
```

### Strategy 3: Cross-Document Synthesis

For questions that explicitly compare or synthesize across documents:

```python
def synthesize_across_documents(
    vectorstore: Chroma,
    question: str,
    doc_names: list[str],
    k_per_doc: int = 3
) -> dict:
    """
    Retrieve from each specified document independently, then synthesize.
    This prevents any one document from dominating the retrieval.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Retrieve independently from each document
    per_doc_context = {}
    for doc_name in doc_names:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": k_per_doc,
                "filter": {"doc_name": doc_name}
            }
        )
        chunks = retriever.invoke(question)
        per_doc_context[doc_name] = chunks

    # Build structured context that clearly separates document sources
    context_parts = []
    for doc_name, chunks in per_doc_context.items():
        doc_text = "\n\n".join([c.page_content for c in chunks])
        context_parts.append(f"=== {doc_name} ===\n{doc_text}")

    structured_context = "\n\n".join(context_parts)

    synthesis_prompt = f"""You are comparing information across multiple documents.
Answer the question by referencing what each document says specifically.
Use the document names in your response to make it clear which document says what.

{structured_context}

Question: {question}

Answer (reference each document explicitly):"""

    response = llm.invoke(synthesis_prompt)

    return {
        "answer": response.content,
        "sources": doc_names,
        "chunks_per_doc": {k: len(v) for k, v in per_doc_context.items()}
    }


# Example: compare EU and US privacy policies
result = synthesize_across_documents(
    vectorstore=vs,
    question="What are the differences in data retention periods between EU and US policies?",
    doc_names=["EU Privacy Policy", "US Privacy Policy"],
    k_per_doc=4
)
print(result["answer"])
```

### Strategy 4: Parent-Child Retrieval

Retrieve small child chunks for precision, expand to parent chunks for context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parent splitter: large chunks that provide context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Child splitter: small chunks for precise retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Vector store for small chunks; docstore for large chunks
vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)
docstore = InMemoryStore()   # use Redis in production

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Index: stores both small (embedded) and large (returned) chunks
retriever.add_documents(documents, ids=None)

# Retrieval: finds small chunks, returns their parent large chunks
# Better context without sacrificing retrieval precision
parent_chunks = retriever.invoke("authentication error handling")
print(f"Retrieved {len(parent_chunks)} parent chunks")
for chunk in parent_chunks:
    print(f"  {len(chunk.page_content)} chars from {chunk.metadata.get('source')}")
```

---

## Best Practices

**Always include document name and type in chunk metadata.** This enables filtering and provides source attribution in answers. Chunks without metadata cannot be reliably attributed to a source.

**Use independent retrieval for comparison queries.** When a question asks to compare two documents, retrieve from each document separately with equal quota. Global retrieval tends to over-represent the document with more content on the topic.

**Add version and date metadata for regulated domains.** Legal, compliance, and policy documents change over time. Storing effective date and version number lets you filter to the current version or retrieve historical versions explicitly.

**Use parent-child retrieval for long technical documents.** Retrieve small, precise child chunks; return their large parent chunks for the LLM. This improves retrieval precision while maintaining enough context for good generation.

**Build a per-document retrieval test suite.** For each key document in your collection, write 10 test questions with known answers. Run this test suite after adding or updating documents to verify retrieval hasn't degraded.

---

## Common Mistakes

**Treating multi-document as just "more documents."** Adding 50 documents to a single flat vector store without namespace separation produces retrieval interference — chunks from Document A rank above the most relevant chunks from Document B for queries that belong to Document B.

**No deduplication after multi-source retrieval.** When fetching from multiple namespaces, the same chunk may appear in multiple result sets. Deduplicate by chunk content or ID before building the LLM context.

**Ignoring document-level recency.** Without date filtering, older versions of documents compete with current versions in retrieval. A user asking about the current return policy should not receive chunks from the 2022 policy revision.

**Not testing cross-document synthesis.** Questions that span documents are the hardest case. Test these explicitly with hand-curated examples that require information from at least two sources.

---

## Summary

Multi-document RAG requires structure beyond a flat vector store. Use metadata filtering for simple scoping, query routing for intent-based document selection, and independent retrieval with structured synthesis for comparison queries. Parent-child retrieval improves precision for large documents.

The common thread across all strategies: make document provenance explicit in chunk metadata, and test retrieval independently from generation so you can debug the right layer when things go wrong.

---

## Related Articles

- [RAG Architecture Guide](/blog/rag-architecture-guide) — foundational RAG system design
- [Chunking Strategies for RAG Pipelines](/blog/rag-chunking-strategies) — preparing documents for effective retrieval
- [Hybrid Search in RAG Systems](/blog/hybrid-search-rag) — combining dense and sparse retrieval

---

## FAQ

**How many documents can a single RAG system handle?**
The limit is primarily your vector database capacity. ChromaDB handles up to ~1M vectors comfortably on a modern machine. Pinecone and Qdrant scale to hundreds of millions. At 100 chunks per document, a 10,000-document system is well within range of any production vector database.

**Should I use one collection or multiple collections per document type?**
Start with one collection and use metadata filtering. Multiple collections add operational complexity (separate connections, separate indexes) without significant performance benefit at scales below a few million vectors. Switch to multiple collections if you need strict isolation or different index configurations per type.

**How do I handle documents that update frequently?**
Delete and reinsert by document ID. Store a canonical ID (e.g., `doc_name + version`) in chunk metadata and delete all chunks with that ID before re-indexing the updated document. This is more reliable than trying to update individual chunks.

**What is the right chunk quota per document in cross-document synthesis?**
Two to four chunks per document is a good starting point. More than five per document in a comparison scenario and the context becomes too long for the LLM to process effectively. Prioritize quality over quantity: use MMR retrieval within each document's pool.

**Can multi-document RAG handle hierarchical document structures (e.g., regulations with sub-sections)?**
Yes, with parent-child retrieval. Store the hierarchy in metadata (section, subsection) and use the parent retriever to return complete sections while using child chunks for precision matching.
