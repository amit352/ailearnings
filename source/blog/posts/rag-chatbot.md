---
title: "Build a RAG Chatbot with Python and Vector Databases"
description: "Build a RAG chatbot with Python, LangChain, and ChromaDB. Add conversation memory, source citations, and streaming — with production-ready code examples."
date: "2026-03-15"
slug: "rag-chatbot"
keywords: ["rag chatbot tutorial", "build rag chatbot python", "langchain chatbot rag", "conversational rag", "chatbot vector database", "chromadb chatbot"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-15"
---

# Build a RAG Chatbot with Python and Vector Databases

Most RAG tutorials build a Q&A system — you ask a question, you get an answer. That's useful, but real users interact with chatbots conversationally. They ask a follow-up. They refer to something they mentioned two messages ago. They rephrase when the first answer is unclear. A stateless RAG pipeline falls apart the moment a user types "can you explain that last part?"

The challenge is that language models are stateless by design. Each call to the OpenAI API is independent. To build a conversational RAG system, you need to manage two kinds of memory: conversation history (what was said) and the document index (what is known). These interact in non-obvious ways.

This tutorial builds a complete RAG chatbot with conversation history, multi-turn awareness, streaming responses, and source citations. The code is production-ready and runs locally. For the foundational architecture concepts, see the [RAG Tutorial](/blog/rag-tutorial).

---

## Concept Overview

A **RAG chatbot** differs from a simple RAG Q&A system in one key way: the retrieval step must account for conversation history, not just the most recent user message.

If a user asks "What's the return policy?" and then asks "Does it apply to digital products too?", the second question makes no sense without the context of the first. A naive RAG pipeline would embed "Does it apply to digital products too?" and retrieve chunks about digital products generally — missing that the user is asking specifically about the return policy.

The solution is **contextual query rewriting**: before retrieving, rewrite the user's latest message into a standalone question that incorporates relevant conversation history. This single addition transforms a stateless Q&A system into a genuinely conversational experience.

---

## How It Works

![Architecture diagram](/assets/diagrams/rag-chatbot-diagram-1.png)

The query rewriter is a lightweight LLM call — it takes the conversation history and the latest message and outputs a reformulated standalone question. This is then embedded for retrieval. The original conversation history is also passed to the final LLM call for coherence.

---

## Implementation Example

### Setup

```bash
pip install langchain langchain-openai langchain-community chromadb pypdf
export OPENAI_API_KEY="sk-..."
```

### Document Indexer

```python
# indexer.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def build_index(docs_dir: str = "./docs", persist_dir: str = "./chroma_db") -> Chroma:
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="chatbot_docs"
    )
    print(f"Indexed {vs._collection.count()} chunks from {len(documents)} pages")
    return vs

def load_index(persist_dir: str = "./chroma_db") -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="chatbot_docs"
    )
```

### Conversation Memory

```python
# memory.py
from dataclasses import dataclass, field

@dataclass
class Message:
    role: str          # "user" or "assistant"
    content: str
    sources: list = field(default_factory=list)

class ConversationMemory:
    """Sliding-window conversation memory."""

    def __init__(self, max_turns: int = 10):
        self.messages: list[Message] = []
        self.max_turns = max_turns

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str, sources: list = None):
        self.messages.append(Message(role="assistant", content=content, sources=sources or []))

    def get_history_text(self, last_n: int = 6) -> str:
        """Format the last N messages as plain text for the rewriter prompt."""
        recent = self.messages[-last_n:] if len(self.messages) > last_n else self.messages
        lines = []
        for msg in recent:
            prefix = "Human" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def get_openai_messages(self, system_prompt: str, last_n: int = 10) -> list[dict]:
        """Format history as OpenAI chat message objects."""
        messages = [{"role": "system", "content": system_prompt}]
        recent = self.messages[-last_n:]
        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    def clear(self):
        self.messages = []
```

### Query Rewriter

```python
# rewriter.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

REWRITE_PROMPT = PromptTemplate(
    template="""Given the conversation history and the user's latest message,
rewrite the latest message as a complete standalone question.

Rules:
- If already standalone, return it unchanged
- Resolve pronouns ("it", "that", "them") to their referents from history
- Preserve the user's exact intent
- Output ONLY the rewritten question, nothing else

Conversation history:
{history}

Latest message: {question}

Standalone question:""",
    input_variables=["history", "question"]
)

class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=128)
        self.chain = REWRITE_PROMPT | self.llm

    def rewrite(self, question: str, history: str) -> str:
        if not history.strip():
            return question

        response = self.chain.invoke({"history": history, "question": question})
        rewritten = response.content.strip()
        return rewritten if rewritten else question
```

### Core Chatbot

```python
# chatbot.py
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

SYSTEM_PROMPT = """You are a helpful assistant that answers questions from company documents.

RULES:
- Answer ONLY using the retrieved context provided in each message
- If the context doesn't have the answer, say: "I don't have that information in my knowledge base."
- Be concise and factual
- Do not use your training knowledge beyond the provided context"""

class RAGChatbot:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.memory = ConversationMemory(max_turns=20)
        self.rewriter = QueryRewriter()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

    def _format_context(self, chunks: list) -> tuple[str, list]:
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            context_parts.append(f"[Source {i+1}: {source}, p.{page}]\n{chunk.page_content}")
            sources.append({"file": source, "page": page})
        return "\n\n---\n\n".join(context_parts), sources

    def chat(self, user_message: str) -> dict:
        # Step 1: Rewrite question to be standalone
        history_text = self.memory.get_history_text(last_n=6)
        standalone_question = self.rewriter.rewrite(user_message, history_text)

        # Step 2: Retrieve
        chunks = self.retriever.invoke(standalone_question)
        context, sources = self._format_context(chunks)

        # Step 3: Build messages with history + context
        messages = self.memory.get_openai_messages(SYSTEM_PROMPT, last_n=8)
        augmented_message = f"Context from knowledge base:\n{context}\n\nUser question: {user_message}"
        messages.append({"role": "user", "content": augmented_message})

        # Step 4: Generate
        response = self.llm.invoke(messages)
        answer = response.content

        # Step 5: Update memory (store original message, not the augmented one)
        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(answer, sources)

        return {"answer": answer, "sources": sources, "rewritten_query": standalone_question}

    def stream_chat(self, user_message: str):
        """Stream tokens. Yields str tokens, then a final metadata dict."""
        history_text = self.memory.get_history_text(last_n=6)
        standalone_question = self.rewriter.rewrite(user_message, history_text)
        chunks = self.retriever.invoke(standalone_question)
        context, sources = self._format_context(chunks)

        messages = self.memory.get_openai_messages(SYSTEM_PROMPT, last_n=8)
        augmented = f"Context:\n{context}\n\nQuestion: {user_message}"
        messages.append({"role": "user", "content": augmented})

        full_response = []
        for chunk in self.llm.stream(messages):
            token = chunk.content
            full_response.append(token)
            yield token

        answer = "".join(full_response)
        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(answer, sources)
        yield {"__metadata__": True, "sources": sources, "rewritten_query": standalone_question}

    def reset(self):
        self.memory.clear()
```

### CLI Interface

```python
# main.py
from pathlib import Path
from indexer import load_index, build_index
from chatbot import RAGChatbot

def run_chatbot():
    if Path("./chroma_db").exists():
        vs = load_index("./chroma_db")
    else:
        vs = build_index("./docs", "./chroma_db")

    bot = RAGChatbot(vs)
    print("RAG Chatbot ready. Type 'quit' to exit, 'reset' to clear history.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("Conversation cleared.\n")
            continue

        result = bot.chat(user_input)
        print(f"\nAssistant: {result['answer']}")

        if result["sources"]:
            unique = {f"{s['file']} (p.{s['page']})" for s in result["sources"]}
            print(f"Sources: {', '.join(unique)}")

        if result["rewritten_query"] != user_input:
            print(f"[Interpreted as: {result['rewritten_query']}]")
        print()

if __name__ == "__main__":
    run_chatbot()
```

### Example Multi-Turn Conversation

```
You: What's the return policy?
Assistant: Products purchased within 30 days can be returned for a full refund. Items must be in original condition.
Sources: handbook.pdf (p.14)

You: Does that apply to software licenses too?
[Interpreted as: Does the return policy apply to software licenses?]
Assistant: Software licenses are non-refundable once the license key has been activated, per section 4.2 of the return policy.
Sources: handbook.pdf (p.15)

You: What about trial versions?
[Interpreted as: Does the return policy apply to trial versions of software licenses?]
Assistant: Trial versions can be upgraded to paid licenses but are not subject to the return policy since no payment is collected during the trial period.
Sources: handbook.pdf (p.15)
```

Notice how each follow-up question gets rewritten to be standalone before retrieval. Without this step, "Does that apply to software licenses too?" would retrieve chunks about software licenses in general, not the return policy context.

---

## Best Practices

**Keep conversation history bounded.** Sending the full conversation history to the LLM on every turn gets expensive fast. A sliding window of the last 6–10 turns is sufficient for most conversational contexts.

**Store original messages, not augmented ones.** The memory should record what the user actually said, not the context-injected version. Otherwise the conversation history grows unmanageably large.

**Rewrite before retrieving, not after.** The query rewriter must run before the vector search. Retrieving against the original follow-up question produces poor results because the question is underspecified.

**Add session IDs for multi-user deployments.** Each user session needs its own `ConversationMemory` instance. In a web application, store memory keyed by session ID in Redis or a similar store with TTL.

**Test degradation under long conversations.** Conversations beyond 20 turns often see retrieval quality drop because the rewriter starts conflating topics from early in the conversation. Set a maximum history window and test it explicitly.

---

## Common Mistakes

**Storing augmented user messages in history.** If you store the full `"Context:\n...\n\nQuestion: ..."` string in memory instead of the original user message, the history quickly becomes thousands of tokens and the rewriter produces nonsense.

**Not testing query rewriting independently.** The rewriter is a separate failure point. Log the original query and the rewritten query for every turn. You'll quickly spot cases where the rewriter changes the meaning of a question.

**Using the same LLM for rewriting and generation.** GPT-4o-mini is a good rewriter but overkill for generation in many cases. Conversely, using a weak model for rewriting while using a strong model for generation wastes the retrieval quality you're working hard to achieve. Match model quality to task complexity.

**No session cleanup.** In-memory `ConversationMemory` objects accumulate over time in long-running servers. Implement TTL-based cleanup or use a Redis backend with expiry.

---

## Summary

A RAG chatbot requires three additions on top of a basic RAG Q&A system: conversation memory, contextual query rewriting, and structured message passing that includes both history and retrieved context. The query rewriter is the most critical piece — it transforms underspecified follow-up questions into retrievable standalone queries.

The code in this tutorial is production-ready. Add a FastAPI wrapper around `RAGChatbot.chat()` and a Redis-backed memory store, and you have the core of a deployable conversational assistant.

---

## Related Articles

- [RAG Tutorial](/blog/rag-tutorial) — foundational RAG pipeline without the chatbot layer
- [RAG Architecture Guide](/blog/rag-architecture-guide) — architectural decisions and component tradeoffs
- [Production RAG System Design](/blog/production-rag) — scaling RAG systems to handle real traffic

---

## FAQ

**How do I handle multiple users simultaneously?**
Each user session needs its own `ConversationMemory` instance. In a FastAPI application, store memories in a dictionary keyed by session token, or use Redis with session-scoped TTL for persistence across server restarts.

**What happens if the query rewriter changes the meaning of a question?**
Log both the original and rewritten queries in production. When you spot rewriting errors, add examples to the rewrite prompt showing the correct behavior. Few-shot prompting in the rewriter is highly effective.

**Can I use this with a local LLM?**
Replace `ChatOpenAI` with `ChatOllama` pointing at a local Ollama instance. For the rewriter, a 7B model works fine. For final generation, use at least a 13B model for acceptable quality. The rest of the code is unchanged.

**How many turns of history should I include?**
Six to eight turns covers the vast majority of conversational reference patterns. Beyond that, including more history increases token costs without improving quality — most users don't reference things said 15 turns ago.

**Should I use LangChain's built-in ConversationalRetrievalChain?**
LangChain's `ConversationalRetrievalChain` handles the same pattern and is a reasonable starting point. The custom implementation above gives you more control over memory management, rewriting strategy, and streaming behavior — which matters in production.