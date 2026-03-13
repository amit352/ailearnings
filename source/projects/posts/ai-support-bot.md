---
title: "Build an AI Customer Support Bot with Knowledge Base"
description: "Build a production-ready customer support chatbot that answers questions from your documentation, escalates complex issues, maintains conversation history, and handles tone appropriately."
date: "2026-03-10"
slug: "ai-support-bot"
level: "Intermediate"
time: "5–6 hours"
stack: "Python, OpenAI API, ChromaDB, FastAPI, Streamlit"
keywords: ["AI customer support bot", "chatbot with knowledge base", "LLM support automation"]
---

## Project Overview

A customer support chatbot that answers questions from a product knowledge base, maintains multi-turn conversation history, detects when to escalate to a human agent, and responds with consistent brand tone.

---

## Learning Goals

- Build a knowledge base RAG system for support
- Manage multi-turn conversation state
- Design escalation logic and intent classification
- Serve the bot via FastAPI for production use

---

## Architecture

```
Knowledge base docs (FAQ, docs, policies)
        ↓ ingest
Vector store (ChromaDB)
        ↓
User message → intent classify → retrieve context
        ↓
System prompt + history + context → GPT-4o-mini
        ↓
Response + escalation check
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai chromadb fastapi uvicorn streamlit pydantic
```

### Step 2: Knowledge Base Ingestion

```python
# knowledge_base.py
import hashlib
import chromadb
from openai import OpenAI
from pathlib import Path

client = OpenAI()
chroma = chromadb.PersistentClient(path="./support_kb")
collection = chroma.get_or_create_collection("support_docs")


def ingest_text(content: str, source: str, chunk_size: int = 400):
    """Ingest text content into the knowledge base."""
    sentences = content.replace("\n\n", "\n").split("\n")
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))

    if not chunks:
        return 0

    embeddings_resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )
    embeddings = [e.embedding for e in embeddings_resp.data]

    ids = [hashlib.md5(f"{source}:{i}".encode()).hexdigest() for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks,
                   metadatas=[{"source": source}] * len(chunks))
    return len(chunks)


def ingest_file(path: str) -> int:
    content = Path(path).read_text(encoding="utf-8")
    return ingest_text(content, Path(path).name)


def search(query: str, n_results: int = 4) -> list[dict]:
    embedding = client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
```

### Step 3: Support Bot Core

```python
# bot.py
from openai import OpenAI
from knowledge_base import search

client = OpenAI()

SYSTEM_PROMPT = """You are a helpful customer support assistant for {company_name}.

Your responsibilities:
- Answer questions based ONLY on the provided knowledge base context
- Be friendly, professional, and empathetic
- If you don't know the answer, say so honestly and offer to escalate
- Never make up information about products, pricing, or policies
- For billing, account-specific, or complex issues, recommend human escalation

Knowledge Base Context:
{context}

If the context doesn't contain enough information to answer, respond with:
"I don't have specific information about that. Let me connect you with a human agent who can help."
"""

ESCALATION_KEYWORDS = [
    "refund", "cancel", "legal", "complaint", "fraud", "billing error",
    "account hacked", "urgent", "lawsuit", "angry", "unacceptable"
]


def should_escalate(message: str, response: str) -> bool:
    msg_lower = message.lower()
    resp_lower = response.lower()
    keyword_match = any(kw in msg_lower for kw in ESCALATION_KEYWORDS)
    uncertainty = "human agent" in resp_lower or "don't have information" in resp_lower
    return keyword_match or uncertainty


def chat(
    user_message: str,
    history: list[dict],
    company_name: str = "Our Company",
) -> dict:
    # Retrieve relevant context
    context_chunks = search(user_message)
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in context_chunks)

    system = SYSTEM_PROMPT.format(company_name=company_name, context=context or "No relevant information found.")

    messages = [{"role": "system", "content": system}]
    messages.extend(history[-8:])  # Keep last 8 turns
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.3,
    )
    reply = response.choices[0].message.content
    escalate = should_escalate(user_message, reply)

    return {
        "reply": reply,
        "escalate": escalate,
        "sources": list({c["source"] for c in context_chunks}),
    }
```

### Step 4: FastAPI Backend

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from bot import chat

app = FastAPI(title="Support Bot API")
sessions: dict[str, list] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    company_name: str = "Our Company"


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    history = sessions.get(req.session_id, [])
    result = chat(req.message, history, req.company_name)
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": result["reply"]})
    sessions[req.session_id] = history[-20:]  # Keep last 20 messages
    return result


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}
```

### Step 5: Streamlit Demo

```python
# app.py
import uuid
import streamlit as st
import requests

st.set_page_config(page_title="Support Bot", page_icon="💬")
st.title("💬 Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.write(msg["content"])
        if msg.get("escalate"):
            st.warning("🔔 This issue has been flagged for human review.")

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    resp = requests.post("http://localhost:8000/chat", json={
        "session_id": st.session_state.session_id,
        "message": prompt,
    }).json()

    msg = {"role": "assistant", "content": resp["reply"], "escalate": resp.get("escalate")}
    st.session_state.messages.append(msg)
    with st.chat_message("assistant"):
        st.write(resp["reply"])
        if resp.get("escalate"):
            st.warning("🔔 Escalating to human agent...")
```

### Step 6: Run

```bash
# Start API
uvicorn api:app --reload

# In another terminal: ingest your docs
python -c "from knowledge_base import ingest_file; ingest_file('faq.txt')"

# Launch UI
streamlit run app.py
```

---

## Extension Ideas

1. **Live handoff** — integrate with Zendesk or Intercom for real escalation
2. **Ticket creation** — auto-create support tickets for escalated issues
3. **Analytics** — track unanswered questions to identify knowledge gaps
4. **Multi-language** — detect and respond in the user's language
5. **Voice support** — add Whisper for voice input + TTS for audio responses

---

## What to Learn Next

- **RAG architecture** → [RAG System Architecture](/blog/rag-system-architecture/)
- **Deploying AI apps** → [Deploying AI Applications](/blog/deploying-ai-applications/)
