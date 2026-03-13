---
title: "Build a ChatGPT Clone from Scratch: Full-Stack AI Chat App"
description: "Build a full-stack conversational AI app with streaming responses, persistent conversation history, and a React chat UI — using FastAPI and the OpenAI API."
date: "2026-03-10"
slug: "build-chatgpt-clone"
level: "Beginner"
time: "3–4 hours"
stack: "Python, FastAPI, React, OpenAI API"
keywords: ["build ChatGPT clone Python", "LLM chat app tutorial", "streaming FastAPI OpenAI"]
---

## Project Overview

Build a fully functional conversational AI application — the kind of thing that powers ChatGPT, Claude.ai, and similar interfaces. By the end you will have a working chat app with streaming token-by-token responses, persistent conversation history, multiple named conversations, and a configurable system prompt per session.

This is the most important first project for any LLM developer. It teaches you the core request/response cycle, how streaming works, how conversation context is managed, and how to wire a backend to a frontend — all the foundations you need before tackling RAG and agents.

---

## Learning Outcomes

After completing this project you will be able to:

- Explain how the **chat completions message array** works and why the full history must be sent on every request
- Implement **token streaming** end-to-end from the OpenAI API to the browser using Server-Sent Events
- Build a **FastAPI async backend** that handles long-lived streaming responses without blocking
- Design a minimal **conversation persistence layer** and understand its token-limit implications
- Use **system prompts** to control model persona, output format, and response constraints
- Handle **context window overflow** by implementing a rolling message truncation strategy

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React + Vite | Chat UI, SSE stream consumption |
| Backend | FastAPI + uvicorn | API endpoints, streaming response |
| Database | SQLite via SQLModel | Conversation and message persistence |
| LLM API | OpenAI gpt-4o-mini | Low-cost, fast, reliable chat completions |
| Streaming | Server-Sent Events | HTTP-native token streaming to the browser |
| HTTP client | openai Python SDK | Handles retries and async streaming |
| Language | Python 3.11+ | Core backend implementation |

---

## Architecture

```
Browser (React)
    │  EventSource (SSE stream)
    ▼
FastAPI backend  ←→  SQLite
    │               (conversations, messages)
    ▼
openai.chat.completions.create(stream=True)
    │
    ▼
OpenAI API  →  gpt-4o-mini
```

**Key design decisions:**

- **Server-Sent Events** over WebSockets — simpler for one-directional token streams, works over standard HTTP
- **Full message history on every request** — how all LLM chat works; teaches context window awareness
- **SQLite** for persistence — zero infrastructure, trivially swappable for Postgres later
- **Background message saving** — save the completed assistant message after the stream finishes, not during

---

## Implementation

### Step 1: Project setup

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn openai sqlmodel python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env

# Frontend
npm create vite@latest frontend -- --template react
cd frontend && npm install
```

### Step 2: Database models

```python
# models.py
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime

class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(default="New conversation")
    system_prompt: str = Field(default="You are a helpful assistant.")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List["Message"] = Relationship(back_populates="conversation")

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    conversation: Optional[Conversation] = Relationship(back_populates="messages")
```

### Step 3: FastAPI endpoints

```python
# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from openai import AsyncOpenAI
import asyncio, json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])
client = AsyncOpenAI()

@app.post("/conversations")
def create_conversation(title: str = "New conversation", system_prompt: str = "You are a helpful assistant."):
    with Session(engine) as session:
        conv = Conversation(title=title, system_prompt=system_prompt)
        session.add(conv)
        session.commit()
        session.refresh(conv)
        return conv

@app.get("/conversations")
def list_conversations():
    with Session(engine) as session:
        return session.exec(select(Conversation).order_by(Conversation.created_at.desc())).all()

@app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: int):
    with Session(engine) as session:
        return session.exec(select(Message).where(Message.conversation_id == conv_id)).all()
```

### Step 4: Streaming chat endpoint

```python
@app.post("/conversations/{conv_id}/chat")
async def chat(conv_id: int, user_message: str):
    with Session(engine) as session:
        conv = session.get(Conversation, conv_id)
        history = session.exec(select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.created_at)).all()

    # Build message array — this is the core of how LLM chat works
    messages = [{"role": "system", "content": conv.system_prompt}]
    # Truncate if approaching context limit (~3000 tokens budget for history)
    for msg in history[-20:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    async def generate():
        full_response = []
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response.append(delta)
                yield f"data: {json.dumps({'token': delta})}\n\n"

        # Save both messages after stream completes
        with Session(engine) as session:
            session.add(Message(conversation_id=conv_id, role="user", content=user_message))
            session.add(Message(conversation_id=conv_id, role="assistant", content="".join(full_response)))
            session.commit()
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Step 5: React chat UI

```jsx
// App.jsx  (key streaming logic)
const sendMessage = async (text) => {
  setMessages(prev => [...prev, { role: 'user', content: text }]);
  setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

  const source = new EventSource(
    `/api/conversations/${activeConv}/chat?user_message=${encodeURIComponent(text)}`
  );

  source.onmessage = (e) => {
    if (e.data === '[DONE]') { source.close(); return; }
    const { token } = JSON.parse(e.data);
    setMessages(prev => {
      const updated = [...prev];
      updated[updated.length - 1].content += token;
      return updated;
    });
  };
};
```

### Step 6: Run and test

```bash
# Start backend
uvicorn main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev

# Test streaming works correctly with curl
curl -N "http://localhost:8000/conversations/1/chat?user_message=Hello"
```

---

## Extension Ideas

1. **Model selector** — let users switch between gpt-4o-mini, gpt-4o, and o1 per conversation
2. **Conversation title generation** — auto-generate a title from the first message using a separate LLM call
3. **Token counter** — display live token usage and estimated cost per message
4. **Export conversation** — download a conversation as Markdown or JSON
5. **Response regeneration** — add a "try again" button that deletes the last assistant message and re-runs the request

---

## What to Learn Next

- **RAG** → [RAG Document Assistant](/projects/rag-document-assistant/)
- **Agents** → [AI Code Review Assistant](/projects/ai-code-review-assistant/)
- **Prompt engineering** → [Prompt Engineering Guide](/blog/prompt-engineering-techniques/)
