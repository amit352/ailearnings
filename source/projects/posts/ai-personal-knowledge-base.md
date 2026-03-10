---
title: "Build an AI Personal Knowledge Base: Your Second Brain with AI"
description: "Build a personal knowledge management system that ingests notes, web clips, and PDFs, auto-links related content, and lets you have conversations with your accumulated knowledge."
date: "2026-03-10"
slug: "ai-personal-knowledge-base"
level: "Advanced"
time: "8–12 hours"
stack: "Python, OpenAI API, ChromaDB, SQLite, Streamlit"
keywords: ["AI knowledge base Python", "personal knowledge management AI", "second brain AI system"]
---

## Project Overview

A personal knowledge management system (second brain) that ingests notes, web articles, PDFs, and Markdown files. Auto-generates tags and links between related notes. Lets you chat with your entire knowledge base and surface forgotten connections.

---

## Learning Goals

- Build a multi-source ingestion pipeline (Markdown, PDF, web URLs)
- Auto-generate tags and link related notes using embeddings
- Implement semantic search + conversation over personal knowledge
- Design a note graph with bidirectional links

---

## Architecture

```
Sources: Markdown / PDF / URL / clipboard
        ↓
Ingest: extract text, generate metadata + tags
        ↓
Embed → ChromaDB vector store
        ↓ also
SQLite: note metadata, tags, bidirectional links
        ↓
Query: semantic search → related notes → LLM synthesis
        ↓
Streamlit UI: search, chat, note graph visualization
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai chromadb pypdf requests beautifulsoup4 streamlit sqlite-utils tiktoken
```

### Step 2: Note Schema + Database

```python
# database.py
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "knowledge.db"


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                source_url TEXT,
                tags TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS links (
                from_id INTEGER,
                to_id INTEGER,
                link_type TEXT DEFAULT 'related',
                similarity REAL,
                PRIMARY KEY (from_id, to_id),
                FOREIGN KEY (from_id) REFERENCES notes(id),
                FOREIGN KEY (to_id) REFERENCES notes(id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
            USING fts5(title, content, content=notes, content_rowid=id);
        """)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_note(title: str, content: str, source: str = "", url: str = "", tags: list = None) -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO notes (title, content, source, source_url, tags) VALUES (?,?,?,?,?)",
            (title, content, source, url or "", json.dumps(tags or []))
        )
        note_id = cursor.lastrowid
        conn.execute("INSERT INTO notes_fts(rowid, title, content) VALUES (?,?,?)",
                     (note_id, title, content))
        return note_id


def get_note(note_id: int) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return dict(row) if row else None


def get_all_notes(limit: int = 500) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, source, tags, created_at FROM notes ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def save_link(from_id: int, to_id: int, similarity: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO links (from_id, to_id, similarity) VALUES (?,?,?)",
            (from_id, to_id, similarity)
        )


def get_related_notes(note_id: int, limit: int = 5) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT n.id, n.title, l.similarity
            FROM links l JOIN notes n ON l.to_id = n.id
            WHERE l.from_id = ? ORDER BY l.similarity DESC LIMIT ?
        """, (note_id, limit)).fetchall()
        return [dict(r) for r in rows]


init_db()
```

### Step 3: Ingestion Pipeline

```python
# ingestion.py
import json
import hashlib
import chromadb
from pathlib import Path
from openai import OpenAI
from database import save_note, save_link, get_all_notes

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_kb")
collection = chroma.get_or_create_collection("knowledge")

METADATA_PROMPT = """Analyze this note/document and return JSON:
{{
  "title": "concise descriptive title",
  "tags": ["tag1", "tag2", "tag3"],
  "summary": "2-3 sentence summary"
}}
Content: {content}"""


def generate_metadata(content: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": METADATA_PROMPT.format(content=content[:3000])}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    import json
    return json.loads(response.choices[0].message.content)


def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in response.data]


def ingest_text(content: str, title: str = None, source: str = "text", url: str = "") -> int:
    metadata = generate_metadata(content)
    final_title = title or metadata["title"]
    tags = metadata.get("tags", [])

    note_id = save_note(final_title, content, source, url, tags)

    # Embed and store in vector DB
    doc_id = f"note_{note_id}"
    embedding = embed([content[:4000]])[0]
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content[:4000]],
        metadatas=[{"note_id": note_id, "title": final_title}],
    )

    # Auto-link to related notes
    find_and_save_links(note_id, content)

    print(f"Ingested: '{final_title}' (id={note_id}, tags={tags})")
    return note_id


def ingest_file(path: str) -> int:
    p = Path(path)
    if p.suffix == ".pdf":
        from pypdf import PdfReader
        content = "\n\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
    else:
        content = p.read_text(encoding="utf-8")
    return ingest_text(content, source=p.name)


def ingest_url(url: str) -> int:
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()
    article = soup.find("article") or soup.find("main") or soup.body
    content = article.get_text(separator="\n") if article else ""
    return ingest_text(content, source="web", url=url)


def find_and_save_links(note_id: int, content: str, threshold: float = 0.75):
    """Find related notes and create bidirectional links."""
    embedding = embed([content[:2000]])[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=6,
    )
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        related_id = meta.get("note_id")
        if related_id and related_id != note_id:
            similarity = 1 - distance
            if similarity >= threshold:
                save_link(note_id, related_id, similarity)
                save_link(related_id, note_id, similarity)
```

### Step 4: Knowledge Chat

```python
# chat.py
from openai import OpenAI
from ingestion import embed, collection
from database import get_note, get_related_notes

client = OpenAI()

CHAT_PROMPT = """You are an AI assistant with access to a personal knowledge base.
Answer questions using the provided notes. Quote specific notes when relevant.
If information is not in the notes, say so.

Relevant notes:
{context}

Question: {question}"""


def search_knowledge(query: str, n: int = 6) -> list[dict]:
    embedding = embed([query])[0]
    results = collection.query(query_embeddings=[embedding], n_results=n)
    return [
        {"note_id": meta["note_id"], "title": meta["title"], "content": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def chat_with_kb(question: str, history: list[dict]) -> dict:
    chunks = search_knowledge(question)
    context = "\n\n---\n\n".join(
        f"**{c['title']}**\n{c['content'][:800]}" for c in chunks
    )
    messages = [{"role": "user", "content": CHAT_PROMPT.format(context=context, question=question)}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=800,
        temperature=0.3,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [{"id": c["note_id"], "title": c["title"]} for c in chunks],
    }
```

### Step 5: Streamlit App

```python
# app.py
import streamlit as st
from ingestion import ingest_text, ingest_url, ingest_file
from chat import chat_with_kb, search_knowledge
from database import get_all_notes, get_note, get_related_notes
import tempfile, os

st.set_page_config(page_title="Knowledge Base", page_icon="🧠", layout="wide")
st.title("🧠 AI Personal Knowledge Base")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📥 Add Notes", "📚 Browse"])

with tab1:
    if "kb_history" not in st.session_state:
        st.session_state.kb_history = []

    for msg in st.session_state.kb_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(s["title"] for s in msg["sources"]))

    if prompt := st.chat_input("Ask your knowledge base..."):
        st.session_state.kb_history.append({"role": "user", "content": prompt})
        with st.spinner("Searching knowledge..."):
            result = chat_with_kb(prompt, st.session_state.kb_history)
        st.session_state.kb_history.append({
            "role": "assistant", "content": result["answer"], "sources": result["sources"]
        })
        st.rerun()

with tab2:
    mode = st.radio("Add from", ["Text/Markdown", "URL", "File"], horizontal=True)
    if mode == "Text/Markdown":
        title = st.text_input("Title (optional)")
        content = st.text_area("Content", height=300)
        if st.button("Add Note") and content:
            with st.spinner("Ingesting..."):
                note_id = ingest_text(content, title or None)
            st.success(f"Added note #{note_id}")
    elif mode == "URL":
        url = st.text_input("URL")
        if st.button("Fetch & Add") and url:
            with st.spinner("Fetching..."):
                note_id = ingest_url(url)
            st.success(f"Added note #{note_id}")
    else:
        uploaded = st.file_uploader("Upload PDF or text file", type=["pdf", "txt", "md"])
        if uploaded and st.button("Ingest File"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            with st.spinner("Ingesting..."):
                note_id = ingest_file(tmp_path)
            os.unlink(tmp_path)
            st.success(f"Added note #{note_id}")

with tab3:
    notes = get_all_notes(100)
    st.write(f"**{len(notes)} notes** in your knowledge base")
    for note in notes:
        with st.expander(f"{note['title']}"):
            import json
            tags = json.loads(note.get("tags", "[]"))
            st.write(f"Source: {note['source']} | Tags: {', '.join(tags)}")
            related = get_related_notes(note["id"])
            if related:
                st.write("Related: " + ", ".join(f"**{r['title']}**" for r in related))
```

### Step 6: Run

```bash
# Ingest existing notes directory
python -c "
from ingestion import ingest_file
from pathlib import Path
for f in Path('my_notes/').glob('*.md'):
    ingest_file(str(f))
"

# Launch app
streamlit run app.py
```

---

## Extension Ideas

1. **Obsidian sync** — watch Obsidian vault for new/changed notes and auto-ingest
2. **Browser extension** — one-click web clipping to your knowledge base
3. **Scheduled digests** — weekly email of "forgotten notes" that are newly relevant
4. **Note graph visualization** — interactive graph of note connections with D3.js
5. **Spaced repetition** — surface notes for review based on forgetting curves

---

## What to Learn Next

- **Multi-agent research** → [Multi-Agent Research System](/projects/multi-agent-research-system/)
- **RAG deep dive** → [RAG Document Assistant](/projects/rag-document-assistant/)
