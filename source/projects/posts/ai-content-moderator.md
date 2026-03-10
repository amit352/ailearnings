---
title: "Build an AI Content Moderator: Classify and Filter Harmful Content"
description: "Build a content moderation system that classifies text and images for toxicity, hate speech, spam, and policy violations. Includes a review queue and confidence thresholds."
date: "2026-03-10"
slug: "ai-content-moderator"
level: "Intermediate"
time: "4–5 hours"
stack: "Python, OpenAI API (Moderation + GPT), FastAPI, SQLite"
keywords: ["AI content moderation", "text classification Python", "LLM moderation pipeline"]
---

## Project Overview

A content moderation pipeline that uses OpenAI's Moderation API for fast flagging and GPT-4o-mini for nuanced classification. Includes a review queue for borderline cases, configurable thresholds, and a moderation dashboard.

---

## Learning Goals

- Use the OpenAI Moderation API effectively
- Combine rule-based + AI moderation for cost efficiency
- Build a review queue for human-in-the-loop moderation
- Design configurable threshold systems

---

## Architecture

```
Content submission
        ↓
OpenAI Moderation API (fast, free) — obvious violations
        ↓ passed
GPT-4o-mini (contextual analysis) — nuanced classification
        ↓
ALLOW / FLAG (auto-reject) / REVIEW (human queue)
        ↓
SQLite: log all decisions + review queue
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai fastapi uvicorn sqlalchemy pydantic
```

### Step 2: Moderation Engine

```python
# moderator.py
import json
from openai import OpenAI

client = OpenAI()

# Thresholds — adjust per use case
MODERATION_THRESHOLDS = {
    "hate": 0.5,
    "harassment": 0.6,
    "self-harm": 0.3,
    "sexual": 0.7,
    "violence": 0.6,
}

CLASSIFY_PROMPT = """You are a content moderator. Classify this content for policy violations.

Return JSON:
{{
  "verdict": "allow" | "flag" | "review",
  "confidence": <0.0-1.0>,
  "categories": {{
    "spam": false,
    "hate_speech": false,
    "harassment": false,
    "misinformation": false,
    "inappropriate": false,
    "off_topic": false
  }},
  "reason": "brief explanation",
  "severity": "none" | "low" | "medium" | "high"
}}

Context: {context}
Content: {content}"""


def run_moderation_api(content: str) -> dict:
    """Fast first pass using OpenAI Moderation API (free)."""
    response = client.moderations.create(input=content)
    result = response.results[0]

    flagged = result.flagged
    violations = []
    for category, score in result.category_scores.__dict__.items():
        threshold = MODERATION_THRESHOLDS.get(category.replace("/", "-").replace("_", "-"), 0.7)
        if score > threshold:
            violations.append({"category": category, "score": round(score, 3)})

    return {
        "flagged": flagged,
        "violations": violations,
        "raw_scores": {k: round(v, 4) for k, v in result.category_scores.__dict__.items()},
    }


def run_contextual_classification(content: str, context: str = "") -> dict:
    """Nuanced classification for borderline content."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(
            content=content[:2000],
            context=context or "General public platform"
        )}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def moderate(content: str, context: str = "", use_gpt: bool = True) -> dict:
    """Full moderation pipeline."""
    # Step 1: OpenAI Moderation API
    mod_result = run_moderation_api(content)

    # Auto-flag obvious violations
    if mod_result["flagged"] and mod_result["violations"]:
        return {
            "verdict": "flag",
            "confidence": 0.95,
            "method": "moderation_api",
            "violations": mod_result["violations"],
            "reason": f"Flagged by Moderation API: {[v['category'] for v in mod_result['violations']]}",
        }

    # Step 2: GPT contextual analysis for borderline content
    if use_gpt:
        gpt_result = run_contextual_classification(content, context)
        gpt_result["method"] = "gpt_contextual"
        return gpt_result

    return {"verdict": "allow", "confidence": 0.8, "method": "moderation_api_only"}
```

### Step 3: Database + Queue

```python
# database.py
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "moderation.db"


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS moderation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                content_hash TEXT,
                verdict TEXT NOT NULL,
                confidence REAL,
                method TEXT,
                reason TEXT,
                reviewer TEXT,
                reviewed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
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


def log_decision(content: str, result: dict) -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO moderation_log (content, verdict, confidence, method, reason) VALUES (?,?,?,?,?)",
            (content[:500], result.get("verdict"), result.get("confidence"),
             result.get("method"), result.get("reason", ""))
        )
        return cursor.lastrowid


def get_review_queue(limit: int = 50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM moderation_log WHERE verdict='review' AND reviewer IS NULL ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def update_decision(item_id: int, verdict: str, reviewer: str = "human"):
    with get_conn() as conn:
        conn.execute(
            "UPDATE moderation_log SET verdict=?, reviewer=?, reviewed_at=? WHERE id=?",
            (verdict, reviewer, datetime.now().isoformat(), item_id)
        )


init_db()
```

### Step 4: FastAPI Service

```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moderator import moderate
from database import log_decision, get_review_queue, update_decision

app = FastAPI(title="Content Moderation API")


class ContentRequest(BaseModel):
    content: str
    context: str = ""
    use_gpt: bool = True


class ReviewRequest(BaseModel):
    verdict: str
    reviewer: str = "human"


@app.post("/moderate")
def moderate_content(req: ContentRequest):
    result = moderate(req.content, req.context, req.use_gpt)
    item_id = log_decision(req.content, result)
    result["id"] = item_id
    return result


@app.get("/queue")
def get_queue(limit: int = 50):
    return {"items": get_review_queue(limit)}


@app.post("/queue/{item_id}/review")
def review_item(item_id: int, req: ReviewRequest):
    if req.verdict not in ("allow", "flag"):
        raise HTTPException(400, "verdict must be 'allow' or 'flag'")
    update_decision(item_id, req.verdict, req.reviewer)
    return {"status": "updated"}
```

### Step 5: Run

```bash
# Start API
uvicorn api:app --reload

# Test
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "Your content here", "context": "Social media platform"}'
```

---

## Extension Ideas

1. **Image moderation** — use GPT-4o vision to moderate uploaded images
2. **Custom policy rules** — let admins define platform-specific rules
3. **Appeals system** — allow users to appeal automated decisions
4. **Analytics dashboard** — track moderation rates, false positive rates
5. **Batch API** — process large content backlogs asynchronously

---

## What to Learn Next

- **AI agents** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
- **Production deployment** → [Deploying AI Applications](/blog/deploying-ai-applications/)
