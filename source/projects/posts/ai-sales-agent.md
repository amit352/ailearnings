---
title: "Build an AI Sales Agent: Automated Lead Qualification and Outreach"
description: "Build an AI sales agent that qualifies leads from a CRM, personalizes outreach emails based on company research, tracks responses, and generates pipeline reports."
date: "2026-03-10"
slug: "ai-sales-agent"
level: "Advanced"
time: "8–10 hours"
stack: "Python, OpenAI API, SQLite, FastAPI, requests"
keywords: ["AI sales agent Python", "lead qualification AI", "automated sales outreach LLM"]
---

## Project Overview

An AI sales agent that pulls leads from a database, researches companies, scores lead quality, writes personalized outreach emails, tracks engagement, and generates pipeline reports — automating the research and writing portions of the sales workflow.

---

## Learning Goals

- Build an agentic loop with external tool calls
- Design multi-step AI workflows with state management
- Use function calling for structured data operations
- Build a complete business workflow with AI

---

## Architecture

```
Lead database (SQLite)
        ↓
Lead qualification agent
  → research company + ICP scoring
        ↓
Outreach agent
  → personalized email generation
        ↓
Follow-up agent
  → response classification + next action
        ↓
Pipeline report generator
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai requests fastapi uvicorn sqlite-utils pydantic
```

### Step 2: Lead Database

```python
# database.py
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "sales.db"


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                company TEXT NOT NULL,
                title TEXT,
                website TEXT,
                industry TEXT,
                company_size TEXT,
                score INTEGER DEFAULT 0,
                qualification_notes TEXT,
                status TEXT DEFAULT 'new',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS outreach (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER,
                email_subject TEXT,
                email_body TEXT,
                sent_at TEXT,
                opened_at TEXT,
                replied_at TEXT,
                reply_text TEXT,
                reply_sentiment TEXT,
                next_action TEXT,
                FOREIGN KEY (lead_id) REFERENCES leads(id)
            );
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


def add_lead(name: str, company: str, email: str = "", title: str = "",
             website: str = "", industry: str = "", company_size: str = "") -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO leads (name, company, email, title, website, industry, company_size) VALUES (?,?,?,?,?,?,?)",
            (name, company, email, title, website, industry, company_size)
        )
        return cursor.lastrowid


def get_leads(status: str = "new", limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM leads WHERE status=? ORDER BY score DESC LIMIT ?",
            (status, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def update_lead(lead_id: int, **kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [lead_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE leads SET {sets} WHERE id=?", values)


def save_outreach(lead_id: int, subject: str, body: str) -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO outreach (lead_id, email_subject, email_body, sent_at) VALUES (?,?,?,?)",
            (lead_id, subject, body, datetime.now().isoformat())
        )
        return cursor.lastrowid


init_db()
```

### Step 3: Company Research Tool

```python
# research.py
import requests
from openai import OpenAI

client = OpenAI()

RESEARCH_PROMPT = """Research this company and return a sales intelligence brief.

Company: {company}
Website: {website}
Industry: {industry}

Return JSON:
{{
  "company_summary": "2-3 sentences about what the company does",
  "key_pain_points": ["pain point 1", "pain point 2"],
  "relevant_use_cases": ["how our product could help them"],
  "talking_points": ["specific personalization hook 1", "hook 2"],
  "icp_score": <0-100>,
  "icp_rationale": "why this score"
}}

Assume we're selling an AI development platform for engineering teams."""


def research_company(company: str, website: str = "", industry: str = "") -> dict:
    import json
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": RESEARCH_PROMPT.format(
            company=company, website=website or "unknown", industry=industry or "unknown"
        )}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)
```

### Step 4: Qualification Agent

```python
# qualification_agent.py
from research import research_company
from database import get_leads, update_lead

ICP_CRITERIA = """Ideal Customer Profile:
- Engineering teams of 10-500 people
- Building AI/ML features or products
- Tech companies or AI-forward enterprises
- Budget: $10k-$100k/year"""


def qualify_leads(batch_size: int = 10):
    """Qualify new leads and update their scores."""
    leads = get_leads("new", batch_size)
    print(f"Qualifying {len(leads)} leads...")

    for lead in leads:
        print(f"  Researching {lead['company']}...")
        research = research_company(
            lead["company"],
            lead.get("website", ""),
            lead.get("industry", ""),
        )

        score = research.get("icp_score", 0)
        notes = f"Score: {score}/100\n{research.get('icp_rationale', '')}"

        update_lead(
            lead["id"],
            score=score,
            qualification_notes=notes,
            status="qualified" if score >= 60 else "disqualified",
        )
        print(f"    → Score: {score} | Status: {'qualified' if score >= 60 else 'disqualified'}")

    return len(leads)
```

### Step 5: Outreach Agent

```python
# outreach_agent.py
from openai import OpenAI
from research import research_company
from database import get_leads, save_outreach, update_lead

client = OpenAI()

EMAIL_PROMPT = """Write a personalized cold outreach email for a sales representative.

Prospect: {name}, {title} at {company}
Company research: {research}

Rules:
- Subject line: specific, relevant, 8 words max
- Opening: reference something specific about their company
- Value prop: 2 sentences on how we help teams like theirs
- CTA: one specific ask (15-min call, demo, etc.)
- Total length: 5-7 sentences MAX
- Tone: conversational, not salesy

Return JSON:
{{
  "subject": "email subject line",
  "body": "full email body (plain text)"
}}"""


def generate_outreach(lead: dict, research: dict) -> dict:
    import json
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EMAIL_PROMPT.format(
            name=lead["name"],
            title=lead.get("title", ""),
            company=lead["company"],
            research=str(research),
        )}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(response.choices[0].message.content)


def run_outreach_batch(batch_size: int = 5, dry_run: bool = True):
    """Generate outreach emails for qualified leads."""
    leads = get_leads("qualified", batch_size)
    print(f"Generating outreach for {len(leads)} leads (dry_run={dry_run})...")

    emails = []
    for lead in leads:
        research = research_company(lead["company"], lead.get("website", ""), lead.get("industry", ""))
        email = generate_outreach(lead, research)
        outreach_id = save_outreach(lead["id"], email["subject"], email["body"])

        if not dry_run:
            # In production: send via SendGrid/SES here
            update_lead(lead["id"], status="contacted")
            print(f"  ✓ Sent to {lead['email']}: {email['subject']}")
        else:
            print(f"\n--- Lead: {lead['name']} ({lead['company']}) ---")
            print(f"Subject: {email['subject']}")
            print(f"Body:\n{email['body']}")

        emails.append({"lead": lead, "email": email, "outreach_id": outreach_id})

    return emails
```

### Step 6: Pipeline Report

```python
# pipeline.py
from database import get_conn

def generate_pipeline_report() -> str:
    with get_conn() as conn:
        stats = conn.execute("""
            SELECT
                status,
                COUNT(*) as count,
                AVG(score) as avg_score
            FROM leads GROUP BY status
        """).fetchall()

        recent_outreach = conn.execute("""
            SELECT l.name, l.company, o.email_subject, o.sent_at
            FROM outreach o JOIN leads l ON o.lead_id = l.id
            ORDER BY o.sent_at DESC LIMIT 10
        """).fetchall()

    lines = ["# Sales Pipeline Report", ""]
    lines.append("## Lead Status Summary")
    for row in stats:
        lines.append(f"- **{row['status'].title()}**: {row['count']} leads (avg score: {row['avg_score'] or 0:.0f})")

    lines.append("\n## Recent Outreach")
    for row in recent_outreach:
        lines.append(f"- {row['name']} ({row['company']}): *{row['email_subject']}* — {row['sent_at'][:10]}")

    return "\n".join(lines)
```

### Step 7: Run

```bash
# Add sample leads
python -c "
from database import add_lead
add_lead('Jane Smith', 'Acme AI', 'jane@acme.ai', 'VP Engineering', 'https://acme.ai', 'AI Software', '50-200')
add_lead('Bob Chen', 'DataFlow Inc', 'bob@dataflow.io', 'CTO', 'https://dataflow.io', 'Data Analytics', '10-50')
"

# Qualify leads
python -c "from qualification_agent import qualify_leads; qualify_leads()"

# Generate outreach (dry run)
python -c "from outreach_agent import run_outreach_batch; run_outreach_batch(dry_run=True)"

# Generate report
python -c "from pipeline import generate_pipeline_report; print(generate_pipeline_report())"
```

---

## Extension Ideas

1. **CRM sync** — integrate with HubSpot or Salesforce APIs
2. **Email sending** — connect SendGrid for actual email delivery + tracking
3. **Reply handling** — classify inbound replies and suggest next actions
4. **A/B testing** — generate multiple subject lines and track open rates
5. **Meeting scheduler** — auto-schedule demos using Calendly API for interested leads

---

## What to Learn Next

- **Multi-agent systems** → [Multi-Agent Research System](/projects/multi-agent-research-system/)
- **AI agent fundamentals** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
