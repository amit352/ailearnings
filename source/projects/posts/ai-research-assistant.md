---
title: "Build an AI Research Assistant: Summarize Papers and Extract Insights"
description: "Build an AI tool that reads academic papers and web pages, extracts key findings, compares multiple sources, and generates research summaries with citations."
date: "2026-03-10"
slug: "ai-research-assistant"
level: "Intermediate"
time: "4–5 hours"
stack: "Python, OpenAI API, requests, BeautifulSoup, Streamlit"
keywords: ["AI research assistant Python", "paper summarizer AI", "LLM literature review"]
---

## Project Overview

A research assistant that fetches papers from URLs (arXiv, PDFs, web pages), extracts key findings, methodology, and limitations, and synthesizes multiple papers into a comparative literature review.

---

## Learning Goals

- Scrape and clean academic content from the web
- Design prompts for structured academic extraction
- Synthesize insights across multiple sources
- Generate properly cited Markdown research reports

---

## Architecture

```
URLs (arXiv / PDF / web page)
        ↓
Text extraction + cleaning
        ↓
Per-paper: findings, methods, limitations, key terms
        ↓
Cross-paper synthesis
        ↓
Research report with citations
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai requests beautifulsoup4 pypdf streamlit
```

### Step 2: Content Fetcher

```python
# fetcher.py
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
import tempfile, os


def fetch_url(url: str) -> str:
    """Fetch and extract clean text from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (research-tool)"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "pdf" in content_type or url.endswith(".pdf"):
        return _extract_pdf_from_bytes(response.content)

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    article = soup.find("article") or soup.find(id="content") or soup.find("main") or soup.body
    text = article.get_text(separator="\n") if article else soup.get_text(separator="\n")
    return _clean(text)


def fetch_arxiv(arxiv_id: str) -> str:
    """Fetch paper from arXiv (e.g., '2303.08774')."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    return fetch_url(url)


def fetch_local_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return _clean(text)


def _extract_pdf_from_bytes(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        text = fetch_local_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
    return text


def _clean(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
```

### Step 3: Paper Analyzer

```python
# analyzer.py
import json
from openai import OpenAI

client = OpenAI()

EXTRACT_PROMPT = """Analyze this academic/research text and extract structured information.

Return JSON:
{{
  "title": "paper/article title",
  "authors": ["author1", "author2"],
  "year": "publication year or 'unknown'",
  "problem_statement": "what problem does this address?",
  "methodology": "what approach/methods were used?",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "contributions": ["contribution 1", "contribution 2"],
  "limitations": ["limitation 1", "limitation 2"],
  "key_terms": ["term1", "term2", "term3"],
  "tldr": "one sentence summary"
}}

Text (first 6000 chars):
{text}"""

SYNTHESIS_PROMPT = """You are a research synthesizer. Given analyses of {n} papers on the topic "{topic}", write a literature review.

Structure:
1. **Overview** — what is this field about, why does it matter (2-3 sentences)
2. **Common Themes** — what do the papers agree on (bullet points)
3. **Contrasting Approaches** — where do the papers differ in methodology or findings
4. **Key Insights** — the most important findings across all papers
5. **Research Gaps** — what questions remain unanswered
6. **Conclusion** — synthesis in 2-3 sentences

Papers analyzed:
{papers_json}

Write in academic but accessible style. Use author/year citations like (Smith, 2023)."""


def analyze_paper(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text[:6000])}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


def synthesize_papers(analyses: list[dict], topic: str) -> str:
    papers_summary = json.dumps([{
        "title": a.get("title", "Unknown"),
        "authors": a.get("authors", []),
        "year": a.get("year", "unknown"),
        "key_findings": a.get("key_findings", []),
        "methodology": a.get("methodology", ""),
        "limitations": a.get("limitations", []),
        "tldr": a.get("tldr", ""),
    } for a in analyses], indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SYNTHESIS_PROMPT.format(
            n=len(analyses), topic=topic, papers_json=papers_summary
        )}],
        max_tokens=1500,
        temperature=0.4,
    )
    return response.choices[0].message.content
```

### Step 4: Streamlit App

```python
# app.py
import streamlit as st
from fetcher import fetch_url, fetch_local_pdf
from analyzer import analyze_paper, synthesize_papers

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 AI Research Assistant")
st.caption("Summarize papers and synthesize literature reviews")

topic = st.text_input("Research topic (for synthesis)", placeholder="e.g., RAG for question answering")

papers_data = []
urls_input = st.text_area("Paper URLs (one per line)", height=150,
    placeholder="https://arxiv.org/abs/2403.01234\nhttps://example.com/paper.pdf")
uploaded_pdfs = st.file_uploader("Or upload PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze Papers", type="primary"):
    urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]

    # Process URLs
    for url in urls:
        with st.spinner(f"Fetching {url[:50]}..."):
            try:
                text = fetch_url(url)
                analysis = analyze_paper(text)
                analysis["url"] = url
                papers_data.append(analysis)
                st.success(f"✓ {analysis.get('title', url[:50])}")
            except Exception as e:
                st.error(f"Failed to fetch {url}: {e}")

    # Process uploaded PDFs
    import tempfile, os
    for f in uploaded_pdfs or []:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            text = fetch_local_pdf(tmp_path)
            analysis = analyze_paper(text)
            analysis["url"] = f.name
            papers_data.append(analysis)
            st.success(f"✓ {analysis.get('title', f.name)}")
        finally:
            os.unlink(tmp_path)

    st.session_state["papers"] = papers_data

if "papers" in st.session_state and st.session_state["papers"]:
    papers = st.session_state["papers"]

    # Individual paper summaries
    st.divider()
    st.subheader(f"Individual Papers ({len(papers)})")
    for paper in papers:
        with st.expander(paper.get("title", "Unknown title")):
            st.markdown(f"**TL;DR:** {paper.get('tldr', '')}")
            st.markdown(f"**Authors:** {', '.join(paper.get('authors', []))}")
            st.markdown("**Key Findings:**")
            for f in paper.get("key_findings", []):
                st.markdown(f"- {f}")
            st.markdown(f"**Limitations:** {', '.join(paper.get('limitations', []))}")

    # Synthesis
    if len(papers) > 1 and topic:
        st.divider()
        if st.button("Generate Literature Review"):
            with st.spinner("Synthesizing..."):
                review = synthesize_papers(papers, topic)
            st.subheader("Literature Review")
            st.markdown(review)
            st.download_button("Download Review (.md)", review, "literature_review.md", "text/markdown")
```

### Step 5: Run

```bash
streamlit run app.py
```

---

## Extension Ideas

1. **Citation graph** — build a graph of which papers cite which
2. **Gap finder** — identify specific unanswered research questions
3. **Automatic arXiv search** — search by keyword, not just manual URL entry
4. **Notion/Obsidian export** — save research notes directly to your PKM system
5. **Follow-up questions** — ask questions about specific papers in a chat interface

---

## What to Learn Next

- **Personal knowledge base** → [AI Personal Knowledge Base](/projects/ai-personal-knowledge-base/)
- **RAG deep dive** → [RAG Document Assistant](/projects/rag-document-assistant/)
