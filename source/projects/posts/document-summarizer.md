---
title: "Build an AI Document Summarizer: PDF, Text, and URL Summarization"
description: "Build an AI-powered document summarizer that handles PDFs, plain text, and web URLs. Supports map-reduce summarization for long documents and outputs key bullet points."
date: "2026-03-10"
slug: "document-summarizer"
level: "Beginner"
time: "2–3 hours"
stack: "Python, OpenAI API, PyPDF2, BeautifulSoup"
keywords: ["AI document summarizer", "PDF summarizer Python", "LLM summarization"]
---

## Project Overview

Build a tool that takes any document — PDF, text file, or web page URL — and returns a concise summary with key points. Handles long documents that exceed the LLM's context window using a map-reduce strategy.

---

## Learning Goals

- Extract text from PDFs and web pages
- Handle long documents with map-reduce summarization
- Prompt engineering for structured summaries
- Build a simple CLI with file arguments

---

## Architecture

```
Input (PDF / text / URL)
        ↓
Text Extraction
        ↓
Chunk if too long (>4000 tokens)
   ↓ short             ↓ long
Direct summarize    Map: summarize each chunk
                    Reduce: combine chunk summaries
        ↓
Structured Output (bullets + TL;DR)
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai pypdf requests beautifulsoup4 tiktoken
```

### Step 2: Text Extractors

```python
# extractors.py
import re
import requests
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

def extract_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_text(text)

def extract_from_text(path: str) -> str:
    return clean_text(Path(path).read_text(encoding="utf-8"))

def extract_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove nav, header, footer, ads
    for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Get main content
    main = soup.find("main") or soup.find("article") or soup.find("body")
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    return clean_text(text)

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
```

### Step 3: Summarizer with Map-Reduce

```python
# summarizer.py
import tiktoken
from openai import OpenAI

client = OpenAI()
enc = tiktoken.encoding_for_model("gpt-4o-mini")

MAX_CHUNK_TOKENS = 3000
SUMMARY_PROMPT = """Summarize the following text. Provide:
1. A TL;DR (one sentence)
2. Key points (4-6 bullet points)
3. Main takeaways (2-3 sentences)

Text:
{text}"""

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """Split text into chunks that fit within token limit."""
    paragraphs = text.split('\n\n')
    chunks, current, current_tokens = [], [], 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append('\n\n'.join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append('\n\n'.join(current))
    return chunks

def summarize_chunk(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(text=text)}],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content

def summarize(text: str) -> str:
    tokens = count_tokens(text)
    print(f"Document: {tokens} tokens")

    if tokens <= MAX_CHUNK_TOKENS:
        # Direct summarization
        print("Summarizing directly...")
        return summarize_chunk(text)

    # Map-reduce for long documents
    chunks = chunk_text(text)
    print(f"Document too long — splitting into {len(chunks)} chunks...")

    # Map: summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
        chunk_summaries.append(summarize_chunk(chunk))

    # Reduce: combine chunk summaries into final summary
    print("Combining summaries...")
    combined = "\n\n---\n\n".join(chunk_summaries)
    final_prompt = f"""Below are summaries of sections of a long document.
Create a unified final summary with:
1. A TL;DR (one sentence)
2. Key points (5-7 bullet points)
3. Main takeaways (2-3 sentences)

Section summaries:
{combined}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=800,
        temperature=0.3,
    )
    return response.choices[0].message.content
```

### Step 4: CLI

```python
# main.py
import argparse
import sys
from extractors import extract_from_pdf, extract_from_text, extract_from_url
from summarizer import summarize

def main():
    parser = argparse.ArgumentParser(description="Summarize any document with AI")
    parser.add_argument("source", help="PDF path, text file path, or URL")
    parser.add_argument("--output", "-o", help="Save summary to file")
    args = parser.parse_args()

    source = args.source
    print(f"Processing: {source}\n")

    # Detect input type
    if source.startswith(("http://", "https://")):
        text = extract_from_url(source)
    elif source.endswith(".pdf"):
        text = extract_from_pdf(source)
    else:
        text = extract_from_text(source)

    if not text.strip():
        print("Error: Could not extract text from source.")
        sys.exit(1)

    summary = summarize(text)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
```

### Step 5: Run

```bash
# Summarize a PDF
python main.py research_paper.pdf

# Summarize a web page
python main.py https://example.com/article

# Summarize text and save
python main.py document.txt --output summary.txt
```

---

## Sample Output

```
Processing: research_paper.pdf
Document: 8,432 tokens
Document too long — splitting into 3 chunks...
  Summarizing chunk 1/3...
  Summarizing chunk 2/3...
  Summarizing chunk 3/3...
Combining summaries...

============================================================
SUMMARY
============================================================
**TL;DR:** This paper introduces a novel attention mechanism that reduces transformer memory usage by 60% while maintaining 98% of baseline performance.

**Key Points:**
• Proposes Sparse Windowed Attention (SWA) for long-context processing
• Achieves 60% memory reduction vs standard attention
• Performance within 2% of full attention on standard benchmarks
• Enables 4× longer context windows on the same hardware
• Evaluated on 5 NLP benchmarks: GLUE, SuperGLUE, SQuAD, TriviaQA, NarrativeQA

**Takeaways:** The SWA mechanism represents a practical improvement for production deployment of large language models...
```

---

## Extension Ideas

1. **Batch processing** — summarize an entire folder of PDFs
2. **Custom prompts** — add `--style academic|executive|bullet` flag
3. **Language support** — add `--language` flag for summaries in other languages
4. **Web app** — wrap in a Gradio or Streamlit UI with file upload
5. **Comparison mode** — summarize two documents and compare them

---

## What to Learn Next

- **RAG for Q&A** → [RAG Document Assistant](/projects/rag-document-assistant/)
- **LLM summarization patterns** → [LangChain Complete Tutorial](/blog/langchain-tutorial-complete/)
