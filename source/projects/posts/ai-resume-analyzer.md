---
title: "Build an AI Resume Analyzer: Score, Critique, and Improve Resumes"
description: "Build a tool that analyzes resumes against job descriptions, scores fit, identifies gaps, and suggests specific improvements using GPT-4o-mini."
date: "2026-03-10"
slug: "ai-resume-analyzer"
level: "Beginner"
time: "3–4 hours"
stack: "Python, OpenAI API, pypdf, Streamlit"
keywords: ["AI resume analyzer", "resume analyzer Python", "LLM resume review"]
---

## Project Overview

A tool that reads a resume (PDF or text) and a job description, then scores the match, identifies missing skills and keywords, and generates specific improvement suggestions.

---

## Learning Goals

- Extract and process PDF text
- Use structured JSON outputs for consistent results
- Design evaluation prompts with scoring rubrics
- Build a Streamlit web app with file upload

---

## Architecture

```
Resume (PDF/text) + Job Description
        ↓
Text extraction
        ↓
Analysis prompt (scoring rubric)
        ↓
LLM → JSON with scores, gaps, suggestions
        ↓
Formatted report (CLI or Streamlit UI)
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai pypdf streamlit
```

### Step 2: Analyzer Core

```python
# analyzer.py
import json
from openai import OpenAI
from pypdf import PdfReader

client = OpenAI()

ANALYSIS_PROMPT = """You are an expert recruiter and career coach.

Analyze this resume against the job description and provide a detailed evaluation.

Return JSON with this exact structure:
{{
  "overall_score": <0-100>,
  "category_scores": {{
    "skills_match": <0-100>,
    "experience_relevance": <0-100>,
    "education_fit": <0-100>,
    "keywords_coverage": <0-100>
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["missing skill/experience 1", "gap 2", "gap 3"],
  "missing_keywords": ["keyword1", "keyword2", "keyword3"],
  "suggestions": [
    {{"priority": "high", "action": "specific improvement 1"}},
    {{"priority": "high", "action": "specific improvement 2"}},
    {{"priority": "medium", "action": "improvement 3"}}
  ],
  "hiring_recommendation": "strong_yes | yes | maybe | no",
  "summary": "2-3 sentence overall assessment"
}}

Job Description:
{job_description}

Resume:
{resume_text}"""


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def analyze_resume(resume_text: str, job_description: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": ANALYSIS_PROMPT.format(
                resume_text=resume_text[:4000],
                job_description=job_description[:2000],
            )}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)
```

### Step 3: Streamlit App

```python
# app.py
import streamlit as st
from analyzer import extract_text_from_pdf, analyze_resume

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📋", layout="wide")
st.title("📋 AI Resume Analyzer")
st.caption("Score your resume against any job description")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload PDF or paste text", type=["pdf", "txt"])
    resume_text_input = st.text_area("Or paste resume text", height=200)

with col2:
    st.subheader("Job Description")
    job_desc = st.text_area("Paste the job description here", height=300)

analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

if analyze_btn:
    # Get resume text
    resume_text = ""
    if resume_file:
        if resume_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = resume_file.read().decode("utf-8")
    elif resume_text_input:
        resume_text = resume_text_input

    if not resume_text:
        st.error("Please upload a resume or paste resume text.")
        st.stop()
    if not job_desc:
        st.error("Please paste a job description.")
        st.stop()

    with st.spinner("Analyzing resume..."):
        result = analyze_resume(resume_text, job_desc)

    # Display results
    st.divider()

    # Score overview
    score = result["overall_score"]
    rec = result.get("hiring_recommendation", "")
    rec_colors = {"strong_yes": "🟢", "yes": "🟡", "maybe": "🟠", "no": "🔴"}

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Overall Score", f"{score}/100")
    col_b.metric("Skills Match", f"{result['category_scores']['skills_match']}/100")
    col_c.metric("Experience", f"{result['category_scores']['experience_relevance']}/100")
    col_d.metric("Keywords", f"{result['category_scores']['keywords_coverage']}/100")

    st.info(f"{rec_colors.get(rec, '⚪')} Hiring Recommendation: **{rec.replace('_', ' ').title()}**")
    st.write(result["summary"])

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("✅ Strengths")
        for s in result.get("strengths", []):
            st.markdown(f"- {s}")

        st.subheader("🔑 Missing Keywords")
        st.markdown(", ".join(f"`{kw}`" for kw in result.get("missing_keywords", [])))

    with col_r:
        st.subheader("⚠️ Gaps")
        for g in result.get("gaps", []):
            st.markdown(f"- {g}")

    st.subheader("💡 Improvement Suggestions")
    for s in result.get("suggestions", []):
        priority = s.get("priority", "medium")
        icon = "🔴" if priority == "high" else "🟡"
        st.markdown(f"{icon} **{priority.upper()}**: {s['action']}")
```

### Step 4: Run

```bash
streamlit run app.py
```

---

## Extension Ideas

1. **ATS scoring** — simulate Applicant Tracking System keyword matching
2. **Multiple job comparison** — analyze one resume against 3-5 jobs, rank them
3. **Rewrite suggestions** — generate a rewritten version of weak bullet points
4. **Cover letter generator** — use analysis results to draft a targeted cover letter
5. **History** — save past analyses to compare improvement over time

---

## What to Learn Next

- **Structured outputs** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
- **Deploying Streamlit apps** → [Deploying AI Applications](/blog/deploying-ai-applications/)
