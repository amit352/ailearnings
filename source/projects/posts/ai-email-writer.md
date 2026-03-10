---
title: "Build an AI Email Writer: Draft Professional Emails in Seconds"
description: "Build an AI tool that drafts professional emails from bullet points or brief descriptions. Supports multiple tones, templates, and a Streamlit web UI."
date: "2026-03-10"
slug: "ai-email-writer"
level: "Beginner"
time: "2–3 hours"
stack: "Python, OpenAI API, Streamlit"
keywords: ["AI email writer", "email generator Python", "LLM email drafting"]
---

## Project Overview

An AI tool that turns brief descriptions, bullet points, or rough drafts into polished professional emails. Supports tone selection, email type templates, and length control.

---

## Learning Goals

- Prompt engineering for consistent tone and style
- Build a Streamlit form-based UI
- Use temperature settings to control creativity
- Handle different email types with templates

---

## Implementation

### Step 1: Setup

```bash
pip install openai streamlit
```

### Step 2: Email Writer Core

```python
# email_writer.py
from openai import OpenAI

client = OpenAI()

EMAIL_PROMPT = """Write a professional email based on the following brief.

Email Type: {email_type}
Tone: {tone}
Key Points to Include:
{key_points}

Additional Context: {context}

Requirements:
- Write a complete email with subject line, greeting, body, and sign-off
- Tone must be: {tone}
- Length: {length}
- Be specific and actionable
- Subject line should be clear and compelling

Format the output as:
Subject: [subject line]

[Full email body]"""

TONES = ["Professional", "Friendly", "Formal", "Persuasive", "Empathetic", "Concise"]
EMAIL_TYPES = [
    "Job Application",
    "Follow-up",
    "Meeting Request",
    "Project Update",
    "Customer Support",
    "Sales Outreach",
    "Networking",
    "Apology / Issue Resolution",
    "Thank You",
    "Introduction",
]
LENGTHS = ["Brief (3-4 sentences)", "Standard (2-3 paragraphs)", "Detailed (4+ paragraphs)"]


def write_email(
    key_points: str,
    email_type: str = "Professional",
    tone: str = "Professional",
    context: str = "",
    length: str = "Standard (2-3 paragraphs)",
) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": EMAIL_PROMPT.format(
                email_type=email_type,
                tone=tone,
                key_points=key_points,
                context=context,
                length=length,
            )}
        ],
        max_tokens=800,
        temperature=0.7,
    )
    return response.choices[0].message.content


def improve_email(original_email: str, improvement_notes: str) -> str:
    """Improve an existing email draft."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"""Improve this email draft based on the feedback.

Original Email:
{original_email}

Improvement Notes: {improvement_notes}

Return the improved email with subject line."""}
        ],
        max_tokens=800,
        temperature=0.5,
    )
    return response.choices[0].message.content
```

### Step 3: Streamlit App

```python
# app.py
import streamlit as st
from email_writer import write_email, improve_email, TONES, EMAIL_TYPES, LENGTHS

st.set_page_config(page_title="AI Email Writer", page_icon="✉️")
st.title("✉️ AI Email Writer")
st.caption("Turn bullet points into polished professional emails")

with st.form("email_form"):
    col1, col2 = st.columns(2)

    with col1:
        email_type = st.selectbox("Email Type", EMAIL_TYPES)
        tone = st.selectbox("Tone", TONES)
        length = st.selectbox("Length", LENGTHS)

    with col2:
        context = st.text_area("Context (optional)",
            placeholder="Who are you writing to? What's the background?",
            height=100)

    key_points = st.text_area(
        "Key Points to Include *",
        placeholder="• Ask for a meeting next week\n• Mention our previous conversation\n• Emphasize my experience with Python",
        height=150,
    )

    submitted = st.form_submit_button("Generate Email ✉️", type="primary", use_container_width=True)

if submitted:
    if not key_points.strip():
        st.error("Please enter at least one key point.")
        st.stop()

    with st.spinner("Writing your email..."):
        email = write_email(key_points, email_type, tone, context, length)

    st.divider()
    st.subheader("Generated Email")

    # Show in editable text area
    edited_email = st.text_area("Edit if needed:", value=email, height=300)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("Download .txt", edited_email, "email.txt", "text/plain")
    with col_b:
        if st.button("Copy to Clipboard"):
            st.write("✅ Copied!")
    with col_c:
        improve_notes = st.text_input("Improve it:", placeholder="Make it shorter and more direct")

    if improve_notes and st.button("Apply Improvement"):
        with st.spinner("Improving..."):
            improved = improve_email(edited_email, improve_notes)
        st.text_area("Improved Email:", value=improved, height=300)
```

### Step 4: Run

```bash
streamlit run app.py
```

---

## Extension Ideas

1. **Gmail integration** — connect to Gmail API to send directly
2. **Reply mode** — paste an incoming email and generate a reply
3. **A/B testing** — generate 2-3 variants and pick the best
4. **Language support** — add language selector for non-English emails
5. **Email thread summarizer** — summarize long email threads

---

## What to Learn Next

- **AI agents with email tools** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
- **Building a full chatbot** → [Build an AI Chatbot](/projects/ai-chatbot-python/)
