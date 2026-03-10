---
title: "Build an AI Meeting Summarizer: Transcribe and Summarize Meetings Automatically"
description: "Build a tool that transcribes audio recordings with Whisper, summarizes with GPT-4o-mini, extracts action items, decisions, and attendee contributions into a structured meeting report."
date: "2026-03-10"
slug: "ai-meeting-summarizer"
level: "Intermediate"
time: "3–5 hours"
stack: "Python, OpenAI API (Whisper + GPT), Streamlit"
keywords: ["AI meeting summarizer", "Whisper transcription Python", "meeting notes AI"]
---

## Project Overview

A tool that takes a meeting audio file or transcript, uses Whisper for transcription, and GPT-4o-mini to produce a structured summary with action items, key decisions, discussion topics, and next steps.

---

## Learning Goals

- Use OpenAI Whisper API for speech-to-text
- Design prompts for structured meeting intelligence
- Handle long transcripts with chunking
- Generate actionable output in multiple formats (Markdown, JSON)

---

## Architecture

```
Audio file (MP3/WAV/M4A)
        ↓
Whisper transcription
        ↓
Long transcript? → chunk + partial summarize
        ↓
Meeting analysis prompt
        ↓
Structured report: summary, action items, decisions, next steps
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai streamlit pydub
```

### Step 2: Transcription

```python
# transcriber.py
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI()
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
MAX_FILE_SIZE_MB = 25  # Whisper API limit


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper API."""
    path = Path(audio_path)

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}. Use: {SUPPORTED_FORMATS}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB. Split the audio first.")

    print(f"Transcribing {path.name} ({size_mb:.1f}MB)...")
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    print(f"Transcription complete: {len(transcript)} characters")
    return transcript


def split_audio_file(audio_path: str, chunk_minutes: int = 10) -> list[str]:
    """Split large audio files into chunks using pydub."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    base = Path(audio_path).stem

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"/tmp/{base}_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    return chunks


def transcribe_large_file(audio_path: str) -> str:
    """Handle files > 25MB by splitting and transcribing chunks."""
    chunks = split_audio_file(audio_path)
    transcripts = []
    for i, chunk_path in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        transcripts.append(transcribe_audio(chunk_path))
        os.unlink(chunk_path)
    return "\n\n".join(transcripts)
```

### Step 3: Meeting Analyzer

```python
# analyzer.py
import json
from openai import OpenAI

client = OpenAI()

MEETING_PROMPT = """Analyze this meeting transcript and return a JSON meeting report.

Return JSON with this structure:
{{
  "title": "Meeting title or best guess from content",
  "duration_estimate": "e.g., ~45 minutes",
  "summary": "3-4 sentence executive summary",
  "key_topics": ["topic 1", "topic 2", "topic 3"],
  "decisions": [
    {{"decision": "...", "owner": "person name or 'Team'"}}
  ],
  "action_items": [
    {{"task": "...", "owner": "person name", "due": "deadline if mentioned or 'TBD'"}}
  ],
  "open_questions": ["unresolved question 1", "unresolved question 2"],
  "next_steps": ["step 1", "step 2"],
  "participants": ["name1", "name2"]
}}

Meeting Transcript:
{transcript}"""


def analyze_meeting(transcript: str) -> dict:
    # If transcript is very long, summarize sections first
    max_chars = 12000
    if len(transcript) > max_chars:
        transcript = summarize_long_transcript(transcript, max_chars)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": MEETING_PROMPT.format(transcript=transcript)}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


def summarize_long_transcript(transcript: str, target_chars: int) -> str:
    """Condense a long transcript by summarizing each section."""
    chunk_size = 4000
    chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
    summaries = []

    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize this meeting section, preserving key decisions, action items, and speaker attributions:\n\n{chunk}"}],
            max_tokens=400,
        )
        summaries.append(response.choices[0].message.content)

    return "\n\n---\n\n".join(summaries)


def format_report(analysis: dict) -> str:
    """Format the JSON analysis as a readable Markdown report."""
    lines = [
        f"# Meeting Summary: {analysis.get('title', 'Untitled')}",
        f"\n**Duration:** {analysis.get('duration_estimate', 'N/A')}",
        f"**Participants:** {', '.join(analysis.get('participants', ['Unknown']))}",
        f"\n## Executive Summary\n{analysis.get('summary', '')}",
        "\n## Key Topics\n" + "\n".join(f"- {t}" for t in analysis.get("key_topics", [])),
        "\n## Decisions Made",
    ]
    for d in analysis.get("decisions", []):
        lines.append(f"- **{d.get('decision', '')}** (Owner: {d.get('owner', 'TBD')})")

    lines.append("\n## Action Items")
    for a in analysis.get("action_items", []):
        lines.append(f"- [ ] **{a.get('task', '')}** — {a.get('owner', 'TBD')} by {a.get('due', 'TBD')}")

    if analysis.get("open_questions"):
        lines.append("\n## Open Questions")
        lines.extend(f"- {q}" for q in analysis["open_questions"])

    lines.append("\n## Next Steps")
    lines.extend(f"- {s}" for s in analysis.get("next_steps", []))

    return "\n".join(lines)
```

### Step 4: Streamlit App

```python
# app.py
import streamlit as st
import tempfile, os
from transcriber import transcribe_audio, transcribe_large_file
from analyzer import analyze_meeting, format_report

st.set_page_config(page_title="AI Meeting Summarizer", page_icon="🎙️")
st.title("🎙️ AI Meeting Summarizer")

input_mode = st.radio("Input type", ["Audio file", "Paste transcript"], horizontal=True)

transcript = ""

if input_mode == "Audio file":
    audio_file = st.file_uploader("Upload meeting recording", type=["mp3", "mp4", "m4a", "wav", "webm"])
    if audio_file and st.button("Transcribe", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("Transcribing audio..."):
            size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            if size_mb > 25:
                transcript = transcribe_large_file(tmp_path)
            else:
                transcript = transcribe_audio(tmp_path)
        os.unlink(tmp_path)
        st.session_state["transcript"] = transcript
        st.success("Transcription complete!")

    if "transcript" in st.session_state:
        with st.expander("View transcript"):
            st.text(st.session_state["transcript"])
        transcript = st.session_state["transcript"]
else:
    transcript = st.text_area("Paste meeting transcript", height=300)

if transcript and st.button("Analyze Meeting", type="primary"):
    with st.spinner("Analyzing meeting..."):
        analysis = analyze_meeting(transcript)
        report = format_report(analysis)

    st.divider()
    st.subheader("Meeting Report")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Topics Covered", len(analysis.get("key_topics", [])))
        st.metric("Action Items", len(analysis.get("action_items", [])))
    with col2:
        st.metric("Decisions Made", len(analysis.get("decisions", [])))
        st.metric("Participants", len(analysis.get("participants", [])))

    st.markdown(report)
    st.download_button("Download Report (.md)", report, "meeting_summary.md", "text/markdown")
```

### Step 5: Run

```bash
streamlit run app.py
```

---

## Extension Ideas

1. **Calendar integration** — read meeting title/attendees from Google Calendar event
2. **Slack bot** — post summary to a channel automatically after each meeting
3. **Speaker diarization** — identify who said what using pyannote.audio
4. **Email draft** — generate a follow-up email with action items
5. **Searchable archive** — store all meeting summaries in a database for search

---

## What to Learn Next

- **Whisper & audio AI** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
- **Building AI agents with tools** → [Tool Use and Function Calling](/blog/tool-use-and-function-calling/)
