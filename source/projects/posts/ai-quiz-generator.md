---
title: "Build an AI Quiz Generator from Any Text or Document"
description: "Build an AI tool that generates multiple-choice quizzes from any text, PDF, or topic. Export to JSON, run interactively, and track scores."
date: "2026-03-10"
slug: "ai-quiz-generator"
level: "Beginner"
time: "2–3 hours"
stack: "Python, OpenAI API, JSON"
keywords: ["AI quiz generator", "quiz from text Python", "LLM quiz generation"]
---

## Project Overview

An AI tool that reads any text or document and generates a multiple-choice quiz with configurable difficulty and question count. Runs interactively in the terminal or exports to JSON for use in other apps.

---

## Learning Goals

- Use structured output (JSON mode) from LLMs
- Validate and parse LLM responses reliably
- Build an interactive quiz engine
- Prompt engineering for specific output formats

---

## Architecture

```
Input (text / topic / PDF)
        ↓
Prompt with JSON schema instruction
        ↓
LLM generates questions
        ↓
Parse + validate JSON
        ↓
Interactive quiz runner OR export
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai pypdf
```

### Step 2: Question Generator

```python
# generator.py
import json
from openai import OpenAI

client = OpenAI()

GENERATION_PROMPT = """Generate {n} multiple-choice quiz questions about the following content.

Rules:
- Each question must have exactly 4 options (A, B, C, D)
- Exactly one option is correct
- Include a brief explanation for the correct answer
- Vary difficulty: mix easy, medium, and hard questions
- Questions should test understanding, not just memorization

Return valid JSON in this exact format:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct": "A",
      "explanation": "Brief explanation of why A is correct."
    }}
  ]
}}

Content:
{content}"""

def generate_quiz(content: str, n_questions: int = 5) -> list[dict]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": GENERATION_PROMPT.format(
                n=n_questions, content=content[:6000]
            )}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    data = json.loads(response.choices[0].message.content)
    questions = data.get("questions", [])

    # Validate structure
    valid = []
    for q in questions:
        if (isinstance(q.get("question"), str)
                and isinstance(q.get("options"), dict)
                and len(q["options"]) == 4
                and q.get("correct") in q["options"]):
            valid.append(q)

    return valid


def generate_from_topic(topic: str, n_questions: int = 5) -> list[dict]:
    """Generate quiz from a topic name without requiring source text."""
    prompt = f"Generate quiz questions about: {topic}"
    return generate_quiz(prompt, n_questions)
```

### Step 3: Quiz Runner

```python
# quiz_runner.py

def run_quiz(questions: list[dict]) -> dict:
    """Interactive quiz in the terminal. Returns results."""
    print(f"\n{'='*60}")
    print(f"Quiz: {len(questions)} questions")
    print("="*60)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}. {q['question']}")
        for letter, text in q["options"].items():
            print(f"  {letter}) {text}")

        while True:
            answer = input("\nYour answer (A/B/C/D): ").strip().upper()
            if answer in ("A", "B", "C", "D"):
                break
            print("Please enter A, B, C, or D")

        correct = q["correct"]
        is_correct = answer == correct

        if is_correct:
            print("✅ Correct!")
        else:
            print(f"❌ Wrong. The correct answer is {correct}: {q['options'][correct]}")

        print(f"💡 {q['explanation']}")
        results.append({"question": q["question"], "your_answer": answer,
                        "correct_answer": correct, "is_correct": is_correct})

    # Final score
    score = sum(1 for r in results if r["is_correct"])
    total = len(questions)
    print(f"\n{'='*60}")
    print(f"Final Score: {score}/{total} ({score/total:.0%})")
    if score == total:
        print("Perfect score! 🎉")
    elif score >= total * 0.8:
        print("Great job! 🌟")
    elif score >= total * 0.6:
        print("Good effort! Keep practicing.")
    else:
        print("Keep studying — you'll get there! 📚")

    return {"score": score, "total": total, "results": results}
```

### Step 4: Main CLI

```python
# main.py
import argparse
import json
import sys
from pathlib import Path
from generator import generate_quiz, generate_from_topic
from quiz_runner import run_quiz

def main():
    parser = argparse.ArgumentParser(description="AI Quiz Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Text or PDF file to quiz from")
    group.add_argument("--topic", help="Topic name (e.g., 'machine learning')")
    parser.add_argument("--questions", "-n", type=int, default=5, help="Number of questions")
    parser.add_argument("--export", help="Export questions to JSON file")
    parser.add_argument("--no-run", action="store_true", help="Only generate, don't run quiz")
    args = parser.parse_args()

    # Generate questions
    if args.topic:
        print(f"Generating {args.questions} questions about: {args.topic}")
        questions = generate_from_topic(args.topic, args.questions)
    else:
        path = Path(args.file)
        if path.suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        else:
            text = path.read_text(encoding="utf-8")
        print(f"Generating {args.questions} questions from: {args.file}")
        questions = generate_quiz(text, args.questions)

    if not questions:
        print("Failed to generate questions. Try a different input.")
        sys.exit(1)

    print(f"Generated {len(questions)} questions.")

    # Export if requested
    if args.export:
        with open(args.export, "w") as f:
            json.dump({"questions": questions}, f, indent=2)
        print(f"Saved to {args.export}")

    # Run quiz
    if not args.no_run:
        run_quiz(questions)

if __name__ == "__main__":
    main()
```

### Step 5: Run

```bash
# Quiz from a topic
python main.py --topic "Python decorators" --questions 5

# Quiz from a file
python main.py --file lecture_notes.txt --questions 10

# Generate and export without running
python main.py --topic "RAG systems" --export quiz.json --no-run
```

---

## Extension Ideas

1. **Difficulty levels** — add `--difficulty easy|medium|hard` flag
2. **Question types** — add True/False and short-answer questions
3. **Leaderboard** — save scores to a SQLite DB, show top scores
4. **Flashcard mode** — export to Anki-compatible format
5. **Web app** — Streamlit UI with file upload and instant quiz

---

## What to Learn Next

- **Structured outputs** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
- **More complex document handling** → [Document Summarizer](/projects/document-summarizer/)
