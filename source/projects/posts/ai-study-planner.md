---
title: "Build an AI Study Planner: Personalized Learning Schedules with Spaced Repetition"
description: "Build an AI study planner that creates personalized schedules, generates practice questions using spaced repetition, tracks progress, and adapts the plan based on performance."
date: "2026-03-10"
slug: "ai-study-planner"
level: "Advanced"
time: "8–10 hours"
stack: "Python, OpenAI API, SQLite, Streamlit, schedule"
keywords: ["AI study planner", "spaced repetition AI", "personalized learning Python"]
---

## Project Overview

An AI-powered study planner that takes a learning goal and deadline, creates a week-by-week curriculum, generates practice questions with spaced repetition scheduling, tracks performance, and adapts the plan based on what's working.

---

## Learning Goals

- Implement a spaced repetition algorithm (SM-2)
- Design adaptive AI prompts that use performance data
- Build a complete study session management system
- Use SQLite for persistent learning state

---

## Architecture

```
Learning goal + deadline input
        ↓
Curriculum generator (LLM)
  → topics, subtopics, resources per week
        ↓
Session scheduler (SM-2 spaced repetition)
  → what to study today
        ↓
Question generator (LLM)
  → practice questions + answers per topic
        ↓
Performance tracker → adapts schedule
        ↓
Streamlit dashboard
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai streamlit schedule
```

### Step 2: Database

```python
# database.py
import sqlite3
import json
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = "study.db"


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                deadline TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER,
                week INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                priority INTEGER DEFAULT 5,
                FOREIGN KEY (goal_id) REFERENCES goals(id)
            );
            CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                difficulty TEXT DEFAULT 'medium',
                interval INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                ease_factor REAL DEFAULT 2.5,
                next_review TEXT,
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id INTEGER,
                quality INTEGER,
                reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (card_id) REFERENCES cards(id)
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


def save_goal(title: str, description: str, deadline: str) -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO goals (title, description, deadline) VALUES (?,?,?)",
            (title, description, deadline)
        )
        return cursor.lastrowid


def save_topic(goal_id: int, week: int, title: str, description: str = "") -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO topics (goal_id, week, title, description) VALUES (?,?,?,?)",
            (goal_id, week, title, description)
        )
        return cursor.lastrowid


def save_card(topic_id: int, question: str, answer: str, difficulty: str = "medium") -> int:
    next_review = datetime.now().isoformat()
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO cards (topic_id, question, answer, difficulty, next_review) VALUES (?,?,?,?,?)",
            (topic_id, question, answer, difficulty, next_review)
        )
        return cursor.lastrowid


def get_due_cards(limit: int = 20) -> list[dict]:
    now = datetime.now().isoformat()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT c.*, t.title as topic_title
            FROM cards c JOIN topics t ON c.topic_id = t.id
            WHERE c.next_review <= ? ORDER BY c.next_review ASC LIMIT ?
        """, (now, limit)).fetchall()
        return [dict(r) for r in rows]


def update_card_sm2(card_id: int, quality: int):
    """Update card using SM-2 spaced repetition algorithm. Quality: 0-5."""
    with get_conn() as conn:
        card = dict(conn.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone())

    ef = card["ease_factor"]
    reps = card["repetitions"]
    interval = card["interval"]

    if quality >= 3:  # Correct response
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        reps += 1
        ef = max(1.3, ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    else:  # Wrong answer — reset
        reps = 0
        interval = 1

    next_review = (datetime.now() + timedelta(days=interval)).isoformat()

    with get_conn() as conn:
        conn.execute(
            "UPDATE cards SET interval=?, repetitions=?, ease_factor=?, next_review=? WHERE id=?",
            (interval, reps, ef, next_review, card_id)
        )
        conn.execute(
            "INSERT INTO reviews (card_id, quality) VALUES (?,?)",
            (card_id, quality)
        )


init_db()
```

### Step 3: Curriculum Generator

```python
# curriculum.py
import json
from openai import OpenAI
from database import save_goal, save_topic

client = OpenAI()

CURRICULUM_PROMPT = """Create a comprehensive study curriculum for this learning goal.

Goal: {goal}
Deadline: {deadline} ({weeks} weeks available)
Current level: {level}

Create a week-by-week curriculum. Return JSON:
{{
  "overview": "2-3 sentence description of the learning path",
  "weeks": [
    {{
      "week": 1,
      "theme": "Week theme/focus",
      "topics": [
        {{
          "title": "Topic name",
          "description": "What to learn and why",
          "time_hours": 2,
          "resources": ["resource suggestion 1", "resource suggestion 2"]
        }}
      ]
    }}
  ]
}}

Make it practical and progressive. Focus on building skills week by week."""

QUESTIONS_PROMPT = """Generate practice questions for this study topic.

Topic: {topic}
Description: {description}
Difficulty: {difficulty}

Generate {n} questions. Return JSON:
{{
  "questions": [
    {{
      "question": "Question text",
      "answer": "Comprehensive answer",
      "difficulty": "easy" | "medium" | "hard",
      "type": "conceptual" | "practical" | "recall"
    }}
  ]
}}"""


def generate_curriculum(goal: str, deadline: str, level: str = "intermediate", weeks: int = 8) -> int:
    """Generate a curriculum and save to database."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": CURRICULUM_PROMPT.format(
            goal=goal, deadline=deadline, level=level, weeks=weeks
        )}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    data = json.loads(response.choices[0].message.content)

    goal_id = save_goal(goal, data.get("overview", ""), deadline)
    for week_data in data.get("weeks", []):
        for topic in week_data.get("topics", []):
            save_topic(goal_id, week_data["week"], topic["title"], topic["description"])

    print(f"Created curriculum: {len(data.get('weeks', []))} weeks, goal_id={goal_id}")
    return goal_id


def generate_questions_for_topic(topic_id: int, topic_title: str, description: str, n: int = 5):
    """Generate practice questions for a topic and save as cards."""
    from database import save_card
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": QUESTIONS_PROMPT.format(
            topic=topic_title, description=description, difficulty="mixed", n=n
        )}],
        response_format={"type": "json_object"},
        temperature=0.5,
    )
    data = json.loads(response.choices[0].message.content)
    count = 0
    for q in data.get("questions", []):
        save_card(topic_id, q["question"], q["answer"], q.get("difficulty", "medium"))
        count += 1
    return count
```

### Step 4: Adaptive Planner

```python
# planner.py
from openai import OpenAI
from database import get_conn

client = OpenAI()

ADAPT_PROMPT = """You are an AI tutor reviewing a student's study progress.

Goal: {goal}
Recent performance data:
{performance}

Based on the performance data, suggest specific adaptations:
1. Topics that need more review (low scores)
2. Topics ready to advance (high scores)
3. Recommended focus for the next 3 days
4. Any curriculum adjustments

Keep suggestions specific and actionable."""


def get_performance_summary(goal_id: int) -> dict:
    with get_conn() as conn:
        stats = conn.execute("""
            SELECT t.title, t.week,
                COUNT(r.id) as reviews,
                AVG(r.quality) as avg_quality,
                MIN(r.quality) as min_quality
            FROM topics t
            JOIN cards c ON c.topic_id = t.id
            LEFT JOIN reviews r ON r.card_id = c.id
            WHERE t.goal_id = ?
            GROUP BY t.id
        """, (goal_id,)).fetchall()

        due_count = conn.execute("""
            SELECT COUNT(*) FROM cards c
            JOIN topics t ON c.topic_id = t.id
            WHERE t.goal_id = ? AND c.next_review <= datetime('now')
        """, (goal_id,)).fetchone()[0]

    return {
        "topic_stats": [dict(r) for r in stats],
        "due_cards": due_count,
    }


def get_adaptive_advice(goal_id: int, goal_title: str) -> str:
    perf = get_performance_summary(goal_id)
    perf_text = f"Due cards: {perf['due_cards']}\n\nTopic performance:\n"
    for t in perf["topic_stats"]:
        avg = t["avg_quality"] or 0
        perf_text += f"- Week {t['week']}: {t['title']} — {t['reviews']} reviews, avg quality: {avg:.1f}/5\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ADAPT_PROMPT.format(
            goal=goal_title, performance=perf_text
        )}],
        max_tokens=400,
        temperature=0.4,
    )
    return response.choices[0].message.content
```

### Step 5: Streamlit App

```python
# app.py
import streamlit as st
from curriculum import generate_curriculum, generate_questions_for_topic
from planner import get_performance_summary, get_adaptive_advice
from database import get_conn, get_due_cards, update_card_sm2, save_goal

st.set_page_config(page_title="AI Study Planner", page_icon="📚", layout="wide")
st.title("📚 AI Study Planner")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Goals", "📖 Study Session", "📊 Progress", "🤖 AI Advice"])

with tab1:
    st.subheader("Create New Learning Goal")
    col1, col2 = st.columns(2)
    with col1:
        goal_title = st.text_input("What do you want to learn?", placeholder="Master LLM fine-tuning")
        level = st.selectbox("Current level", ["Beginner", "Intermediate", "Advanced"])
    with col2:
        deadline = st.date_input("Target completion date")
        weeks = st.number_input("Study weeks", min_value=1, max_value=52, value=8)

    if st.button("Generate Curriculum", type="primary") and goal_title:
        with st.spinner("Generating personalized curriculum..."):
            goal_id = generate_curriculum(goal_title, str(deadline), level.lower(), int(weeks))
            st.session_state["active_goal_id"] = goal_id
            st.session_state["active_goal_title"] = goal_title
        st.success(f"Curriculum created! Goal ID: {goal_id}")

    # List existing goals
    st.subheader("Your Goals")
    with get_conn() as conn:
        goals = conn.execute("SELECT * FROM goals ORDER BY created_at DESC").fetchall()
    for g in goals:
        if st.button(f"📌 {g['title']} (due: {g['deadline']})", key=f"goal_{g['id']}"):
            st.session_state["active_goal_id"] = g["id"]
            st.session_state["active_goal_title"] = g["title"]

with tab2:
    due = get_due_cards(10)
    st.subheader(f"Study Session — {len(due)} cards due")

    if not due:
        st.success("🎉 No cards due! All caught up.")
    elif "card_idx" not in st.session_state:
        st.session_state.card_idx = 0
        st.session_state.show_answer = False

    if due and st.session_state.get("card_idx", 0) < len(due):
        card = due[st.session_state.card_idx]
        st.progress(st.session_state.card_idx / len(due))
        st.write(f"**Topic:** {card['topic_title']} | Card {st.session_state.card_idx + 1}/{len(due)}")
        st.markdown(f"### {card['question']}")

        if not st.session_state.get("show_answer"):
            if st.button("Show Answer"):
                st.session_state.show_answer = True
                st.rerun()
        else:
            st.info(f"**Answer:** {card['answer']}")
            st.write("How well did you know this?")
            cols = st.columns(5)
            labels = ["0 - Blackout", "1 - Wrong", "2 - Hard", "3 - OK", "4 - Easy", "5 - Perfect"]
            for i, (col, label) in enumerate(zip(cols, labels[1:])):
                if col.button(label, key=f"q_{i+1}"):
                    update_card_sm2(card["id"], i + 1)
                    st.session_state.card_idx += 1
                    st.session_state.show_answer = False
                    st.rerun()

with tab3:
    goal_id = st.session_state.get("active_goal_id")
    if goal_id:
        perf = get_performance_summary(goal_id)
        st.metric("Cards Due Now", perf["due_cards"])
        st.subheader("Topic Performance")
        for t in perf["topic_stats"]:
            avg = t["avg_quality"] or 0
            col1, col2 = st.columns([3, 1])
            col1.write(f"Week {t['week']}: {t['title']}")
            col2.progress(avg / 5, text=f"{avg:.1f}/5")

with tab4:
    goal_id = st.session_state.get("active_goal_id")
    goal_title = st.session_state.get("active_goal_title", "")
    if goal_id and st.button("Get AI Advice", type="primary"):
        with st.spinner("Analyzing your progress..."):
            advice = get_adaptive_advice(goal_id, goal_title)
        st.markdown(advice)
```

### Step 6: Run

```bash
streamlit run app.py
```

---

## Extension Ideas

1. **Anki sync** — export/import cards from Anki for existing decks
2. **Resource integration** — auto-fetch YouTube videos and articles for each topic
3. **Study streak tracking** — GitHub-style contribution calendar for study activity
4. **Exam simulator** — timed exam mode with randomized questions
5. **Social study groups** — share decks with friends and compare performance

---

## What to Learn Next

- **AI agents** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
- **Structured outputs** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
