---
title: "Build a Multi-Agent Research System: Parallel AI Agents for Deep Research"
description: "Build a multi-agent system where a Planner agent breaks research questions into sub-tasks, Researcher agents gather information in parallel, and a Synthesizer agent produces a comprehensive report."
date: "2026-03-10"
slug: "multi-agent-research-system"
level: "Advanced"
time: "8–10 hours"
stack: "Python, OpenAI API, asyncio, Tavily Search API, Streamlit"
keywords: ["multi-agent system Python", "AI research agent", "LLM agent orchestration"]
---

## Project Overview

A multi-agent research pipeline: a Planner decomposes complex questions into research sub-tasks, multiple Researcher agents execute them in parallel (web search + synthesis), and a Synthesizer agent produces a structured research report with citations.

---

## Learning Goals

- Design agent roles and inter-agent communication
- Use asyncio for parallel agent execution
- Implement tool use (web search) in agents
- Orchestrate multi-step agentic workflows

---

## Architecture

```
User: "Research question..."
        ↓
Planner Agent
  → breaks into 4-6 sub-questions
        ↓ (parallel)
Researcher Agents (one per sub-question)
  → web search + synthesize findings
        ↓
Synthesizer Agent
  → combine all findings + citations
        ↓
Structured report (Markdown)
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai tavily-python streamlit asyncio
```

Get a free Tavily API key at https://tavily.com for web search.

### Step 2: Tool Definitions

```python
# tools.py
import os
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return structured results."""
    try:
        results = tavily.search(query=query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        return [{"error": str(e)}]


def format_search_results(results: list[dict]) -> str:
    if not results or "error" in results[0]:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"**{r['title']}** ({r['url']})\n{r['content']}")
    return "\n\n---\n\n".join(lines)
```

### Step 3: Agent Definitions

```python
# agents.py
import json
import asyncio
from openai import AsyncOpenAI
from tools import web_search, format_search_results

client = AsyncOpenAI()

# --- Planner Agent ---

PLANNER_PROMPT = """You are a research planner. Break down this research question into specific sub-questions that together will provide a comprehensive answer.

Rules:
- Generate 4-6 focused sub-questions
- Each sub-question should be independently researchable
- Cover different aspects: background, current state, examples, implications, future trends
- Make sub-questions specific enough for targeted web searches

Return JSON:
{{
  "sub_questions": [
    {{"id": 1, "question": "...", "focus": "background|current|examples|implications|trends"}}
  ],
  "research_approach": "brief strategy note"
}}

Main question: {question}"""


async def plan_research(question: str) -> dict:
    print(f"[Planner] Decomposing: {question[:60]}...")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PLANNER_PROMPT.format(question=question)}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)


# --- Researcher Agent ---

RESEARCHER_PROMPT = """You are a research agent. Your job is to answer a specific research question by searching the web and synthesizing findings.

Research question: {question}
Context (main topic): {context}

Search results:
{search_results}

Write a comprehensive answer to the research question based on the search results.
Include specific facts, data points, and examples. Note the sources.
Format as: key finding, supporting evidence, sources."""


async def research_sub_question(sub_q: dict, main_question: str) -> dict:
    question = sub_q["question"]
    print(f"[Researcher-{sub_q['id']}] Researching: {question[:50]}...")

    # Search for information
    results = web_search(question)
    formatted = format_search_results(results)

    # Synthesize findings
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": RESEARCHER_PROMPT.format(
            question=question,
            context=main_question,
            search_results=formatted,
        )}],
        max_tokens=600,
        temperature=0.3,
    )
    findings = response.choices[0].message.content
    sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results if "url" in r]

    return {
        "sub_question_id": sub_q["id"],
        "sub_question": question,
        "focus": sub_q.get("focus", ""),
        "findings": findings,
        "sources": sources,
    }


# --- Synthesizer Agent ---

SYNTHESIZER_PROMPT = """You are a research synthesizer. Combine these research findings into a comprehensive, well-structured report.

Main question: {question}

Research findings:
{findings_text}

Write a complete research report with:
## Executive Summary
(3-4 sentences covering the key answer)

## Background
(Context and definitions)

## Key Findings
(Numbered list of the most important findings with evidence)

## Analysis
(Connect findings, identify patterns, contradictions)

## Conclusion
(Direct answer to the main question + implications)

## Sources
(List all cited sources)

Use clear headings, be specific, cite sources as [Source Title](URL)."""


async def synthesize_report(question: str, research_results: list[dict]) -> str:
    print("[Synthesizer] Generating final report...")
    findings_parts = []
    for r in research_results:
        findings_parts.append(f"### Sub-question {r['sub_question_id']}: {r['sub_question']}\n{r['findings']}")

    all_sources = []
    for r in research_results:
        all_sources.extend(r.get("sources", []))

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SYNTHESIZER_PROMPT.format(
            question=question,
            findings_text="\n\n---\n\n".join(findings_parts),
        )}],
        max_tokens=2000,
        temperature=0.4,
    )
    return response.choices[0].message.content
```

### Step 4: Orchestrator

```python
# orchestrator.py
import asyncio
from agents import plan_research, research_sub_question, synthesize_report


async def run_research(question: str) -> dict:
    """Full multi-agent research pipeline."""
    print(f"\n{'='*60}")
    print(f"Research: {question}")
    print('='*60)

    # Step 1: Plan
    plan = await plan_research(question)
    sub_questions = plan["sub_questions"]
    print(f"\n[Planner] Created {len(sub_questions)} sub-questions")
    for sq in sub_questions:
        print(f"  {sq['id']}. {sq['question']}")

    # Step 2: Research in parallel
    print(f"\n[Orchestrator] Starting {len(sub_questions)} researcher agents in parallel...")
    research_tasks = [research_sub_question(sq, question) for sq in sub_questions]
    research_results = await asyncio.gather(*research_tasks)
    print(f"\n[Orchestrator] All researchers complete")

    # Step 3: Synthesize
    report = await synthesize_report(question, list(research_results))

    return {
        "question": question,
        "plan": plan,
        "research_results": list(research_results),
        "report": report,
    }


def research(question: str) -> dict:
    """Synchronous wrapper for the async pipeline."""
    return asyncio.run(run_research(question))
```

### Step 5: Streamlit App

```python
# app.py
import streamlit as st
from orchestrator import run_research
import asyncio

st.set_page_config(page_title="Multi-Agent Research", page_icon="🔬", layout="wide")
st.title("🔬 Multi-Agent Research System")
st.caption("Parallel AI agents research complex topics comprehensively")

question = st.text_area(
    "Research question",
    placeholder="What are the current limitations of RAG systems and what approaches are being used to overcome them?",
    height=100,
)

depth = st.select_slider("Research depth", options=["Quick (3 agents)", "Standard (5 agents)", "Deep (7 agents)"], value="Standard (5 agents)")
n_agents = int(depth.split("(")[1].split(" ")[0])

if st.button("Start Research", type="primary") and question:
    progress = st.progress(0, "Initializing agents...")

    async def run_with_progress():
        progress.progress(10, "Planning research...")
        plan = await __import__("agents").plan_research(question)
        sub_questions = plan["sub_questions"][:n_agents]

        progress.progress(30, f"Researching {len(sub_questions)} sub-questions in parallel...")
        from agents import research_sub_question
        tasks = [research_sub_question(sq, question) for sq in sub_questions]
        results = await asyncio.gather(*tasks)

        progress.progress(80, "Synthesizing report...")
        from agents import synthesize_report
        report = await synthesize_report(question, list(results))
        progress.progress(100, "Complete!")
        return plan, list(results), report

    plan, results, report = asyncio.run(run_with_progress())

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Research Report")
        st.markdown(report)
        st.download_button("Download Report (.md)", report, "research_report.md", "text/markdown")
    with col2:
        st.subheader("Research Plan")
        for sq in plan["sub_questions"]:
            st.markdown(f"**{sq['id']}.** {sq['question']}")
        st.caption(plan.get("research_approach", ""))
```

### Step 6: Run

```bash
export OPENAI_API_KEY=your-key
export TAVILY_API_KEY=your-key

streamlit run app.py

# Or use directly:
python -c "
from orchestrator import research
result = research('How does mixture of experts work in LLMs?')
print(result['report'])
"
```

---

## Extension Ideas

1. **Critic agent** — add a fact-checking agent that validates claims in the report
2. **Memory agent** — cache previous research results to avoid redundant searches
3. **Debate mode** — two researcher agents with opposing viewpoints synthesized by arbitrator
4. **Source scoring** — rank source reliability (academic > news > blog)
5. **Iterative research** — Planner reviews findings and spawns additional sub-questions

---

## What to Learn Next

- **Agent fundamentals** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
- **Tool use** → [Tool Use and Function Calling](/blog/tool-use-and-function-calling/)
- **Multi-agent theory** → [Multi-Agent Systems](/blog/multi-agent-systems/)
