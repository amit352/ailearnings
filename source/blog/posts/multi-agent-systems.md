---
title: "Multi-Agent Systems: Architectures for Coordinating LLM Agents"
description: "Build multi-agent AI systems — orchestrator-worker patterns, agent handoffs, shared memory, debate patterns, and production-ready multi-agent orchestration with LangGraph."
date: "2026-03-10"
slug: "multi-agent-systems"
keywords: ["multi-agent systems", "multi-agent AI", "LangGraph agents", "AI agent orchestration"]
---

## Learning Objectives

- Understand when multi-agent systems are better than single agents
- Implement orchestrator-worker, pipeline, and debate patterns
- Use LangGraph for stateful multi-agent workflows
- Handle agent failures and coordination errors
- Design safe, auditable multi-agent systems

---

## Why Multi-Agent?

Single agents struggle with:
- **Long tasks** — context gets full; quality degrades
- **Parallel work** — one agent does everything sequentially
- **Specialization** — one agent with many tools is confused about which to use
- **Verification** — the same agent can't reliably critique its own work

Multi-agent systems solve these by distributing work across specialized agents.

---

## Core Patterns

### 1. Orchestrator-Worker

A controller agent breaks down tasks and delegates to specialized worker agents.

```python
from openai import OpenAI
import json

client = OpenAI()

# Specialized worker agents
def research_agent(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research specialist. Provide factual, detailed information."},
            {"role": "user",   "content": f"Research this topic thoroughly: {topic}"},
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def writer_agent(research: str, style: str = "technical") -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a {style} writer. Write clear, engaging content."},
            {"role": "user",   "content": f"Write an article based on this research:\n\n{research}"},
        ],
        max_tokens=1500,
    )
    return response.choices[0].message.content


def editor_agent(draft: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an editor. Improve clarity, fix errors, ensure consistency."},
            {"role": "user",   "content": f"Edit and improve this draft:\n\n{draft}"},
        ],
        max_tokens=1500,
    )
    return response.choices[0].message.content


# Orchestrator
def orchestrate(task: str) -> dict:
    print(f"Orchestrator: Breaking down task: {task}")

    # Step 1: Research
    print("  → Research agent working...")
    research = research_agent(task)

    # Step 2: Write
    print("  → Writer agent working...")
    draft = writer_agent(research)

    # Step 3: Edit
    print("  → Editor agent reviewing...")
    final = editor_agent(draft)

    return {
        "research": research,
        "draft": draft,
        "final": final,
    }


result = orchestrate("Explain how Mixture of Experts (MoE) works in LLMs")
print(result["final"])
```

### 2. Pipeline Pattern

Agents form a linear chain, each processing the previous agent's output.

```python
from dataclasses import dataclass

@dataclass
class PipelineState:
    raw_input: str
    extracted_data: dict = None
    validated_data: dict = None
    transformed_data: dict = None
    final_output: str = None


def extraction_agent(state: PipelineState) -> PipelineState:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract structured data from text. Return JSON only."},
            {"role": "user",   "content": f"Extract key information:\n{state.raw_input}"},
        ],
        response_format={"type": "json_object"},
    )
    state.extracted_data = json.loads(response.choices[0].message.content)
    return state


def validation_agent(state: PipelineState) -> PipelineState:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Validate this data. Return {valid: bool, errors: [], cleaned: {}}"},
            {"role": "user",   "content": json.dumps(state.extracted_data)},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    state.validated_data = result.get("cleaned", state.extracted_data)
    return state


def formatting_agent(state: PipelineState) -> PipelineState:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Format this data into a professional summary."},
            {"role": "user",   "content": json.dumps(state.validated_data)},
        ],
    )
    state.final_output = response.choices[0].message.content
    return state


def run_pipeline(raw_text: str) -> str:
    state = PipelineState(raw_input=raw_text)
    for agent_fn in [extraction_agent, validation_agent, formatting_agent]:
        state = agent_fn(state)
    return state.final_output
```

### 3. Debate Pattern

Two agents argue opposing positions, then a judge synthesizes the best answer.

```python
def debate_pattern(question: str) -> str:
    # Agent A argues one position
    pos_a = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Present the strongest case FOR the following position."},
            {"role": "user",   "content": question},
        ],
    ).choices[0].message.content

    # Agent B argues against
    pos_b = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Present the strongest case AGAINST the following position."},
            {"role": "user",   "content": question},
        ],
    ).choices[0].message.content

    # Judge synthesizes
    verdict = client.chat.completions.create(
        model="gpt-4o",  # use stronger model for synthesis
        messages=[
            {"role": "system", "content": "Synthesize both perspectives into a balanced, well-reasoned conclusion."},
            {"role": "user",   "content": f"Question: {question}\n\nPro:\n{pos_a}\n\nCon:\n{pos_b}"},
        ],
    ).choices[0].message.content

    return verdict


print(debate_pattern("Should AI systems be given autonomous decision-making authority in healthcare?"))
```

---

## LangGraph: Stateful Multi-Agent Workflows

LangGraph models agent workflows as directed graphs with shared state.

```bash
pip install langgraph langchain-openai
```

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List
import operator

# Define shared state
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # messages accumulate
    current_agent: str
    task_complete: bool
    result: str


llm = ChatOpenAI(model="gpt-4o-mini")

# Agent nodes
def researcher_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm.invoke([
        HumanMessage(content=f"Research this topic and provide key facts: {messages[-1].content}")
    ])
    return {
        "messages": [AIMessage(content=f"[Researcher]: {response.content}")],
        "current_agent": "writer",
        "task_complete": False,
        "result": "",
    }


def writer_node(state: AgentState) -> AgentState:
    # Get researcher output
    research = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"Write a concise summary based on this research:\n{research}")
    ])
    return {
        "messages": [AIMessage(content=f"[Writer]: {response.content}")],
        "current_agent": "done",
        "task_complete": True,
        "result": response.content,
    }


# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# Define edges
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="Explain vector databases")],
    "current_agent": "researcher",
    "task_complete": False,
    "result": "",
})

print(result["result"])
```

### Conditional Routing in LangGraph

```python
def should_continue(state: AgentState) -> str:
    """Route to next agent based on state."""
    if state["task_complete"]:
        return "end"
    elif state["current_agent"] == "writer":
        return "writer"
    else:
        return "researcher"


workflow.add_conditional_edges(
    "researcher",
    should_continue,
    {
        "writer": "writer",
        "end": END,
    }
)
```

---

## Shared Memory and Communication

```python
from threading import Lock
from collections import defaultdict

class SharedMemory:
    """Thread-safe shared memory for multi-agent systems."""
    def __init__(self):
        self._store = defaultdict(list)
        self._lock = Lock()

    def write(self, key: str, value: any):
        with self._lock:
            self._store[key].append(value)

    def read(self, key: str) -> list:
        with self._lock:
            return list(self._store[key])

    def read_latest(self, key: str) -> any:
        with self._lock:
            items = self._store[key]
            return items[-1] if items else None


memory = SharedMemory()

def agent_1(task: str):
    result = f"Agent 1 processed: {task}"
    memory.write("task_results", result)
    return result

def agent_2():
    previous = memory.read("task_results")
    return f"Agent 2 sees: {previous}"
```

---

## Error Handling and Reliability

```python
import time
from functools import wraps

def with_retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(agent_fn):
        @wraps(agent_fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return agent_fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Agent {agent_fn.__name__} failed (attempt {attempt+1}): {e}")
                    time.sleep(delay * (2 ** attempt))
        return wrapper
    return decorator


@with_retry(max_retries=3)
def reliable_agent(task: str) -> str:
    return research_agent(task)
```

---

## Safety and Guardrails

```python
BLOCKED_ACTIONS = ["delete_database", "send_mass_email", "deploy_to_production"]

def safe_execute_tool(name: str, args: dict) -> str:
    # Blocklist check
    if name in BLOCKED_ACTIONS:
        return json.dumps({"error": f"Tool '{name}' is blocked for safety reasons."})

    # Require confirmation for destructive actions
    destructive = ["delete", "remove", "drop", "truncate"]
    if any(d in name.lower() for d in destructive):
        print(f"⚠️  Destructive action requested: {name}({args})")
        confirm = input("Confirm? (yes/no): ")
        if confirm.lower() != "yes":
            return json.dumps({"error": "Action cancelled by user."})

    return execute_tool(name, args)
```

---

## Troubleshooting

**Agents contradict each other**
- Use a judge/synthesis agent to resolve conflicts
- Define clear authority hierarchy — orchestrator has final say
- Use structured outputs to enforce consistent formats

**Infinite loops**
- Always set `max_iterations` or use LangGraph's `recursion_limit`
- Add a termination condition checker as a separate node
- Log agent state transitions for debugging

**High costs from multi-agent systems**
- Use `gpt-4o-mini` for worker agents; `gpt-4o` only for orchestration/synthesis
- Cache repeated sub-tasks
- Use smaller models for simple routing decisions

---

## FAQ

**How many agents is too many?**
Start with 2–3. Each agent adds latency, cost, and failure points. Add agents only when a single agent demonstrably fails at the task.

**When should I use LangGraph vs a custom loop?**
LangGraph for complex graphs with branching, cycles, and persistent state. Custom loops for simple linear or orchestrator-worker patterns.

---

## What to Learn Next

- **Single agent fundamentals** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
- **Tool use** → [Tool Use and Function Calling](/blog/tool-use-and-function-calling/)
- **Deploy agents** → [Deploying AI Applications](/blog/deploying-ai-applications/)
