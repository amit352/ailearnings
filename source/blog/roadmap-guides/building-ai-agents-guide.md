---
title: "Building AI Agents: ReAct, Tool Use, and Agentic Workflows"
description: "A practical guide to building AI agents that can reason and act — covering the ReAct pattern, tool definition, agent loops, memory strategies, and multi-step task execution with OpenAI function calling."
date: "2026-03-10"
slug: "building-ai-agents-guide"
keywords: ["building AI agents Python", "ReAct agent tutorial", "LLM function calling agent"]
---

## What Makes an LLM an "Agent"?

A chatbot responds to a single message. An agent takes a goal, decides what actions to take, executes them, observes results, and repeats until the goal is achieved:

```
Goal: "Find the current price of Apple stock and compare to last month"

Step 1: Think → "I need to search for AAPL current price"
Step 2: Act  → search("AAPL stock price today")
Step 3: Observe → "$189.50 as of March 10, 2026"
Step 4: Think → "Now I need last month's price"
Step 5: Act  → search("AAPL stock price February 2026")
Step 6: Observe → "$182.30 as of February 10, 2026"
Step 7: Think → "I have both values, can now compare"
Step 8: Respond → "Apple stock is $189.50 (+3.9% from last month's $182.30)"
```

This Think → Act → Observe loop is called **ReAct** (Reasoning + Acting).

---

## Building a ReAct Agent from Scratch

```python
from openai import OpenAI
import json

client = OpenAI()

# ── Tool Definitions ─────────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """Search the web. In production, use Tavily or SerpAPI."""
    # Placeholder — integrate real search API here
    return f"[Search results for: {query}] (integrate real search API)"

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    """Read a local file."""
    from pathlib import Path
    try:
        return Path(path).read_text(encoding="utf-8")[:3000]
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    from pathlib import Path
    Path(path).write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {path}"


TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "read_file": read_file,
    "write_file": write_file,
}

# ── OpenAI Tool Schemas ───────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, facts, news, or prices",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions: '2 + 2', '100 * 1.07'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a local file by path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

# ── Agent Loop ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI agent with access to tools.
Use tools to gather information and complete tasks accurately.
Think step by step. Be explicit about your reasoning before calling a tool.
When you have enough information to answer, do so directly without further tool calls."""


def run_agent(task: str, max_steps: int = 10, verbose: bool = True) -> str:
    """Run the agent loop until task is complete or max_steps reached."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):
        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # If no tool calls, agent has finished
        if not msg.tool_calls:
            if verbose:
                print(f"\n[Final Answer] {msg.content}")
            return msg.content

        # Execute each tool call
        messages.append(msg)  # add assistant message with tool_calls

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n[Step {step+1}] Tool: {fn_name}({fn_args})")

            # Execute the tool
            if fn_name in TOOLS:
                result = TOOLS[fn_name](**fn_args)
            else:
                result = f"Error: unknown tool '{fn_name}'"

            if verbose:
                print(f"[Result] {str(result)[:200]}")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return "Max steps reached without completion."


# Example
result = run_agent("What is 15% of 847, and what is 847 + that amount?")
```

---

## Agent Memory Strategies

### 1. Short-Term (Conversation) Memory

```python
class ConversationAgent:
    def __init__(self, system_prompt: str, max_history: int = 20):
        self.history = [{"role": "system", "content": system_prompt}]
        self.max_history = max_history

    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        # Trim if too long (keep system prompt + last N turns)
        if len(self.history) > self.max_history + 1:
            self.history = [self.history[0]] + self.history[-(self.max_history):]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        # ... handle tool calls ...
        assistant_msg = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg
```

### 2. External Memory (Semantic Memory)

```python
import chromadb
from datetime import datetime

class MemoryAgent:
    def __init__(self):
        self.chroma = chromadb.PersistentClient("./agent_memory")
        self.memory = self.chroma.get_or_create_collection("memories")
        self.turn_count = 0

    def remember(self, content: str, memory_type: str = "conversation"):
        """Store a memory."""
        embedding = embed(content)  # your embed function
        self.memory.add(
            ids=[f"mem_{self.turn_count}_{datetime.now().timestamp()}"],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"type": memory_type, "timestamp": datetime.now().isoformat()}],
        )
        self.turn_count += 1

    def recall(self, query: str, n: int = 3) -> list[str]:
        """Retrieve relevant memories."""
        results = self.memory.query(
            query_embeddings=[embed(query)],
            n_results=n,
        )
        return results["documents"][0]

    def chat(self, message: str) -> str:
        memories = self.recall(message)
        context = "\n".join(f"- {m}" for m in memories) if memories else "No relevant memories."

        messages = [
            {"role": "system", "content": f"Relevant memories:\n{context}"},
            {"role": "user", "content": message},
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        answer = response.choices[0].message.content
        self.remember(f"User: {message}\nAgent: {answer}")
        return answer
```

---

## Task Planning: Breaking Down Complex Goals

```python
PLANNER_PROMPT = """Break down this complex task into specific, sequential steps.

Task: {task}

Return JSON:
{{
  "steps": [
    {{"step": 1, "action": "specific action to take", "tool": "tool_name or null"}},
    ...
  ],
  "estimated_complexity": "simple | moderate | complex"
}}"""


def plan_and_execute(task: str) -> str:
    """Plan a task then execute each step."""
    # Plan
    plan_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PLANNER_PROMPT.format(task=task)}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    plan = json.loads(plan_resp.choices[0].message.content)

    print(f"Plan ({plan['estimated_complexity']}):")
    for s in plan["steps"]:
        print(f"  Step {s['step']}: {s['action']}")

    # Execute as a single agent run with the plan as context
    enriched_task = f"{task}\n\nPlan to follow:\n" + "\n".join(
        f"{s['step']}. {s['action']}" for s in plan["steps"]
    )
    return run_agent(enriched_task)
```

---

## Common Agent Patterns

### Code-Writing Agent

```python
def coding_agent(task: str) -> str:
    """Agent that writes and tests code."""
    tools_with_exec = TOOL_SCHEMAS + [{
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return output",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }]

    def execute_python(code: str) -> str:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout or result.stderr

    local_tools = dict(TOOLS, execute_python=execute_python)
    # Run agent with extended tools...
```

### Guard Rails

```python
BLOCKED_TOOLS = {"delete_database", "send_email", "deploy_production"}

def safe_tool_call(fn_name: str, fn_args: dict) -> str:
    if fn_name in BLOCKED_TOOLS:
        return f"Tool '{fn_name}' is not allowed without explicit authorization."
    if fn_name not in TOOLS:
        return f"Unknown tool: {fn_name}"

    # Validate arguments
    if fn_name == "write_file" and ".." in fn_args.get("path", ""):
        return "Error: path traversal not allowed"

    return TOOLS[fn_name](**fn_args)
```

---

## What to Learn Next

- **AI agent evaluation** → [AI Agent Evaluation](/blog/roadmap-guides/ai-agent-evaluation/)
- **Multi-agent systems** → [Multi-Agent Systems](/blog/multi-agent-systems/)
- **Build an agent project** → [Multi-Agent Research System](/projects/multi-agent-research-system/)
