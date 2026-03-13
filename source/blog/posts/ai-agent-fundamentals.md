---
title: "AI Agent Fundamentals: How LLM Agents Think, Plan, and Act"
description: "Learn the building blocks of AI agents — ReAct loops, tool use, planning, memory, and multi-agent coordination. Includes hands-on examples with LangChain and OpenAI."
date: "2026-03-10"
slug: "ai-agent-fundamentals"
keywords: ["AI agents", "LLM agents", "AI agent tutorial", "ReAct agent", "agentic AI"]
---

## Learning Objectives

- Understand what an AI agent is and how it differs from a chatbot
- Implement the ReAct (Reasoning + Acting) loop
- Give agents tools and handle tool call results
- Add memory so agents can reference prior context
- Design multi-step task flows

---

## What Is an AI Agent?

A chatbot generates a single response. An **AI agent** takes actions in a loop — reasoning about what to do, using tools, observing results, and repeating until a goal is achieved.

```
Chatbot:    User → LLM → Response
Agent:      User → [LLM → Tool → Observe → LLM → Tool → Observe → ...] → Final Answer
```

The key difference: an agent can use external tools (web search, code execution, APIs, databases) and decide autonomously which tools to use and in what order.

---

## The ReAct Pattern

ReAct (Reasoning + Acting) interleaves reasoning traces with tool calls:

```
Thought: I need to find the current price of Bitcoin.
Action: search("Bitcoin price today")
Observation: Bitcoin is trading at $67,432 as of March 10, 2026.
Thought: I have the price. Now I can answer.
Action: finish("The current Bitcoin price is $67,432.")
```

This pattern dramatically improves reliability on multi-step tasks compared to single-shot prompting.

---

## Agent Loop from Scratch

```python
from openai import OpenAI
import json

client = OpenAI()

# ── Tool definitions ──────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Mock weather tool."""
    weather_data = {
        "london": "15°C, cloudy",
        "tokyo": "22°C, sunny",
        "new york": "8°C, rainy",
    }
    return weather_data.get(city.lower(), f"No data for {city}")

TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
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
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    },
]

# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
        {"role": "user",   "content": user_message},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        # No tool call — final answer
        if not msg.tool_calls:
            return msg.content

        # Execute all tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if name in TOOLS:
                result = TOOLS[name](**args)
            else:
                result = f"Unknown tool: {name}"

            print(f"  [Tool] {name}({args}) → {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Max iterations reached."


# Test
print(run_agent("What's 15% of 847?"))
print(run_agent("Is it a good day to visit Tokyo? What's the weather like?"))
print(run_agent("If I spend 2 hours commuting and 8 hours working, how many hours are left in my day?"))
```

---

## Agent Memory

Without memory, agents can't reference earlier conversations. Add memory by maintaining a message history.

### Short-Term Memory (In-Context)

```python
class AgentWithMemory:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def run(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            self.messages.append(msg)

            if not msg.tool_calls:
                return msg.content

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = TOOLS.get(name, lambda **k: "Unknown tool")(**args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

    def clear_memory(self):
        self.messages = [self.messages[0]]  # keep system prompt


agent = AgentWithMemory("You are a helpful research assistant.")
print(agent.run("What's 15% of 850?"))
print(agent.run("What was the number I just asked about?"))  # remembers 850
```

### Long-Term Memory (External Store)

For persistent memory across sessions, store important facts in a vector database:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class MemoryStore:
    def __init__(self):
        self.db = Chroma(embedding_function=OpenAIEmbeddings())

    def remember(self, fact: str):
        self.db.add_texts([fact])

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        results = self.db.similarity_search(query, k=top_k)
        return [r.page_content for r in results]

memory = MemoryStore()
memory.remember("User prefers Python over JavaScript.")
memory.remember("User is building a RAG system for legal documents.")

recalled = memory.recall("What is the user's project?")
print(recalled)  # ["User is building a RAG system for legal documents."]
```

---

## Planning: Breaking Down Complex Tasks

For complex, multi-step tasks, have the agent create a plan before executing:

```python
def plan_and_execute(task: str) -> str:
    # Step 1: Generate a plan
    plan_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """Break down the task into a numbered list of concrete steps.
Each step should be independently executable. Be specific."""},
            {"role": "user", "content": f"Task: {task}"},
        ],
    )
    plan = plan_response.choices[0].message.content
    print(f"Plan:\n{plan}\n")

    # Step 2: Execute each step
    results = []
    agent = AgentWithMemory("You are executing a step in a larger task. Be concise and precise.")

    for line in plan.split('\n'):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        step = line.split('.', 1)[-1].strip()
        print(f"Executing: {step}")
        result = agent.run(step)
        results.append(f"{step}: {result}")

    # Step 3: Synthesize results
    synthesis = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Synthesize the step results into a coherent final answer."},
            {"role": "user", "content": f"Original task: {task}\n\nResults:\n" + "\n".join(results)},
        ],
    )
    return synthesis.choices[0].message.content
```

---

## Common Agent Patterns

### Router Agent
Routes queries to specialized sub-agents:

```python
AGENTS = {
    "math":    AgentWithMemory("You are a math expert."),
    "weather": AgentWithMemory("You are a weather assistant."),
    "general": AgentWithMemory("You are a general assistant."),
}

def route(query: str) -> str:
    router_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Route the query to one of: {list(AGENTS.keys())}. Reply with just the agent name."},
            {"role": "user",   "content": query},
        ],
    )
    agent_name = router_response.choices[0].message.content.strip().lower()
    agent = AGENTS.get(agent_name, AGENTS["general"])
    return agent.run(query)
```

### Reflection Pattern
Agent evaluates its own output and improves it:

```python
def agent_with_reflection(task: str) -> str:
    agent = AgentWithMemory("You are a careful assistant. Think step by step.")

    # First attempt
    response = agent.run(task)

    # Self-critique
    critique = agent.run(f"""
Review your previous response and identify:
1. Any errors or inaccuracies
2. Missing important information
3. Ways to make it clearer

Previous response: {response}
""")

    # Refined answer
    refined = agent.run("Based on your critique, provide an improved final answer.")
    return refined
```

---

## Troubleshooting

**Agent loops forever**
- Set `max_iterations` and return a graceful message when exceeded
- Add a "finish" tool that the agent must call when done

**Agent calls the wrong tool**
- Improve tool descriptions — be explicit about when to use each tool
- Add examples in the tool description
- Reduce the number of tools (fewer choices = clearer decisions)

**Agent ignores available tools**
- Set `tool_choice="required"` to force tool use when appropriate
- Move tool descriptions earlier in the system prompt

---

## FAQ

**What is the difference between an AI agent and a chatbot?**
A chatbot responds once. An agent acts in a loop, uses tools, and can autonomously complete multi-step tasks.

**When should I use agents vs standard RAG?**
Use RAG when you need to answer questions about documents. Use agents when you need to take actions, use multiple tools, or complete tasks that require multi-step planning.

**Are agents reliable enough for production?**
It depends on task complexity. Well-defined, narrow tasks (e.g., "look up a customer record and draft an email") can be production-ready. Open-ended, complex tasks have higher failure rates and need human oversight.

---

## What to Learn Next

- **LangChain agents** → langchain-agents-tutorial
- **Multi-agent systems** → multi-agent-systems
- **Tool use and function calling** → tool-use-and-function-calling
- **AI agent projects** → [AI Projects for Developers](/ai-projects/)
