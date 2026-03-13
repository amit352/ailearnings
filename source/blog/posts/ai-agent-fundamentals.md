---
title: "AI Agent Fundamentals: How LLM Agents Think, Plan, and Act"
description: "Learn the building blocks of AI agents — ReAct loops, tool use, planning, memory, and multi-agent coordination. Includes hands-on examples with LangChain and OpenAI."
date: "2026-03-10"
slug: "ai-agent-fundamentals"
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
keywords: ["AI agents", "LLM agents", "AI agent tutorial", "ReAct agent", "agentic AI"]
---

# AI Agent Fundamentals: How LLM Agents Think, Plan, and Act

You have built a chatbot that answers questions, but your users keep asking it to do things — check a price, look up a record, run a calculation. A chatbot gives one response. An **AI agent** takes action: it reasons about what to do, calls a tool, observes the result, and continues until the task is complete. This guide walks through the building blocks you need to build reliable agents.

---

## What Is an AI Agent?

A chatbot generates a single response and stops. An **AI agent** operates in a loop — reasoning about what to do next, using tools to take action, observing the result, and repeating until a goal is achieved.

```
Chatbot:    User → LLM → Response
Agent:      User → [LLM → Tool → Observe → LLM → Tool → Observe → ...] → Final Answer
```

The key difference: an agent can use external tools (web search, code execution, APIs, databases) and decides autonomously which tools to use and in what order. The agent is not executing a fixed script — it is reasoning through the problem in real time.

A minimal agent has three components:
- An **LLM** — the reasoning engine
- A set of **tools** — functions the LLM can call
- An **agent loop** — the runtime that orchestrates reasoning and execution

---

## Why AI Agent Fundamentals Matter for Developers

Understanding agent architecture at the component level lets you build agents that are reliable, debuggable, and scoped correctly. Developers who treat agents as black boxes struggle to diagnose failures, tune behavior, and control costs.

Knowing the fundamentals helps you:
- Decide when to use an agent versus a simpler chain
- Design tools that the agent uses correctly
- Set appropriate stopping conditions and error handling
- Understand why an agent chose a particular path
- Build memory systems that scale appropriately

For most real-world tasks — research, analysis, data lookup, automation — well-designed agents with focused tools outperform elaborate prompts against a single model call.

---

## The ReAct Pattern

**ReAct** (Reasoning + Acting) is the most common agent architecture. It interleaves explicit reasoning traces with tool calls:

```
Thought: I need to find the current price of Bitcoin.
Action: search("Bitcoin price today")
Observation: Bitcoin is trading at $67,432 as of March 2026.
Thought: I have the price. I can now answer.
Final Answer: The current Bitcoin price is $67,432.
```

The model generates the Thought step to reason about what it needs. The Action step calls a tool. The runtime executes the tool and injects the result as an Observation. The model then reasons again. This loop continues until it produces a Final Answer.

This pattern dramatically improves reliability on multi-step tasks compared to asking for the answer in a single prompt.

---

## Practical Example

### Building an Agent Loop from Scratch

```python
from openai import OpenAI
import json

client = OpenAI()

# Tool implementations
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Return current weather for a city (mock)."""
    data = {
        "london": "15°C, cloudy",
        "tokyo": "22°C, sunny",
        "new york": "8°C, rainy",
    }
    return data.get(city.lower(), f"No data for {city}")

TOOLS = {"calculator": calculator, "get_weather": get_weather}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for any arithmetic or numeric calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Valid Python math expression"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Use when the user asks about weather conditions.",
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

def run_agent(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed to answer accurately."},
        {"role": "user", "content": user_message},
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

        # No tool call — this is the final answer
        if not msg.tool_calls:
            return msg.content

        # Execute all requested tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = TOOLS.get(name, lambda **k: "Unknown tool")(**args)
            print(f"  [Tool] {name}({args}) → {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Max iterations reached without a final answer."

# Tests
print(run_agent("What's 15% of 847?"))
print(run_agent("Is it warm in Tokyo today?"))
print(run_agent("If I spend 2 hours commuting and 8 hours working, how many hours are left?"))
```

### Adding Short-Term Memory

```python
class AgentWithMemory:
    """Agent that remembers conversation history within a session."""

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
        self.messages = [self.messages[0]]  # keep system prompt only

agent = AgentWithMemory("You are a helpful research assistant.")
print(agent.run("What's 15% of 850?"))
print(agent.run("What number did I just ask about?"))  # should recall 850
```

---

## Real-World Applications

AI agents are the foundation of several categories of production AI systems:

**Research assistants** — Search the web, read pages, and synthesize findings into structured reports. The agent decides which sources are authoritative and when it has gathered enough information.

**Code assistants** — Read files, write code, run tests, observe the test output, and iterate. The loop continues until tests pass or the agent declares that it cannot solve the problem.

**Customer support automation** — Look up customer records in a CRM, check order status via an API, and compose a response grounded in real data. Reduces the hallucination risk of pure generation.

**Data analysis agents** — Query databases, compute statistics, generate charts, and summarize findings. The agent routes numerical questions to a code execution tool rather than reasoning about them.

**Document processing** — Extract structured fields, validate against schemas, flag exceptions for human review. Each step uses the right tool rather than relying on a single model call.

---

## Common Mistakes Developers Make

1. **No stopping condition** — Without a maximum iteration limit, a stuck agent runs forever and accumulates cost. Always set `max_iterations` and return a graceful message when exceeded. Also consider wall-clock timeouts.

2. **Trusting all tool outputs blindly** — Tools return strings. Malicious or malformed data in tool outputs can influence agent behavior. Validate tool outputs before injecting them back into the agent's context.

3. **Overly broad tool descriptions** — The agent selects tools based on their description. Vague descriptions like "does stuff with data" cause the agent to select the wrong tool. Be precise about what each tool does and when to use it.

4. **No error handling in tools** — Tools that raise unhandled exceptions crash the agent loop. Wrap every tool in try/except and return informative error strings instead of propagating exceptions.

5. **Using agents for simple tasks** — If the steps are fixed and known in advance, use a chain. Agents add latency from multiple LLM calls and introduce unpredictability. Match the pattern to the task complexity.

---

## Best Practices

- **Start with two or three tools** — Validate the core loop with minimal tools before adding complexity. More tools create more decision points and more places to fail.
- **Log every tool call and result** — In production, log every tool invocation including inputs, outputs, and timing. Agent behavior is nearly impossible to debug without a complete trace.
- **Test adversarial inputs** — Users will try to manipulate agents. Test for prompt injection, unexpected input types, and edge cases that cause tools to return errors.
- **Give agents clear stopping criteria** — Describe in the system prompt when the agent should consider the task complete and return a final answer.
- **Use human-in-the-loop for high-stakes actions** — For irreversible actions (sending emails, executing code that modifies data), require human confirmation before proceeding.

---

## FAQ

**What is the difference between an AI agent and a chatbot?**
A chatbot responds once per user message. An agent acts in a loop across multiple steps, using tools to gather information and take actions autonomously.

**When should I use agents vs standard RAG?**
Use RAG when you need to answer questions about documents. Use agents when you need to take actions, use multiple tools in flexible combinations, or complete tasks requiring multi-step planning.

**Are agents reliable enough for production?**
It depends on task complexity and how narrowly scoped the tools are. Well-defined tasks (look up a record, draft an email) can be production-ready. Open-ended tasks have higher failure rates and need human oversight or fallback mechanisms.

**How do I prevent an agent from looping forever?**
Set a `max_iterations` limit. Add a "finish" tool the agent must call when done. Use LangGraph's recursion limit for graph-based agents. Always set a wall-clock timeout as a final backstop.

**What models work best for agents?**
GPT-4o and GPT-4o-mini for OpenAI. Claude 3.5 Sonnet for Anthropic. For local models, Llama 3.1 70B performs well on tool selection. Smaller models under 7B often struggle with correct tool use.

---

## Further Reading

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [HuggingFace Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction)

---

## What to Learn Next

- [AI Agents Guide](/blog/ai-agents-guide/) — architecture patterns and design decisions for agent systems
- [Build AI Agents Step-by-Step](/blog/build-ai-agents/) — hands-on implementation with LangChain and LangGraph
- [Tool Use and Function Calling](/blog/tool-use-and-function-calling/) — deep dive on defining and handling tool calls
