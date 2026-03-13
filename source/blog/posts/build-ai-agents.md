---
title: "How to Build AI Agents Step-by-Step"
description: "Learn how developers build AI agents using LLM frameworks."
date: "2026-03-13"
slug: "build-ai-agents"
keywords: ["build AI agents", "AI agent tutorial", "how to build AI agent", "LangChain agent build", "AI agent development"]
---

# How to Build AI Agents Step-by-Step

Building an AI agent means giving a language model the ability to take actions — run tools, access data, and complete multi-step tasks autonomously. This guide walks through building a production-ready agent from scratch, covering tool design, the agent loop, memory, and error handling.

---

## What is Building an AI Agent

Building an AI agent means combining an LLM with a set of tools and a runtime loop that allows the model to reason, act, and observe in cycles until a task is complete. The result is a system that can handle open-ended tasks that cannot be solved in a single prompt response.

For background on how agents work conceptually, see [AI agents guide](/blog/ai-agents-guide/).

---

## Why Building AI Agents Matters for Developers

Agents are how AI moves from answering questions to getting work done. A well-built agent can:
- Autonomously research topics using search tools
- Write and test code by running a Python interpreter
- Query databases and synthesize results
- Coordinate workflows across multiple systems

The barrier to building agents has dropped significantly. With LangChain, LangGraph, or the OpenAI Assistants API, a working agent can be built in under 50 lines of code.

---

## How to Build an AI Agent

The core steps:
1. Choose an LLM
2. Define tools
3. Create the agent and executor
4. Add memory (optional)
5. Handle errors and add guardrails

---

## Practical Examples

### Step 1: Install Dependencies

```bash
pip install langchain langchain-openai langchain-community langgraph
```

### Step 2: Define Tools

Tools are the actions your agent can take. Each tool needs a clear name, description, and well-defined input/output.

```python
from langchain.tools import tool
import requests

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic.
    Use this when you need up-to-date facts or recent events."""
    # In production, use Tavily or SerpAPI
    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun().run(query)

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output.
    Use this for calculations, data analysis, or code testing.
    Only use safe, non-destructive operations."""
    import io, sys, contextlib
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len}})
        return output.getvalue() or "Code executed successfully with no output."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

@tool
def read_file(filepath: str) -> str:
    """Read the contents of a text file.
    Input should be a valid file path."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except Exception as e:
        return f"Error reading file: {e}"

tools = [search_web, run_python, read_file]
```

### Step 3: Create the Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)
```

### Step 4: Add Memory

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # Remember last 5 exchanges
    return_messages=True
)

# Use a prompt that includes chat_history
prompt_with_memory = hub.pull("hwchase17/react-chat")

agent = create_react_agent(llm, tools, prompt_with_memory)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=10,
)
```

### Step 5: Build with LangGraph (Production)

LangGraph gives you explicit control over agent state and transitions — critical for production.

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated
import operator

SYSTEM_PROMPT = """You are a helpful research assistant.
Use your tools to answer questions thoroughly.
When you have a complete answer, provide it as your final response."""

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="Research the top 3 open-source LLMs released in 2025")]
})
print(result["messages"][-1].content)
```

### Step 6: Running and Testing

```python
def run_agent(task: str, verbose: bool = False) -> str:
    try:
        result = executor.invoke(
            {"input": task},
            config={"max_concurrency": 1}
        )
        return result["output"]
    except Exception as e:
        return f"Agent failed: {e}"

# Test cases
test_tasks = [
    "What is 847 * 293?",
    "Search for the latest developments in AI agents",
    "Write Python code to calculate the first 10 Fibonacci numbers",
]

for task in test_tasks:
    print(f"Task: {task}")
    print(f"Result: {run_agent(task)}\n")
```

---

## Tools and Frameworks

**LangChain** — The most accessible starting point for agent development. Rich ecosystem of built-in tools and integrations. See [LangChain agents](/blog/langchain-agents/) for the full guide.

**LangGraph** — Best for production agents. Gives you control over state, transitions, and human-in-the-loop checkpoints.

**OpenAI Assistants API** — Managed agent runtime. Built-in file search, code interpreter, and thread management. Good if you want to minimize infrastructure.

**AutoGen** — Microsoft's multi-agent framework. Best for workflows requiring multiple specialized agents collaborating.

---

## Common Mistakes

**No timeout or iteration limit** — Agents can loop indefinitely on ambiguous tasks. Always set `max_iterations` and handle `StopIteration` errors.

**Tools that throw exceptions** — An unhandled exception inside a tool crashes the agent loop. Wrap every tool in try/except.

**Not validating tool outputs** — Tool outputs are injected directly into the agent's context. Large or malformed outputs can exceed the context window or confuse the model.

**Testing only happy paths** — Test what happens when search returns no results, when files don't exist, and when the model misuses a tool.

**No observability** — Without logging each tool call and result, debugging agent failures is nearly impossible.

---

## Best Practices

- **Start with two or three tools** — Validate the agent loop before adding more complexity.
- **Write tool descriptions as if writing documentation for the model** — The LLM chooses tools based on their descriptions. Be precise about what each tool does and when to use it.
- **Log intermediate steps in production** — Set `return_intermediate_steps=True` and log every action and observation.
- **Add input validation to tools** — Reject malformed inputs early with a descriptive error message.
- **Use LangGraph for anything stateful** — Memory, checkpoints, and human-in-the-loop flows all require explicit state management that LangGraph handles cleanly.

---

## Summary

Building an AI agent involves four steps: define the tools, create the agent with an LLM and prompt, wrap it in an executor with iteration limits, and optionally add memory.

Start simple with two or three tools and the ReAct pattern. Validate that the basic loop works before adding complexity. Migrate to LangGraph when you need production-grade state management and observability.

For the conceptual foundation, see [AI agents guide](/blog/ai-agents-guide/). For LangChain-specific agent patterns, see [LangChain agents](/blog/langchain-agents/). For building AI applications more broadly, see [how to build your first AI app](/blog/build-ai-app/).
