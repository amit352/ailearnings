---
title: "How to Build AI Agents Step-by-Step"
description: "Learn how developers build AI agents using LLM frameworks."
date: "2026-03-13"
slug: "build-ai-agents"
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
keywords: ["build AI agents", "AI agent tutorial", "how to build AI agent", "LangChain agent build", "AI agent development"]
---

# How to Build AI Agents Step-by-Step

A chatbot answers a question and stops. An agent answers a question, runs a calculation, searches for current information, and then synthesizes everything into a final answer — all without you specifying each step. The difference is not just a matter of architecture; it changes what problems you can solve. This guide shows you exactly how to build a production-ready agent from tool definitions through state management.

---

## What is Building an AI Agent

**Building an AI agent** means combining an LLM with a set of tools and a runtime loop that allows the model to reason, act, and observe in cycles until a task is complete. Unlike a simple API call, an agent chooses which tools to use, in what order, based on what it observes after each action.

The result is a system that can handle open-ended tasks that cannot be solved in a single prompt response — research tasks, multi-step workflows, and tasks requiring access to live data or external systems. For background on how agents work conceptually, see [AI agents guide](/blog/ai-agents-guide/).

---

## Why Building AI Agents Matters for Developers

Agents are how AI moves from answering questions to getting work done. A well-built agent can autonomously research topics using search tools, write and test code by running a Python interpreter, query databases and synthesize results, and coordinate workflows across multiple systems.

The barrier to building agents has dropped significantly. With LangChain, LangGraph, or the OpenAI Assistants API, a working agent can be built in under 50 lines of code. The challenge is making agents reliable and observable in production — which is what this guide focuses on.

---

## How to Build an AI Agent

The core steps are: choose an LLM, define tools, create the agent and executor, add memory, and add error handling with guardrails. Each step has specific design decisions that determine whether the agent works reliably or fails unpredictably.

**Tool design is the most critical step.** The LLM decides which tool to use solely based on the tool's name and description. A vague description produces unpredictable tool selection. A precise description produces reliable behavior.

---

## Practical Example

### Step 1: Install Dependencies

```bash
pip install langchain langchain-openai langchain-community langgraph
```

### Step 2: Define Tools

Tools are the actions your agent can take. Each tool needs a clear name, description, and well-defined input/output contract.

```python
from langchain.tools import tool
import io, sys, contextlib

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic.
    Use this when you need up-to-date facts, recent events, or information
    that may have changed since your training data cutoff."""
    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun().run(query)

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output.
    Use this for calculations, data analysis, or verifying logic.
    Only use safe, non-destructive operations."""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "sum": sum}})
        return output.getvalue() or "Code executed with no output."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

@tool
def read_file(filepath: str) -> str:
    """Read the contents of a text file.
    Input should be a valid relative or absolute file path."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except Exception as e:
        return f"Error reading file: {e}"

tools = [search_web, run_python, read_file]
```

### Step 3: Create the Agent with LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Pull the standard ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # log each reasoning step
    max_iterations=10,      # prevent infinite loops
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

result = executor.invoke({"input": "What is 1234 * 5678? Verify with Python."})
print(result["output"])
```

### Step 4: Build with LangGraph for Production

LangGraph gives you explicit control over agent state and transitions — critical for production workflows that need checkpoints, human-in-the-loop pauses, or complex branching.

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated
import operator

SYSTEM_PROMPT = """You are a helpful research assistant.
Use your tools to answer questions thoroughly and accurately.
When you have a complete answer, provide it directly."""

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
    "messages": [HumanMessage(content="Research the top 3 open-source LLMs and compare their context windows")]
})
print(result["messages"][-1].content)
```

---

## Real-World Applications

**Research automation** — Agents that search the web, read pages, and produce structured reports eliminate hours of manual research. The same agent pattern powers tools like Perplexity and deep research assistants.

**Code generation and testing** — Agents with a Python interpreter tool can write code, run it, observe the output, and fix bugs iteratively. GitHub Copilot Workspace uses this pattern for multi-file edits.

**Data analysis pipelines** — An agent given database query tools and a Python execution tool can answer analytical questions by writing SQL, running it, and interpreting the results.

**Customer support automation** — Agents with access to CRM data, knowledge bases, and ticketing tools can resolve common support requests without human involvement, escalating only when confidence is low.

---

## Common Mistakes Developers Make

1. **No iteration or timeout limit** — Agents can loop indefinitely on ambiguous tasks. Always set `max_iterations` in the executor and add wall-clock timeouts in production.

2. **Tools that throw unhandled exceptions** — An unhandled exception inside a tool crashes the agent loop mid-task. Wrap every tool body in a try/except and return descriptive error messages.

3. **Vague tool descriptions** — The LLM chooses tools based entirely on descriptions. "Run code" is worse than "Execute Python code and return stdout. Use for calculations and data verification." Specificity drives correct tool selection.

4. **No observability on intermediate steps** — Without logging each tool call and observation, debugging production agent failures is guesswork. Set `return_intermediate_steps=True` and log every action.

5. **Testing only happy paths** — Test what happens when search returns no results, files do not exist, or the model misuses a tool. Agents need adversarial testing beyond the expected workflow.

---

## Best Practices

- **Start with two or three tools** — Validate the agent loop before adding complexity. A 10-tool agent that misbehaves is much harder to debug than a 2-tool agent.
- **Write tool descriptions as if writing docs for a junior developer** — The LLM uses descriptions exactly as written. Be explicit about when to use the tool, what inputs it expects, and what it returns.
- **Use LangGraph for stateful or multi-step production agents** — Memory, checkpoints, and human-in-the-loop flows require explicit state management. LangGraph handles these patterns cleanly where `AgentExecutor` does not.
- **Add input validation inside tools** — Reject malformed inputs immediately with a descriptive error. Better to fail fast with context than silently produce wrong output.
- **Monitor token consumption** — Agents can accumulate large context windows through tool observations. Set context size limits and summarize intermediate results when they exceed a threshold.

---

## FAQ

**Which LLM works best for agents?**
GPT-4o or Claude Sonnet for production — they follow tool-calling instructions reliably. GPT-4o-mini and Claude Haiku are good for development and cost-sensitive use cases. Smaller open-source models (7B–13B) often struggle with consistent tool selection.

**When should I use LangGraph instead of AgentExecutor?**
Use LangGraph when you need: explicit state that persists across steps, human-in-the-loop pauses, conditional branching based on tool results, or sub-agents within a larger workflow. Use AgentExecutor for simple single-agent prototypes.

**How do I prevent agents from going off-task?**
Add explicit constraints to the system prompt ("Only answer questions related to X. If asked about Y, decline and explain."). Add a tool that signals task completion with a structured output. Set `max_iterations` conservatively.

**How do I handle tool failures gracefully?**
Return an error string (not raise an exception) from the tool function. The agent will observe the error, adapt its approach, and try an alternative. Raising an exception bypasses the reasoning loop.

---

## Further Reading

- [LangChain Agents Documentation](https://python.langchain.com/docs/concepts/agents/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

---

## What to Learn Next

- [AI Agents Guide](/blog/ai-agents-guide/) — the conceptual foundation for agent architectures
- [LangChain Agents](/blog/langchain-agents/) — LangChain-specific agent patterns and tooling
- [Multi-Agent Systems](/blog/multi-agent-systems/) — coordinating multiple specialized agents in a pipeline
