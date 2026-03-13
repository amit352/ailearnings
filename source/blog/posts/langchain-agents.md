---
title: "LangChain Agents Explained for Developers"
description: "Learn how LangChain agents automate workflows in AI applications."
date: "2026-03-13"
slug: "langchain-agents"
keywords: ["LangChain agents", "LangChain agent tutorial", "AI agents LangChain", "ReAct agent", "LangGraph agents"]
---

# LangChain Agents Explained for Developers

A LangChain agent is an LLM that decides what to do next — which tools to call, in what order, and when to stop. Unlike a chain, where the sequence of steps is fixed at build time, an agent determines the sequence dynamically based on the task. This makes agents powerful for open-ended tasks that cannot be anticipated in advance.

---

## What is a LangChain Agent

A LangChain agent consists of three components:
- An **LLM** — the reasoning engine that decides what to do
- A set of **tools** — functions the LLM can call (search, code execution, database queries, APIs)
- An **agent executor** — the runtime loop that runs the LLM, executes tools, and feeds results back

The LLM reasons about the task, selects a tool to call, observes the result, and continues until it reaches a final answer. This is the ReAct (Reason + Act) pattern.

---

## Why LangChain Agents Matter for Developers

Chains are great when you know the steps ahead of time. Agents are necessary when:
- The number of steps is unknown (depends on the data)
- The path depends on intermediate results
- The task requires multiple different tools in flexible combinations
- You want the model to handle ambiguous requests without hard-coded routing

Common agent use cases: research assistants that search and synthesize, data analysis agents that query databases and generate charts, coding assistants that read files and run tests.

For the broader concept of AI agents, see [AI agents guide](/blog/ai-agents-guide/).

---

## How LangChain Agents Work

### The ReAct Loop

```
Thought: What do I need to do?
Action: tool_name("input")
Observation: [tool result]
Thought: What did I learn? What next?
Action: ...
Final Answer: [answer to original question]
```

The LLM generates this text. The agent executor parses out the action, calls the tool, and injects the observation back into the prompt. This continues until the model outputs "Final Answer."

### Basic Agent Setup

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain import hub

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Built-in tool
search = DuckDuckGoSearchRun()

# Custom tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [search, calculator]

# Pull a ReAct prompt template
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### Running the Agent

```python
result = executor.invoke({
    "input": "What is the current population of Japan divided by the number of letters in 'artificial intelligence'?"
})
print(result["output"])
```

---

## Practical Examples

### Research Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def summarize_findings(text: str) -> str:
    """Summarize a block of research findings into 3 bullet points."""
    response = llm.invoke(f"Summarize in 3 bullet points:\n{text}")
    return response.content

tools = [wikipedia, summarize_findings]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

result = executor.invoke({"input": "Research the history of the transformer architecture in AI"})
```

### LangGraph Agent (Stateful)

LangGraph extends LangChain with graph-based agent orchestration. It gives you explicit control over state and transitions.

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Search for the latest LLM releases")]})
```

---

## Tools and Frameworks

**LangChain built-in tools** — Dozens of pre-built tools including web search (DuckDuckGo, Tavily), Wikipedia, Python REPL, shell execution, and file operations.

**LangGraph** — Graph-based agent orchestration. Better than the basic AgentExecutor for multi-agent systems, long-running workflows, and stateful agents with memory.

**Tavily Search** — A search API designed for AI agents. Returns structured, relevant results better suited to agent consumption than raw web scraping.

**LangSmith** — Traces every agent step, including tool inputs and outputs. Invaluable for debugging agents that take unexpected paths.

---

## Common Mistakes

**No max_iterations limit** — Without a limit, a stuck agent runs forever. Always set `max_iterations=10` or similar in AgentExecutor.

**Tools with ambiguous names or descriptions** — The LLM selects tools based on their name and description. Vague descriptions lead to wrong tool selection. Be specific about what each tool does and when to use it.

**Not handling tool errors** — Tools fail. APIs return errors. Structure tools to return descriptive error messages rather than raising exceptions that crash the agent.

**Using agents when a chain suffices** — Agents add latency and unpredictability. If the steps are known in advance, use a chain. Reserve agents for genuinely dynamic tasks.

**Too many tools** — Giving an agent 20 tools creates decision paralysis. Each call the model must evaluate all options. Provide the minimum set of tools that covers the task.

---

## Best Practices

- **Write clear tool descriptions** — The LLM reads them to decide what to call. Treat them as documentation for the model, not for humans.
- **Test tools independently** — Every tool should work correctly when called directly before being attached to an agent.
- **Use `verbose=True` during development** — See exactly what the agent is thinking and doing. Turn it off in production.
- **Set a timeout** — Agents can loop or make expensive API calls. Set a wall-clock timeout in addition to max_iterations.
- **Use LangGraph for production agents** — The basic AgentExecutor is good for prototyping. LangGraph gives you the control, observability, and reliability needed for production.

---

## Summary

LangChain agents combine an LLM with tools and a reasoning loop to handle tasks that cannot be expressed as fixed chains. The model decides which tools to call based on the task and intermediate results.

Use agents when the task is open-ended, when the path depends on data discovered at runtime, or when you need flexible tool composition. For simpler, predictable workflows, stick with chains.

For the broader agent design space, see [AI agents guide](/blog/ai-agents-guide/). For the LangChain foundation these agents build on, see [LangChain tutorial](/blog/langchain-tutorial/).
