---
title: "AI Agents Tutorial: Build Autonomous AI Systems (2026)"
description: "Step-by-step tutorial for building a working AI agent with Python and LangChain. Build a research agent with full ReAct loop in runnable code."
date: "2026-03-15"
updatedAt: "2026-03-15"
slug: "/blog/ai-agents-tutorial"
keywords: ["ai agents tutorial", "build ai agent python", "langchain agent tutorial"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
level: "intermediate"
time: "15 min"
stack: ["Python", "LangChain"]
---

# AI Agents Tutorial: Build Autonomous AI Systems (2026)

The gap between understanding how agents work conceptually and actually getting one running is wider than most tutorials admit. Documentation explains the happy path. Production shows you the edge cases — malformed tool calls, observations that exceed the context window, loops that don't terminate, and tools that return HTML when you expected JSON.

This tutorial takes a different approach. We build a real research agent that searches the web, reads files, and writes a structured report. Every code block runs. Every decision — model choice, tool selection, iteration limits — comes with a reason. By the end, you will have a working agent and the mental model to adapt it.

The stack is Python 3.11+, LangChain 0.3, and OpenAI GPT-4o. The same patterns work with Anthropic's Claude using `langchain-anthropic`.

---

## Concept Overview

A research agent is a good first agent to build because the task is well-defined: given a topic, find relevant information from multiple sources and synthesize it into a structured report. The agent needs at least three tools — web search, file reading, and file writing — and a loop that continues until the report is complete.

The architecture follows the ReAct (Reason + Act) pattern:

1. The agent receives a research topic
2. It reasons about what information it needs
3. It calls a search tool, reads results, and decides what to search next
4. It accumulates enough information to write a report
5. It writes the report to a file and returns a summary

This is a multi-step, multi-tool workflow that exercises everything core to agent design.

---

## How It Works

The research agent follows this flow:

![Architecture diagram](/assets/diagrams/ai-agents-tutorial-diagram-1.png)

The agent does not follow a fixed script. It decides how many searches to perform, what to look for in each one, and when it has gathered enough to write a coherent report. This is what distinguishes an agent from a chain — the control flow is determined by the model, not the developer.

---

## Implementation Example

### Step 1: Environment Setup

```python
# requirements.txt
# langchain==0.3.x
# langchain-openai==0.2.x
# langchain-community==0.3.x
# duckduckgo-search==6.x
# wikipedia==1.4.x

import os
from pathlib import Path

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Directory for agent output
OUTPUT_DIR = Path("./agent_output")
OUTPUT_DIR.mkdir(exist_ok=True)
```

### Step 2: Define the Tools

```python
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Web search tool
web_search = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for current information on a topic. Input should be a specific search query string."
)

# Wikipedia tool
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000),
    name="wikipedia",
    description="Look up detailed background information on a topic from Wikipedia. Best for established concepts, historical context, and technical definitions."
)

@tool
def write_report(filename: str, content: str) -> str:
    """
    Write a research report to a file.

    Args:
        filename: Name of the output file (without extension, e.g., 'ai_agents_report')
        content: The full text content of the report in markdown format

    Returns:
        Confirmation message with file path
    """
    output_path = OUTPUT_DIR / f"{filename}.md"
    output_path.write_text(content, encoding="utf-8")
    return f"Report saved to {output_path}. Word count: {len(content.split())} words."

@tool
def read_file(filepath: str) -> str:
    """
    Read the contents of a local text or markdown file.

    Args:
        filepath: Absolute or relative path to the file to read

    Returns:
        File contents as a string, or an error message if the file is not found
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File not found at {filepath}"
        if path.stat().st_size > 50000:  # 50KB limit
            return path.read_text(encoding="utf-8")[:5000] + "\n[... truncated for length]"
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"

tools = [web_search, wikipedia, write_report, read_file]
```

Tool descriptions are load-bearing. The agent reads these descriptions to decide which tool to call. Notice that `web_search` specifies it handles "current information" while `wikipedia` specifies it is best for "established concepts" — this distinction helps the model route queries correctly.

### Step 3: Build the Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,          # Zero temperature for consistent, factual research
    max_tokens=4096         # Give the model room to write substantial responses
)

system_prompt = """You are a research agent. Your job is to research topics thoroughly and produce well-structured reports.

When given a research topic:
1. Plan your research approach
2. Search for primary information using web_search
3. Look up background context using wikipedia where appropriate
4. Gather enough information to write a comprehensive report (aim for 3-5 searches minimum)
5. Write the report using write_report with a descriptive filename
6. Return a brief summary of what you found and where the report was saved

Research quality standards:
- Cover multiple perspectives, not just the first result
- Include specific examples, numbers, and dates where available
- Note any conflicting information you find
- Structure the report with clear sections: Overview, Key Findings, Analysis, Conclusion
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,          # Research tasks may need more steps
    max_execution_time=120,     # 2-minute wall-clock limit
    handle_parsing_errors=True,
    return_intermediate_steps=True
)
```

### Step 4: Run the Agent

```python
def run_research_agent(topic: str) -> dict:
    """Run the research agent on a given topic and return results."""
    print(f"\n{'='*60}")
    print(f"Starting research on: {topic}")
    print(f"{'='*60}\n")

    try:
        result = agent_executor.invoke({
            "input": f"Research the following topic and write a comprehensive report: {topic}"
        })

        print(f"\n{'='*60}")
        print("AGENT SUMMARY:")
        print(result["output"])
        print(f"{'='*60}")

        # Show what tools were called
        print(f"\nTools called: {len(result['intermediate_steps'])}")
        for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
            print(f"  Step {i}: {action.tool}({str(action.tool_input)[:60]}...)")

        return result

    except Exception as e:
        print(f"Agent error: {e}")
        return {"error": str(e)}

# Run it
result = run_research_agent("The current state of AI agent frameworks in 2026")
```

### Step 5: Inspecting the ReAct Loop

The `verbose=True` flag shows the full reasoning trace. Here is what a typical research agent run looks like:

```
> Entering new AgentExecutor chain...

Invoking: `web_search` with `{'query': 'AI agent frameworks 2026 comparison LangChain LangGraph CrewAI'}`

[Search results returned...]

Invoking: `wikipedia` with `{'query': 'AI agent software architecture'}`

[Wikipedia content returned...]

Invoking: `web_search` with `{'query': 'LangGraph vs CrewAI production use cases 2026'}`

[Search results returned...]

Invoking: `write_report` with `{'filename': 'ai_agent_frameworks_2026', 'content': '# AI Agent Frameworks in 2026\n\n## Overview\n...'}`

Report saved to agent_output/ai_agent_frameworks_2026.md. Word count: 847 words.

> Finished chain.
```

Each `Invoking:` line represents one iteration of the ReAct loop. The model decided on its own to search twice before writing, which gave it enough information to produce a meaningful report.

### Step 6: Adding Conversation Memory

For interactive use where the user can ask follow-up questions:

```python
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

def run_research_agent_with_memory(topic: str, history: list) -> tuple[str, list]:
    """Run the research agent with conversation history."""
    result = agent_executor.invoke({
        "input": topic,
        "chat_history": history
    })

    # Update history
    history.append(HumanMessage(content=topic))
    history.append(AIMessage(content=result["output"]))

    return result["output"], history

# First query
output, chat_history = run_research_agent_with_memory(
    "Research the current state of AI agent frameworks",
    chat_history
)

# Follow-up using context from previous run
output, chat_history = run_research_agent_with_memory(
    "Now focus specifically on LangGraph — what makes it different from the others?",
    chat_history
)
```

---

## Best Practices

**Start with verbose mode always on during development.** The reasoning trace is the fastest way to understand why an agent made a specific decision. Turn it off in production, but log the intermediate steps to your observability platform.

**Calibrate `max_iterations` to your task.** A simple question-answering agent needs 3-5 iterations. A research task might need 10-15. A complex multi-tool workflow could need 20+. Set the limit higher than you expect and monitor actual usage to tune it down.

**Make tool error messages informative.** When a tool fails, the error message becomes an observation that the agent reads. "Error: connection timeout after 5s" is more useful than "Error: something went wrong" because the agent can decide to retry or try an alternative.

**Test with `return_intermediate_steps=True` in staging.** Review every tool call for each test case. Agents often find unexpected paths to answers, and some of those paths are inefficient or incorrect even when the final output looks right.

---

## Common Mistakes

1. **Running the agent without any output directory setup.** The `write_report` tool will fail if the output directory does not exist. Always initialize directories before the agent runs.

2. **Using temperature > 0 for research agents.** Research tasks need consistency, not creativity. High temperature causes the agent to phrase tool inputs differently across runs, which reduces reproducibility and makes debugging harder.

3. **Not limiting observation length.** Web search can return thousands of characters per result. If the agent performs five searches and each returns 5,000 characters, you have used 25,000 tokens just on observations. Truncate at the tool level.

4. **Trusting the agent to format reports correctly every time.** The system prompt tells the agent what sections to include, but it will not always comply. Add post-processing validation that checks for required sections before saving.

5. **Forgetting to handle the case where the agent reaches `max_iterations`.** When the agent hits its limit, it stops without necessarily completing the task. Check the output for completion indicators and alert the user if the task was not finished.

---

## Summary

Building a research agent reveals the practical engineering work behind agent systems. Tools need careful descriptions and error handling. Iteration limits are mandatory, not optional. The system prompt shapes agent behavior more than any other single factor. Starting with verbose logging and `return_intermediate_steps=True` makes the difference between debugging in minutes versus hours.

The patterns here — ReAct loop, function-calling, tool design, memory for follow-ups — transfer directly to any agent task. Research is a good starting point because the task is clear and the tools are safe (read-only except for writing local files). Once you have this working, the next step is adding persistence and multi-step planning for more complex workflows.

---

## Related Articles

- [AI Agents Guide: Architecture and Design Patterns](/blog/ai-agents-guide)
- [Building Agents with LangChain: Complete Tutorial](/blog/langchain-agents)
- [Tool Use in AI Agents](/blog/agent-tools)
- [Memory Systems in AI Agents](/blog/agent-memory)
- [Autonomous Task Execution in AI Agents](/blog/autonomous-agents)
- [LangChain Tutorial for Beginners](/blog/langchain-tutorial)

---

## FAQ

**Do I need GPT-4o or can I use a cheaper model?**
GPT-4o is recommended for multi-step research agents. GPT-4o-mini can work for simple tasks but struggles with complex planning and reliable function calling across many steps. Claude 3.5 Haiku is a good cheaper alternative with strong function-calling performance.

**How do I run this without an OpenAI API key?**
Replace `ChatOpenAI` with `ChatAnthropic` from `langchain-anthropic` and use `create_openai_functions_agent` with Anthropic's tool-use format, or use the Ollama integration for local models. Local models generally produce less reliable agents for complex tasks.

**Why does my agent sometimes fail to write the report?**
The most common causes are: the agent reached `max_iterations` before completing, the `write_report` tool received invalid arguments, or the output directory did not exist. Enable `verbose=True` and check the trace to identify which step failed.

**Can the agent run indefinitely without my `max_iterations` setting?**
Yes. Without `max_iterations`, a confused or looping agent will run until it hits the model's context limit or your API rate limit. This can be expensive. Always set both `max_iterations` and `max_execution_time`.

**How do I make the agent faster?**
Cache search results for repeated queries. Use streaming to show partial results to the user. Consider using a faster model (GPT-4o-mini) for sub-tasks that do not require deep reasoning. Parallelize independent tool calls where your framework supports it.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "Do I need GPT-4o or can I use a cheaper model?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "GPT-4o is recommended for multi-step research agents. GPT-4o-mini can work for simple tasks but struggles with complex planning and reliable function calling across many steps. Claude 3.5 Haiku is a good cheaper alternative with strong function-calling performance."
      }
    },
    {
      "@type": "Question",
      "name": "Why does my agent sometimes fail to write the report?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "The most common causes are: the agent reached max_iterations before completing, the write_report tool received invalid arguments, or the output directory did not exist. Enable verbose=True and check the trace to identify which step failed."
      }
    },
    {
      "@type": "Question",
      "name": "Can the agent run indefinitely without max_iterations setting?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes. Without max_iterations, a confused or looping agent will run until it hits the model's context limit or your API rate limit. Always set both max_iterations and max_execution_time."
      }
    },
    {
      "@type": "Question",
      "name": "How do I make the agent faster?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Cache search results for repeated queries. Use streaming to show partial results to the user. Consider using a faster model for sub-tasks that do not require deep reasoning. Parallelize independent tool calls where your framework supports it."
      }
    },
    {
      "@type": "Question",
      "name": "How do I run this without an OpenAI API key?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Replace ChatOpenAI with ChatAnthropic from langchain-anthropic and use Anthropic's tool-use format, or use the Ollama integration for local models. Local models generally produce less reliable agents for complex tasks."
      }
    }
  ]
}
</script>
