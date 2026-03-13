---
title: "LangChain Complete Tutorial: Chains, Agents, Tools, and Memory"
description: "Master LangChain from scratch — LLM wrappers, prompt templates, chains, LCEL, agents with tools, memory, and building production applications."
date: "2026-03-10"
slug: "langchain-tutorial-complete"
keywords: ["LangChain tutorial", "LangChain guide", "LangChain chains agents", "LCEL LangChain"]
---

## Learning Objectives

- Understand LangChain's core abstractions
- Build chains with LCEL (LangChain Expression Language)
- Create agents with tools
- Add memory to conversations
- Build and deploy a complete LangChain application

---

## Why LangChain?

LangChain provides:
- **Unified interfaces** — same code works with OpenAI, Anthropic, Ollama, etc.
- **Composable primitives** — chain prompts, LLMs, parsers, retrievers together
- **Pre-built integrations** — 100+ document loaders, vector stores, tools
- **Observability** — built-in tracing with LangSmith

Trade-offs: adds abstraction overhead; debugging can be harder than raw API calls.

---

## Setup

```bash
pip install langchain langchain-openai langchain-community langchain-chroma
export OPENAI_API_KEY="sk-..."
```

---

## Core Concepts

### LLM Wrappers

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Direct invocation
response = llm.invoke("What is LangChain?")
print(response.content)

# With messages
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LCEL?"),
]
response = llm.invoke(messages)
print(response.content)
```

### Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Simple template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Be concise and technical."),
    ("human", "{question}"),
])

# Format it
formatted = prompt.invoke({"domain": "machine learning", "question": "What is gradient descent?"})
print(formatted.messages)

# Few-shot template
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "15% of 200", "output": "30"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)
```

### Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# String output
str_parser = StrOutputParser()

# Structured output
class TechConcept(BaseModel):
    name: str = Field(description="Concept name")
    definition: str = Field(description="One-sentence definition")
    use_cases: List[str] = Field(description="2-3 use cases")

json_parser = JsonOutputParser(pydantic_object=TechConcept)

prompt_with_format = ChatPromptTemplate.from_messages([
    ("system", "Extract structured information.\n{format_instructions}"),
    ("human", "Explain {concept}"),
]).partial(format_instructions=json_parser.get_format_instructions())
```

---

## LCEL: LangChain Expression Language

LCEL uses the `|` operator to compose chains. It's the modern way to build LangChain applications.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Simple chain: prompt | llm | parser
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize technical content concisely."),
        ("human", "Summarize: {text}"),
    ])
    | llm
    | parser
)

result = chain.invoke({"text": "LangChain is a framework for building LLM applications..."})
print(result)

# Streaming
for chunk in chain.stream({"text": "LangChain is a framework..."}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([
    {"text": "Python is a programming language."},
    {"text": "Docker is a containerization tool."},
])
```

### Parallel Chains (RunnableParallel)

```python
from langchain_core.runnables import RunnableParallel

# Run two chains in parallel on the same input
parallel_chain = RunnableParallel(
    summary=chain,
    keywords=(
        ChatPromptTemplate.from_messages([("human", "List 5 keywords from: {text}")])
        | llm
        | parser
    ),
)

result = parallel_chain.invoke({"text": "Machine learning uses algorithms to learn from data..."})
print(result["summary"])
print(result["keywords"])
```

### Conditional Chains (RunnableBranch)

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

classify_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify the question as: 'technical', 'general', or 'math'. Reply with one word only."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

technical_chain = (
    ChatPromptTemplate.from_messages([("system", "You are a senior engineer. Be technical."), ("human", "{question}")])
    | llm | parser
)

general_chain = (
    ChatPromptTemplate.from_messages([("system", "You are a friendly assistant."), ("human", "{question}")])
    | llm | parser
)

router = RunnableBranch(
    (lambda x: "technical" in x["type"].lower(), technical_chain),
    general_chain,  # default
)

def route_question(input_dict):
    question = input_dict["question"]
    q_type = classify_chain.invoke({"question": question})
    return router.invoke({"type": q_type, "question": question})
```

---

## Memory

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Session-based memory
store = {}  # session_id → ChatMessageHistory

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

chain = prompt | llm | parser

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Each call with the same session_id remembers previous messages
session = {"configurable": {"session_id": "user_123"}}
print(chain_with_history.invoke({"input": "My name is Alice."}, config=session))
print(chain_with_history.invoke({"input": "What is my name?"}, config=session))
# → "Your name is Alice."
```

---

## Tools and Agents

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
import requests

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Use for real-time data."""
    # Use DuckDuckGo or SerpAPI in production
    return f"Search results for '{query}': [simulated results]"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_web, calculate, get_current_time]

# Create agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools. Use them when needed."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What's 15% of 4,280? Also, what time is it?"})
print(result["output"])
```

---

## RAG with LCEL

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context only:\n\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | parser
)

answer = rag_chain.invoke("What is the authentication flow?")
print(answer)
```

---

## Observability with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."  # from smith.langchain.com
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# Now all chain invocations are automatically traced
# View traces at https://smith.langchain.com
```

---

## Troubleshooting

**Chain output is not as expected**
- Use `.invoke()` step by step to debug: run each component separately
- Enable `verbose=True` on AgentExecutor
- Check LangSmith traces

**Memory not persisting between sessions**
- Ensure same `session_id` in `configurable` dict
- For production persistence: use `SQLChatMessageHistory` instead of in-memory

**Tool not being called**
- Improve the `@tool` docstring — the LLM uses it to decide when to call the tool
- Check if the tool schema is correct with `tool.args_schema`

---

## FAQ

**LCEL vs the old chain syntax?**
Use LCEL. The old `LLMChain`, `ConversationChain`, etc. are legacy and being deprecated.

**LangChain vs LlamaIndex?**
LangChain: broader ecosystem, better for agents and custom pipelines. LlamaIndex: better for document-heavy RAG with advanced indexing.

---

## What to Learn Next

- **RAG with LangChain** → [LangChain RAG Tutorial](/blog/langchain-rag-tutorial/)
- **Agents** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
- **Full AI roadmap** → [AI Roadmap for Developers](/blog/ai-roadmap-for-developers/)
