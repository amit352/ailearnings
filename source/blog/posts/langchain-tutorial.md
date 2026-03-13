---
title: "LangChain Tutorial – Build AI Applications with LLMs"
description: "Learn how to use LangChain to build AI applications with LLMs."
date: "2026-03-13"
slug: "langchain-tutorial"
keywords: ["LangChain tutorial", "LangChain guide", "build with LangChain", "LangChain chains", "LangChain Python"]
---

# LangChain Tutorial – Build AI Applications with LLMs

LangChain is the most widely used framework for building LLM-powered applications. It provides abstractions for chaining prompts, connecting to data sources, integrating tools, and orchestrating agents. This tutorial covers the core concepts and patterns developers use most in production.

---

## What is LangChain

LangChain is an open-source Python (and JavaScript) framework that simplifies building applications with large language models. It provides:

- **Model wrappers** — Unified interface for OpenAI, Anthropic, Gemini, and local models
- **Prompt templates** — Parameterized, reusable prompt structures
- **Chains** — Sequences of operations connecting prompts, models, and data
- **Retrievers** — Components that fetch relevant documents for RAG
- **Agents** — LLMs that decide which tools to call and in what order
- **Memory** — Conversation history management

LangChain handles the boilerplate so you focus on application logic.

---

## Why LangChain Matters for Developers

Building AI applications requires connecting many moving parts: prompt management, API calls, context retrieval, output parsing, and error handling. LangChain provides tested abstractions for each of these, reducing the amount of glue code you write.

Key benefits:
- **Model-agnostic** — Swap OpenAI for Claude or a local model with one line change
- **Built-in integrations** — 100+ document loaders, vector stores, and tool connectors
- **LCEL (LangChain Expression Language)** — Composable, pipeable syntax for building chains
- **LangSmith integration** — Observability and debugging built in

For RAG specifically, LangChain is the most common choice. See [RAG explained](/blog/rag-explained/) for the underlying pattern.

---

## How LangChain Works

### Installation

```bash
pip install langchain langchain-openai langchain-community
```

### Basic LLM Call

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("What is the difference between RAG and fine-tuning?")
print(response.content)
```

### Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains {topic} to developers."),
    ("human", "{question}")
])

chain = prompt | llm

response = chain.invoke({
    "topic": "machine learning",
    "question": "What is gradient descent?"
})
print(response.content)
```

The `|` operator is LCEL — it pipes the output of one component as input to the next.

### Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel

# String output
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "AI", "question": "What is RAG?"})

# Structured JSON output
class Summary(BaseModel):
    key_points: list[str]
    difficulty: str
    one_liner: str

parser = JsonOutputParser(pydantic_object=Summary)
json_chain = prompt | llm | parser
```

---

## Practical Examples

### Simple RAG Chain

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

result = qa_chain.invoke({"query": "How do I reset my password?"})
print(result["result"])
```

### Conversational Chain with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="My name is Alex.")
response = conversation.predict(input="What's my name?")
print(response)  # Remembers "Alex"
```

### Multi-Step Processing Chain

```python
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Extract facts
extract_prompt = PromptTemplate.from_template(
    "Extract the 3 most important facts from this text:\n{text}"
)

# Step 2: Summarize from facts
summarize_prompt = PromptTemplate.from_template(
    "Write a 2-sentence summary based on these facts:\n{facts}"
)

parser = StrOutputParser()

# Chain steps together with LCEL
pipeline = (
    extract_prompt | llm | parser
    | (lambda facts: {"facts": facts})
    | summarize_prompt | llm | parser
)

result = pipeline.invoke({"text": long_article})
print(result)
```

---

## Tools and Frameworks

**LangSmith** — The observability and debugging companion to LangChain. Traces every prompt, intermediate step, and response. Essential for production debugging and prompt iteration.

**LangGraph** — Graph-based agent orchestration built on top of LangChain. Enables stateful, multi-step agents with cycles and conditional routing. Best for complex agent workflows.

**LangServe** — Deployment layer for LangChain applications. Exposes chains as REST APIs with auto-generated docs.

**Chroma / FAISS / Pinecone** — Vector stores used with LangChain's retriever interface. For details, see [vector databases explained](/blog/vector-database-explained/).

---

## Common Mistakes

**Hardcoding model names** — Store model names in config or environment variables. Switching models should be a one-line change, not a refactor.

**Not using LCEL** — The older `LLMChain` and `SequentialChain` APIs are verbose and harder to extend. LCEL is more composable and better supported.

**Ignoring token limits** — LangChain does not automatically handle context overflow. Track token counts manually or use `ConversationSummaryMemory` for long conversations.

**Skipping LangSmith** — Running production chains without observability makes debugging extremely difficult. Set up LangSmith tracing from the start.

**Treating the framework as magic** — LangChain simplifies the boilerplate but does not fix bad prompts or wrong retrieval. Understand what each component does before using it.

---

## Best Practices

- **Use LCEL for all new chains** — It is cleaner, more composable, and better integrated with LangSmith tracing than the legacy chain classes.
- **Test each component independently** — Prompt templates, retrievers, and output parsers can each be tested in isolation before wiring them together.
- **Set `temperature=0` for deterministic tasks** — Extraction, classification, and structured output should be deterministic. Use higher temperature only for generative tasks.
- **Use streaming for long responses** — LangChain supports streaming out of the box. For user-facing applications, stream responses to reduce perceived latency.
- **Pin LangChain versions** — LangChain evolves rapidly. Pin the version in your requirements file and upgrade deliberately.

---

## Summary

LangChain provides the building blocks for production LLM applications: prompt templates, model wrappers, chains, retrievers, agents, and memory. The LCEL syntax makes composing these components declarative and readable.

Start with simple chains using LCEL. Add retrieval for document Q&A. Graduate to agents when your application needs to decide which tools to call. Use LangSmith for observability throughout.

For building a complete retrieval application, see [build a RAG app](/blog/build-rag-app/). For agent-specific patterns, see [LangChain agents](/blog/langchain-agents/). For deploying what you build, see [how to build your first AI app](/blog/build-ai-app/).
