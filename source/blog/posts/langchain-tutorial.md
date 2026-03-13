---
title: "LangChain Tutorial – Build AI Applications with LLMs"
description: "Learn how to use LangChain to build AI applications with LLMs."
date: "2026-03-13"
slug: "langchain-tutorial"
keywords: ["LangChain tutorial", "LangChain guide", "build with LangChain", "LangChain chains", "LangChain Python"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
---

# LangChain Tutorial – Build AI Applications with LLMs

Every LLM application eventually runs into the same wall: raw API calls get messy fast. You need prompt templates, context retrieval, output parsing, memory across turns, and tool use — and wiring those together yourself produces fragile glue code that breaks every time the underlying model changes. LangChain exists to solve exactly this problem. It is the most widely adopted framework for building production LLM applications, and this tutorial shows you how to use it correctly from day one.

---

## What is LangChain

LangChain is an open-source Python (and JavaScript) framework that provides composable abstractions for building LLM-powered applications. Rather than working with raw API calls and string concatenation, you work with typed components that plug together.

Its core building blocks are:

- **Model wrappers** — A unified interface for OpenAI, Anthropic, Gemini, and local models via Ollama or HuggingFace
- **Prompt templates** — Parameterized, reusable prompt structures with typed inputs
- **Chains** — Sequences of operations connecting prompts, models, parsers, and data
- **Retrievers** — Components that fetch relevant documents for RAG applications
- **Agents** — LLMs that decide which tools to call, in what order, and when to stop
- **Memory** — Abstractions for managing conversation history across turns

LangChain handles the boilerplate. You focus on application logic and prompt quality.

---

## Why LangChain Matters for Developers

The practical value of LangChain shows up most clearly when your application needs to do more than one thing. A single-turn Q&A call needs almost no framework. But a multi-turn assistant that retrieves documents, formats structured output, and calls external APIs suddenly has ten moving parts. LangChain's composable design handles this without requiring you to write custom orchestration logic.

Key practical benefits:

- **Model-agnostic** — Swap OpenAI for Claude or a local model with a single line change. Your chains work unchanged.
- **Built-in integrations** — 100+ document loaders, vector store connectors, and tool integrations maintained by the community
- **LCEL (LangChain Expression Language)** — A declarative, pipeable syntax for building chains that is more readable than nested callbacks
- **LangSmith integration** — Every step of every chain can be traced, logged, and debugged without additional instrumentation

For RAG specifically, LangChain is the dominant choice in production because it handles the entire pipeline from document loading to answer generation. See [RAG explained](/blog/rag-explained/) for the underlying architecture.

---

## Installation and Setup

```bash
pip install langchain langchain-openai langchain-community
export OPENAI_API_KEY="sk-..."
```

### Basic LLM Call

The simplest possible LangChain usage: wrap a model and call it.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("What is the difference between RAG and fine-tuning?")
print(response.content)
```

This already gives you more than a raw API call — the response is a typed `AIMessage` object with metadata, and you can swap `ChatOpenAI` for `ChatAnthropic` or `ChatOllama` without changing the rest of your code.

---

## Prompt Templates and LCEL

Hardcoded prompt strings are the first thing you should replace. Prompt templates make prompts reusable, testable, and version-controlled.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains {topic} to developers."),
    ("human", "{question}")
])

# LCEL: pipe components together with |
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "topic": "machine learning",
    "question": "What is gradient descent?"
})
print(response)
```

The `|` operator is LangChain Expression Language (LCEL). It passes the output of the left side as input to the right side. The pipeline above: formats the prompt → sends it to the model → parses the response to a plain string.

This is more readable than the equivalent nested function calls, and it integrates automatically with LangSmith tracing.

---

## Output Parsers and Structured Data

Raw string output from LLMs is hard to work with programmatically. Output parsers give you typed, structured results.

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class ContentSummary(BaseModel):
    key_points: List[str] = Field(description="The 3 most important takeaways")
    difficulty: str = Field(description="beginner, intermediate, or advanced")
    one_liner: str = Field(description="One sentence describing the topic")

parser = JsonOutputParser(pydantic_object=ContentSummary)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a content analyst. Respond with valid JSON only."),
    ("human", "Summarize this topic: {topic}\n\n{format_instructions}"),
])

chain = prompt | llm | parser

result = chain.invoke({
    "topic": "vector databases in AI applications",
    "format_instructions": parser.get_format_instructions()
})

print(result["key_points"])  # List[str] — typed output
print(result["difficulty"])  # 'intermediate'
```

For stricter guarantees, use `with_structured_output` directly on the model — it uses OpenAI's JSON mode under the hood and retries automatically on malformed responses.

---

## Building Chains with LCEL

### Multi-Step Processing Pipeline

LCEL lets you compose multi-step pipelines declaratively. Here is a two-stage pipeline: extract key facts, then summarize them.

```python
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Stage 1: Extract facts
extract_prompt = PromptTemplate.from_template(
    "Extract the 3 most important facts from this text:\n\n{text}"
)

# Stage 2: Summarize from facts
summarize_prompt = PromptTemplate.from_template(
    "Write a 2-sentence executive summary based on these facts:\n\n{facts}"
)

# Wire stages together — lambda bridges the two prompts
pipeline = (
    extract_prompt
    | llm
    | parser
    | (lambda facts: {"facts": facts})
    | summarize_prompt
    | llm
    | parser
)

long_article = """
Large language models have fundamentally changed how developers build software.
They can generate code, summarize documents, answer questions, and reason across
complex problems. The cost per token has dropped 100x since 2020, making AI
features economically viable for most applications...
"""

result = pipeline.invoke({"text": long_article})
print(result)
```

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

result = qa_chain.invoke({"query": "What is the company's vacation policy?"})
print(result["result"])
for doc in result["source_documents"]:
    print(f"  Source: {doc.metadata.get('source')} page {doc.metadata.get('page')}")
```

### Conversational Memory

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# Keep only the last 5 turns to control context size
memory = ConversationBufferWindowMemory(k=5)
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="My name is Alex and I'm building a RAG app.")
response = conversation.predict(input="What should I focus on first?")
print(response)  # Model knows your name is Alex and your project context
```

---

## Tools and the LangChain Ecosystem

**LangSmith** — The observability and debugging companion. It traces every prompt, intermediate output, and response in your chains. Essential for debugging why a retrieval chain returns wrong answers. Set it up by adding two environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls__..."
```

Every chain invocation then appears in the LangSmith UI with full input/output visibility for each step.

**LangGraph** — Graph-based agent orchestration built on LangChain. Enables stateful, multi-step agents with cycles, conditional routing, and human-in-the-loop checkpoints. The right choice when your agent needs to loop, branch, or maintain complex state.

**LangServe** — Deployment layer for LangChain applications. Exposes any chain as a REST API with auto-generated OpenAPI docs and a built-in playground UI.

**Chroma / FAISS / Pinecone** — Vector stores used with LangChain's retriever interface. For a comparison of options, see [vector databases explained](/blog/vector-database-explained/).

---

## Common Mistakes

**Hardcoding model names in chains** — Store model names in environment variables or config files. Switching from `gpt-4o-mini` to `gpt-4o` should be a one-line change in configuration, not a hunt through your codebase.

**Using the legacy chain classes** — The older `LLMChain`, `SequentialChain`, and `ConversationChain` APIs still work but are being phased out. LCEL is more composable, better integrated with LangSmith, and actively maintained. New code should use LCEL.

**Ignoring token limits** — LangChain does not automatically truncate inputs that exceed the model's context window. For long conversations, switch from `ConversationBufferMemory` to `ConversationSummaryMemory`, which summarizes older turns instead of keeping all of them verbatim.

**Skipping LangSmith for development** — It takes two minutes to enable and saves hours of debugging. When a chain returns a wrong answer, LangSmith shows you exactly which step produced bad output.

**Treating the framework as magic** — LangChain reduces boilerplate but does not fix bad prompts or poor retrieval quality. If your RAG application gives wrong answers, the problem is almost always in chunking strategy, retrieval parameters, or prompt quality — not in LangChain itself.

**Not pinning the version** — LangChain releases breaking changes frequently. Pin the version in your `requirements.txt` and upgrade deliberately. Running `pip install --upgrade langchain` in production can break your application.

---

## Best Practices

1. **Start with LCEL for all new chains** — It is more readable, better tested, and integrates with all modern LangChain features.
2. **Test each component in isolation** — A prompt template, a retriever, and an output parser can each be unit-tested independently before wiring them together.
3. **Set `temperature=0` for structured tasks** — Extraction, classification, and JSON output should be deterministic. Use higher temperatures only when you actually want creative variation.
4. **Use streaming for user-facing applications** — LangChain supports streaming with `.stream()` instead of `.invoke()`. Users perceive streamed responses as faster, even when total latency is the same.
5. **Log token usage per chain** — Track which chains consume the most tokens. The cost profile of a production application is rarely what you expect at design time.
6. **Use `RunnableParallel` for concurrent steps** — If two retrieval steps are independent, run them in parallel with `RunnableParallel` to reduce total latency.

---

## What to Learn Next

LangChain is a means to an end. Once you understand how chains and retrievers work, apply them to real problems:

- **Build a complete retrieval app end-to-end** → [How to Build a RAG Application](/blog/build-rag-app/)
- **Go deeper on the RAG architecture** → [RAG Explained](/blog/rag-explained/)
- **Connect an LLM to your own data with OpenAI directly** → [OpenAI API Tutorial](/blog/openai-api-tutorial/)
