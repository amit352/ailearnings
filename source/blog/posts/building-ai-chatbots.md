---
title: "Building AI Chatbots: From Simple Q&A to Contextual Conversations"
description: "Step-by-step guide to building AI chatbots — multi-turn conversations, conversation memory, RAG-powered bots, streaming responses, and deploying a chat UI."
date: "2026-03-10"
slug: "building-ai-chatbots"
keywords: ["building AI chatbots", "AI chatbot tutorial", "LLM chatbot", "chatbot with memory"]
---

## Learning Objectives

- Build a multi-turn conversational chatbot
- Add long-term conversation memory
- Ground chatbot responses in documents using RAG
- Stream responses for a better user experience
- Deploy a chatbot with a simple web UI

---

## Types of Chatbots

| Type | Memory | Data Source | Use Case |
|------|--------|-------------|----------|
| Simple Q&A | None | LLM training data | FAQs, basic info |
| Conversational | Short-term | LLM training data | Customer support, assistants |
| RAG Chatbot | Short-term | Your documents | Enterprise knowledge base |
| Agent Chatbot | Long-term | Tools + documents | Complex task automation |

---

## Step 1: Basic Multi-Turn Chatbot

```python
from openai import OpenAI

client = OpenAI()

def run_chatbot():
    print("AI Chatbot — type 'quit' to exit\n")
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise and friendly."}
    ]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=512,
        )

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"AI: {reply}\n")

        # Keep memory manageable — trim to last 20 messages (+ system prompt)
        if len(messages) > 21:
            messages = [messages[0]] + messages[-20:]


run_chatbot()
```

---

## Step 2: Streaming Chatbot

```python
def run_streaming_chatbot():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})
        print("AI: ", end="", flush=True)

        # Collect streamed response
        full_reply = ""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                full_reply += delta.content

        print()  # newline after streaming
        messages.append({"role": "assistant", "content": full_reply})
```

---

## Step 3: Chatbot with Persistent Memory

Store conversation summaries so the bot remembers important context across sessions.

```python
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

class PersistentChatbot:
    def __init__(self, session_id: str, system_prompt: str):
        self.session_id = session_id
        self.memory_file = Path(f"./memory_{session_id}.json")
        self.system_prompt = system_prompt
        self.messages = self._load_history()

    def _load_history(self) -> list:
        if self.memory_file.exists():
            history = json.loads(self.memory_file.read_text())
            print(f"Resuming session with {len(history)} messages")
            return [{"role": "system", "content": self.system_prompt}] + history
        return [{"role": "system", "content": self.system_prompt}]

    def _save_history(self):
        # Save everything except system prompt
        history = [m for m in self.messages if m["role"] != "system"]
        # Keep last 50 exchanges
        self.memory_file.write_text(json.dumps(history[-100:], indent=2))

    def _summarize_if_needed(self):
        # Summarize old messages if conversation gets too long
        if len(self.messages) < 40:
            return

        old_messages = self.messages[1:20]  # skip system prompt, take first 20
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize this conversation in 3-5 bullet points."},
                {"role": "user",   "content": str(old_messages)},
            ],
        )
        summary = summary_response.choices[0].message.content

        # Replace old messages with summary
        self.messages = (
            [self.messages[0]]  # system prompt
            + [{"role": "system", "content": f"Earlier conversation summary:\n{summary}"}]
            + self.messages[20:]  # keep recent messages
        )

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._summarize_if_needed()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            max_tokens=512,
        )

        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        self._save_history()
        return reply


# Usage
bot = PersistentChatbot("alice_session_1", "You are Alice's personal coding assistant.")
print(bot.chat("My name is Alice and I'm learning Python."))
print(bot.chat("What's a list comprehension?"))

# Later, in a new Python session:
bot2 = PersistentChatbot("alice_session_1", "You are Alice's personal coding assistant.")
print(bot2.chat("Do you remember my name?"))  # Should recall Alice
```

---

## Step 4: RAG-Powered Document Chatbot

A chatbot that answers questions from your documents:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

class DocumentChatbot:
    def __init__(self, vectorstore_path: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=OpenAIEmbeddings(),
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        self.chat_history = []

        # Question reformulation (makes follow-ups work)
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given chat history, rephrase the follow-up question as a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt
        )

        # Answer generation
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the question using only the provided context.
If the answer is not in the context, say "I don't have that information."
Be concise and helpful.

Context:
{context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        answer_chain = create_stuff_documents_chain(self.llm, answer_prompt)
        self.rag_chain = create_retrieval_chain(history_retriever, answer_chain)

    def chat(self, question: str) -> dict:
        result = self.rag_chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })

        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=result["answer"]),
        ])

        return {
            "answer": result["answer"],
            "sources": list({d.metadata.get("source", "?") for d in result["context"]}),
        }


# Run it
bot = DocumentChatbot("./chroma_db")
result = bot.chat("What are the key features of the product?")
print(result["answer"])
print("Sources:", result["sources"])
```

---

## Step 5: Web UI with Gradio

```bash
pip install gradio
```

```python
import gradio as gr
from openai import OpenAI

client = OpenAI()

def chat(user_message: str, history: list) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Convert Gradio history format to OpenAI messages format
    for user, assistant in history:
        messages.append({"role": "user",      "content": user})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512,
    )
    return response.choices[0].message.content


demo = gr.ChatInterface(
    fn=chat,
    title="AI Assistant",
    description="Ask me anything!",
    theme=gr.themes.Soft(),
    examples=["What is machine learning?", "Explain transformers in simple terms"],
)

demo.launch(share=True)  # share=True creates a public URL
```

### Gradio with RAG and Streaming

```python
def chat_streaming(user_message: str, history: list):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user, assistant in history:
        messages.append({"role": "user",      "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )

    partial_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message  # Gradio updates the UI as text arrives


demo = gr.ChatInterface(fn=chat_streaming, title="Streaming Chatbot")
demo.launch()
```

---

## Troubleshooting

**Bot loses context after a few messages**
- Check that history is being passed correctly to each API call
- Verify the summarization logic isn't removing too much context
- Increase `max_tokens` if replies are being cut short

**Bot answers questions it shouldn't (hallucination)**
- Add guardrails to the system prompt: "Only answer questions about [topic]. For off-topic questions, politely decline."
- For RAG bots: increase specificity of "only use provided context" instruction

**Bot response is too slow**
- Use streaming (`stream=True`) for perceived responsiveness
- Switch to `gpt-4o-mini` if using `gpt-4o`
- Cache common questions

---

## FAQ

**How do I build a chatbot for my website?**
Options: (1) Gradio embedded in an iframe, (2) Next.js/React frontend + FastAPI backend, (3) Vercel AI SDK for Next.js with streaming support.

**How do I prevent the bot from going off-topic?**
Add explicit instructions in the system prompt: "You are a customer support bot for [Company]. Only answer questions about [topic]. Politely decline all other requests."

---

## What to Learn Next

- **Deploy your chatbot** → [Deploying AI Applications](/blog/deploying-ai-applications/)
- **RAG for documents** → [LangChain RAG Tutorial](/blog/langchain-rag-tutorial/)
- **AI agent chatbot** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
