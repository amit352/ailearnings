---
title: "Build an AI Chatbot in Python: Complete Beginner Project"
description: "Build a multi-turn AI chatbot in Python using the OpenAI API. Includes conversation memory, streaming, a CLI interface, and a simple Gradio web UI."
date: "2026-03-10"
slug: "ai-chatbot-python"
level: "Beginner"
time: "2–3 hours"
stack: "Python, OpenAI API, Gradio"
keywords: ["AI chatbot Python", "build chatbot OpenAI", "Python chatbot project"]
---

## Project Overview

Build a conversational AI chatbot that remembers context across messages, streams responses in real-time, and can be used from both the command line and a browser-based UI.

This is the ideal starting project for AI engineering — it teaches the core patterns (LLM API calls, message history, streaming) used in every AI application.

---

## Learning Goals

- Call the OpenAI chat completions API
- Maintain multi-turn conversation history
- Stream tokens for real-time output
- Build a simple web UI with Gradio
- Handle errors and edge cases gracefully

---

## Architecture

```
User Input (CLI or Browser)
        ↓
Message History Manager
        ↓
OpenAI Chat Completions API
        ↓
Streamed Response
        ↓
Display + Update History
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai gradio python-dotenv
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
```

### Step 2: Core Chatbot Class

```python
# chatbot.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self, system_prompt: str = "You are a helpful assistant.", model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_history = 20  # keep last 20 messages

    def chat(self, user_input: str) -> str:
        if not user_input.strip():
            return ""

        self.messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})

        # Trim history (keep system prompt + last N messages)
        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]

        return reply

    def chat_stream(self, user_input: str):
        """Yields tokens as they arrive."""
        self.messages.append({"role": "user", "content": user_input})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_reply += delta.content
                yield delta.content

        self.messages.append({"role": "assistant", "content": full_reply})

    def reset(self):
        system = self.messages[0]
        self.messages = [system]
```

### Step 3: CLI Interface

```python
# cli.py
from chatbot import Chatbot

def run_cli():
    print("AI Chatbot — type 'quit' to exit, 'reset' to clear history\n")
    bot = Chatbot(system_prompt="You are a helpful coding assistant.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("Conversation reset.\n")
            continue

        print("AI: ", end="", flush=True)
        for token in bot.chat_stream(user_input):
            print(token, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    run_cli()
```

### Step 4: Gradio Web UI

```python
# app.py
import gradio as gr
from chatbot import Chatbot

bot = Chatbot(system_prompt="You are a helpful assistant.")

def respond(message: str, history: list) -> str:
    return bot.chat(message)

def respond_streaming(message: str, history: list):
    partial = ""
    for token in bot.chat_stream(message):
        partial += token
        yield partial

def clear_history():
    bot.reset()
    return None

with gr.Blocks(title="AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Chatbot\nPowered by GPT-4o-mini")

    chatbot_ui = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
    clear_btn = gr.Button("Clear conversation", size="sm")

    msg.submit(respond_streaming, [msg, chatbot_ui], chatbot_ui)
    clear_btn.click(clear_history, outputs=chatbot_ui)

demo.launch(share=False)
```

### Step 5: Run

```bash
# CLI version
python cli.py

# Web UI
python app.py
# Open http://localhost:7860
```

---

## Deployment Ideas

- **Hugging Face Spaces**: Free hosting for Gradio apps. Push to a Space repo.
- **Streamlit Cloud**: Alternative UI framework with easy deployment.
- **Railway / Render**: Deploy as a FastAPI + Gradio app on free tier.

---

## Extension Ideas

1. **Domain specialization** — give it a custom system prompt for a specific use case (coding, cooking, travel)
2. **Conversation export** — save chat history to JSON or PDF
3. **Custom personas** — let users choose from multiple bot personalities
4. **Response rating** — add thumbs up/down to collect feedback
5. **RAG integration** — let the bot answer questions from your documents

---

## What to Learn Next

- **Add document Q&A** → [RAG Document Assistant](/projects/rag-document-assistant/)
- **Deploy properly** → [Deploying AI Applications](/blog/deploying-ai-applications/)
- **API deep dive** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
