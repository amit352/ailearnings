import { useState } from "react";

const phases = [
  {
    id: 1,
    emoji: "🌱",
    title: "Phase 1 – AI Foundations",
    duration: "4–6 weeks",
    tag: "Understand the landscape",
    color: "from-green-500 to-emerald-600",
    tagColor: "bg-green-100 text-green-700",
    goal: "Understand how AI/ML works conceptually. No heavy math — just intuition and vocabulary.",
    topics: [
      "What is AI, ML, Deep Learning, and GenAI — and how they relate",
      "Neural networks: inputs, weights, layers, outputs",
      "How models learn (gradient descent, loss functions)",
      "What LLMs are and how they generate text",
      "Key terms: tokens, embeddings, parameters, inference",
    ],
    resources: [
      { label: "3Blue1Brown – Neural Networks (YouTube)", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", time: "2 hrs" },
      { label: "Fast.ai – Practical Deep Learning", url: "https://course.fast.ai", time: "self-paced" },
      { label: "Andrej Karpathy – Intro to LLMs (YouTube)", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g", time: "1 hr" },
      { label: "Google ML Crash Course", url: "https://developers.google.com/machine-learning/crash-course", time: "~15 hrs" },
    ],
    project: "Explore pre-built models on Hugging Face Spaces — try text generation, sentiment analysis, image captioning.",
    milestone: "You can explain what an LLM is and how it works to a non-technical person.",
  },
  {
    id: 2,
    emoji: "⚙️",
    title: "Phase 2 – LLM Setup & Configuration",
    duration: "2–3 weeks",
    tag: "Get your environment ready",
    color: "from-slate-500 to-gray-600",
    tagColor: "bg-slate-100 text-slate-700",
    goal: "Set up your local and cloud AI environment. Know the difference between running, hosting, and calling an LLM.",
    topics: [
      "Python environment setup (pyenv, venv, conda)",
      "GPU vs CPU inference — when you need a GPU",
      "Running LLMs locally via Ollama or LM Studio (free)",
      "Cloud LLM APIs: OpenAI, Anthropic, Google Gemini, Groq",
      "Hugging Face Transformers library basics",
      "Key config parameters: temperature, top-p, top-k, max_tokens, context window",
      "Quantization: what it is and why it lets you run big models on small hardware (GGUF, 4-bit, 8-bit)",
      "Model cards: how to read and pick the right model",
    ],
    config: {
      title: "Minimum Hardware to Run LLMs Locally",
      rows: [
        ["7B model (e.g. Llama 3 8B)", "8 GB RAM, no GPU needed (CPU slow)", "Ollama + llama3"],
        ["13B model", "16 GB RAM or 8 GB VRAM GPU", "Ollama / LM Studio"],
        ["34B model", "32 GB RAM or 16–24 GB VRAM", "Ollama / llama.cpp"],
        ["70B model", "64 GB RAM or 40+ GB VRAM", "llama.cpp / vLLM"],
        ["Cloud API (any size)", "Just internet + API key", "OpenAI / Anthropic / Groq"],
      ],
      note: "💡 For most developers, start with cloud APIs (free tiers available) and run 7B–13B models locally via Ollama for privacy or offline use.",
    },
    resources: [
      { label: "Ollama – Run LLMs locally (free)", url: "https://ollama.com", time: "setup" },
      { label: "LM Studio – GUI for local models (free)", url: "https://lmstudio.ai", time: "setup" },
      { label: "Hugging Face Transformers Quickstart", url: "https://huggingface.co/docs/transformers/quicktour", time: "1 hr" },
      { label: "DeepLearning.AI – Open Source Models with HF (free)", url: "https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/", time: "1.5 hrs" },
    ],
    project: "Run Llama 3 locally via Ollama. Call it from a Python script. Compare output vs Claude/GPT API.",
    milestone: "You have a local LLM running and can call multiple LLM APIs from code.",
  },
  {
    id: 3,
    emoji: "🔧",
    title: "Phase 3 – Prompt Engineering & LLM APIs",
    duration: "3–4 weeks",
    tag: "Start building immediately",
    color: "from-blue-500 to-indigo-600",
    tagColor: "bg-blue-100 text-blue-700",
    goal: "As a dev, you can start building real AI-powered apps right now using APIs — no training needed.",
    topics: [
      "Prompt design: zero-shot, few-shot, chain-of-thought",
      "System prompts and role-based prompting",
      "Using OpenAI / Anthropic / Gemini APIs",
      "Structured outputs (JSON mode)",
      "Handling context windows and token limits",
    ],
    resources: [
      { label: "DeepLearning.AI – Prompt Engineering for Devs (free)", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/", time: "1.5 hrs" },
      { label: "Anthropic Prompt Engineering Docs", url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview", time: "reference" },
      { label: "DeepLearning.AI – Building Systems with LLM API (free)", url: "https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/", time: "1.5 hrs" },
    ],
    project: "Build a CLI or web tool powered by an LLM API — a code reviewer, doc summarizer, or Q&A bot.",
    milestone: "You have a working AI-powered app you built yourself using an LLM API.",
  },
  {
    id: 4,
    emoji: "📚",
    title: "Phase 4 – RAG & Working with Your Own Data",
    duration: "4–5 weeks",
    tag: "Make AI know your domain",
    color: "from-purple-500 to-violet-600",
    tagColor: "bg-purple-100 text-purple-700",
    goal: "Learn to feed your own documents, data, and knowledge into AI systems — critical for real-world apps.",
    topics: [
      "Embeddings and vector databases (Chroma, Pinecone, Weaviate)",
      "Document parsing, chunking strategies",
      "Retrieval-Augmented Generation (RAG) pipeline",
      "Semantic search vs keyword search",
      "RAG evaluation (relevance, faithfulness, correctness)",
    ],
    resources: [
      { label: "DeepLearning.AI – LangChain: Chat with Your Data (free)", url: "https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/", time: "1.5 hrs" },
      { label: "DeepLearning.AI – Building & Evaluating Advanced RAG (free)", url: "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/", time: "1.5 hrs" },
      { label: "Hugging Face NLP Course – Ch. 1–4 (free)", url: "https://huggingface.co/learn/nlp-course/chapter1/1", time: "self-paced" },
    ],
    project: "Build a chatbot that answers questions from your own PDF documents or a knowledge base you care about.",
    milestone: "You can build a RAG pipeline from scratch and evaluate its quality.",
  },
  {
    id: 5,
    emoji: "🤖",
    title: "Phase 5 – Agentic AI",
    duration: "4–5 weeks",
    tag: "AI that thinks and acts",
    color: "from-orange-500 to-amber-600",
    tagColor: "bg-orange-100 text-orange-700",
    goal: "Build AI systems that don't just answer — they plan, use tools, and execute multi-step tasks autonomously.",
    topics: [
      "Agents vs chatbots vs agentic systems",
      "Tool calling and function calling",
      "ReACT pattern: Reason → Act → Observe loop",
      "Agentic workflows: prompt chaining, routing, parallelization, reflection",
      "Memory: short-term (context), long-term (vector store), episodic",
      "Multi-agent systems: orchestrator + worker pattern",
      "MCP (Model Context Protocol) — connect agents to external services",
      "Agent evaluation and failure modes",
    ],
    agentic: {
      title: "How Agentic AI Works — The Core Loop",
      steps: [
        { icon: "💬", step: "1. User Goal", desc: "User gives a high-level task: 'Research the top 5 AI tools launched this month and write a report.'" },
        { icon: "🧠", step: "2. LLM Plans", desc: "The LLM breaks the goal into sub-tasks: search web → read pages → extract info → summarize → format report." },
        { icon: "🔧", step: "3. Tool Use", desc: "Agent calls tools: web_search(), fetch_url(), read_file(), write_file(), call_api() — anything it's been given access to." },
        { icon: "👁️", step: "4. Observe", desc: "Agent receives tool results, evaluates if they're sufficient, and decides the next action." },
        { icon: "🔁", step: "5. Iterate", desc: "Repeats the Reason → Act → Observe loop until the goal is complete or it hits a stop condition." },
        { icon: "✅", step: "6. Output", desc: "Returns final result to user. Optionally asks for clarification or approval before continuing." },
      ],
      patterns: [
        ["Prompt Chaining", "Output of one LLM call feeds into the next. Linear pipeline."],
        ["Routing", "LLM classifies input and routes to the right sub-agent or tool."],
        ["Parallelization", "Multiple agents run simultaneously and results are merged."],
        ["Reflection", "Agent critiques its own output and self-corrects before responding."],
        ["Orchestrator-Worker", "One LLM plans and delegates; worker LLMs execute sub-tasks."],
        ["ReACT", "Reason + Act loop — the most common single-agent pattern."],
      ],
    },
    resources: [
      { label: "DeepLearning.AI – AI Agents in LangGraph (free)", url: "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/", time: "2 hrs" },
      { label: "DeepLearning.AI – Functions, Tools & Agents (free)", url: "https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/", time: "2 hrs" },
      { label: "Anthropic MCP Documentation", url: "https://docs.anthropic.com/en/docs/agents-and-tools/mcp", time: "reference" },
    ],
    project: "Build an agent that can search the web, read a URL, and write a summary report — a mini Perplexity.",
    milestone: "You understand agentic design patterns and have built a working multi-step agent.",
  },
  {
    id: 6,
    emoji: "🏗️",
    title: "Phase 6 – Building & Training LLMs",
    duration: "6–8 weeks",
    tag: "Go deep under the hood",
    color: "from-rose-500 to-pink-600",
    tagColor: "bg-rose-100 text-rose-700",
    goal: "Understand how LLMs are actually built and trained. Know when to fine-tune vs prompt vs RAG.",
    topics: [
      "Transformer architecture deep dive: attention, positional encoding, MLP layers",
      "Pre-training: data collection, cleaning, tokenization (BPE), training loop",
      "Supervised Fine-Tuning (SFT) on custom datasets",
      "RLHF: reward models, PPO, preference data",
      "Parameter-Efficient Fine-Tuning: LoRA, QLoRA, PEFT",
      "Evaluation: perplexity, BLEU, benchmarks, human eval",
      "Inference optimization: quantization, batching, KV cache",
    ],
    training: {
      title: "LLM Training — From Scratch vs Fine-Tuning",
      table: [
        ["Approach", "Data Needed", "Compute", "When to Use"],
        ["Pre-train from scratch", "Billions of tokens", "Thousands of GPUs / months", "Building a foundation model (rare, expensive)"],
        ["Continued Pre-training", "Millions of tokens", "Tens of GPUs / days–weeks", "Specializing in a domain (e.g. medical, legal)"],
        ["Supervised Fine-Tuning (SFT)", "1K–100K examples", "1–8 GPUs / hours–days", "Teaching a specific task or style"],
        ["LoRA / QLoRA Fine-Tuning", "1K–50K examples", "1 GPU / 1–few hours", "Efficient fine-tuning on consumer hardware"],
        ["RLHF", "Preference pairs", "Multiple GPUs / days", "Aligning model behavior with human preferences"],
        ["Prompt Engineering / RAG", "None (zero-shot)", "No training compute", "Most practical apps — use this first"],
      ],
      note: "💡 As a developer, you'll almost always use LoRA/QLoRA fine-tuning or RAG — full pre-training is only for large organizations.",
      stack: [
        ["Hugging Face Transformers", "Load, fine-tune, and run models"],
        ["PEFT library", "LoRA, QLoRA fine-tuning"],
        ["Unsloth", "2x faster fine-tuning, less VRAM (free)"],
        ["Axolotl", "Config-based fine-tuning framework"],
        ["Google Colab / Kaggle", "Free GPU for fine-tuning small models"],
        ["Weights & Biases", "Track training runs (free tier)"],
      ],
    },
    resources: [
      { label: "Andrej Karpathy – Let's build GPT from scratch (YouTube)", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", time: "2 hrs" },
      { label: "DeepLearning.AI – Finetuning LLMs (free)", url: "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/", time: "1 hr" },
      { label: "Hugging Face PEFT Docs (free)", url: "https://huggingface.co/docs/peft/index", time: "reference" },
      { label: "Unsloth – Fast fine-tuning (free, Colab notebooks)", url: "https://github.com/unslothai/unsloth", time: "hands-on" },
      { label: "The Illustrated Transformer – Jay Alammar", url: "https://jalammar.github.io/illustrated-transformer/", time: "1 hr" },
    ],
    project: "Fine-tune Llama 3 8B on a custom dataset using QLoRA + Unsloth on free Google Colab GPU.",
    milestone: "You can fine-tune an open-source model, understand what happened under the hood, and evaluate the result.",
  },
  {
    id: 7,
    emoji: "🚀",
    title: "Phase 7 – Build & Ship Real Projects",
    duration: "Ongoing",
    tag: "Where knowledge becomes mastery",
    color: "from-teal-500 to-cyan-600",
    tagColor: "bg-teal-100 text-teal-700",
    goal: "Real mastery comes from building. Pick projects that excite you and ship them.",
    topics: [
      "Multimodal AI (images, audio, video generation)",
      "Reasoning models (o1, DeepSeek-R1, Chain-of-Thought)",
      "AI in your existing dev stack (Next.js, FastAPI, etc.)",
      "Observability & evals for production AI apps",
      "Staying current: papers, blogs, community",
    ],
    resources: [
      { label: "Hugging Face Diffusion Models Course (free)", url: "https://huggingface.co/learn/diffusion-course/unit0/1", time: "self-paced" },
      { label: "DeepLearning.AI – Reasoning with o1 (free)", url: "https://www.deeplearning.ai/short-courses/reasoning-with-o1/", time: "1 hr" },
      { label: "Vercel AI SDK (for web devs)", url: "https://sdk.vercel.ai/docs", time: "reference" },
      { label: "Latent Space Podcast (stay current)", url: "https://www.latent.space/podcast", time: "ongoing" },
    ],
    project: "Pick one meaningful personal project — a research assistant, coding tool, or AI for your hobby — and launch it publicly.",
    milestone: "You have shipped 2–3 real AI projects and can discuss AI topics with genuine depth.",
  },
];

const tools = [
  { name: "Google Colab", desc: "Free cloud GPUs for running code", url: "https://colab.research.google.com" },
  { name: "Hugging Face", desc: "Models, datasets, free Spaces", url: "https://huggingface.co" },
  { name: "Ollama", desc: "Run LLMs locally for free", url: "https://ollama.com" },
  { name: "LM Studio", desc: "GUI to run local models", url: "https://lmstudio.ai" },
  { name: "LangChain", desc: "Build RAG & agent apps", url: "https://python.langchain.com" },
  { name: "Unsloth", desc: "Fast free fine-tuning", url: "https://github.com/unslothai/unsloth" },
];

export default function Roadmap() {
  const [open, setOpen] = useState(null);
  const [tab, setTab] = useState({});

  const setPhaseTab = (id, t, e) => { e.stopPropagation(); setTab(prev => ({ ...prev, [id]: t })); };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">🧠 AI Zero → Hero</h1>
          <p className="text-gray-400 text-sm">For Software Developers · 4–6 hrs/week · Domain Knowledge + Building</p>
          <div className="flex flex-wrap justify-center gap-2 mt-3">
            <span className="bg-gray-800 text-gray-300 text-xs px-3 py-1 rounded-full">~14 months total</span>
            <span className="bg-gray-800 text-gray-300 text-xs px-3 py-1 rounded-full">Mostly free resources</span>
            <span className="bg-gray-800 text-gray-300 text-xs px-3 py-1 rounded-full">Project-based</span>
          </div>
        </div>

        <div className="space-y-3 mb-10">
          {phases.map((p, i) => {
            const activeTab = tab[p.id] || "learn";
            return (
              <div key={p.id} className="relative">
                {i < phases.length - 1 && <div className="absolute left-6 top-full w-0.5 h-3 bg-gray-700 z-10" />}
                <div
                  className={`rounded-xl border cursor-pointer transition-all duration-200 ${open === p.id ? "border-gray-500 bg-gray-900" : "border-gray-800 bg-gray-900 hover:border-gray-600"}`}
                  onClick={() => setOpen(open === p.id ? null : p.id)}
                >
                  <div className="flex items-center gap-3 p-4">
                    <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${p.color} flex items-center justify-center text-lg flex-shrink-0`}>{p.emoji}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-semibold text-sm">{p.title}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${p.tagColor}`}>{p.tag}</span>
                      </div>
                      <p className="text-gray-500 text-xs mt-0.5">{p.duration}</p>
                    </div>
                    <div className="text-gray-500 text-lg flex-shrink-0">{open === p.id ? "▲" : "▼"}</div>
                  </div>

                  {open === p.id && (
                    <div className="border-t border-gray-800">
                      {/* Tabs */}
                      <div className="flex gap-1 p-3 pb-0">
                        {["learn", "resources", ...(p.config ? ["setup"] : []), ...(p.agentic ? ["how it works"] : []), ...(p.training ? ["training"] : []), "project"].map(t => (
                          <button key={t} onClick={e => setPhaseTab(p.id, t, e)}
                            className={`text-xs px-3 py-1.5 rounded-t-lg capitalize transition-colors ${activeTab === t ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"}`}>
                            {t}
                          </button>
                        ))}
                      </div>

                      <div className="p-4 space-y-3">
                        <p className="text-gray-300 text-sm">{p.goal}</p>

                        {activeTab === "learn" && (
                          <ul className="space-y-1.5">
                            {p.topics.map((t, i) => (
                              <li key={i} className="text-sm text-gray-300 flex gap-2">
                                <span className="text-gray-600 flex-shrink-0 mt-0.5">•</span>{t}
                              </li>
                            ))}
                          </ul>
                        )}

                        {activeTab === "resources" && (
                          <div className="space-y-1.5">
                            {p.resources.map((r, i) => (
                              <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
                                className="flex items-center justify-between bg-gray-800 hover:bg-gray-700 rounded-lg px-3 py-2 transition-colors group"
                                onClick={e => e.stopPropagation()}>
                                <span className="text-sm text-blue-400 group-hover:text-blue-300">{r.label}</span>
                                <span className="text-xs text-gray-500 ml-2 flex-shrink-0">{r.time}</span>
                              </a>
                            ))}
                          </div>
                        )}

                        {activeTab === "setup" && p.config && (
                          <div className="space-y-3">
                            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{p.config.title}</p>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="text-gray-500 border-b border-gray-700">
                                    <th className="text-left pb-2 pr-3">Model Size</th>
                                    <th className="text-left pb-2 pr-3">Hardware</th>
                                    <th className="text-left pb-2">Tool</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {p.config.rows.map((r, i) => (
                                    <tr key={i} className="border-b border-gray-800">
                                      <td className="py-2 pr-3 text-blue-400 font-medium">{r[0]}</td>
                                      <td className="py-2 pr-3 text-gray-300">{r[1]}</td>
                                      <td className="py-2 text-gray-400">{r[2]}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                            <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-300">{p.config.note}</div>
                          </div>
                        )}

                        {activeTab === "how it works" && p.agentic && (
                          <div className="space-y-4">
                            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{p.agentic.title}</p>
                            <div className="space-y-2">
                              {p.agentic.steps.map((s, i) => (
                                <div key={i} className="flex gap-3 bg-gray-800 rounded-lg p-3">
                                  <span className="text-xl flex-shrink-0">{s.icon}</span>
                                  <div>
                                    <p className="text-sm font-semibold text-white">{s.step}</p>
                                    <p className="text-xs text-gray-400 mt-0.5">{s.desc}</p>
                                  </div>
                                </div>
                              ))}
                            </div>
                            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mt-4">Agentic Design Patterns</p>
                            <div className="grid grid-cols-1 gap-2">
                              {p.agentic.patterns.map(([name, desc], i) => (
                                <div key={i} className="bg-gray-800 rounded-lg px-3 py-2 flex gap-2">
                                  <span className="text-orange-400 font-semibold text-xs w-36 flex-shrink-0">{name}</span>
                                  <span className="text-gray-400 text-xs">{desc}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {activeTab === "training" && p.training && (
                          <div className="space-y-4">
                            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{p.training.title}</p>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="text-gray-500 border-b border-gray-700">
                                    {p.training.table[0].map((h, i) => (
                                      <th key={i} className="text-left pb-2 pr-3">{h}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {p.training.table.slice(1).map((r, i) => (
                                    <tr key={i} className="border-b border-gray-800">
                                      <td className="py-2 pr-3 text-rose-400 font-medium">{r[0]}</td>
                                      <td className="py-2 pr-3 text-gray-300">{r[1]}</td>
                                      <td className="py-2 pr-3 text-gray-300">{r[2]}</td>
                                      <td className="py-2 text-gray-400">{r[3]}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                            <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-300">{p.training.note}</div>
                            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Free Training Stack</p>
                            <div className="grid grid-cols-1 gap-1.5">
                              {p.training.stack.map(([tool, desc], i) => (
                                <div key={i} className="bg-gray-800 rounded-lg px-3 py-2 flex gap-2">
                                  <span className="text-rose-400 font-semibold text-xs w-40 flex-shrink-0">{tool}</span>
                                  <span className="text-gray-400 text-xs">{desc}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {activeTab === "project" && (
                          <div className="space-y-3">
                            <div className="bg-gray-800 rounded-lg p-3">
                              <p className="text-xs font-semibold text-yellow-400 uppercase tracking-wider mb-1">🛠 Phase Project</p>
                              <p className="text-sm text-gray-200">{p.project}</p>
                            </div>
                            <div className="bg-gray-800 rounded-lg p-3">
                              <p className="text-xs font-semibold text-green-400 uppercase tracking-wider mb-1">✅ Milestone</p>
                              <p className="text-sm text-gray-200">{p.milestone}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Tools */}
        <div className="mb-8">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">🛠 Essential Free Tools</p>
          <div className="grid grid-cols-2 gap-2">
            {tools.map((t, i) => (
              <a key={i} href={t.url} target="_blank" rel="noopener noreferrer"
                className="bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-lg p-3 transition-colors">
                <p className="text-sm font-medium text-blue-400">{t.name}</p>
                <p className="text-xs text-gray-500 mt-0.5">{t.desc}</p>
              </a>
            ))}
          </div>
        </div>

        {/* Principles */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">💡 Principles for the Journey</p>
          <div className="space-y-2">
            {[
              ["Build early, build often", "Don't wait until you feel 'ready'. Ship something after Phase 3."],
              ["Concepts > memorization", "Understand the why. Tools change fast, intuition doesn't."],
              ["Your dev skills are a superpower", "You can go from idea to working AI app faster than most beginners."],
              ["Prompt → RAG → Fine-tune", "Always try the simplest approach first before going deeper."],
              ["Projects are your portfolio", "Even personal projects signal domain expertise better than certificates."],
            ].map(([title, desc], i) => (
              <div key={i} className="flex gap-2">
                <span className="text-gray-600 flex-shrink-0 mt-0.5">→</span>
                <div>
                  <span className="text-sm font-medium text-gray-200">{title}: </span>
                  <span className="text-sm text-gray-400">{desc}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        <p className="text-center text-gray-600 text-xs mt-6">Click each phase to expand · Use tabs to navigate sections</p>
      </div>
    </div>
  );
}
