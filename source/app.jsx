    const { useState } = React;

    // ── SEO helper — call at the top of each page component ──────────────────
    function useSeo(title, description) {
      React.useEffect(() => {
        document.title = title;
        const el = document.querySelector('meta[name="description"]');
        if (el) el.setAttribute('content', description);
      }, []);
    }

    // ─────────────────────────────────────────────────────────────
    // MASTER APP — defined last, but forward-declared here as shell
    // Real App function is at the bottom of this script
    // ─────────────────────────────────────────────────────────────

    // ═══════════════════════════════════════════════════════════
    // ROADMAP COMPONENT  (ai_roadmap.tsx)
    // ═══════════════════════════════════════════════════════════
    const roadmapPhases = [
      {
        id: 1, icon: Sprout, title: "Phase 1 – AI Foundations", duration: "4–6 weeks",
        tag: "Understand the landscape", color: "from-green-500 to-emerald-600",
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
          { label: "Karpathy – Neural Networks: Zero to Hero (YouTube)", url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ", time: "~10 hrs", note: "The gold standard. Builds intuition from scratch — best free resource for developers." },
          { label: "Andrej Karpathy – Intro to LLMs (YouTube)", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g", time: "1 hr", note: "1-hour masterclass on how LLMs work. Best single video for any beginner." },
          { label: "3Blue1Brown – Neural Networks (YouTube)", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", time: "2 hrs", note: "Beautiful visual explanations of neural network math — no prior knowledge needed." },
          { label: "Google ML Crash Course", url: "https://developers.google.com/machine-learning/crash-course", time: "~15 hrs", note: "Structured hands-on ML fundamentals from Google. Great for systematic learners." },
          { label: "Fast.ai – Practical Deep Learning", url: "https://course.fast.ai", time: "self-paced", note: "Top-down approach — build first, understand later. Best for developers who learn by doing." },
        ],
        project: "Explore pre-built models on Hugging Face Spaces — try text generation, sentiment analysis, image captioning.",
        milestone: "You can explain what an LLM is and how it works to a non-technical person.",
      },
      {
        id: 2, icon: Settings, title: "Phase 2 – LLM Setup & Configuration", duration: "2–3 weeks",
        tag: "Get your environment ready", color: "from-slate-500 to-gray-600",
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
          { label: "Ollama – Run LLMs locally (free)", url: "https://ollama.com", time: "setup", note: "Easiest way to run any open-source LLM locally. One command to get started." },
          { label: "LM Studio – GUI for local models (free)", url: "https://lmstudio.ai", time: "setup", note: "No-code desktop app for running and chatting with local models. Great for non-CLI setup." },
          { label: "Hugging Face Transformers Quickstart", url: "https://huggingface.co/docs/transformers/quicktour", time: "1 hr", note: "Official HF docs — load and run any model in a few lines of Python." },
          { label: "DeepLearning.AI – Open Source Models with HF (free)", url: "https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/", time: "1.5 hrs", note: "Hands-on course on loading, configuring, and comparing open-source models." },
        ],
        project: "Run Llama 3 locally via Ollama. Call it from a Python script. Compare output vs Claude/GPT API.",
        milestone: "You have a local LLM running and can call multiple LLM APIs from code.",
      },
      {
        id: 3, icon: Wrench, title: "Phase 3 – Prompt Engineering & LLM APIs", duration: "3–4 weeks",
        tag: "Start building immediately", color: "from-blue-500 to-indigo-600",
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
          { label: "DeepLearning.AI – Prompt Engineering for Devs (free)", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/", time: "1.5 hrs", note: "Canonical course by Andrew Ng. Covers every core technique with hands-on Python notebooks." },
          { label: "DeepLearning.AI – Building Systems with LLM API (free)", url: "https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/", time: "1.5 hrs", note: "Goes from single prompts to multi-step LLM pipelines. Best bridge into real app development." },
          { label: "Anthropic Prompt Engineering Docs", url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview", time: "reference", note: "Official Anthropic guide — the most detailed and up-to-date reference for production prompting." },
          { label: "OpenAI Prompt Engineering Guide", url: "https://platform.openai.com/docs/guides/prompt-engineering", time: "reference", note: "Official OpenAI reference for prompt techniques. Complements Anthropic's guide well." },
        ],
        project: "Build a CLI or web tool powered by an LLM API — a code reviewer, doc summarizer, or Q&A bot.",
        milestone: "You have a working AI-powered app you built yourself using an LLM API.",
      },
      {
        id: 4, icon: BookOpen, title: "Phase 4 – RAG & Working with Your Own Data", duration: "4–5 weeks",
        tag: "Make AI know your domain", color: "from-purple-500 to-violet-600",
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
          { label: "DeepLearning.AI – LangChain: Chat with Your Data (free)", url: "https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/", time: "1.5 hrs", note: "Best hands-on intro to RAG — document loading, chunking, retrieval, and generation in one course." },
          { label: "DeepLearning.AI – Building & Evaluating Advanced RAG (free)", url: "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/", time: "1.5 hrs", note: "Goes beyond naive RAG — sentence-window retrieval, auto-merging, and evaluation metrics." },
          { label: "Hugging Face NLP Course – Ch. 1–4 (free)", url: "https://huggingface.co/learn/nlp-course/chapter1/1", time: "self-paced", note: "Official HF course on transformers and embeddings — the theory behind why RAG works." },
        ],
        project: "Build a chatbot that answers questions from your own PDF documents or a knowledge base you care about.",
        milestone: "You can build a RAG pipeline from scratch and evaluate its quality.",
      },
      {
        id: 5, icon: Bot, title: "Phase 5 – Agentic AI", duration: "4–5 weeks",
        tag: "AI that thinks and acts", color: "from-orange-500 to-amber-600",
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
            { icon: MessageSquare, step: "1. User Goal", desc: "User gives a high-level task: 'Research the top 5 AI tools launched this month and write a report.'" },
            { icon: Brain, step: "2. LLM Plans", desc: "The LLM breaks the goal into sub-tasks: search web → read pages → extract info → summarize → format report." },
            { icon: "🔧", step: "3. Tool Use", desc: "Agent calls tools: web_search(), fetch_url(), read_file(), write_file(), call_api() — anything it's been given access to." },
            { icon: Eye, step: "4. Observe", desc: "Agent receives tool results, evaluates if they're sufficient, and decides the next action." },
            { icon: RotateCcw, step: "5. Iterate", desc: "Repeats the Reason → Act → Observe loop until the goal is complete or it hits a stop condition." },
            { icon: Check, step: "6. Output", desc: "Returns final result to user. Optionally asks for clarification or approval before continuing." },
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
          { label: "DeepLearning.AI – AI Agents in LangGraph (free)", url: "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/", time: "2 hrs", note: "Best practical agent course — builds ReACT agents and multi-agent systems step by step." },
          { label: "DeepLearning.AI – AI Agentic Design Patterns with AutoGen (free)", url: "https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/", time: "1 hr", note: "Covers orchestrator-worker, reflection, and multi-agent patterns using Microsoft AutoGen." },
          { label: "DeepLearning.AI – Functions, Tools & Agents (free)", url: "https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/", time: "2 hrs", note: "Deep dive into tool calling and function use — the foundation of every agentic system." },
          { label: "Anthropic MCP Documentation", url: "https://docs.anthropic.com/en/docs/agents-and-tools/mcp", time: "reference", note: "Official spec for Model Context Protocol — how to connect agents to external services and tools." },
        ],
        project: "Build an agent that can search the web, read a URL, and write a summary report — a mini Perplexity.",
        milestone: "You understand agentic design patterns and have built a working multi-step agent.",
      },
      {
        id: 6, icon: Building2, title: "Phase 6 – Building & Training LLMs", duration: "6–8 weeks",
        tag: "Go deep under the hood", color: "from-rose-500 to-pink-600",
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
          { label: "Andrej Karpathy – Let's build GPT from scratch (YouTube)", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", time: "2 hrs", note: "Legendary 2-hour video — builds a working GPT in Python from first principles." },
          { label: "DeepLearning.AI – Finetuning LLMs (free)", url: "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/", time: "1 hr", note: "Concise practical walkthrough of SFT, LoRA fine-tuning and how to evaluate results." },
          { label: "Hugging Face PEFT Docs (free)", url: "https://huggingface.co/docs/peft/index", time: "reference", note: "Official reference for LoRA, QLoRA, and all parameter-efficient fine-tuning methods." },
          { label: "Unsloth – Fast fine-tuning (free, Colab notebooks)", url: "https://github.com/unslothai/unsloth", time: "hands-on", note: "2× faster, 60% less VRAM than stock HF. Best way to fine-tune on free Colab GPUs." },
          { label: "The Illustrated Transformer – Jay Alammar", url: "https://jalammar.github.io/illustrated-transformer/", time: "1 hr", note: "The most-read visual explanation of transformer architecture. Required reading for Phase 6." },
        ],
        project: "Fine-tune Llama 3 8B on a custom dataset using QLoRA + Unsloth on free Google Colab GPU.",
        milestone: "You can fine-tune an open-source model, understand what happened under the hood, and evaluate the result.",
      },
      {
        id: 7, icon: Rocket, title: "Phase 7 – Build & Ship Real Projects", duration: "Ongoing",
        tag: "Where knowledge becomes mastery", color: "from-teal-500 to-cyan-600",
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
          { label: "Hugging Face Diffusion Models Course (free)", url: "https://huggingface.co/learn/diffusion-course/unit0/1", time: "self-paced", note: "Best free course on image generation models — covers diffusion theory and Stable Diffusion." },
          { label: "DeepLearning.AI – Reasoning with o1 (free)", url: "https://www.deeplearning.ai/short-courses/reasoning-with-o1/", time: "1 hr", note: "Covers chain-of-thought reasoning models and how to use them differently from standard LLMs." },
          { label: "Vercel AI SDK (for web devs)", url: "https://sdk.vercel.ai/docs", time: "reference", note: "Best way to add streaming LLM responses to a Next.js or React app. Production-ready." },
          { label: "Latent Space Podcast (free)", url: "https://www.latent.space/podcast", time: "ongoing", note: "Best podcast for AI engineers — interviews with people building real AI systems and research." },
        ],
        project: "Pick one meaningful personal project — a research assistant, coding tool, or AI for your hobby — and launch it publicly.",
        milestone: "You have shipped 2–3 real AI projects and can discuss AI topics with genuine depth.",
      },
    ];

    const roadmapTools = [
      { name: "Google Colab", desc: "Free cloud GPUs for running code", url: "https://colab.research.google.com" },
      { name: "Hugging Face", desc: "Models, datasets, free Spaces", url: "https://huggingface.co" },
      { name: "Ollama", desc: "Run LLMs locally for free", url: "https://ollama.com" },
      { name: "LM Studio", desc: "GUI to run local models", url: "https://lmstudio.ai" },
      { name: "LangChain", desc: "Build RAG & agent apps", url: "https://python.langchain.com" },
      { name: "Unsloth", desc: "Fast free fine-tuning", url: "https://github.com/unslothai/unsloth" },
    ];

    function Roadmap() {
      useSeo(
        "AI Learning Hub – Zero to Hero Roadmap for Developers",
        "Free 7-phase AI roadmap for software developers. Master LLMs, Prompt Engineering, RAG, and Agentic AI with curated free resources and hands-on projects."
      );
      const [open, setOpen] = useState(1);
      const [tab, setTab] = useState({});
      const [visited, setVisited] = useState({ "1-learn": true });
      const setPhaseTab = (id, t, e) => {
        e.stopPropagation();
        setTab(prev => ({ ...prev, [id]: t }));
        setVisited(prev => ({ ...prev, [`${id}-${t}`]: true }));
      };

      const [done, setDone] = React.useState(() => {
        try { return JSON.parse(localStorage.getItem("ai_progress") || "{}"); } catch { return {}; }
      });
      const toggleTopic = (phaseId, idx, e) => {
        e.stopPropagation();
        setDone(prev => {
          const key = `${phaseId}-${idx}`;
          const next = { ...prev, [key]: !prev[key] };
          localStorage.setItem("ai_progress", JSON.stringify(next));
          return next;
        });
      };
      const phaseProgress = (p) => {
        const total = p.topics.length;
        const completed = p.topics.filter((_, i) => done[`${p.id}-${i}`]).length;
        return { completed, total, pct: total ? Math.round((completed / total) * 100) : 0 };
      };
      const totalTopics = roadmapPhases.reduce((s, p) => s + p.topics.length, 0);
      const totalDone = roadmapPhases.reduce((s, p) => s + p.topics.filter((_, i) => done[`${p.id}-${i}`]).length, 0);
      const overallPct = totalTopics ? Math.round((totalDone / totalTopics) * 100) : 0;

      const [tipDismissed, setTipDismissed] = React.useState(() => {
        try { return localStorage.getItem("ai_tips_dismissed") === "1"; } catch { return false; }
      });
      const dismissTip = () => {
        setTipDismissed(true);
        try { localStorage.setItem("ai_tips_dismissed", "1"); } catch {}
      };

      const scrollToPhase1 = () => {
        setOpen(1);
        setTimeout(() => {
          const el = document.getElementById("roadmap-start");
          if (el) {
            const top = el.getBoundingClientRect().top + window.scrollY - 64;
            window.scrollTo({ top, behavior: "smooth" });
          }
        }, 50);
      };

      return (
        <div className="text-gray-100 font-sans">

          {/* ── HERO — full viewport ── */}
          <div className="min-h-[calc(100vh-52px)] flex flex-col justify-between px-4 pt-8 pb-6 max-w-3xl mx-auto">

            {/* Top: badge + headline + subtitle + stats */}
            <div className="text-center">
              <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 rounded-full px-3 py-1 mb-5">
                <BrainCircuit size={12} className="text-blue-400"/>
                <span className="text-xs text-blue-400 font-medium">Updated March 2026 · Free · For software developers</span>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold mb-4 leading-[1.1] tracking-tight">
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">AI Engineering</span>
                <br />
                <span className="text-white">Roadmap</span>
              </h1>
              <p className="text-gray-300 text-lg max-w-xl mx-auto mb-1 leading-relaxed">
                The structured path from zero to{" "}
                <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent font-semibold">production-ready AI</span>
              </p>
              <p className="text-gray-400 text-sm max-w-lg mx-auto mb-6">
                Curated free resources · hands-on projects · milestones at every phase
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[[Layers,"blue","7","Phases"],[Clock,"purple","~14","months"],[BookOpen,"green","50+","Free resources"],[Wrench,"orange","7","Projects"]].map(([Icon,color,val,label]) => (
                  <span key={label} className="inline-flex items-center gap-1.5 bg-gray-800/60 border border-white/8 text-gray-300 text-xs px-3 py-1.5 rounded-full">
                    <Icon size={12} className={`text-${color}-400`}/><strong className="text-white">{val}</strong> {label}
                  </span>
                ))}
              </div>
            </div>

            {/* Middle: USP cards + section pills */}
            <div className="space-y-6 my-8">
              {/* USP — 3 cards horizontal */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {[
                  { icon: BookOpen, color: "text-blue-400",   bg: "bg-blue-500/8 border-blue-500/20",   title: "Not just links",       desc: "Topics, projects & milestones — not a bookmark list." },
                  { icon: Code2,    color: "text-purple-400", bg: "bg-purple-500/8 border-purple-500/20", title: "Built for developers", desc: "No math papers. Ship AI products from day one." },
                  { icon: Check,    color: "text-green-400",  bg: "bg-green-500/8 border-green-500/20",  title: "Track progress",       desc: "Check off topics. Saved in your browser forever." },
                ].map(({icon: Icon, color, bg, title, desc}) => (
                  <div key={title} className={`flex flex-col gap-2 rounded-xl border ${bg} px-4 py-4`}>
                    <Icon size={18} className={color}/>
                    <p className="text-sm font-semibold text-white">{title}</p>
                    <p className="text-xs text-gray-400 leading-relaxed">{desc}</p>
                  </div>
                ))}
              </div>

              {/* Section links — pills with hover tooltip */}
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wider mb-3 text-center">Also inside this tool</p>
                <div className="flex flex-wrap justify-center gap-2">
                  {[
                    { slug: "prep-plan",      icon: Calendar,    color: "text-orange-400", label: "Prep Plan",       when: "Already know some AI basics?", why: "A focused 6-week sprint to go from zero to building your first AI app — faster than the full roadmap." },
                    { slug: "prompt-eng",     icon: Zap,         color: "text-yellow-400", label: "Prompt Eng",      when: "Starting Phase 3?",            why: "Master techniques, templates, and a 6-week plan to become genuinely good at prompting — the skill every AI app depends on." },
                    { slug: "genai-guide",    icon: Cpu,         color: "text-purple-400", label: "GenAI Guide",     when: "Curious how AI actually works?", why: "A deep dive into text, code, image, and audio AI — how each domain works under the hood, with tools to explore." },
                    { slug: "readiness",      icon: CheckCircle, color: "text-green-400",  label: "Readiness Check", when: "About to move phases?",        why: "Before moving on, check the green flags and move-on rules to make sure you're actually ready — not just rushed." },
                    { slug: "resources",      icon: BookOpen,    color: "text-blue-400",   label: "Books & Courses", when: "Want to go deeper?",            why: "The best books and paid/free courses mapped to each phase — for when the free resources aren't enough." },
                    { slug: "assessment",     icon: BarChart2,   color: "text-pink-400",   label: "Assessment",      when: "Planning your journey?",       why: "See exactly where you'll stand as an AI engineer after completing the roadmap — what roles you'll be ready for." },
                    { slug: "beyond-roadmap", icon: Compass,     color: "text-teal-400",   label: "Beyond Roadmap",  when: "Finished the roadmap?",        why: "What to learn next, what gaps remain, and what projects will take you from competent to expert." },
                  ].map(({ slug, icon: Icon, color, label, when, why }) => (
                    <div key={slug} className="relative group">
                      <a href={tabPath(slug)}
                        className="inline-flex items-center gap-1.5 bg-gray-800/60 hover:bg-gray-700/60 border border-white/8 hover:border-white/15 text-gray-300 hover:text-white text-xs px-3 py-1.5 rounded-full transition-colors">
                        <Icon size={11} className={color}/>{label}
                      </a>
                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-50">
                        <div className="bg-gray-900 border border-white/10 rounded-xl px-3 py-2.5 shadow-xl">
                          <p className="text-xs font-semibold text-white mb-0.5">{label}</p>
                          <p className="text-xs text-blue-400 mb-1">{when}</p>
                          <p className="text-xs text-gray-400 leading-relaxed">{why}</p>
                        </div>
                        <div className="w-2 h-2 bg-gray-900 border-r border-b border-white/10 rotate-45 mx-auto -mt-1"/>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Bottom: progress → CTA → scroll hint */}
            <div className="flex flex-col items-center gap-5">
              {totalDone > 0 ? (
                <div className="w-full max-w-sm bg-gray-800/60 border border-blue-500/20 rounded-xl px-4 py-3">
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-gray-400 font-medium">Your progress</span>
                    <span className="text-blue-400 font-semibold">{totalDone}/{totalTopics} topics · {overallPct}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500" style={{width: `${overallPct}%`}}/>
                  </div>
                </div>
              ) : null}

              <div className="flex flex-col sm:flex-row items-center gap-3">
                <button onClick={scrollToPhase1}
                  className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-bold px-8 py-3 rounded-xl transition-all shadow-[0_0_24px_rgba(59,130,246,0.35)] hover:shadow-[0_0_32px_rgba(59,130,246,0.5)] text-sm">
                  {totalDone > 0 ? "Continue learning" : "Start the roadmap"} <ArrowRight size={15}/>
                </button>
                <a href="https://github.com/amit352/ailearnings/discussions" target="_blank" rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-gray-400 hover:text-white text-sm transition-colors">
                  <MessageSquare size={14}/> Share feedback
                </a>
              </div>

              <button onClick={scrollToPhase1} className="flex flex-col items-center gap-1 text-gray-400 hover:text-gray-200 transition-colors">
                <span className="text-xs tracking-wider uppercase">Scroll to roadmap</span>
                <ChevronDown size={16} className="animate-bounce"/>
              </button>
            </div>
          </div>

          {/* ── ROADMAP SECTION ── */}
          <div className="px-4 pb-8 max-w-3xl mx-auto">
            <div id="roadmap-start" className="flex items-center gap-3 mb-6 pt-2">
              <div className="flex-1 h-px bg-gray-800"/>
              <span className="text-xs text-gray-400 uppercase tracking-widest font-medium">The Roadmap</span>
              <div className="flex-1 h-px bg-gray-800"/>
            </div>

            {!tipDismissed && (
              <div className="mb-6 bg-gray-800/60 border border-blue-500/20 rounded-xl px-4 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <Lightbulb size={14} className="text-blue-400 flex-shrink-0"/>
                    <span className="text-xs font-semibold text-blue-400">How to use this roadmap</span>
                  </div>
                  <button onClick={dismissTip} className="text-gray-400 hover:text-gray-300 flex-shrink-0 transition-colors" aria-label="Dismiss">
                    <X size={14}/>
                  </button>
                </div>
                <div className="mt-2.5 grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {[
                    [Check, "Check off topics", "Click any topic in the Learn tab to track your progress. It's saved in your browser."],
                    [Layers, "Explore tabs per phase", "Each phase has Learn, Resources, and Project tabs — the project tells you what to build."],
                    [BookOpen, "More sections above", "Use the top navigation to explore Alt Resources, Knowledge Gaps, and Prompt Engineering guides."],
                  ].map(([Icon, title, desc]) => (
                    <div key={title} className="flex items-start gap-2">
                      <Icon size={12} className="text-gray-400 flex-shrink-0 mt-0.5"/>
                      <div>
                        <p className="text-xs font-medium text-gray-300">{title}</p>
                        <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">{desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="relative mb-10">
              {/* Vertical timeline line */}
              <div className="absolute left-6 top-6 bottom-6 w-0.5 bg-gradient-to-b from-green-500 via-blue-500 to-purple-500 opacity-30" />
              <div className="space-y-6">
              {roadmapPhases.map((p, i) => {
                const activeTab = tab[p.id] || "learn";
                return (
                  <div key={p.id} id={`phase-card-${p.id}`} className="relative flex gap-4">
                    {/* Timeline node */}
                    <div className="flex flex-col items-center flex-shrink-0 z-10">
                      <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${p.color} flex items-center justify-center text-xl shadow-lg ring-2 ring-gray-900`}><p.icon size={20} className="text-white"/></div>
                    </div>
                    {/* Card */}
                    <div className="flex-1 min-w-0">
                    <div
                      className={`rounded-xl border cursor-pointer transition-all duration-200 backdrop-blur-sm ${open === p.id ? "border-blue-500/30 bg-gray-900/80 shadow-[0_0_30px_rgba(59,130,246,0.08)]" : "border-white/8 bg-gray-900/60 hover:border-white/15 hover:bg-gray-900/80"}`}
                      onClick={() => setOpen(open === p.id ? null : p.id)}
                    >
                      <div className="flex items-start gap-3 p-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-semibold text-sm">{p.title}</span>
                            <span className={`text-xs px-2 py-0.5 rounded-full ${p.tagColor}`}>{p.tag}</span>
                          </div>
                          <div className="flex items-center gap-2 mt-1 flex-wrap">
                            <span className="text-gray-400 text-xs">{p.duration}</span>
                            <span className="text-gray-400 text-xs">·</span>
                            <span className="text-gray-400 text-xs">{p.topics.length} topics · {p.resources.length} resources</span>
                            {(() => { const pg = phaseProgress(p); return pg.completed > 0 ? (
                              <span className="text-xs text-green-400 font-medium">{pg.completed}/{pg.total} done</span>
                            ) : null; })()}
                          </div>
                          {(() => { const pg = phaseProgress(p); return pg.completed > 0 ? (
                            <div className="h-1 bg-gray-700 rounded-full mt-1.5 overflow-hidden w-full">
                              <div className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full transition-all duration-300" style={{width: `${pg.pct}%`}}/>
                            </div>
                          ) : null; })()}
                          {open !== p.id && (
                            <div className="mt-2 flex items-start gap-1.5">
                              <Check size={11} className="text-green-400 flex-shrink-0 mt-0.5"/>
                              <p className="text-xs text-gray-400 italic leading-snug">{p.milestone}</p>
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-2 flex-shrink-0 mt-0.5">
                          {open !== p.id && <span className="text-xs text-gray-400 hidden sm:inline">expand</span>}
                          <div className={`text-gray-400 ${open !== p.id ? "animate-bounce" : ""}`}>{open === p.id ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</div>
                        </div>
                      </div>

                      {open === p.id && (
                        <div className="border-t border-gray-800">
                          <div className="flex gap-1 p-3 pb-0 flex-wrap">
                            {["learn", "resources", ...(p.config ? ["setup"] : []), ...(p.agentic ? ["how it works"] : []), ...(p.training ? ["training"] : []), "project"].map(t => {
                              const isVisited = visited[`${p.id}-${t}`];
                              const isActive = activeTab === t;
                              return (
                                <button key={t} onClick={e => setPhaseTab(p.id, t, e)}
                                  className={`relative text-xs px-3 py-1.5 rounded-t-lg capitalize transition-colors ${isActive ? "bg-gray-700 text-white" : "text-gray-400 hover:text-gray-300"}`}>
                                  {t}
                                  {!isVisited && !isActive && (
                                    <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"/>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                          <div className="p-4 space-y-3">
                            <p className="text-gray-300 text-sm">{p.goal}</p>
                            {activeTab === "learn" && (
                              <ul className="space-y-1.5">
                                {totalDone === 0 && (
                                  <li className="flex items-center gap-1.5 text-xs text-gray-400 pb-1">
                                    <MousePointer size={11} className="flex-shrink-0"/>
                                    Click any topic to check it off and track your progress
                                  </li>
                                )}
                                {p.topics.map((t, i) => {
                                  const checked = !!done[`${p.id}-${i}`];
                                  return (
                                    <li key={i}
                                      className={`text-sm flex gap-2 items-start cursor-pointer select-none rounded-lg px-2 py-1.5 transition-colors ${checked ? "text-gray-400 bg-green-900/10" : "text-gray-300 hover:bg-gray-800"}`}
                                      onClick={e => toggleTopic(p.id, i, e)}>
                                      <span className={`flex-shrink-0 mt-0.5 w-4 h-4 rounded border flex items-center justify-center transition-colors ${checked ? "bg-green-600 border-green-600" : "border-gray-600"}`}>
                                        {checked && <Check size={10} className="text-white"/>}
                                      </span>
                                      <span className={checked ? "line-through" : ""}>{t}</span>
                                    </li>
                                  );
                                })}
                              </ul>
                            )}
                            {activeTab === "learn" && (
                              <button onClick={e => setPhaseTab(p.id, "resources", e)}
                                className="w-full mt-2 flex items-center justify-center gap-2 text-xs text-blue-400 hover:text-blue-300 bg-blue-500/8 hover:bg-blue-500/15 border border-blue-500/20 rounded-lg py-2 transition-colors">
                                Ready? See curated resources for this phase <ArrowRight size={12}/>
                              </button>
                            )}
                            {activeTab === "resources" && (
                              <div className="space-y-1.5">
                                <p className="text-xs text-gray-400 mb-2">Ordered by recommendation — start from the top.</p>
                                {p.resources.map((r, i) => (
                                  <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
                                    className={`flex items-start justify-between rounded-lg px-3 py-2.5 transition-colors group ${i === 0 ? "bg-blue-900/20 border border-blue-500/25 hover:bg-blue-900/30" : "bg-gray-800 hover:bg-gray-700"}`}
                                    onClick={e => e.stopPropagation()}>
                                    <div className="flex items-start gap-2.5 min-w-0">
                                      <span className={`flex-shrink-0 text-xs font-bold w-5 text-center mt-0.5 ${i === 0 ? "text-yellow-400" : "text-gray-600"}`}>
                                        {i === 0 ? "★" : `${i + 1}`}
                                      </span>
                                      <div>
                                        <div className="flex items-center gap-2 flex-wrap">
                                          <span className={`text-sm group-hover:text-blue-300 ${i === 0 ? "text-blue-300 font-medium" : "text-blue-400"}`}>{r.label}</span>
                                          {i === 0 && <span className="text-xs bg-yellow-500/15 text-yellow-400 border border-yellow-500/25 px-1.5 py-0.5 rounded-full">Start here</span>}
                                        </div>
                                        {r.note && <p className="text-xs text-gray-400 mt-0.5">{r.note}</p>}
                                      </div>
                                    </div>
                                    <span className="text-xs text-gray-400 ml-2 flex-shrink-0">{r.time}</span>
                                  </a>
                                ))}
                                <button onClick={e => setPhaseTab(p.id, "project", e)}
                                  className="w-full mt-2 flex items-center justify-center gap-2 text-xs text-green-400 hover:text-green-300 bg-green-500/8 hover:bg-green-500/15 border border-green-500/20 rounded-lg py-2 transition-colors">
                                  Now see what to build — the phase project <ArrowRight size={12}/>
                                </button>
                              </div>
                            )}
                            {activeTab === "setup" && p.config && (
                              <div className="space-y-3">
                                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{p.config.title}</p>
                                <div className="overflow-x-auto">
                                  <table className="w-full text-xs">
                                    <thead>
                                      <tr className="text-gray-400 border-b border-gray-700">
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
                                      <span className="flex-shrink-0">{React.createElement(s.icon, {size:20})}</span>
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
                                      <tr className="text-gray-400 border-b border-gray-700">
                                        {p.training.table[0].map((h, i) => <th key={i} className="text-left pb-2 pr-3">{h}</th>)}
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
                                  <p className="text-xs font-semibold text-yellow-400 uppercase tracking-wider mb-1"><Wrench size={12} className="inline mr-1.5 align-middle"/>Phase Project</p>
                                  <p className="text-sm text-gray-200">{p.project}</p>
                                </div>
                                <div className="bg-gray-800 rounded-lg p-3">
                                  <p className="text-xs font-semibold text-green-400 uppercase tracking-wider mb-1"><Check size={11} className="inline mr-1 align-middle"/>Milestone</p>
                                  <p className="text-sm text-gray-200">{p.milestone}</p>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                    </div>
                  </div>
                );
              })}
              </div>
            </div>

            <div className="mb-8">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3"><Wrench size={12} className="inline mr-1.5 align-middle"/>Essential Free Tools</p>
              <div className="grid grid-cols-2 gap-2">
                {roadmapTools.map((t, i) => (
                  <a key={i} href={t.url} target="_blank" rel="noopener noreferrer"
                    className="bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-lg p-3 transition-colors">
                    <p className="text-sm font-medium text-blue-400">{t.name}</p>
                    <p className="text-xs text-gray-400 mt-0.5">{t.desc}</p>
                  </a>
                ))}
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3"><Lightbulb size={12} className="inline mr-1.5 align-middle"/>Principles for the Journey</p>
              <div className="space-y-2">
                {[
                  ["Build early, build often", "Don't wait until you feel 'ready'. Ship something after Phase 3."],
                  ["Concepts > memorization", "Understand the why. Tools change fast, intuition doesn't."],
                  ["Your dev skills are a superpower", "You can go from idea to working AI app faster than most beginners."],
                  ["Prompt → RAG → Fine-tune", "Always try the simplest approach first before going deeper."],
                  ["Projects are your portfolio", "Even personal projects signal domain expertise better than certificates."],
                ].map(([title, desc], i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-gray-400 flex-shrink-0 mt-0.5"><ArrowRight size={13} className="flex-shrink-0"/></span>
                    <div>
                      <span className="text-sm font-medium text-gray-200">{title}: </span>
                      <span className="text-sm text-gray-400">{desc}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            {/* Community feedback card */}
            <div className="mt-6 rounded-xl border border-blue-500/20 bg-blue-500/5 p-5">
              <div className="flex items-center gap-2 mb-3">
                <MessageSquare size={15} className="text-blue-400" />
                <p className="text-sm font-semibold text-white">Help make this better</p>
              </div>
              <p className="text-sm text-gray-400 mb-4">This roadmap is community-driven. If you spot something outdated, missing, or wrong — please say so.</p>
              <div className="grid grid-cols-1 gap-2 mb-4">
                {[
                  { icon: Lightbulb, label: "Suggest a missing resource", href: "https://github.com/amit352/ailearnings/discussions/new?category=ideas", color: "text-yellow-400" },
                  { icon: AlertTriangle, label: "Report an outdated or broken link", href: "https://github.com/amit352/ailearnings/discussions/new?category=general", color: "text-orange-400" },
                  { icon: Check, label: "Share your experience after completing a phase", href: "https://github.com/amit352/ailearnings/discussions/new?category=general", color: "text-green-400" },
                ].map(({ icon: Icon, label, href, color }, i) => (
                  <a key={i} href={href} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2.5 bg-gray-900/60 hover:bg-gray-800/80 border border-white/8 hover:border-white/15 rounded-lg px-3 py-2.5 transition-all group">
                    <Icon size={13} className={`${color} flex-shrink-0`} />
                    <span className="text-sm text-gray-300 group-hover:text-white transition-colors">{label}</span>
                    <ArrowRight size={12} className="text-gray-600 group-hover:text-gray-400 ml-auto flex-shrink-0 transition-colors" />
                  </a>
                ))}
              </div>
              <a href="https://github.com/amit352/ailearnings/discussions" target="_blank" rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 text-xs text-gray-400 hover:text-gray-200 transition-colors">
                <Github size={12} /> View all discussions on GitHub
              </a>
            </div>

            <p className="text-center text-gray-400 text-xs mt-6">Click each phase to expand · Use tabs to navigate sections</p>
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // ALT RESOURCES COMPONENT  (alt_resources.tsx)
    // ═══════════════════════════════════════════════════════════
    const FREE = "free";
    const PAID = "paid";
    const OREILLY = "oreilly";

    const AltBadge = ({ type }) => {
      const map = {
        free: "bg-green-900 text-green-300 border border-green-700",
        paid: "bg-gray-800 text-gray-400 border border-gray-600",
        oreilly: "bg-orange-900 text-orange-300 border border-orange-700",
      };
      const label = { free: "Free", paid: "~$40–60", oreilly: "O'Reilly" };
      return <span className={`text-xs px-2 py-0.5 rounded-full ${map[type]}`}>{label[type]}</span>;
    };

    const altPhases = [
      {
        id: 1, icon: Sprout, title: "Phase 1 – AI Foundations", color: "from-green-500 to-emerald-600",
        books: [
          { title: "Hands-On Large Language Models", authors: "Jay Alammar & Maarten Grootendorst", publisher: "O'Reilly, 2024", type: OREILLY, why: "The #1 beginner book for devs. Visual, intuitive, covers LLM internals without heavy math. Perfect Phase 1 read.", url: "https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/", rating: "⭐ 4.7" },
          { title: "The Hundred-Page Language Models Book", authors: "Andriy Burkov", publisher: "True Positive Inc., 2025", type: PAID, why: "Dense but accessible. Great conceptual coverage of transformers, training, and inference in under 200 pages.", url: "https://www.thelmbook.com", rating: "⭐ 4.8" },
        ],
        videos: [
          { title: "Neural Networks: Zero to Hero", author: "Andrej Karpathy (YouTube)", type: FREE, why: "The gold standard. Builds intuition from scratch. Watch this before anything else.", url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ", duration: "~10 hrs total" },
          { title: "Intro to Large Language Models", author: "Andrej Karpathy (YouTube)", type: FREE, why: "1-hour masterclass on how LLMs work. Best single video for beginners.", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g", duration: "1 hr" },
          { title: "Generative AI for Beginners", author: "Microsoft (GitHub + YouTube)", type: FREE, why: "21 structured lessons covering GenAI fundamentals, prompting, RAG, agents. 74K+ GitHub stars.", url: "https://github.com/microsoft/generative-ai-for-beginners", duration: "Self-paced" },
          { title: "Generative AI for Everyone", author: "Andrew Ng / DeepLearning.AI (free audit)", type: FREE, why: "Non-technical introduction to GenAI — great for building mental models before diving into code. 4.8★, 450K+ enrolled.", url: "https://www.deeplearning.ai/courses/generative-ai-for-everyone/", duration: "~4 hrs" },
          { title: "StatQuest – ML Fundamentals", author: "Josh Starmer (YouTube)", type: FREE, why: "Best visual explanations of statistics and ML concepts on YouTube. Perfect companion for building intuition.", url: "https://www.youtube.com/c/joshstarmer", duration: "Self-paced" },
        ],
      },
      {
        id: 2, icon: Settings, title: "Phase 2 – LLM Setup & Configuration", color: "from-slate-500 to-gray-600",
        books: [
          { title: "Designing Large Language Model Applications", authors: "Suhas Pai", publisher: "O'Reilly, 2024", type: OREILLY, why: "Covers loading LLMs, decoding strategies (top-k, top-p, beam search), quantization, Ollama, and HF Accelerate. Exactly what Phase 2 needs.", url: "https://www.oreilly.com/library/view/designing-large-language/9781098150495/", rating: "⭐ 4.5" },
        ],
        videos: [
          { title: "Open Source Models with Hugging Face", author: "DeepLearning.AI (free short course)", type: FREE, why: "Covers loading, configuring, and running open-source models. Directly addresses setup needs.", url: "https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/", duration: "1.5 hrs" },
          { title: "Running LLMs Locally with Ollama (freeCodeCamp)", author: "freeCodeCamp (YouTube)", type: FREE, why: "Practical walkthrough on setting up local models, quantization, and config params.", url: "https://www.youtube.com/results?search_query=ollama+local+llm+freecodecamp", duration: "~2 hrs" },
        ],
      },
      {
        id: 3, icon: Wrench, title: "Phase 3 – Prompt Engineering & LLM APIs", color: "from-blue-500 to-indigo-600",
        books: [
          { title: "Prompt Engineering for Generative AI", authors: "James Phoenix & Mike Taylor", publisher: "O'Reilly, 2024", type: OREILLY, why: "The definitive O'Reilly book on prompting. Covers zero-shot, few-shot, CoT, role prompting, and advanced patterns. Rated 4.5★.", url: "https://www.oreilly.com/library/view/prompt-engineering-for/9781098153427/", rating: "⭐ 4.5" },
          { title: "AI Engineering", authors: "Chip Huyen", publisher: "O'Reilly, 2025", type: OREILLY, why: "The most-read book on O'Reilly since launch. Covers the full AI app stack — evaluation, RAG, agents, fine-tuning. A must-read across Phases 3–6.", url: "https://www.oreilly.com/library/view/ai-engineering/9781098166298/", rating: "⭐ 4.7" },
        ],
        videos: [
          { title: "ChatGPT Prompt Engineering for Developers", author: "DeepLearning.AI (free short course)", type: FREE, why: "The canonical prompt engineering course. Covers all techniques with hands-on notebooks.", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/", duration: "1.5 hrs" },
          { title: "Building Systems with the ChatGPT API", author: "DeepLearning.AI (free short course)", type: FREE, why: "Goes from prompting to building multi-step LLM-powered apps. Great bridge into real development.", url: "https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/", duration: "1.5 hrs" },
          { title: "DSPy – Programmatic Prompting (Stanford)", author: "Stanford NLP (GitHub)", type: FREE, why: "22K+ stars. Replaces hand-crafted prompts with declarative modules — the direction prompting is heading.", url: "https://github.com/stanfordnlp/dspy", duration: "Reference" },
          { title: "Instructor – Structured LLM Outputs", author: "Jason Liu (GitHub)", type: FREE, why: "10K+ stars. The standard library for getting typed, validated JSON from any LLM. Used widely in production.", url: "https://github.com/jxnl/instructor", duration: "Reference" },
        ],
      },
      {
        id: 4, icon: BookOpen, title: "Phase 4 – RAG & Working with Data", color: "from-purple-500 to-violet-600",
        books: [
          { title: "Designing Large Language Model Applications", authors: "Suhas Pai", publisher: "O'Reilly, 2024", type: OREILLY, why: "Has dedicated chapters on RAG pipelines, chunking, embedding strategies, vector DBs, and RAG vs fine-tuning comparisons.", url: "https://www.oreilly.com/library/view/designing-large-language/9781098150495/", rating: "⭐ 4.5" },
          { title: "Building AI Agents with LLMs, RAG, and Knowledge Graphs", authors: "Multiple Authors", publisher: "O'Reilly / Packt, 2024", type: OREILLY, why: "Covers naive RAG, advanced RAG (chunking, embedding strategies, evaluation), and how RAG compares to fine-tuning.", url: "https://www.oreilly.com/library/view/building-ai-agents/9781835087060/", rating: "⭐ 4.4" },
        ],
        videos: [
          { title: "LangChain: Chat with Your Data", author: "DeepLearning.AI (free short course)", type: FREE, why: "Hands-on RAG from document loading to retrieval and generation. The best intro to RAG in practice.", url: "https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/", duration: "1.5 hrs" },
          { title: "Building & Evaluating Advanced RAG", author: "DeepLearning.AI (free short course)", type: FREE, why: "Goes deeper — sentence-window retrieval, auto-merging, evaluation metrics. Great follow-up.", url: "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/", duration: "1.5 hrs" },
          { title: "Knowledge Graphs for RAG", author: "DeepLearning.AI (free short course)", type: FREE, why: "Advanced RAG technique — connect your retrieval pipeline to knowledge graphs for structured context.", url: "https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/", duration: "1.5 hrs" },
          { title: "Generative AI with LLMs (Coursera/AWS)", author: "DeepLearning.AI + AWS", type: FREE, why: "3-week course covering transformers, RAG, fine-tuning, and deployment. Audit for free.", url: "https://www.coursera.org/learn/generative-ai-with-llms", duration: "~16 hrs" },
          { title: "LlamaIndex Docs & Tutorials", author: "LlamaIndex (free)", type: FREE, why: "Leading alternative to LangChain for RAG. Excellent for document pipelines and query engines.", url: "https://docs.llamaindex.ai", duration: "Reference" },
          { title: "Maxime Labonne – LLM Course", author: "GitHub (free, 44K⭐)", type: FREE, why: "Comprehensive free course covering quantization, fine-tuning, and RAG with hands-on notebooks.", url: "https://github.com/mlabonne/llm-course", duration: "Self-paced" },
          { title: "RAGAS – RAG Evaluation Framework", author: "Exploding Gradients (GitHub)", type: FREE, why: "7K+ stars. The standard open-source library for evaluating RAG pipelines (faithfulness, relevance, correctness).", url: "https://github.com/explodinggradients/ragas", duration: "Reference" },
          { title: "Weaviate Academy – Vector DB Course", author: "Weaviate (free)", type: FREE, why: "Free structured course on vector databases from the team that builds one. Covers embeddings, search, and RAG.", url: "https://weaviate.io/learn/academy", duration: "Self-paced" },
        ],
      },
      {
        id: 5, icon: Bot, title: "Phase 5 – Agentic AI", color: "from-orange-500 to-amber-600",
        books: [
          { title: "Building Applications with AI Agents", authors: "Multiple Authors", publisher: "O'Reilly, 2025", type: OREILLY, why: "Most up-to-date O'Reilly agent book. Covers LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, multi-agent coordination, and real-world use cases.", url: "https://www.oreilly.com/library/view/building-applications-with/9781098176495/", rating: "⭐ 4.5" },
          { title: "Building AI Agents with LLMs, RAG, and Knowledge Graphs", authors: "Multiple Authors", publisher: "O'Reilly / Packt, 2024", type: OREILLY, why: "Covers agent frameworks (LangChain, LlamaIndex, AutoGen), tool usage, planning, and multi-agent systems end-to-end.", url: "https://www.oreilly.com/library/view/building-ai-agents/9781835087060/", rating: "⭐ 4.4" },
        ],
        videos: [
          { title: "AI Agents in LangGraph", author: "DeepLearning.AI (free short course)", type: FREE, why: "Builds ReACT agents, tool-calling agents, and multi-agent systems using LangGraph. Hands-on and practical.", url: "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/", duration: "2 hrs" },
          { title: "AI Agentic Design Patterns with AutoGen", author: "Microsoft / DeepLearning.AI (free)", type: FREE, why: "Covers orchestrator-worker, reflection, and multi-agent patterns using Microsoft's AutoGen framework.", url: "https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/", duration: "1 hr" },
          { title: "Full Stack LLM Bootcamp – Agents & Tools", author: "Full Stack Deep Learning (YouTube, free)", type: FREE, why: "University-quality lectures covering agents, tool use, and production patterns. Free on YouTube.", url: "https://www.youtube.com/@FullStackDeepLearning", duration: "Self-paced" },
          { title: "LangGraph Official Tutorials", author: "LangChain (free)", type: FREE, why: "Best hands-on resource for building stateful agents. Start here after the DeepLearning.AI course.", url: "https://langchain-ai.github.io/langgraph/tutorials/", duration: "Self-paced" },
          { title: "CrewAI – Multi-Agent Framework", author: "CrewAI (GitHub + Docs)", type: FREE, why: "28K+ stars. Most popular multi-agent framework. Great for orchestrator-worker patterns.", url: "https://docs.crewai.com", duration: "Reference" },
          { title: "OpenAI Agents SDK", author: "OpenAI (GitHub)", type: FREE, why: "Official OpenAI framework for building agents with tool use, handoffs, and guardrails.", url: "https://github.com/openai/openai-agents-python", duration: "Reference" },
          { title: "Smolagents – Minimal Agent Framework", author: "Hugging Face (GitHub)", type: FREE, why: "HuggingFace's own lean agent library. Clean, minimal, great for learning agent internals.", url: "https://github.com/huggingface/smolagents", duration: "Reference" },
        ],
      },
      {
        id: 6, icon: Building2, title: "Phase 6 – Building & Training LLMs", color: "from-rose-500 to-pink-600",
        books: [
          { title: "Build a Large Language Model (From Scratch)", authors: "Sebastian Raschka", publisher: "Manning, 2024", type: PAID, why: "THE book for understanding LLM internals by building one from scratch in PyTorch. Covers tokenization, attention, pre-training, SFT, RLHF. Rated 4.6★ — best in class for this topic.", url: "https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167", rating: "⭐ 4.6" },
          { title: "LLM Engineer's Handbook", authors: "Paul Iusztin & Maxime Labonne", publisher: "Packt, 2024", type: PAID, why: "Covers LLM engineering end-to-end: fine-tuning with LoRA/QLoRA, RAG, evaluation, and production deployment. Highly practical.", url: "https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072", rating: "⭐ 4.6" },
          { title: "Natural Language Processing with Transformers", authors: "Lewis Tunstall et al. (Hugging Face team)", publisher: "O'Reilly, 2022", type: OREILLY, why: "Still the best deep dive into transformer architecture and fine-tuning with Hugging Face. Foundational for anyone who wants to understand the mechanics.", url: "https://www.oreilly.com/library/view/natural-language-processing/9781098136789/", rating: "⭐ 4.6" },
        ],
        videos: [
          { title: "Let's Build GPT from Scratch", author: "Andrej Karpathy (YouTube)", type: FREE, why: "Legendary 2-hour video. Builds a GPT model from scratch in Python. Best hands-on transformer tutorial that exists.", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", duration: "2 hrs" },
          { title: "Finetuning Large Language Models", author: "DeepLearning.AI (free short course)", type: FREE, why: "Practical fine-tuning walkthrough covering SFT, LoRA, and evaluation.", url: "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/", duration: "1 hr" },
          { title: "Stanford CS336: Language Modeling from Scratch", author: "Stanford University (YouTube, free)", type: FREE, why: "Full university course on building LLMs from scratch. One of the most rigorous free resources available.", url: "https://www.youtube.com/@stanfordonline", duration: "Full course" },
          { title: "Maxime Labonne – LLM Course", author: "GitHub (free, 44K⭐)", type: FREE, why: "Comprehensive free course with notebooks on quantization, LoRA/QLoRA fine-tuning, and RLHF. One of the best structured free resources.", url: "https://github.com/mlabonne/llm-course", duration: "Self-paced" },
          { title: "DataTalksClub – LLM Zoomcamp", author: "DataTalksClub (GitHub, free)", type: FREE, why: "Free 10-week cohort-style course covering RAG, fine-tuning, and LLM deployment end-to-end.", url: "https://github.com/DataTalksClub/llm-zoomcamp", duration: "10 weeks" },
          { title: "HuggingFace TRL – RLHF & DPO Training", author: "Hugging Face (Docs)", type: FREE, why: "The standard HuggingFace library for RLHF, DPO, and PPO fine-tuning. Reference for Phase 6 training techniques.", url: "https://huggingface.co/docs/trl", duration: "Reference" },
        ],
      },
      {
        id: 7, icon: Rocket, title: "Phase 7 – Production & Staying Current", color: "from-teal-500 to-cyan-600",
        books: [
          { title: "AI Engineering", authors: "Chip Huyen", publisher: "O'Reilly, 2025", type: OREILLY, why: "If you read one book across this entire roadmap, make it this. Covers evaluation, RAG, agents, fine-tuning, and production — all from a systems perspective. Most read book on O'Reilly in 2025.", url: "https://www.oreilly.com/library/view/ai-engineering/9781098166298/", rating: "⭐ 4.7" },
          { title: "Building LLMs for Production", authors: "Louis-François Bouchard & Louie Peters", publisher: "Independent, 2024", type: PAID, why: "Focuses on prompting, fine-tuning, RAG, and reliability in production. Practical and concise.", url: "https://www.amazon.com/Building-LLMs-Production-Reliability-Fine-Tuning/dp/B0D4FFPFW5", rating: "⭐ 4.4" },
        ],
        videos: [
          { title: "Latent Space Podcast", author: "swyx & Alessio (YouTube + Podcast)", type: FREE, why: "Best podcast to stay current on AI engineering. Interviews with researchers and engineers building real AI systems.", url: "https://www.latent.space/podcast", duration: "Ongoing" },
          { title: "Two Minute Papers", author: "Károly Zsolnai-Fehér (YouTube)", type: FREE, why: "Bite-sized breakdowns of the latest AI research papers. Great for keeping up with the field without reading papers yourself.", url: "https://www.youtube.com/@TwoMinutePapers", duration: "Ongoing" },
          { title: "Yannic Kilcher", author: "YouTube", type: FREE, why: "Deep technical paper walkthroughs. Best for understanding what's happening at the research frontier (transformers, reasoning models, etc.).", url: "https://www.youtube.com/@YannicKilcher", duration: "Ongoing" },
          { title: "AI Explained", author: "YouTube", type: FREE, why: "338K+ subscribers. Breaks down cutting-edge AI research and product releases clearly and quickly.", url: "https://www.youtube.com/@aiexplained-official", duration: "Ongoing" },
          { title: "The Batch – Andrew Ng Newsletter", author: "DeepLearning.AI (free)", type: FREE, why: "1M+ weekly readers. Andrew Ng's curation of the most important AI developments each week. Signal over noise.", url: "https://www.deeplearning.ai/the-batch/", duration: "Weekly" },
          { title: "TLDR AI Newsletter", author: "TLDR (free)", type: FREE, why: "620K subscribers. Best daily AI digest — 5 minutes to stay current on models, tools, and research.", url: "https://tldr.tech/ai", duration: "Daily" },
          { title: "Hugging Face Blog", author: "Hugging Face (free)", type: FREE, why: "First place to read about new model releases, techniques, and open-source AI developments.", url: "https://huggingface.co/blog", duration: "Ongoing" },
          { title: "Papers With Code", author: "Meta AI (free)", type: FREE, why: "Track state-of-the-art benchmarks across every AI task. See what's winning and find the code.", url: "https://paperswithcode.com", duration: "Reference" },
          { title: "LangSmith – LLM Observability", author: "LangChain (free tier)", type: FREE, why: "Trace, debug, and evaluate LLM app behavior in production. Essential for moving beyond prototypes.", url: "https://smith.langchain.com", duration: "Reference" },
          { title: "Weights & Biases Weave", author: "W&B (free tier)", type: FREE, why: "LLM-specific tracking layer on top of W&B. Logs prompts, completions, costs, and evals.", url: "https://wandb.ai/site/weave", duration: "Reference" },
        ],
      },
    ];

    const oreillyCost = {
      title: "O'Reilly Subscription – Is It Worth It?",
      points: [
        ["Monthly plan", "~$50/month — cancellable anytime. Read 2–3 books in a month, cancel."],
        ["Annual plan", "~$500/year — better value if you're serious across all phases."],
        ["Free alternative", "Many O'Reilly books are available at local/national public libraries via apps like Libby or O'Reilly for Public Libraries."],
        ["Trial", "O'Reilly often offers 10-day free trials — enough to read 1 full book."],
        ["University access", "If you have any university affiliation, check if they provide O'Reilly access (many do, for free)."],
      ],
    };

    function AltResources() {
      useSeo(
        "AI Learning Resources – Books & Courses by Phase | AI Learning Hub",
        "Curated books, video courses, and references mapped to each phase of the AI roadmap. Includes free and O'Reilly resources for every level."
      );
      const [open, setOpen] = useState(null);
      const [view, setView] = useState({});
      const toggle = (id) => setOpen(open === id ? null : id);
      const setV = (id, v, e) => { e.stopPropagation(); setView(p => ({ ...p, [id]: v })); };

      return (
        <div className="min-h-screen text-gray-100 p-4 font-sans">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-6">
              <h1 className="text-2xl font-bold mb-1"><BookOpen size={20} className="inline mr-2 align-middle text-green-400"/>Curated Books & Courses</h1>
              <p className="text-gray-400 text-sm max-w-lg mx-auto">Curated books and courses mapped to each phase — use these to go deeper beyond the free resources in the main roadmap.</p>
              <p className="text-xs text-gray-400 mt-2">Each phase has a recommended book and video course. Expand a phase to see why each was chosen.</p>
            </div>

            <div className="bg-gray-900 border border-orange-800 rounded-xl p-4 mb-6">
              <p className="text-orange-400 font-semibold text-sm mb-2">{oreillyCost.title}</p>
              <div className="space-y-1.5">
                {oreillyCost.points.map(([k, v], i) => (
                  <div key={i} className="flex gap-2 text-sm">
                    <span className="text-orange-500 font-medium w-28 flex-shrink-0">{k}:</span>
                    <span className="text-gray-300">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-3 mb-8">
              {altPhases.map((p) => {
                const v = view[p.id] || "books";
                return (
                  <div key={p.id} className={`rounded-xl border cursor-pointer transition-all ${open === p.id ? "border-gray-500 bg-gray-900" : "border-gray-800 bg-gray-900 hover:border-gray-600"}`}
                    onClick={() => toggle(p.id)}>
                    <div className="flex items-center gap-3 p-4">
                      <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${p.color} flex items-center justify-center text-base flex-shrink-0`}><p.icon size={20} className="text-white"/></div>
                      <div className="flex-1">
                        <p className="font-semibold text-sm">{p.title}</p>
                        <p className="text-gray-400 text-xs mt-0.5">{p.books.length} book{p.books.length > 1 ? "s" : ""} · {p.videos.length} video course{p.videos.length > 1 ? "s" : ""}</p>
                      </div>
                      <span className="text-gray-400">{open === p.id ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
                    </div>
                    {open === p.id && (
                      <div className="border-t border-gray-800">
                        <div className="flex gap-1 px-4 pt-3">
                          {["books", "videos"].map(t => (
                            <button key={t} onClick={e => setV(p.id, t, e)}
                              className={`text-xs px-4 py-1.5 rounded-lg capitalize transition-colors ${v === t ? "bg-gray-700 text-white" : "text-gray-400 hover:text-gray-300"}`}>
                              {t === "books" ? <><BookOpen size={12} className="inline mr-1"/>Books</> : <><Video size={12} className="inline mr-1"/>Courses & References</>}
                            </button>
                          ))}
                        </div>
                        <div className="p-4 space-y-3">
                          {v === "books" && p.books.map((b, i) => (
                            <a key={i} href={b.url} target="_blank" rel="noopener noreferrer"
                              className="block bg-gray-800 rounded-xl p-4 transition-colors border border-gray-700 hover:border-gray-500"
                              onClick={e => e.stopPropagation()}>
                              <div className="flex items-start justify-between gap-2 mb-1">
                                <p className="text-blue-400 font-semibold text-sm leading-snug">{b.title}</p>
                                <AltBadge type={b.type} />
                              </div>
                              <p className="text-gray-400 text-xs mb-2">{b.authors} · {b.publisher} · {b.rating}</p>
                              <p className="text-gray-300 text-xs leading-relaxed">{b.why}</p>
                            </a>
                          ))}
                          {v === "videos" && p.videos.map((vid, i) => (
                            <a key={i} href={vid.url} target="_blank" rel="noopener noreferrer"
                              className="block bg-gray-800 rounded-xl p-4 transition-colors border border-gray-700 hover:border-gray-500"
                              onClick={e => e.stopPropagation()}>
                              <div className="flex items-start justify-between gap-2 mb-1">
                                <p className="text-blue-400 font-semibold text-sm leading-snug">{vid.title}</p>
                                <AltBadge type={vid.type} />
                              </div>
                              <div className="flex items-center gap-2 mb-2">
                                <p className="text-gray-400 text-xs">{vid.author}</p>
                                <span className="text-gray-400">·</span>
                                <p className="text-gray-400 text-xs">{vid.duration}</p>
                              </div>
                              <p className="text-gray-300 text-xs leading-relaxed">{vid.why}</p>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-6">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3"><Trophy size={12} className="inline mr-1.5 align-middle"/>If You Only Pick 3 Resources Total</p>
              <div className="space-y-3">
                {[
                  { emoji: "🥇", label: "Best Single Book", rec: "AI Engineering – Chip Huyen (O'Reilly, 2025)", why: "Covers the full AI app stack. Most read on O'Reilly. Timeless principles, not just tools.", url: "https://www.oreilly.com/library/view/ai-engineering/9781098166298/" },
                  { emoji: "🥈", label: "Best Deep-Dive Book", rec: "Build a Large Language Model From Scratch – Sebastian Raschka (Manning)", why: "If you want to truly understand how LLMs work, this is the one.", url: "https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167" },
                  { emoji: "🥉", label: "Best Free Video Series", rec: "Neural Networks: Zero to Hero – Andrej Karpathy (YouTube)", why: "The best free education in AI that exists. Period.", url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ" },
                ].map((r, i) => (
                  <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
                    className="flex gap-3 bg-gray-800 rounded-lg p-3 hover:border-gray-500 border border-gray-700 transition-colors">
                    <span className="text-xl flex-shrink-0">{r.emoji}</span>
                    <div>
                      <p className="text-xs text-gray-400 mb-0.5">{r.label}</p>
                      <p className="text-blue-400 text-sm font-semibold">{r.rec}</p>
                      <p className="text-gray-400 text-xs mt-1">{r.why}</p>
                    </div>
                  </a>
                ))}
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3"><BarChart size={12} className="inline mr-1.5 align-middle"/>Paid Bootcamp vs This Path</p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Paid Bootcamp", pros: ["Live cohort + community", "Instructor feedback", "Structured schedule", "Certificate"], cons: ["Expensive", "Fixed schedule", "One-time coverage"] },
                  { label: "This Path", pros: ["Mostly free or ~$100 total", "Self-paced", "Go deeper on any topic", "Referenceable forever"], cons: ["Self-discipline needed", "No live feedback", "No certificate"] },
                ].map((col, i) => (
                  <div key={i} className="bg-gray-800 rounded-lg p-3">
                    <p className="font-semibold text-sm text-white mb-2">{col.label}</p>
                    {col.pros.map((p, j) => <p key={j} className="text-xs text-green-400 mb-1 flex items-center gap-1"><Check size={11}/>{p}</p>)}
                    {col.cons.map((c, j) => <p key={j} className="text-xs text-gray-400 mb-1"><X size={11} className="inline mr-0.5 align-middle"/>{c}</p>)}
                  </div>
                ))}
              </div>
            </div>
            <p className="text-center text-gray-400 text-xs mt-6">Click each phase → toggle Books / Courses & References</p>
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // KNOWLEDGE ASSESSMENT COMPONENT  (knowledge_assessment.tsx)
    // ═══════════════════════════════════════════════════════════
    const afterRoadmap = {
      overall: 65, label: "AI Engineer (Solid)",
      sublabel: "Upper end of the 'Engineer' tier — approaching 'Specialist' in some areas",
      color: "from-purple-500 to-orange-500",
      areas: [
        { name: "LLM Concepts & Internals", pct: 80, note: "Strong. You can explain transformers, attention, training, RLHF — not just use them." },
        { name: "Prompt Engineering", pct: 90, note: "Near expert. You'll have spent real hands-on time with multiple techniques." },
        { name: "RAG Systems", pct: 80, note: "Strong. Can design, build, and evaluate a production RAG pipeline." },
        { name: "Agentic AI / Tool Use", pct: 75, note: "Good practitioner level. Can build complex agents, understand design patterns." },
        { name: "Fine-Tuning / Training", pct: 60, note: "Solid understanding, practical LoRA/QLoRA experience. Not a full ML researcher." },
        { name: "Multimodal AI", pct: 40, note: "Awareness level. Know how diffusion works conceptually, limited hands-on." },
        { name: "ML Research / Math", pct: 30, note: "Surface level. Can read papers but not write or advance them." },
        { name: "Production / MLOps", pct: 50, note: "Moderate. Know the concepts, some hands-on but not deep deployment experience." },
        { name: "AI Safety / Ethics", pct: 40, note: "Awareness. Know the issues, not a practitioner in alignment or safety research." },
      ],
    };

    const honest = [
      { icon: Check, title: "What you CAN do confidently", color: "text-green-400", items: [
        "Build end-to-end AI-powered applications from scratch",
        "Design and evaluate RAG pipelines for real use cases",
        "Build multi-step agents with tool calling and memory",
        "Choose the right model (API vs local vs fine-tuned) for a problem",
        "Fine-tune open-source LLMs using LoRA/QLoRA on modest hardware",
        "Read and roughly understand most AI research paper abstracts",
        "Evaluate LLM outputs meaningfully (not just 'does it look good')",
        "Discuss AI architecture decisions with engineers and researchers",
        "Contribute meaningfully to AI discussions and design reviews",
      ]},
      { icon: AlertTriangle, title: "What you still can't do (yet)", color: "text-yellow-400", items: [
        "Train a foundation model from scratch (needs massive compute + team)",
        "Write or publish novel AI research",
        "Deeply understand all the math (backprop derivations, probability theory)",
        "Specialize in computer vision, audio, or robotics without extra study",
        "Build highly optimized inference pipelines (vLLM, CUDA kernels, etc.)",
        "Work as an ML researcher at a frontier lab (OpenAI, Anthropic, DeepMind)",
        "Architect enterprise-scale AI systems without production ML experience",
      ]},
    ];

    const nextSteps = [
      {
        id: 1, icon: Building2, title: "Deepen by Building Publicly", when: "Immediately & ongoing",
        color: "from-blue-500 to-indigo-600",
        desc: "The fastest way to solidify knowledge is to build something real and share it. This creates feedback loops that no course can give you.",
        actions: [
          "Pick 1 meaningful project that combines RAG + Agents (your strongest areas)",
          "Write about what you built — blog posts, LinkedIn, or GitHub README",
          "Open-source your code and invite feedback",
          "Build in public: share progress, failures, and learnings",
        ],
        examples: ["A RAG chatbot over a domain you care about (legal, medical, finance)", "An AI agent that automates a tedious personal workflow", "A tool that analyzes documents and generates structured reports"],
      },
      {
        id: 2, icon: BookMarked, title: "Fill the Math Gap (Selectively)", when: "After Phase 4–5, if curious",
        color: "from-purple-500 to-violet-600",
        desc: "You don't need a PhD, but some targeted math will help you read papers and understand model behavior more deeply.",
        actions: [
          "Linear algebra: vectors, matrices, dot products (3Blue1Brown 'Essence of Linear Algebra')",
          "Probability & statistics: distributions, Bayes theorem, entropy",
          "Calculus: chain rule, gradients (just enough to understand backprop)",
          "Don't go deep on all of it — learn it when a specific paper or concept demands it",
        ],
        examples: ["'Mathematics for Machine Learning' (free PDF, Cambridge)", "3Blue1Brown Essence of Linear Algebra (YouTube, free)"],
      },
      {
        id: 3, icon: FlaskConical, title: "Start Reading Papers", when: "After Phase 5",
        color: "from-rose-500 to-pink-600",
        desc: "AI moves at paper speed. Even skimming 2–3 papers a month will put you ahead of 90% of practitioners.",
        actions: [
          "Start with landmark papers: Attention is All You Need, GPT-3, InstructGPT, LoRA, RAG",
          "Follow Arxiv Sanity Preserver or Papers With Code for new releases",
          "Use the 'abstract + intro + results' method — skip the math first",
          "Yannic Kilcher on YouTube explains recent papers visually",
          "Join AI communities (r/MachineLearning, Hugging Face Discord, Latent Space)",
        ],
        examples: [],
      },
      {
        id: 4, icon: Target, title: "Pick a Specialization", when: "After completing the roadmap",
        color: "from-orange-500 to-amber-600",
        desc: "At this point you're a generalist AI engineer. Real depth — and real opportunity — comes from going deep in one area.",
        specializations: [
          { name: "AI Application Builder", icon: Wrench, desc: "Go deep on LangChain/LangGraph, evals, production, observability. Build AI products.", fit: "Best fit for your goals" },
          { name: "RAG Specialist", icon: "📚", desc: "Advanced chunking, hybrid search, knowledge graphs, multi-hop RAG.", fit: "High demand in enterprises" },
          { name: "Agent Systems", icon: "🤖", desc: "Multi-agent coordination, planning, MCP integrations, autonomous workflows.", fit: "Fastest growing area" },
          { name: "Fine-Tuning / Alignment", icon: "⚙️", desc: "SFT, RLHF, DPO, preference optimization, model behavior.", fit: "More ML depth needed" },
          { name: "Multimodal AI", icon: Palette, desc: "Image/video generation, vision-language models, diffusion systems.", fit: "Creative + technical" },
          { name: "AI Infra / MLOps", icon: "🏗️", desc: "Serving, inference optimization, monitoring, scaling AI in production.", fit: "Strong eng background helps" },
        ],
      },
      {
        id: 5, icon: Globe, title: "Engage with the AI Community", when: "Throughout the journey",
        color: "from-teal-500 to-cyan-600",
        desc: "AI moves too fast to learn alone. Community keeps you current, connected, and motivated.",
        actions: [
          "Follow key people: Andrej Karpathy, Chip Huyen, Sebastian Raschka, Simon Willison",
          "Subscribe to 1–2 newsletters: Import AI (Jack Clark), The Batch (Andrew Ng), Latent Space",
          "Join Hugging Face Discord or local AI meetups",
          "Contribute to open source — even docs or issue reports matter",
          "Attend free AI events: NeurIPS workshops stream free, many AI conf talks on YouTube",
        ],
        examples: [],
      },
      {
        id: 6, icon: RotateCw, title: "Keep a 'Learning Radar'", when: "Ongoing forever",
        color: "from-green-500 to-emerald-600",
        desc: "AI evolves every 3–6 months. The goal isn't to learn everything — it's to know what's changing and when to dive in.",
        actions: [
          "Every quarter: assess what new techniques/tools are becoming mainstream",
          "Learn new things at the concept level first, then go hands-on only if relevant to your work",
          "Don't chase every shiny new model release — focus on paradigm shifts",
          "Key radar items right now: reasoning models, MCP ecosystem, multimodal agents, on-device AI",
        ],
        examples: [],
      },
    ];

    const comparisons = [
      { role: "AI Hobbyist", pct: 10, desc: "Uses ChatGPT, maybe tried some tools" },
      { role: "Prompt Engineer", pct: 30, desc: "Good at prompting, no engineering depth" },
      { role: "After this roadmap →", pct: 65, desc: "Builds apps, understands systems, reads papers", highlight: true },
      { role: "Senior ML Engineer", pct: 80, desc: "5+ yrs exp, production systems, ML depth" },
      { role: "AI Researcher", pct: 90, desc: "PhD level, novel research, frontier models" },
    ];

    function KnowledgeAssessment() {
      useSeo(
        "AI Knowledge Assessment – Where You'll Stand After the Roadmap | AI Learning Hub",
        "An honest assessment of your AI engineer skill level after completing all 7 phases — what you can do, what gaps remain, and the best next steps."
      );
      const [openStep, setOpenStep] = useState(null);

      return (
        <div className="min-h-screen text-gray-100 p-4 font-sans">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-2xl font-bold mb-1"><Brain size={20} className="inline mr-2 align-middle text-purple-400"/>After the Roadmap</h1>
              <p className="text-gray-400 text-sm max-w-lg mx-auto">An honest assessment of where you'll stand after completing all 7 phases — what you'll know, what gaps remain, and what roles you'll be ready for.</p>
              <p className="text-xs text-gray-400 mt-2">This is a benchmark, not a score. Use it to set expectations and identify where to focus after the roadmap.</p>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-2xl p-5 mb-6">
              <div className="flex items-center justify-between mb-1">
                <p className="text-lg font-bold text-white">{afterRoadmap.label}</p>
                <span className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-orange-400 bg-clip-text text-transparent">{afterRoadmap.overall}%</span>
              </div>
              <p className="text-gray-400 text-xs mb-4">{afterRoadmap.sublabel}</p>
              <div className="w-full bg-gray-800 rounded-full h-3 mb-1">
                <div className={`h-3 rounded-full bg-gradient-to-r ${afterRoadmap.color} transition-all`} style={{ width: `${afterRoadmap.overall}%` }} />
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Hobbyist</span><span>Beginner</span><span>Practitioner</span><span>Engineer</span><span>Specialist</span><span>Researcher</span>
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-6">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">How You Compare</p>
              <div className="space-y-3">
                {comparisons.map((c, i) => (
                  <div key={i} className={`${c.highlight ? "bg-purple-950 border border-purple-700 rounded-lg p-2 -mx-2" : ""}`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className={`text-xs font-medium ${c.highlight ? "text-purple-300" : "text-gray-400"}`}>{c.role}</span>
                      <span className={`text-xs ${c.highlight ? "text-purple-300 font-bold" : "text-gray-400"}`}>{c.pct}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-1.5">
                      <div className={`h-1.5 rounded-full ${c.highlight ? "bg-purple-500" : "bg-gray-600"}`} style={{ width: `${c.pct}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-0.5">{c.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-6">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Knowledge by Area</p>
              <div className="space-y-3">
                {afterRoadmap.areas.map((a, i) => (
                  <div key={i}>
                    <div className="flex justify-between mb-1">
                      <span className="text-xs text-gray-300">{a.name}</span>
                      <span className="text-xs font-semibold" style={{ color: a.pct >= 75 ? "#86efac" : a.pct >= 55 ? "#fbbf24" : "#f87171" }}>{a.pct}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-1.5 mb-1">
                      <div className="h-1.5 rounded-full" style={{ width: `${a.pct}%`, background: a.pct >= 75 ? "#22c55e" : a.pct >= 55 ? "#f59e0b" : "#ef4444" }} />
                    </div>
                    <p className="text-xs text-gray-400">{a.note}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 mb-6">
              {honest.map((h, i) => (
                <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl p-4">
                  <p className={`text-xs font-semibold uppercase tracking-wider mb-3 ${h.color}`}>{React.createElement(h.icon, {size:13, className:"inline mr-1.5 align-middle"})}{h.title}</p>
                  <div className="space-y-1.5">
                    {h.items.map((item, j) => (
                      <div key={j} className="flex gap-2 text-sm">
                        <span className={`flex-shrink-0 ${h.color}`}>{h.icon === Check ? <Check size={13}/> : <X size={13}/>}</span>
                        <span className="text-gray-300 text-xs">{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3"><Rocket size={12} className="inline mr-1.5 align-middle"/>What's Next</p>
            <div className="space-y-3 mb-8">
              {nextSteps.map((s) => (
                <div key={s.id}
                  className={`rounded-xl border cursor-pointer transition-all ${openStep === s.id ? "border-gray-500 bg-gray-900" : "border-gray-800 bg-gray-900 hover:border-gray-600"}`}
                  onClick={() => setOpenStep(openStep === s.id ? null : s.id)}>
                  <div className="flex items-center gap-3 p-4">
                    <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${s.color} flex items-center justify-center text-base flex-shrink-0`}>{React.createElement(s.icon, {size:20, className:"text-white"})}</div>
                    <div className="flex-1">
                      <p className="font-semibold text-sm">{s.title}</p>
                      <p className="text-gray-400 text-xs mt-0.5">{s.when}</p>
                    </div>
                    <span className="text-gray-400">{openStep === s.id ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
                  </div>
                  {openStep === s.id && (
                    <div className="border-t border-gray-800 p-4 space-y-3">
                      <p className="text-gray-300 text-sm">{s.desc}</p>
                      {s.actions && s.actions.length > 0 && (
                        <div>
                          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Actions</p>
                          <div className="space-y-1.5">
                            {s.actions.map((a, i) => (
                              <div key={i} className="flex gap-2">
                                <span className="text-gray-400 flex-shrink-0"><ArrowRight size={13} className="flex-shrink-0"/></span>
                                <span className="text-xs text-gray-300">{a}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {s.examples && s.examples.length > 0 && (
                        <div>
                          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Examples</p>
                          <div className="space-y-1">
                            {s.examples.map((e, i) => (
                              <div key={i} className="flex gap-2">
                                <span className="text-blue-500 flex-shrink-0">•</span>
                                <span className="text-xs text-gray-400">{e}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {s.specializations && (
                        <div>
                          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Choose Your Path</p>
                          <div className="grid grid-cols-1 gap-2">
                            {s.specializations.map((sp, i) => (
                              <div key={i} className="bg-gray-800 rounded-lg p-3 flex gap-3">
                                <span className="text-xl flex-shrink-0">{sp.icon}</span>
                                <div>
                                  <div className="flex items-center gap-2 flex-wrap mb-0.5">
                                    <p className="text-sm font-semibold text-white">{sp.name}</p>
                                    {sp.fit === "Best fit for your goals" && (
                                      <span className="text-xs bg-green-900 text-green-300 border border-green-700 px-2 py-0.5 rounded-full">⭐ Best fit for you</span>
                                    )}
                                  </div>
                                  <p className="text-xs text-gray-400">{sp.desc}</p>
                                  <p className="text-xs text-gray-400 mt-0.5">{sp.fit}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="bg-gray-900 border border-yellow-800 rounded-xl p-4">
              <p className="text-yellow-400 font-semibold text-sm mb-3"><MessageSquare size={14} className="inline mr-1.5 align-middle"/>The Honest Reality</p>
              <div className="space-y-2 text-sm text-gray-300">
                <p>After this roadmap, you'll have hands-on experience with every major layer of the AI stack — from <span className="text-white font-semibold">prompting to production</span>.</p>
                <p className="text-gray-400">But AI is a <span className="text-white">field</span>, not a course. The best practitioners treat it as a continuous practice — building things, reading papers, and re-learning as the landscape shifts every 6 months.</p>
                <p className="text-gray-400">The gap between a <span className="text-white">65% AI Engineer</span> and a <span className="text-white">90% Researcher</span> isn't really about courses. It's about <span className="text-white">years of building real systems</span> and going deep on one specific problem that matters to you.</p>
                <p className="mt-2 text-gray-400 text-xs">Your developer instincts are your biggest asset. Most AI courses are taught to people who can't code. You can. That alone puts you 2 years ahead.</p>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // PROMPT ENGINEERING COMPONENT  (prompt_eng.tsx)
    // ═══════════════════════════════════════════════════════════
    const PromptBadge = ({ label, color }) => (
      <span className={`text-xs px-2 py-0.5 rounded-full border ${color}`}>{label}</span>
    );

    const techniques = [
      {
        id: 1, tier: "Foundation", icon: Layers, color: "from-blue-600 to-blue-800", border: "border-blue-700",
        title: "Core Prompting Techniques",
        desc: "These are non-negotiable. Master these before anything else. Most people stop here — experts start here.",
        techniques: [
          { name: "Zero-Shot Prompting", difficulty: "Beginner", use: "All use cases", what: "Ask directly with no examples. The baseline of everything.", bad: "Fix this code.", good: "You are a senior Python engineer. Review this function for bugs, performance issues, and readability. Output a list of issues with explanations and a corrected version.\n\n[paste code]", insight: "Adding a role, goal, and output format to a plain ask typically improves output quality by 40–60%. Most people never do this." },
          { name: "Few-Shot Prompting", difficulty: "Beginner", use: "Writing, classification, formatting", what: "Give the model 2–5 examples of input→output before your real request. Teaches format and tone without fine-tuning.", bad: "Write a commit message for this diff.", good: "Write a commit message for this diff. Follow this format:\n\nExample 1:\nDiff: Added null check to user login\nMessage: fix(auth): handle null user in login flow\n\nExample 2:\nDiff: Refactored payment module into separate service\nMessage: refactor(payments): extract payment logic to PaymentService\n\nNow write one for:\n[paste your diff]", insight: "Few-shot is the single most powerful technique for getting consistent output format — far easier than fine-tuning for most tasks." },
          { name: "Chain-of-Thought (CoT)", difficulty: "Beginner", use: "Reasoning, debugging, research, decisions", what: "Tell the model to think step by step before answering. Dramatically improves accuracy on complex tasks.", bad: "Which database should I use for this app?", good: "I'm building a real-time chat app expecting 100k concurrent users. Think step by step:\n1. Analyze the requirements (read/write patterns, latency needs, scale)\n2. Compare relevant database types\n3. Evaluate tradeoffs for my specific use case\n4. Give a final recommendation with reasoning", insight: "Adding 'think step by step' or breaking into numbered reasoning steps has been shown to improve LLM accuracy by 30–50% on complex tasks. Works even without examples." },
          { name: "Role / Persona Prompting", difficulty: "Beginner", use: "All use cases", what: "Assign the model an expert identity. Changes vocabulary, depth, assumptions, and communication style.", bad: "Review my resume.", good: "You are a senior engineering hiring manager at a top-tier tech company (FAANG level). You've reviewed 500+ resumes. Review this resume ruthlessly — identify what would get it rejected in under 10 seconds, what's missing, and what's weak. Be direct and specific.", insight: "The role you assign determines the frame of reference. 'Senior engineer' vs 'engineering manager' gives completely different feedback on the same resume." },
          { name: "Output Format Control", difficulty: "Beginner", use: "Coding, automation, structured data", what: "Explicitly specify the output format — JSON, markdown table, bullet points, specific length, etc.", bad: "Extract the key points from this article.", good: 'Extract key points from this article. Return ONLY a JSON array with this structure:\n[\n  {\n    "point": "...",\n    "importance": "high|medium|low",\n    "category": "..."\n  }\n]\nNo preamble. No explanation. JSON only.', insight: "For app development, always specify JSON output + 'no preamble' to get parseable responses without extra parsing logic." },
        ]
      },
      {
        id: 2, tier: "Intermediate", icon: Settings, color: "from-purple-600 to-purple-800", border: "border-purple-700",
        title: "Power Techniques",
        desc: "Where good prompt engineers separate from great ones. These unlock reliability, depth, and consistency.",
        techniques: [
          { name: "System Prompt Design", difficulty: "Intermediate", use: "Apps, automation, consistent assistants", what: "The system prompt defines persistent identity, constraints, tone, and behavior. It's the most leveraged prompt you'll write.", bad: "You are a helpful assistant.", good: "## Identity\nYou are CodeReviewer, a senior software engineer specializing in Python and system design.\n\n## Behavior\n- Always identify bugs before suggesting improvements\n- Cite specific line numbers\n- Distinguish between 'must fix' and 'nice to have'\n- Ask clarifying questions if requirements are unclear\n\n## Output Format\nStructure every review as:\n1. Critical Issues\n2. Suggested Improvements\n3. Positive Observations\n\n## Constraints\n- Never rewrite entire files — give targeted diffs\n- Don't suggest changes outside the stated scope", insight: "A well-designed system prompt can replace 80% of per-request prompting. Build one for every recurring use case you have." },
          { name: "Prompt Chaining", difficulty: "Intermediate", use: "Research, complex writing, automation", what: "Break a complex task into sequential prompts where output of one feeds into the next. More reliable than one giant prompt.", bad: "Research this topic, write a summary, then create a blog post and add SEO metadata.", good: "Step 1 prompt: 'Research [topic]. Output: 5 key insights with source context.'\n→ feed output into →\nStep 2 prompt: 'Given these insights: [output], write a 600-word blog post for a technical audience.'\n→ feed output into →\nStep 3 prompt: 'Given this blog post: [output], generate: title, meta description, 5 tags, and a tweet.'", insight: "Chaining with validation between steps (asking the model to check its own step before proceeding) dramatically reduces compounding errors." },
          { name: "Negative Prompting", difficulty: "Intermediate", use: "Writing, coding, content quality", what: "Explicitly tell the model what NOT to do. Eliminates common failure modes before they happen.", bad: "Summarize this article.", good: "Summarize this article in 3 bullet points.\n\nDo NOT:\n- Use filler phrases like 'The article discusses...'\n- Add your own opinions or analysis\n- Include information not in the article\n- Use bullet points longer than 2 sentences\n- Start with 'Sure!' or any affirmation", insight: "Negative constraints are often more powerful than positive instructions. They eliminate the model's default bad habits directly." },
          { name: "Self-Consistency & Verification", difficulty: "Intermediate", use: "Research, critical decisions, debugging", what: "Ask the model to verify, critique, or challenge its own output in a follow-up prompt.", bad: "(accept first answer)", good: "First prompt: 'Explain why [X approach] is the best solution for [problem].'\n\nSecond prompt: 'Now steelman the opposing view. What are the strongest arguments AGAINST the approach you just recommended? What have you missed?'\n\nThird prompt: 'Given both perspectives, what is your updated final recommendation?'", insight: "LLMs are sycophantic by default — they agree with themselves. Forcing adversarial self-review surfaces blind spots that a single prompt never catches." },
          { name: "Contextual Priming", difficulty: "Intermediate", use: "Research, writing, code review", what: "Front-load rich context before your ask. The quality of context determines the quality of output more than prompt wording.", bad: "Write a technical spec for my feature.", good: "## Context\nApp: B2B SaaS project management tool\nUsers: Engineering teams at mid-sized companies\nStack: Next.js, PostgreSQL, REST API\nTeam: 3 engineers, 1 designer\nTimeline: 2-week sprint\n\n## Feature\nReal-time notifications when task status changes\n\n## Audience for this spec\nJunior engineers who will implement it\n\nWrite a technical specification covering: overview, user stories, API design, database schema changes, edge cases, and acceptance criteria.", insight: "The GIGO principle (garbage in, garbage out) applies more to context than prompt wording. Rich context → dramatically better output." },
        ]
      },
      {
        id: 3, tier: "Advanced", icon: FlaskConical, color: "from-orange-600 to-red-700", border: "border-orange-700",
        title: "Expert Techniques",
        desc: "What separates prompt engineers who build reliable systems from those who get lucky sometimes.",
        techniques: [
          { name: "Tree of Thoughts (ToT)", difficulty: "Advanced", use: "Complex decisions, architecture, research", what: "Ask the model to explore multiple reasoning paths simultaneously, then select the best one.", bad: "How should I architect this system?", good: "I need to architect [system]. Generate 3 completely different high-level architectural approaches. For each:\n1. Name and brief description\n2. Core assumptions\n3. Key advantages\n4. Fatal weaknesses\n5. When this is the right choice\n\nThen evaluate all 3 against my constraints: [list constraints] and recommend one with justification.", insight: "ToT prevents the model from anchoring on its first idea. Exploring multiple paths before committing surfaces options you'd never think to ask about." },
          { name: "Meta-Prompting", difficulty: "Advanced", use: "Building prompt systems, optimization", what: "Use the model to write, improve, or evaluate prompts. Ask the model to be the prompt engineer.", bad: "(struggle to write prompts yourself)", good: "I need a prompt that makes Claude act as a code reviewer for Python. The prompt should:\n- Produce consistent structured output every time\n- Handle both small functions and large modules\n- Distinguish bugs from style issues\n- Be reusable across projects\n\nWrite me the best possible system prompt for this use case. Then explain why you made each design decision.", insight: "Claude and GPT-4 are excellent prompt engineers. Using them to design prompts for themselves produces better results than most humans can write manually." },
          { name: "Constitutional Prompting", difficulty: "Advanced", use: "Content moderation, consistent AI behavior, apps", what: "Define a set of principles or rules the model must follow, then ask it to self-evaluate against those principles.", bad: "Write a response to this customer complaint.", good: "## Principles\n1. Never blame the customer\n2. Always offer a concrete next step\n3. Acknowledge emotion before explaining\n4. Avoid corporate jargon\n5. Response must be under 100 words\n\nWrite a response to this complaint: [complaint]\n\nThen score your response against each principle (1–5). If any score is below 4, rewrite and re-score until all principles are met.", insight: "This is how Anthropic trains Claude — having it evaluate its own outputs against a constitution. You can apply the same technique at prompt time." },
          { name: "Retrieval-Augmented Prompting", difficulty: "Advanced", use: "Research, knowledge-grounded tasks", what: "Inject specific reference material into the prompt context and instruct the model to ground its answer strictly in that material.", bad: "What does our API documentation say about rate limits?", good: "## Reference Material\n[paste exact docs section]\n\n## Instructions\nAnswer the following question using ONLY the reference material above. If the answer is not in the reference material, say 'Not covered in provided documentation' — do not use your general knowledge.\n\n## Question\nWhat are the rate limits for the /v2/messages endpoint?", insight: "This is manual RAG. By grounding the model in specific text, you eliminate hallucinations and get citable, reliable answers." },
          { name: "Prompt Evaluation & Testing", difficulty: "Advanced", use: "App development, reliability engineering", what: "Treat prompts like code — version control them, test against edge cases, measure consistency across runs.", bad: "(change prompts based on gut feel)", good: "1. Define success criteria before writing the prompt ('output must be valid JSON, < 200 words, contain 3 action items')\n2. Build a test set of 10–20 representative inputs including edge cases\n3. Run all inputs through your prompt, score against criteria\n4. A/B test prompt variants — change one variable at a time\n5. Track prompt versions in Git with changelogs\n6. Use tools: PromptFoo, LangSmith, or simple Python scripts", insight: "The #1 difference between amateur and professional prompt engineering is systematic testing. One-shot prompts that 'seem to work' fail unpredictably in production." },
        ]
      },
    ];

    const useCases = [
      {
        id: "coding", icon: Code2, title: "Coding & Debugging", color: "from-blue-600 to-cyan-700",
        tips: [
          { title: "The Debug Template", prompt: "You are a senior [language] engineer debugging a production issue.\n\nCode:\n```\n[paste code]\n```\n\nError:\n```\n[paste error]\n```\n\nContext: [what you expected vs what happened]\n\nThink step by step:\n1. What is this code trying to do?\n2. What does the error indicate?\n3. What are the 3 most likely root causes?\n4. What is the fix for each?\n5. Which fix do you recommend and why?" },
          { title: "Code Review Template", prompt: "Review this [language] code as a senior engineer. Grade each dimension 1–5 and explain:\n- Correctness: Does it do what it intends?\n- Performance: Any O(n²) issues or unnecessary operations?\n- Security: Any vulnerabilities?\n- Readability: Is it self-documenting?\n- Edge cases: What inputs would break this?\n\nThen provide a corrected version with comments explaining each change.\n\n```\n[paste code]\n```" },
          { title: "Architecture Advisor", prompt: "I'm building [describe system]. Current scale: [users/requests]. Expected growth: [timeline].\n\nI'm considering [approach A] vs [approach B].\n\nEvaluate both approaches against:\n1. Scalability to [target scale]\n2. Developer complexity\n3. Operational overhead\n4. Cost at scale\n5. Migration difficulty from current state\n\nRecommend one and justify your choice." },
        ]
      },
      {
        id: "writing", icon: PenLine, title: "Writing & Content", color: "from-purple-600 to-violet-700",
        tips: [
          { title: "Tone Calibration", prompt: "Rewrite the following in my voice. My writing style:\n- Direct and confident, never hedging\n- Short sentences. No fluff.\n- Technical but accessible\n- I use dry humor occasionally\n- I never use: 'leverage', 'synergy', 'delve', 'it's worth noting'\n\nOriginal:\n[paste text]\n\nRewrite it maintaining my style. Don't change the meaning." },
          { title: "Content Reviewer", prompt: "Review this [blog post/email/doc] as a critical editor.\n\nEvaluate:\n1. Opening — does it hook immediately or does it warm up too slowly?\n2. Clarity — any sentences that require re-reading?\n3. Flow — where does the reader's momentum break?\n4. Fluff — which sentences add no value?\n5. Ending — does it close strongly or fade out?\n\nProvide: (a) your overall verdict, (b) 3 specific changes that would most improve it, (c) a rewritten opening if the current one is weak." },
          { title: "Structured Research Summary", prompt: "Summarize [topic or paste article] for a technical audience who is time-constrained.\n\nFormat:\n**TL;DR** (2 sentences max)\n\n**Key Points** (5 bullets, each 1 sentence)\n\n**Why It Matters** (2–3 sentences connecting to [my context])\n\n**Open Questions** (2–3 things this doesn't answer)\n\n**Recommended Action** (1 sentence — what should I do with this information?)" },
        ]
      },
      {
        id: "research", icon: Search, title: "Research & Summarizing", color: "from-green-600 to-teal-700",
        tips: [
          { title: "The Socratic Researcher", prompt: "I want to deeply understand [topic]. Don't give me an overview.\n\nInstead:\n1. What is the single most important thing to understand about this topic?\n2. What is the most common misconception?\n3. What do experts disagree about?\n4. What question should I be asking that I'm not asking?\n5. What would change my mind about the conventional view?\n\nThen ask me 3 questions to understand my specific context before going deeper." },
          { title: "Second-Order Thinking", prompt: "I'm considering [decision/action].\n\nMap out:\n1. First-order effects (immediate, obvious consequences)\n2. Second-order effects (what happens as a result of the first-order effects?)\n3. Third-order effects (longer term ripple effects)\n4. What assumptions am I making that could be wrong?\n5. What would make this decision clearly wrong in hindsight?" },
          { title: "Comparative Analysis", prompt: "Compare [A] vs [B] for my use case: [describe use case].\n\nEvaluate on:\n| Dimension | A | B | Winner |\n|---|---|---|---|\n[fill table]\n\nAfter the table:\n- When would you clearly choose A over B?\n- When would you clearly choose B over A?\n- Is there a C option I haven't considered?\n- Your final recommendation for my specific use case." },
        ]
      },
      {
        id: "automation", icon: Settings, title: "Work Automation", color: "from-orange-600 to-amber-700",
        tips: [
          { title: "The Reusable System Prompt", prompt: "Build me a reusable system prompt for an AI assistant that helps with [specific task].\n\nThe assistant should:\n- [behavior 1]\n- [behavior 2]\n- [behavior 3]\n\nAlways output: [format]\nNever: [anti-behaviors]\n\nThe prompt should be robust enough that I can use it every day without modification." },
          { title: "Document Processor", prompt: 'Process the following document and extract structured information.\n\nDocument type: [invoice/contract/email/report]\n\nExtract and return as JSON:\n{\n  "summary": "2-sentence summary",\n  "key_entities": [...],\n  "action_items": [...],\n  "deadlines": [...],\n  "flags": ["anything unusual or requiring attention"]\n}\n\nDocument:\n[paste document]' },
          { title: "Meeting/Notes Processor", prompt: "Transform these raw meeting notes into a structured document.\n\nNotes:\n[paste notes]\n\nOutput format:\n## Meeting Summary (3 sentences)\n\n## Decisions Made\n- [decision] — Owner: [name] \n\n## Action Items\n- [ ] [task] — Owner: [name] — Due: [date]\n\n## Open Questions\n- [question] — Who resolves: [name]\n\n## Next Meeting\nDate: / Agenda:" },
        ]
      },
    ];

    const promptResources = [
      { type: "Free", label: "Anthropic Prompt Engineering Docs", url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview", note: "Best official reference for Claude specifically" },
      { type: "Free", label: "OpenAI Prompt Engineering Guide", url: "https://platform.openai.com/docs/guides/prompt-engineering", note: "Official GPT-4 guidance, very practical" },
      { type: "Free", label: "DeepLearning.AI – Prompt Engineering for Devs", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/", note: "Best structured intro course (1.5 hrs)" },
      { type: "Free", label: "Learn Prompting (open source guide)", url: "https://learnprompting.org", note: "Community-maintained, covers 50+ techniques" },
      { type: "Free", label: "Prompt Engineering Guide (DAIR.AI)", url: "https://www.promptingguide.ai", note: "Most comprehensive free reference — includes research citations" },
      { type: "Book", label: "Prompt Engineering for Generative AI – O'Reilly", url: "https://www.oreilly.com/library/view/prompt-engineering-for/9781098153427/", note: "Best book on the topic. ~$50 or O'Reilly sub" },
      { type: "Tool", label: "PromptFoo – Prompt Testing Framework", url: "https://promptfoo.dev", note: "Free tool to test and compare prompts systematically" },
      { type: "Tool", label: "Anthropic Workbench", url: "https://console.anthropic.com/workbench", note: "Free prompt playground with system prompt support" },
      { type: "Tool", label: "OpenAI Playground", url: "https://platform.openai.com/playground", note: "Test prompts with full parameter control" },
      { type: "Practice", label: "Prompt Engineering subreddit", url: "https://www.reddit.com/r/PromptEngineering/", note: "Real examples and community feedback" },
    ];

    const promptMilestones = [
      { week: "Week 1–2", title: "Foundation Mastery", tasks: ["Master all 5 core techniques", "Build a personal template library for your top 3 daily tasks", "Compare zero-shot vs few-shot on the same task 10 times"] },
      { week: "Week 3–4", title: "Power Techniques", tasks: ["Build 1 reusable system prompt per use case (coding, writing, research)", "Practice prompt chaining on a real research task", "Start a 'prompt journal' — log what works and why"] },
      { week: "Week 5–6", title: "Advanced + Evaluation", tasks: ["Use meta-prompting to improve your existing prompts", "Set up PromptFoo and test 3 prompts systematically", "Build a 'prompt library' in Notion or GitHub for reuse"] },
      { week: "Month 2–3", title: "Expert Territory", tasks: ["Build a small AI-powered tool using your prompt library", "Write about your learnings — blog post or LinkedIn article", "Start reading prompt engineering research papers (CoT, ToT papers)"] },
    ];

    const universalTemplate = `[ROLE] You are a [expert identity] with [relevant experience].

[CONTEXT] I am [your situation]. The goal is [what you're trying to achieve].

[TASK] [Clear, specific instruction verb — Write / Review / Analyze / Compare / Extract]

[CONSTRAINTS]
- Do NOT: [anti-behavior 1], [anti-behavior 2]
- Always: [required behavior]

[OUTPUT FORMAT]
Return your response as: [format — JSON / bullet list / table / prose]
Length: [word count or structure]

[INPUT]
[paste your actual content here]`;

    function PromptEngineering() {
      useSeo(
        "Prompt Engineering Guide – Techniques & Templates | AI Learning Hub",
        "Master prompt engineering with 15 techniques from zero-shot to tree-of-thoughts, copy-paste templates for coding, writing, and research, plus a 6-week practice plan."
      );
      const [openTech, setOpenTech] = useState(null);
      const [openTier, setOpenTier] = useState(1);
      const [openUC, setOpenUC] = useState(null);
      const [openTemplate, setOpenTemplate] = useState(null);
      const [tab, setTab] = useState("learn");
      const [copied, setCopied] = useState(null);

      const copy = (text, id) => {
        navigator.clipboard.writeText(text).then(() => {
          setCopied(id); setTimeout(() => setCopied(null), 1500);
        });
      };

      return (
        <div className="min-h-screen text-gray-100 p-4 font-sans">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-6">
              <h1 className="text-2xl font-bold mb-1"><Zap size={20} className="inline mr-2 align-middle text-yellow-400"/>Prompt Engineering Mastery</h1>
              <p className="text-gray-400 text-sm max-w-lg mx-auto">Master the skill that makes or breaks every AI app. Work through techniques in order — Foundation first, then Advanced.</p>
              <p className="text-xs text-gray-400 mt-2">Use the Techniques tab to learn, Templates to copy-paste, and Plan for a structured 6-week practice schedule.</p>
            </div>

            <div className="flex gap-1 bg-gray-900 rounded-xl p-1 mb-6 border border-gray-800">
              {[["learn", "Techniques"], ["templates", "Templates"], ["milestones", "Plan"], ["resources", "📚 Resources"]].map(([k, v]) => (
                <button key={k} onClick={() => setTab(k)}
                  className={`flex-1 text-xs py-2 rounded-lg transition-colors ${tab === k ? "bg-gray-700 text-white font-semibold" : "text-gray-400 hover:text-gray-300"}`}>
                  {v}
                </button>
              ))}
            </div>

            {tab === "learn" && (
              <div className="space-y-4">
                <div className="bg-gray-900 border border-yellow-800 rounded-xl p-4 mb-4">
                  <p className="text-yellow-400 font-semibold text-sm mb-1"><Lightbulb size={14} className="inline mr-1.5 align-middle"/>The Expert's Mental Model</p>
                  <p className="text-gray-300 text-sm">A prompt is a <span className="text-white font-semibold">specification</span>, not a question. The more precisely you specify Role + Context + Task + Format + Constraints, the more reliably you get expert-level output. Every technique below is just a way to add one of these dimensions.</p>
                </div>
                {techniques.map(tier => (
                  <div key={tier.id} className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden">
                    <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setOpenTier(openTier === tier.id ? null : tier.id)}>
                      <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${tier.color} flex items-center justify-center text-base flex-shrink-0`}>{React.createElement(tier.icon, {size:20, className:"text-white"})}</div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <p className="font-semibold text-sm">{tier.title}</p>
                          <span className={`text-xs px-2 py-0.5 rounded-full border ${tier.border} text-gray-300`}>{tier.tier}</span>
                        </div>
                        <p className="text-gray-400 text-xs mt-0.5">{tier.techniques.length} techniques</p>
                      </div>
                      <span className="text-gray-400">{openTier === tier.id ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
                    </div>
                    {openTier === tier.id && (
                      <div className="border-t border-gray-800 p-4 space-y-3">
                        <p className="text-gray-400 text-xs">{tier.desc}</p>
                        {tier.techniques.map((t, i) => (
                          <div key={i} className={`rounded-lg border overflow-hidden ${openTech === `${tier.id}-${i}` ? "border-gray-500" : "border-gray-700"}`}>
                            <div className="flex items-center gap-3 p-3 cursor-pointer bg-gray-800"
                              onClick={() => setOpenTech(openTech === `${tier.id}-${i}` ? null : `${tier.id}-${i}`)}>
                              <div className="flex-1">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <p className="text-sm font-semibold text-white">{t.name}</p>
                                  <span className="text-xs text-gray-400 bg-gray-700 px-2 py-0.5 rounded">{t.use}</span>
                                </div>
                                <p className="text-xs text-gray-400 mt-0.5">{t.what}</p>
                              </div>
                              <span className="text-gray-400 text-sm">{openTech === `${tier.id}-${i}` ? "▲" : "▼"}</span>
                            </div>
                            {openTech === `${tier.id}-${i}` && (
                              <div className="p-3 space-y-3 bg-gray-900">
                                <div className="grid grid-cols-2 gap-2">
                                  <div className="bg-red-950 border border-red-900 rounded-lg p-3">
                                    <p className="text-xs text-red-400 font-semibold mb-1"><X size={11} className="inline mr-1 align-middle text-red-400"/>Weak Prompt</p>
                                    <p className="text-xs text-gray-300 font-mono whitespace-pre-wrap">{t.bad}</p>
                                  </div>
                                  <div className="bg-green-950 border border-green-900 rounded-lg p-3">
                                    <p className="text-xs text-green-400 font-semibold mb-1"><Check size={11} className="inline mr-1 align-middle"/>Strong Prompt</p>
                                    <p className="text-xs text-gray-300 font-mono whitespace-pre-wrap">{t.good}</p>
                                  </div>
                                </div>
                                <div className="bg-yellow-950 border border-yellow-900 rounded-lg p-3">
                                  <p className="text-xs text-yellow-400 font-semibold mb-1"><Lightbulb size={11} className="inline mr-1 align-middle"/>Key Insight</p>
                                  <p className="text-xs text-gray-300">{t.insight}</p>
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {tab === "templates" && (
              <div className="space-y-4">
                <p className="text-gray-400 text-xs mb-2">Ready-to-use templates for your specific daily use cases. Click to expand, copy to use.</p>
                {useCases.map(uc => (
                  <div key={uc.id} className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden">
                    <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setOpenUC(openUC === uc.id ? null : uc.id)}>
                      <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${uc.color} flex items-center justify-center text-base flex-shrink-0`}>{React.createElement(uc.icon, {size:20, className:"text-white"})}</div>
                      <p className="font-semibold text-sm flex-1">{uc.title}</p>
                      <span className="text-gray-400">{openUC === uc.id ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
                    </div>
                    {openUC === uc.id && (
                      <div className="border-t border-gray-800 p-4 space-y-3">
                        {uc.tips.map((tip, i) => (
                          <div key={i} className={`rounded-lg border overflow-hidden ${openTemplate === `${uc.id}-${i}` ? "border-gray-500" : "border-gray-700"}`}>
                            <div className="flex items-center justify-between p-3 bg-gray-800 cursor-pointer"
                              onClick={() => setOpenTemplate(openTemplate === `${uc.id}-${i}` ? null : `${uc.id}-${i}`)}>
                              <p className="text-sm font-semibold text-white">{tip.title}</p>
                              <span className="text-gray-400 text-sm">{openTemplate === `${uc.id}-${i}` ? "▲" : "▼"}</span>
                            </div>
                            {openTemplate === `${uc.id}-${i}` && (
                              <div className="p-3 bg-gray-900">
                                <div className="bg-gray-800 rounded-lg p-3 font-mono text-xs text-gray-300 whitespace-pre-wrap mb-2">{tip.prompt}</div>
                                <button onClick={() => copy(tip.prompt, `${uc.id}-${i}`)}
                                  className="text-xs bg-blue-700 hover:bg-blue-600 text-white px-3 py-1.5 rounded-lg transition-colors">
                                  {copied === `${uc.id}-${i}` ? "Copied!" : "Copy Template"}
                                </button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2"><Sparkles size={12} className="inline mr-1.5 align-middle"/>The Universal Prompt Formula</p>
                  <div className="bg-gray-800 rounded-lg p-3 font-mono text-xs text-gray-200 whitespace-pre-wrap mb-2">{universalTemplate}</div>
                  <button onClick={() => copy(universalTemplate, "universal")}
                    className="text-xs bg-blue-700 hover:bg-blue-600 text-white px-3 py-1.5 rounded-lg transition-colors">
                    {copied === "universal" ? "Copied!" : "Copy Universal Template"}
                  </button>
                </div>
              </div>
            )}

            {tab === "milestones" && (
              <div className="space-y-4">
                <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-2">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1"><Zap size={12} className="inline mr-1.5 align-middle"/>Time to Expert</p>
                  <p className="text-gray-300 text-sm">At your current level (daily AI user, software dev), expect <span className="text-white font-semibold">6–8 weeks</span> of deliberate practice to reach expert level. The key word is <span className="text-white font-semibold">deliberate</span> — random usage doesn't build expertise. Intentional practice does.</p>
                </div>
                {promptMilestones.map((m, i) => (
                  <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center text-xs font-bold flex-shrink-0">{i + 1}</div>
                      <div>
                        <p className="font-semibold text-sm text-white">{m.title}</p>
                        <p className="text-xs text-gray-400">{m.week}</p>
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      {m.tasks.map((t, j) => (
                        <div key={j} className="flex gap-2">
                          <span className="text-gray-400 flex-shrink-0 mt-0.5"><ArrowRight size={13} className="flex-shrink-0"/></span>
                          <p className="text-xs text-gray-300">{t}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
                <div className="bg-gray-900 border border-green-800 rounded-xl p-4">
                  <p className="text-green-400 font-semibold text-sm mb-2"><Trophy size={14} className="inline mr-1.5 align-middle"/>How You Know You're Expert Level</p>
                  <div className="space-y-1.5">
                    {[
                      "You instinctively know which technique to apply for any task",
                      "Your prompts work reliably on the first try 80%+ of the time",
                      "You can diagnose why a prompt failed and fix it in one iteration",
                      "You have a personal library of 20+ reusable prompt templates",
                      "You can explain prompt design decisions to others clearly",
                      "You test and version your prompts like code",
                    ].map((s, i) => (
                      <div key={i} className="flex gap-2">
                        <span className="text-green-400 flex-shrink-0"><Check size={13}/></span>
                        <p className="text-xs text-gray-300">{s}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {tab === "resources" && (
              <div className="space-y-3">
                {["Free", "Book", "Tool", "Practice"].map(type => (
                  <div key={type}>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                      {type === "Free" ? <><GraduationCap size={12} className="inline mr-1"/>Free Courses & Guides</> : type === "Book" ? <><BookOpen size={12} className="inline mr-1"/>Books</> : type === "Tool" ? <><Wrench size={12} className="inline mr-1"/>Tools</> : <><MessageSquare size={12} className="inline mr-1"/>Community</>}
                    </p>
                    <div className="space-y-2">
                      {promptResources.filter(r => r.type === type).map((r, i) => (
                        <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
                          className="flex items-start gap-3 bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-lg p-3 transition-colors">
                          <div className="flex-1">
                            <p className="text-blue-400 text-sm font-medium">{r.label}</p>
                            <p className="text-gray-400 text-xs mt-0.5">{r.note}</p>
                          </div>
                        </a>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // PREP PLAN COMPONENT
    // ═══════════════════════════════════════════════════════════
    const prepWeeks = [
      {
        week: 1, title: "How LLMs Actually Work",
        phase: "Phase 1 — Foundations", phaseColor: "text-green-400 bg-green-500/10 border-green-500/20",
        icon: Brain, color: "from-green-500 to-emerald-600",
        goal: "Establish conceptual foundations — not implementation depth. By the end of this week you should be able to explain what an LLM is and how it generates text to a non-technical colleague.",
        resources: [
          { label: "3Blue1Brown – Neural Networks playlist", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", time: "~1.5 hrs", type: "video" },
          { label: "Andrej Karpathy – Intro to LLMs", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g", time: "1 hr", type: "video" },
          { label: "The Illustrated Transformer – Jay Alammar", url: "https://jalammar.github.io/illustrated-transformer/", time: "30 min", type: "article" },
        ],
        project: "Explore 3 pre-built AI demos on Hugging Face Spaces — text generation, sentiment analysis, image captioning.",
      },
      {
        week: 2, title: "Set Up Your AI Environment",
        phase: "Phase 2 — LLM Setup", phaseColor: "text-slate-400 bg-slate-500/10 border-slate-500/20",
        icon: Settings, color: "from-slate-500 to-gray-600",
        goal: "Move from theory to practice. The objective is to have at least one LLM running and callable from code — both locally and via a cloud API.",
        resources: [
          { label: "Ollama – run a 7B model locally in 10 minutes", url: "https://ollama.com", time: "setup", type: "tool" },
          { label: "Open Source Models with Hugging Face (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/", time: "1.5 hrs", type: "course" },
          { label: "Anthropic API Quickstart", url: "https://docs.anthropic.com/en/docs/quickstart", time: "reference", type: "article" },
        ],
        project: "Run Llama 3 locally via Ollama and call the same prompt against a cloud API. Compare speed, quality, and cost.",
      },
      {
        week: 3, title: "Prompt Engineering",
        phase: "Phase 3 — Prompting & APIs", phaseColor: "text-blue-400 bg-blue-500/10 border-blue-500/20",
        icon: Zap, color: "from-blue-500 to-indigo-600",
        goal: "Prompt engineering is where most developers can add immediate value without deep ML knowledge. This week covers the core patterns used in production systems.",
        resources: [
          { label: "Prompt Engineering for Devs (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/", time: "1.5 hrs", type: "course" },
          { label: "Anthropic Prompt Engineering Docs", url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview", time: "reference", type: "article" },
          { label: "Building Systems with LLM APIs (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/", time: "1.5 hrs", type: "course" },
        ],
        project: "Build a simple CLI tool powered by an LLM API — a code reviewer, doc summarizer, or Q&A bot.",
      },
      {
        week: 4, title: "RAG — Making AI Know Your Data",
        phase: "Phase 4 — RAG & Data", phaseColor: "text-purple-400 bg-purple-500/10 border-purple-500/20",
        icon: BookOpen, color: "from-purple-500 to-violet-600",
        goal: "Retrieval-Augmented Generation (RAG) is how production AI systems connect to private data, documentation, and domain knowledge. It is the most widely deployed pattern in enterprise AI applications.",
        resources: [
          { label: "LangChain: Chat with Your Data (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/", time: "1.5 hrs", type: "course" },
          { label: "Building & Evaluating Advanced RAG (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/", time: "1.5 hrs", type: "course" },
          { label: "Hugging Face NLP Course – Ch. 1–4", url: "https://huggingface.co/learn/nlp-course/chapter1/1", time: "self-paced", type: "course" },
        ],
        project: "Build a chatbot that answers questions from a PDF you care about — a technical doc, book, or internal guide.",
      },
      {
        week: 5, title: "Agentic AI & Tool Use",
        phase: "Phase 5 — Agentic AI", phaseColor: "text-orange-400 bg-orange-500/10 border-orange-500/20",
        icon: Bot, color: "from-orange-500 to-amber-600",
        goal: "Agentic systems go beyond question-answering — they plan, invoke tools, and execute multi-step workflows with minimal human intervention. This represents the current frontier of practical AI development.",
        resources: [
          { label: "AI Agents in LangGraph (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/", time: "2 hrs", type: "course" },
          { label: "Functions, Tools and Agents with LangChain (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/", time: "2 hrs", type: "course" },
          { label: "Anthropic MCP Documentation", url: "https://docs.anthropic.com/en/docs/agents-and-tools/mcp", time: "reference", type: "article" },
        ],
        project: "Build a simple ReACT agent that can search the web and write a summary report.",
      },
      {
        week: 6, title: "Fine-Tuning & Multimodal AI",
        phase: "Phases 6–7 — Training & Beyond", phaseColor: "text-rose-400 bg-rose-500/10 border-rose-500/20",
        icon: Rocket, color: "from-rose-500 to-pink-600",
        goal: "A survey of the deeper end of the stack — how models are trained, adapted, and extended to handle images, audio, and video. The objective this week is informed awareness, not implementation mastery.",
        resources: [
          { label: "Finetuning Large Language Models (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/", time: "1 hr", type: "course" },
          { label: "Diffusion Models Course (Hugging Face)", url: "https://huggingface.co/learn/diffusion-course/unit0/1", time: "self-paced", type: "course" },
          { label: "Andrej Karpathy – Let's build GPT from scratch", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", time: "2 hrs", type: "video" },
        ],
        project: "Fine-tune a small model on Google Colab using the DeepLearning.AI notebook, or run Stable Diffusion in a Hugging Face Space.",
      },
    ];

    const prepTools = [
      { name: "Google Colab", desc: "Run Python/AI code in browser, free GPU", url: "https://colab.research.google.com" },
      { name: "Ollama", desc: "Run LLMs locally for free", url: "https://ollama.com" },
      { name: "Hugging Face", desc: "Models, datasets, free Spaces", url: "https://huggingface.co" },
      { name: "LangChain", desc: "Build RAG & agent apps", url: "https://python.langchain.com" },
      { name: "Claude.ai", desc: "Prompt engineering practice", url: "https://claude.ai" },
      { name: "ChatGPT", desc: "Prompt engineering practice", url: "https://chat.openai.com" },
    ];

    const resourceTypeStyle = {
      video:   { label: "Video",   cls: "bg-red-500/10 text-red-400 border border-red-500/20" },
      course:  { label: "Course",  cls: "bg-blue-500/10 text-blue-400 border border-blue-500/20" },
      article: { label: "Read",    cls: "bg-gray-500/10 text-gray-400 border border-gray-500/20" },
      tool:    { label: "Tool",    cls: "bg-green-500/10 text-green-400 border border-green-500/20" },
    };

    function PrepPlan() {
      useSeo(
        "AI Interview Prep Plan – 6-Week Fast Track | AI Learning Hub",
        "Structured 6-week AI prep plan for software developers. Cover LLMs, Prompt Engineering, RAG, and Agentic AI with free resources — 4–6 hours per week."
      );
      const [open, setOpen] = useState(null);

      return (
        <div className="min-h-screen text-gray-100 p-4 font-sans">
          <div className="max-w-3xl mx-auto">

            {/* Header */}
            <div className="text-center mb-8">
              <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 rounded-full px-3 py-1 mb-4">
                <Calendar size={12} className="text-blue-400"/>
                <span className="text-xs text-blue-400 font-medium">6-Week Fast Track</span>
              </div>
              <h1 className="text-3xl md:text-4xl font-bold mb-3">AI Engineering Fast Track</h1>
              <p className="text-gray-400 text-sm max-w-lg mx-auto">
                A structured 6-week overview of the full AI engineering stack — free resources only.{" "}
                Inspired by the{" "}
                <a href="https://bytebyteai.com" target="_blank" rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 transition-colors">ByteByteAI course</a>.
              </p>
              <div className="mt-3 inline-flex items-center gap-2 bg-orange-500/10 border border-orange-500/20 rounded-full px-3 py-1">
                <AlertTriangle size={12} className="text-orange-400"/>
                <span className="text-xs text-orange-400">Use this if you need results fast — not a replacement for the full roadmap</span>
              </div>
            </div>

            {/* Goal callout */}
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 mb-8">
              <p className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Target size={12}/> Goal
              </p>
              <p className="text-gray-200 text-sm leading-relaxed">
                This plan is <strong className="text-white">not a replacement</strong> for the ByteByteAI course — it's preparation for it.
                Complete this first and you will either match the course pace from day one, or progress through it
                significantly faster because the core concepts will already be familiar.
              </p>
            </div>

            {/* Before you start — treated as week 0 in the timeline */}
            <div className="relative mb-2">
              <div className="flex gap-4">
                <div className="flex flex-col items-center flex-shrink-0 z-10">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-gray-600 to-gray-700 flex items-center justify-center shadow-lg ring-2 ring-gray-900">
                    <Check size={20} className="text-white"/>
                  </div>
                </div>
                <div className="flex-1 min-w-0 pb-6">
                  <div className="bg-gray-900/60 border border-white/8 rounded-xl p-4">
                    <p className="font-semibold text-sm text-white mb-3">Before You Start</p>
                    <div className="space-y-2">
                      {[
                        ["Google Colab account", "No local setup needed — runs Python in your browser for free", "https://colab.research.google.com"],
                        ["Claude.ai or ChatGPT account", "Free tier is enough for prompt practice", "https://claude.ai"],
                        ["4–6 hours per week", "Enough to complete the resources and mini-project each week", null],
                      ].map(([title, desc, url], i) => (
                        <div key={i} className="flex gap-3 items-start">
                          <Check size={13} className="text-green-400 mt-0.5 flex-shrink-0"/>
                          <div>
                            {url
                              ? <a href={url} target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-blue-400 hover:text-blue-300 transition-colors">{title}</a>
                              : <span className="text-sm font-medium text-white">{title}</span>
                            }
                            <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Timeline */}
            <div className="relative mb-10">
              {/* Vertical line */}
              <div className="absolute left-6 top-6 bottom-6 w-0.5 bg-gradient-to-b from-green-500 via-blue-500 via-purple-500 to-rose-500 opacity-30" />
              <div className="space-y-6">
                {prepWeeks.map((w) => (
                  <div key={w.week} className="relative flex gap-4">
                    {/* Timeline node */}
                    <div className="flex flex-col items-center flex-shrink-0 z-10">
                      <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${w.color} flex items-center justify-center shadow-lg ring-2 ring-gray-900`}>
                        <w.icon size={20} className="text-white"/>
                      </div>
                    </div>
                    {/* Card */}
                    <div className="flex-1 min-w-0">
                      <div
                        className={`rounded-xl border cursor-pointer transition-all duration-200 backdrop-blur-sm ${open === w.week ? "border-blue-500/30 bg-gray-900/80 shadow-[0_0_30px_rgba(59,130,246,0.08)]" : "border-white/8 bg-gray-900/60 hover:border-white/15 hover:bg-gray-900/80"}`}
                        onClick={() => setOpen(open === w.week ? null : w.week)}>

                        {/* Card header */}
                        <div className="flex items-center gap-3 p-4">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-semibold text-sm">{w.title}</span>
                              <span className={`text-xs px-2 py-0.5 rounded-full border ${w.phaseColor}`}>{w.phase}</span>
                            </div>
                            <p className="text-gray-400 text-xs mt-0.5">Week {w.week} · ~4–6 hrs</p>
                          </div>
                          <span className="text-gray-400 flex-shrink-0">{open === w.week ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
                        </div>

                        {/* Expanded */}
                        {open === w.week && (
                          <div className="border-t border-gray-800 p-4 space-y-4">
                            <p className="text-gray-300 text-sm">{w.goal}</p>

                            {/* Resources */}
                            <div className="space-y-1.5">
                              {w.resources.map((r, i) => (
                                <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
                                  className="flex items-center justify-between bg-gray-800 hover:bg-gray-700 rounded-lg px-3 py-2 transition-colors group"
                                  onClick={e => e.stopPropagation()}>
                                  <span className="text-sm text-blue-400 group-hover:text-blue-300 flex items-center gap-1.5">
                                    <ExternalLink size={12} className="flex-shrink-0 opacity-60"/>{r.label}
                                  </span>
                                  <div className="flex items-center gap-2 flex-shrink-0 ml-3">
                                    <span className="text-xs text-gray-400">{r.time}</span>
                                    <span className={`text-xs px-1.5 py-0.5 rounded ${resourceTypeStyle[r.type].cls}`}>{resourceTypeStyle[r.type].label}</span>
                                  </div>
                                </a>
                              ))}
                            </div>

                            {/* Project */}
                            <div className="bg-gray-800 rounded-lg p-3">
                              <p className="text-xs font-semibold text-yellow-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                                <Wrench size={11}/> Week {w.week} Project
                              </p>
                              <p className="text-sm text-gray-200">{w.project}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Tools */}
            <div className="mb-8">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Wrench size={12}/> Essential Free Tools
              </p>
              <div className="grid grid-cols-2 gap-2">
                {prepTools.map((t, i) => (
                  <a key={i} href={t.url} target="_blank" rel="noopener noreferrer"
                    className="bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-lg p-3 transition-colors">
                    <p className="text-sm font-medium text-blue-400">{t.name}</p>
                    <p className="text-xs text-gray-400 mt-0.5">{t.desc}</p>
                  </a>
                ))}
              </div>
            </div>

            {/* Roadmap callout */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <ArrowRight size={12}/> How This Connects to the Roadmap
              </p>
              <p className="text-gray-300 text-sm leading-relaxed">
                This 6-week plan is a <strong className="text-white">deliberate survey</strong> — one structured pass across every major layer of the AI stack.
                The Roadmap tab then guides you through each phase at the appropriate depth, with curated resources and hands-on projects to build production-level expertise.
              </p>
              <p className="text-gray-400 text-xs mt-2">
                Treat this plan as orientation — a map of the terrain before committing to the full journey.
              </p>
            </div>

            <p className="text-center text-gray-400 text-xs mt-6">Click each week to expand · Resources open in a new tab</p>
          </div>
        </div>
      );
    }

    // ─── NEW COMPONENTS + UPDATED APP APPENDED BELOW ───



    // ─── GENAI GUIDE COMPONENT ───
const genaiTabIds = ["overview", "text", "code", "image", "audio", "tools", "roadmap"];

const domainData = {
  text: {
    icon: MessageSquare, title: "Text & LLMs", color: "from-blue-600 to-indigo-700", border: "border-blue-700",
    tagline: "The foundation of all GenAI. Everything else builds on these ideas.",
    howItWorks: [
      { step: "1. Tokenization", desc: "Text is broken into tokens (word pieces). 'Unbelievable' → ['Un','believ','able']. The model never sees raw text — only token IDs." },
      { step: "2. Embeddings", desc: "Each token is mapped to a high-dimensional vector (e.g. 4096 numbers). Similar meanings → similar vectors. This is how meaning is encoded numerically." },
      { step: "3. Transformer Attention", desc: "Each token 'attends' to every other token in context. The attention mechanism figures out which tokens are most relevant to each other — this is where understanding happens." },
      { step: "4. Feed-Forward Layers", desc: "After attention, each position passes through a neural network that transforms representations. This is where 'knowledge' is stored — billions of these weights encode world knowledge." },
      { step: "5. Next Token Prediction", desc: "The model outputs a probability distribution over all tokens in its vocabulary. The highest probability token is selected (or sampled). Repeat. This is how generation works — one token at a time." },
      { step: "6. Sampling Parameters", desc: "Temperature (creativity), Top-p (diversity), Top-k (vocabulary restriction) control how the next token is chosen from the probability distribution. This is why the same prompt gives different answers each time." },
    ],
    keyConceptsTitle: "Key Concepts to Master",
    keyConcepts: [
      { name: "Context Window", desc: "How many tokens the model can 'see' at once. GPT-4: 128k. Claude: 200k. Determines memory, document size, conversation length." },
      { name: "Temperature", desc: "0 = deterministic/focused. 1 = creative/varied. 2 = chaotic. For code: use 0–0.3. For creative writing: 0.7–1.0." },
      { name: "RLHF", desc: "Reinforcement Learning from Human Feedback. How models learn to be helpful and safe — humans rate outputs, a reward model is trained, then used to guide the LLM." },
      { name: "Hallucination", desc: "Models generate plausible-sounding but false information. Caused by probability-based generation, not factual lookup. Solution: RAG, grounding, verification." },
      { name: "Emergent Abilities", desc: "At scale, models suddenly gain capabilities not present in smaller versions — reasoning, code, translation. These aren't designed, they emerge from scale." },
      { name: "System Prompt", desc: "Instructions that persist throughout a conversation, defining identity, constraints, and behavior. The most powerful lever for consistent AI behavior." },
    ],
    models: [
      { name: "GPT-4o", maker: "OpenAI", best: "General purpose, vision, tool use", access: "API + ChatGPT" },
      { name: "Claude 3.5 Sonnet", maker: "Anthropic", best: "Long context, analysis, coding, safety", access: "API + Claude.ai" },
      { name: "Gemini 1.5 Pro", maker: "Google", best: "Multimodal, very long context (1M tokens)", access: "API + Gemini" },
      { name: "Llama 3 70B", maker: "Meta (open)", best: "Open source, local/private use", access: "Ollama / HuggingFace" },
      { name: "Mistral Large", maker: "Mistral (open)", best: "Efficient, multilingual, open weights", access: "API / local" },
      { name: "DeepSeek R1", maker: "DeepSeek (open)", best: "Reasoning, math, coding, open source", access: "API / local" },
    ],
    resources: [
      { label: "Andrej Karpathy – Intro to LLMs", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g", free: true, time: "1 hr" },
      { label: "3Blue1Brown – Transformers explained visually", url: "https://www.youtube.com/watch?v=wjZofJX0v4M", free: true, time: "30 min" },
      { label: "Hands-On LLMs – Jay Alammar (O'Reilly)", url: "https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/", free: false, time: "Book" },
      { label: "The Illustrated Transformer – Jay Alammar", url: "https://jalammar.github.io/illustrated-transformer/", free: true, time: "30 min" },
    ]
  },
  code: {
    icon: Code2, title: "Code Generation", color: "from-green-600 to-teal-700", border: "border-green-700",
    tagline: "The most immediately useful GenAI for software developers. Already transforming how code is written.",
    howItWorks: [
      { step: "1. Code Pre-training", desc: "Models trained on massive code corpora (GitHub, Stack Overflow, docs). They learn syntax, patterns, and the relationships between natural language descriptions and code." },
      { step: "2. Fill-in-the-Middle (FIM)", desc: "Code models are trained not just to predict next tokens but to fill in missing sections given prefix AND suffix. This powers inline autocomplete in editors." },
      { step: "3. Instruction Fine-tuning", desc: "On top of raw code training, models are fine-tuned on (instruction → code) pairs. This is what makes them follow 'write a function that...' style prompts." },
      { step: "4. Context Awareness", desc: "Modern code models ingest open files, imports, function signatures, and even documentation to generate contextually relevant code — not just isolated snippets." },
      { step: "5. Tool Use / Agentic Coding", desc: "The frontier: AI agents that can run code, read error messages, search docs, edit files, and iterate autonomously. Claude Code, Cursor, and Devin operate this way." },
    ],
    keyConceptsTitle: "Key Capabilities",
    keyConcepts: [
      { name: "Autocomplete", desc: "Line/block completion as you type. Copilot, Cursor. Low latency, context-aware." },
      { name: "Code Generation", desc: "Generate entire functions, classes, or files from a natural language description." },
      { name: "Refactoring", desc: "Transform existing code — extract functions, rename, optimize, change patterns — while preserving behavior." },
      { name: "Debugging", desc: "Given code + error, identify root cause and suggest fixes. Most models are excellent at this." },
      { name: "Test Generation", desc: "Generate unit tests, edge cases, and integration tests from source code." },
      { name: "Code Review", desc: "Review PRs or functions for bugs, security issues, performance, and style." },
    ],
    models: [
      { name: "Claude 3.5 Sonnet", maker: "Anthropic", best: "Best overall for complex coding + reasoning", access: "API / Claude.ai / Claude Code" },
      { name: "GPT-4o", maker: "OpenAI", best: "Strong all-around, great with tool use", access: "API / ChatGPT / Copilot" },
      { name: "DeepSeek Coder V2", maker: "DeepSeek (open)", best: "Best open-source coding model", access: "Local / API" },
      { name: "Codestral", maker: "Mistral (open)", best: "Fast, efficient, 80+ languages", access: "API / local" },
      { name: "Gemini 1.5 Pro", maker: "Google", best: "Great for full codebase context (1M tokens)", access: "API / Gemini" },
    ],
    tools: [
      { name: "Cursor", desc: "Best AI code editor. Claude/GPT-4 built in. Codebase-aware.", url: "https://cursor.sh" },
      { name: "GitHub Copilot", desc: "Industry standard autocomplete. Works in VS Code.", url: "https://github.com/features/copilot" },
      { name: "Claude Code", desc: "Anthropic's agentic CLI — reads/writes files, runs code.", url: "https://claude.ai/code" },
      { name: "Cody (Sourcegraph)", desc: "Codebase-aware AI across large repos.", url: "https://sourcegraph.com/cody" },
    ],
    resources: [
      { label: "DeepLearning.AI – Pair Programming with LLMs", url: "https://www.deeplearning.ai/short-courses/pair-programming-llm/", free: true, time: "1 hr" },
      { label: "Cursor Docs – Getting started", url: "https://docs.cursor.com", free: true, time: "reference" },
      { label: "Claude Code documentation", url: "https://docs.anthropic.com/en/docs/claude-code/overview", free: true, time: "reference" },
    ]
  },
  image: {
    icon: Palette, title: "Image & Video Generation", color: "from-purple-600 to-pink-700", border: "border-purple-700",
    tagline: "From pixels to meaning — how AI learned to paint, and now to direct films.",
    howItWorks: [
      { step: "1. VAE – Compress to Latent Space", desc: "A Variational Autoencoder compresses images into a compact latent representation (much smaller than raw pixels). Generation happens in this compressed space — far more efficient." },
      { step: "2. Diffusion – Forward Process", desc: "Training: Take a real image, progressively add random Gaussian noise over many steps until it's pure noise. The model learns to reverse this — to denoise." },
      { step: "3. Diffusion – Reverse Process (Generation)", desc: "At generation time: Start from pure random noise. Iteratively apply the learned denoising steps. After ~20–50 steps, a coherent image emerges from nothing." },
      { step: "4. CLIP Text Conditioning", desc: "Text prompts are encoded into the same vector space as images (via CLIP). During denoising, the model is guided toward images that match the text embedding — this is how text-to-image works." },
      { step: "5. U-Net / DiT Architecture", desc: "The denoising network is either a U-Net (older, Stable Diffusion 1.x) or a Diffusion Transformer/DiT (newer — DALL-E 3, Flux, SD3). DiT architectures scale better and produce higher quality." },
      { step: "6. Video: Temporal Consistency", desc: "Video generation adds a time dimension — frames must be consistent across time. Models like Sora use 3D attention over space and time. The main challenge: motion coherence and physics." },
    ],
    keyConceptsTitle: "Key Concepts",
    keyConcepts: [
      { name: "CFG Scale", desc: "Classifier-Free Guidance. How strictly the image follows the prompt. Low (1–5): artistic freedom. High (10–15): literal prompt following. Sweet spot: 7–9." },
      { name: "Sampling Steps", desc: "More steps = higher quality but slower. 20–30 steps usually sufficient. Diminishing returns beyond 50." },
      { name: "LoRA for Images", desc: "Train a small adapter on a person's face or art style. Inject into diffusion model to generate images of that specific person/style. Powers most face/style customization." },
      { name: "ControlNet", desc: "Control generation with structural guides — pose, depth map, edge detection. Lets you generate images that follow a specific composition." },
      { name: "Inpainting", desc: "Edit specific regions of an image while preserving the rest. Mask an area, describe what should replace it." },
      { name: "Prompt Weighting", desc: "(word:1.5) increases emphasis. (word:0.5) decreases it. Lets you control how much each element influences the output." },
    ],
    models: [
      { name: "DALL-E 3", maker: "OpenAI", best: "Best prompt following, high quality, safe", access: "ChatGPT / API" },
      { name: "Midjourney v6", maker: "Midjourney", best: "Best aesthetic quality, photorealism", access: "Discord / Web" },
      { name: "Flux 1.1 Pro", maker: "Black Forest Labs (open)", best: "Best open model, excellent realism", access: "fal.ai / local" },
      { name: "Stable Diffusion 3.5", maker: "Stability AI (open)", best: "Open, customizable, LoRA/ControlNet", access: "Local / ComfyUI" },
      { name: "Ideogram 2.0", maker: "Ideogram", best: "Best at text in images", access: "ideogram.ai" },
      { name: "Sora / Kling / Runway", maker: "OpenAI / Kuaishou / Runway", best: "Text-to-video generation", access: "Web apps" },
    ],
    resources: [
      { label: "HuggingFace – Diffusion Models Course (free)", url: "https://huggingface.co/learn/diffusion-course/unit0/1", free: true, time: "self-paced" },
      { label: "DeepLearning.AI – How Diffusion Models Work", url: "https://www.deeplearning.ai/short-courses/how-diffusion-models-work/", free: true, time: "1.5 hrs" },
      { label: "Computerphile – How Stable Diffusion Works", url: "https://www.youtube.com/watch?v=1CIpzeNxIhU", free: true, time: "20 min" },
    ]
  },
  audio: {
    icon: Music, title: "Audio & Music Generation", color: "from-orange-600 to-red-700", border: "border-orange-700",
    tagline: "The most rapidly evolving GenAI domain — voice cloning, music generation, and real-time speech are all here now.",
    howItWorks: [
      { step: "1. Audio as Spectrogram", desc: "Audio is represented as a spectrogram — a 2D image of frequency vs time. Many audio models are essentially image diffusion models operating on spectrograms." },
      { step: "2. Codec Models (EnCodec)", desc: "Modern TTS and music models compress audio into discrete tokens (like text tokens) using neural codecs. This allows transformer architectures to be applied directly to audio." },
      { step: "3. Text-to-Speech (TTS)", desc: "Text → phonemes → acoustic features → waveform. Modern TTS (ElevenLabs, Bark) adds emotional context, speaker identity, and prosody control via transformer models." },
      { step: "4. Voice Cloning", desc: "A reference audio clip (3–30 seconds) is encoded into a speaker embedding. This embedding conditions the TTS model to match that voice's characteristics." },
      { step: "5. Music Generation", desc: "Models like MusicGen and Suno treat music as long audio sequences. Trained on vast music corpora, they learn the structure of rhythm, melody, harmony, and genre." },
      { step: "6. Real-Time Voice AI", desc: "The frontier: full-duplex voice conversation with <300ms latency. GPT-4o Voice and Hume AI use end-to-end audio models — no intermediate text step, preserving tone and emotion." },
    ],
    keyConceptsTitle: "Key Capabilities",
    keyConcepts: [
      { name: "Text-to-Speech (TTS)", desc: "Natural-sounding voice from text. Now indistinguishable from human in many cases. Used for content, accessibility, automation." },
      { name: "Voice Cloning", desc: "Clone any voice from a short sample. Ethical use: your own voice for content creation. Major deepfake risk." },
      { name: "Speech-to-Text (STT)", desc: "Transcription. Whisper (OpenAI) is state of the art, runs locally, handles accents well." },
      { name: "Music Generation", desc: "Generate full songs from text descriptions. Suno can generate 2-min songs with vocals, instruments, lyrics from a simple prompt." },
      { name: "Audio Enhancement", desc: "Remove background noise, enhance clarity, separate vocals from instruments. Adobe Enhance Speech, Lalal.ai." },
      { name: "Voice Agents", desc: "Real-time conversational AI with voice I/O. Vapi, Bland AI, Retell AI enable phone/voice automation." },
    ],
    models: [
      { name: "ElevenLabs", maker: "ElevenLabs", best: "Best TTS quality + voice cloning", access: "API / Web" },
      { name: "Whisper v3", maker: "OpenAI (open)", best: "Best open-source STT, runs locally", access: "Local / API" },
      { name: "Suno v4", maker: "Suno", best: "Best full song generation with lyrics", access: "suno.com" },
      { name: "Udio", maker: "Udio", best: "High quality music, genre control", access: "udio.com" },
      { name: "MusicGen", maker: "Meta (open)", best: "Open source music generation", access: "HuggingFace / local" },
      { name: "Bark", maker: "Suno (open)", best: "Open TTS with emotions, laughter, sounds", access: "HuggingFace / local" },
    ],
    resources: [
      { label: "Whisper by OpenAI – GitHub", url: "https://github.com/openai/whisper", free: true, time: "setup" },
      { label: "ElevenLabs Docs – Voice cloning guide", url: "https://elevenlabs.io/docs", free: true, time: "reference" },
      { label: "AudioCraft by Meta (MusicGen)", url: "https://github.com/facebookresearch/audiocraft", free: true, time: "hands-on" },
    ]
  },
};

const allTools = [
  {
    category: "Text & Chat", icon: MessageSquare,
    tools: [
      { name: "Claude.ai", maker: "Anthropic", free: "Free tier", best: "Long docs, analysis, coding, nuanced writing", url: "https://claude.ai" },
      { name: "ChatGPT", maker: "OpenAI", free: "Free tier", best: "General use, image gen (DALL-E), browsing", url: "https://chat.openai.com" },
      { name: "Gemini Advanced", maker: "Google", free: "Free tier", best: "Google Workspace integration, 1M context", url: "https://gemini.google.com" },
      { name: "Perplexity", maker: "Perplexity AI", free: "Free tier", best: "Research with citations, web search", url: "https://perplexity.ai" },
    ]
  },
  {
    category: "Code", icon: Code2,
    tools: [
      { name: "Cursor", maker: "Cursor", free: "Free tier", best: "Best AI code editor, codebase-aware", url: "https://cursor.sh" },
      { name: "GitHub Copilot", maker: "GitHub", free: "$10/mo", best: "Inline autocomplete, VS Code native", url: "https://github.com/features/copilot" },
      { name: "Claude Code", maker: "Anthropic", free: "Usage-based", best: "Agentic coding CLI, file operations", url: "https://claude.ai/code" },
      { name: "Replit AI", maker: "Replit", free: "Free tier", best: "Browser-based, deploy instantly", url: "https://replit.com" },
    ]
  },
  {
    category: "Image Generation", icon: Palette,
    tools: [
      { name: "Midjourney", maker: "Midjourney", free: "Paid ($10/mo)", best: "Best aesthetic quality, photorealism", url: "https://midjourney.com" },
      { name: "DALL-E 3", maker: "OpenAI", free: "In ChatGPT free", best: "Best prompt adherence, safe", url: "https://openai.com/dall-e-3" },
      { name: "Adobe Firefly", maker: "Adobe", free: "Free credits", best: "Commercial safe, Photoshop integrated", url: "https://firefly.adobe.com" },
      { name: "Flux (fal.ai)", maker: "Black Forest Labs", free: "Pay-per-use", best: "Best open model, API accessible", url: "https://fal.ai" },
      { name: "ComfyUI", maker: "Open source", free: "Free (local)", best: "Maximum control, local, LoRA/ControlNet", url: "https://github.com/comfyanonymous/ComfyUI" },
      { name: "Ideogram", maker: "Ideogram", free: "Free tier", best: "Text in images, posters, logos", url: "https://ideogram.ai" },
    ]
  },
  {
    category: "Video Generation", icon: Video,
    tools: [
      { name: "Runway Gen-3", maker: "Runway", free: "Free credits", best: "Best quality, professional use", url: "https://runwayml.com" },
      { name: "Kling AI", maker: "Kuaishou", free: "Free tier", best: "Realistic motion, free generous tier", url: "https://klingai.com" },
      { name: "Sora", maker: "OpenAI", free: "In ChatGPT Plus", best: "Longest clips, good physics", url: "https://sora.com" },
      { name: "Pika Labs", maker: "Pika", free: "Free tier", best: "Animate images, easy to use", url: "https://pika.art" },
    ]
  },
  {
    category: "Audio & Voice", icon: Music,
    tools: [
      { name: "ElevenLabs", maker: "ElevenLabs", free: "Free tier", best: "Best TTS + voice cloning", url: "https://elevenlabs.io" },
      { name: "Suno", maker: "Suno", free: "Free tier", best: "Full song generation with lyrics", url: "https://suno.com" },
      { name: "Whisper", maker: "OpenAI", free: "Free (local)", best: "Best open-source transcription", url: "https://github.com/openai/whisper" },
      { name: "Udio", maker: "Udio", free: "Free tier", best: "High quality music, genre control", url: "https://udio.com" },
    ]
  },
  {
    category: "Productivity & Workflow", icon: Zap,
    tools: [
      { name: "NotebookLM", maker: "Google", free: "Free", best: "Chat with documents, podcast generation", url: "https://notebooklm.google.com" },
      { name: "Notion AI", maker: "Notion", free: "Add-on", best: "AI inside your notes/docs/wiki", url: "https://notion.so/product/ai" },
      { name: "v0 by Vercel", maker: "Vercel", free: "Free tier", best: "Generate UI components from descriptions", url: "https://v0.dev" },
      { name: "Gamma", maker: "Gamma", free: "Free tier", best: "AI-generated presentations/slides", url: "https://gamma.app" },
    ]
  },
];

const genaiRoadmapPhases = [
  {
    phase: "Phase 1", title: "Foundations Across All Domains", duration: "3–4 weeks",
    color: "from-blue-600 to-indigo-700",
    goal: "Build mental models for how each GenAI domain works. Not hands-on yet — concepts first.",
    tasks: [
      { domain: "Text", icon: MessageSquare, task: "Watch Karpathy's Intro to LLMs (1hr). Understand: tokens, attention, next-token prediction, temperature." },
      { domain: "Image", icon: Palette, task: "Watch Computerphile's Stable Diffusion video (20min). Understand: latent space, diffusion process, text conditioning." },
      { domain: "Audio", icon: Music, task: "Read ElevenLabs blog on how TTS works. Try Whisper for transcription. Understand: spectrograms, codec models." },
      { domain: "Code", icon: Code2, task: "Install Cursor or Copilot. Use it for a real coding task. Notice what it does well and where it fails." },
    ]
  },
  {
    phase: "Phase 2", title: "Hands-On With Every Domain", duration: "4–5 weeks",
    color: "from-purple-600 to-violet-700",
    goal: "Get practical experience with the best tools in each domain. Build something small in each.",
    tasks: [
      { domain: "Text", icon: MessageSquare, task: "Build a prompt library for your top 5 use cases. Experiment with system prompts. Compare Claude vs GPT-4 on the same tasks." },
      { domain: "Image", icon: Palette, task: "Use Midjourney or DALL-E 3 for 2 weeks daily. Learn CFG scale, negative prompts, style references. Try Flux locally via ComfyUI." },
      { domain: "Audio", icon: Music, task: "Clone your own voice with ElevenLabs. Generate a full song with Suno. Run Whisper locally on a long audio file." },
      { domain: "Code", icon: Code2, task: "Build a small full-stack feature using only AI assistance. Use Claude Code for a refactoring task. Measure time saved vs unassisted." },
    ]
  },
  {
    phase: "Phase 3", title: "Build Across Modalities", duration: "4–6 weeks",
    color: "from-orange-600 to-amber-700",
    goal: "Combine domains. The real power of GenAI comes from chaining modalities together.",
    tasks: [
      { domain: "Text + Code", icon: Code2, task: "Build an AI app using LLM APIs. Add structured output parsing. Deploy it." },
      { domain: "Text + Image", icon: Palette, task: "Build an automated image generation pipeline: text prompt → LLM refines prompt → image model generates → auto-saved." },
      { domain: "Text + Audio", icon: Music, task: "Build a document-to-podcast pipeline: PDF → LLM summarizes → ElevenLabs narrates → audio file output." },
      { domain: "Full Pipeline", icon: "🔗", task: "Pick one ambitious project that combines 3+ modalities. E.g.: video script generator → voiceover → auto image/video selection." },
    ]
  },
  {
    phase: "Phase 4", title: "Go Deep on One Domain", duration: "6–8 weeks",
    color: "from-teal-600 to-cyan-700",
    goal: "Generalist foundation → specialize in the domain most relevant to your work and goals.",
    tasks: [
      { domain: "If Text/LLMs", icon: MessageSquare, task: "Study RAG, fine-tuning (LoRA), and agent systems. Build a production-grade LLM app with evals." },
      { domain: "If Image/Video", icon: Palette, task: "Learn ComfyUI workflows, LoRA training, ControlNet. Take HuggingFace Diffusion Course." },
      { domain: "If Audio", icon: Music, task: "Build a voice agent with Vapi or ElevenLabs Conversational AI. Study real-time speech models." },
      { domain: "If Code", icon: Code2, task: "Build an agentic coding pipeline. Study how Claude Code/Devin work. Contribute to an open source AI coding tool." },
    ]
  },
];

const overview = {
  domains: [
    { icon: MessageSquare, name: "Text & LLMs", status: "Mature", desc: "Transformers → next token prediction. GPT, Claude, Gemini. Powers everything.", color: "blue" },
    { icon: Code2, name: "Code Generation", status: "Mature", desc: "Specialized LLMs trained on code. Copilot, Cursor, Claude Code. Already transforming dev.", color: "green" },
    { icon: Palette, name: "Image & Video", status: "Rapidly evolving", desc: "Diffusion models. DALL-E, Midjourney, Sora. Going from images → films.", color: "purple" },
    { icon: Music, name: "Audio & Music", status: "Fast growing", desc: "Codec + transformer models. ElevenLabs, Suno, Whisper. Voice is nearly perfect now.", color: "orange" },
  ],
  convergence: "All four domains are converging. GPT-4o processes text, image, and audio in one model. Gemini 1.5 is natively multimodal. The future is a single model that sees, hears, speaks, reads, and writes — and the components you learn today are the building blocks of that future.",
};

const TabBtn = ({ id, label, active, onClick }) => (
  <button onClick={() => onClick(id)}
    className={`text-xs px-3 py-2 rounded-lg whitespace-nowrap transition-colors ${active ? "bg-gray-700 text-white font-semibold" : "text-gray-400 hover:text-gray-300"}`}>
    {label}
  </button>
);

const Resource = ({ r }) => (
  <a href={r.url} target="_blank" rel="noopener noreferrer"
    className="flex items-center justify-between bg-gray-800 hover:bg-gray-700 rounded-lg px-3 py-2 transition-colors">
    <span className="text-blue-400 text-xs">{r.label}</span>
    <div className="flex items-center gap-2 flex-shrink-0 ml-2">
      <span className="text-xs text-gray-400">{r.time}</span>
      <span className={`text-xs px-1.5 py-0.5 rounded ${r.free ? "bg-green-900 text-green-400" : "bg-gray-700 text-gray-400"}`}>{r.free ? "Free" : "Paid"}</span>
    </div>
  </a>
);

const DomainDetail = ({ d }) => {
  const [sec, setSec] = useState("how");
  return (
    <div className="space-y-4">
      <div className={`bg-gradient-to-r ${d.color} rounded-xl p-4`}>
        <p className="text-lg font-bold">{d.title}</p>
        <p className="text-white text-opacity-80 text-sm mt-1">{d.tagline}</p>
      </div>
      <div className="flex gap-1 flex-wrap">
        {[["how", "How It Works"], ["concepts", d.keyConceptsTitle], ["models", "Models"], ["resources", "Resources"]].map(([k, v]) => (
          <button key={k} onClick={() => setSec(k)}
            className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${sec === k ? "bg-gray-700 text-white" : "text-gray-400 hover:text-gray-300 bg-gray-900"}`}>{v}</button>
        ))}
      </div>
      {sec === "how" && (
        <div className="space-y-2">
          {d.howItWorks.map((s, i) => (
            <div key={i} className="bg-gray-900 border border-gray-800 rounded-lg p-3 flex gap-3">
              <div className="w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center text-xs font-bold flex-shrink-0">{i + 1}</div>
              <div><p className="text-sm font-semibold text-white mb-0.5">{s.step}</p><p className="text-xs text-gray-400">{s.desc}</p></div>
            </div>
          ))}
        </div>
      )}
      {sec === "concepts" && (
        <div className="grid grid-cols-1 gap-2">
          {d.keyConcepts.map((c, i) => (
            <div key={i} className="bg-gray-900 border border-gray-800 rounded-lg p-3">
              <p className="text-sm font-semibold text-white mb-0.5">{c.name}</p>
              <p className="text-xs text-gray-400">{c.desc}</p>
            </div>
          ))}
        </div>
      )}
      {sec === "models" && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left pb-2 pr-3">Model</th><th className="text-left pb-2 pr-3">Maker</th><th className="text-left pb-2 pr-3">Best For</th><th className="text-left pb-2">Access</th>
            </tr></thead>
            <tbody>{d.models.map((m, i) => (
              <tr key={i} className="border-b border-gray-800">
                <td className="py-2 pr-3 font-semibold text-blue-400">{m.name}</td>
                <td className="py-2 pr-3 text-gray-400">{m.maker}</td>
                <td className="py-2 pr-3 text-gray-300">{m.best}</td>
                <td className="py-2 text-gray-400">{m.access}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {sec === "resources" && (
        <div className="space-y-2">{d.resources.map((r, i) => <Resource key={i} r={r} />)}</div>
      )}
      {d.tools && sec === "tools" && (
        <div className="space-y-2">{d.tools.map((t, i) => (
          <a key={i} href={t.url} target="_blank" rel="noopener noreferrer" className="flex items-center justify-between bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-lg p-3">
            <div><p className="text-blue-400 text-sm font-medium">{t.name}</p><p className="text-gray-400 text-xs">{t.desc}</p></div>
          </a>
        ))}</div>
      )}
    </div>
  );
};

function GenAIGuide() {
  useSeo(
    "Generative AI Guide – Text, Code, Image & Audio | AI Learning Hub",
    "Complete overview of Generative AI domains — text, code, image, audio. How each works, top models, tools, and a learning roadmap for developers."
  );
  const [tab, setTab] = useState("overview");

  return (
    <div className="min-h-screen text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-5">
          <h1 className="text-2xl font-bold mb-1"><Cpu size={20} className="inline mr-2 align-middle text-purple-400"/>Generative AI — Complete Guide</h1>
          <p className="text-gray-400 text-sm max-w-lg mx-auto">A deep dive into how each GenAI domain works — text, code, image, audio, and tools. Use this alongside the roadmap or as a standalone reference.</p>
          <p className="text-xs text-gray-400 mt-2">Start with the Overview tab, then explore the domain most relevant to what you're building.</p>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 overflow-x-auto bg-gray-900 rounded-xl p-1 mb-6 border border-gray-800">
          {[["overview","Overview"],["text","Text"],["code","Code"],["image","Image/Video"],["audio","Audio"],["tools","All Tools"],["roadmap","Roadmap"]].map(([k,v]) => (
            <TabBtn key={k} id={k} label={v} active={tab===k} onClick={setTab} />
          ))}
        </div>

        {/* OVERVIEW */}
        {tab === "overview" && (
          <div className="space-y-4">
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">What is Generative AI?</p>
              <p className="text-gray-300 text-sm">Generative AI refers to models that <span className="text-white font-semibold">create new content</span> — text, images, audio, video, code — rather than just classifying or predicting from existing data. The key breakthrough: instead of hard-coding rules, these models learn the underlying distribution of human-created content and sample from it.</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {overview.domains.map((d, i) => (
                <button key={i} onClick={() => setTab(["overview","text","code","image","audio"][i+1])}
                  className="bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-xl p-3 text-left transition-colors">
                  
                  <p className="font-semibold text-sm text-white">{d.name}</p>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${d.status === "Mature" ? "bg-green-900 text-green-400" : "bg-yellow-900 text-yellow-400"}`}>{d.status}</span>
                  <p className="text-xs text-gray-400 mt-1">{d.desc}</p>
                </button>
              ))}
            </div>
            <div className="bg-gray-900 border border-blue-900 rounded-xl p-4">
              <p className="text-blue-400 font-semibold text-sm mb-2">🔮 Where It's All Going</p>
              <p className="text-gray-300 text-sm">{overview.convergence}</p>
            </div>
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">The Common Thread Across All Domains</p>
              <div className="space-y-2">
                {[
                  ["Training", "Show the model millions of examples of human-created content. It learns the patterns."],
                  ["Latent Space", "All domains compress content into a mathematical space of meaning. Generation = sampling from this space."],
                  ["Conditioning", "Guide generation toward what you want — via text prompt, reference image, style embedding, etc."],
                  ["Sampling", "Generation is probabilistic — outputs are sampled from a learned distribution, not looked up. This is why it's creative."],
                ].map(([k, v], i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-blue-400 font-semibold text-xs w-20 flex-shrink-0">{k}</span>
                    <p className="text-xs text-gray-400">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab === "text" && <DomainDetail d={domainData.text} />}
        {tab === "code" && <DomainDetail d={domainData.code} />}
        {tab === "image" && <DomainDetail d={domainData.image} />}
        {tab === "audio" && <DomainDetail d={domainData.audio} />}

        {/* ALL TOOLS */}
        {tab === "tools" && (
          <div className="space-y-5">
            <p className="text-gray-400 text-xs">Best tools by category — covering free tiers and paid options across all GenAI domains.</p>
            {allTools.map((cat, i) => (
              <div key={i}>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">{React.createElement(cat.icon, {size:14, className:"inline mr-1 align-middle"})} {cat.category}</p>
                <div className="space-y-2">
                  {cat.tools.map((t, j) => (
                    <a key={j} href={t.url} target="_blank" rel="noopener noreferrer"
                      className="flex items-start gap-3 bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-xl p-3 transition-colors">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 flex-wrap mb-0.5">
                          <p className="text-blue-400 font-semibold text-sm">{t.name}</p>
                          <span className="text-xs text-gray-400">by {t.maker}</span>
                          <span className={`text-xs px-2 py-0.5 rounded-full ${t.free === "Free" ? "bg-green-900 text-green-400" : "bg-gray-800 text-gray-400"}`}>{t.free}</span>
                        </div>
                        <p className="text-xs text-gray-400">{t.best}</p>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ROADMAP */}
        {tab === "roadmap" && (
          <div className="space-y-4">
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-2">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1"><Zap size={12} className="inline mr-1.5 align-middle"/>Timeline</p>
              <p className="text-gray-300 text-sm">At 4–6 hrs/week: <span className="text-white font-semibold">4–5 months</span> to strong generalist across all 4 domains. This roadmap is designed to run <span className="text-white">in parallel with</span> the main AI learning roadmap — not after it.</p>
            </div>
            {genaiRoadmapPhases.map((p, i) => (
              <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
                <div className={`bg-gradient-to-r ${p.color} p-4`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-bold text-white">{p.phase}: {p.title}</p>
                      <p className="text-white text-opacity-70 text-xs mt-0.5">{p.duration}</p>
                    </div>
                  </div>
                  <p className="text-white text-opacity-80 text-xs mt-2">{p.goal}</p>
                </div>
                <div className="p-4 space-y-2">
                  {p.tasks.map((t, j) => (
                    <div key={j} className="flex gap-3 bg-gray-800 rounded-lg p-3">
                      <span className="text-base flex-shrink-0">{t.icon}</span>
                      <div>
                        <span className="text-xs font-semibold text-gray-400">{t.domain}: </span>
                        <span className="text-xs text-gray-300">{t.task}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            <div className="bg-gray-900 border border-yellow-800 rounded-xl p-4">
              <p className="text-yellow-400 font-semibold text-sm mb-2">💡 The 80/20 of GenAI Mastery</p>
              <div className="space-y-1.5">
                {[
                  "Text/LLMs is the foundation — if you understand it deeply, the other domains make more sense",
                  "Code generation gives you the highest immediate ROI as a developer — start there for daily use",
                  "Image/audio are best learned by doing — pick a creative project and let curiosity drive it",
                  "The real skill isn't knowing individual tools — it's knowing which tool to chain with which for a given outcome",
                  "Tools change every 3–6 months. Learn the underlying concepts; the tools will follow naturally",
                ].map((s, i) => (
                  <div key={i} className="flex gap-2"><span className="text-yellow-500 flex-shrink-0"><ArrowRight size={13} className="flex-shrink-0"/></span><p className="text-xs text-gray-300">{s}</p></div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


    // ─── READINESS CHECKER COMPONENT ───
const readinessPhases = [
  {
    id: 1, icon: Sprout, title: "Phase 1 – AI Foundations",
    color: "from-green-600 to-emerald-700", border: "border-green-700",
    duration: "4–6 weeks", nextPhase: "Phase 2 – LLM Setup",
    readinessThreshold: 75,
    greenFlags: [
      "You can explain what a token is and why it matters",
      "You can explain how attention works conceptually (not mathematically)",
      "You understand the difference between AI, ML, Deep Learning, and GenAI",
      "You know what a transformer is and why it replaced RNNs",
      "You can explain what 'next token prediction' means to a non-technical person",
      "You understand what embeddings are and why similar words cluster together",
      "You've watched at least Karpathy's Intro to LLMs all the way through",
    ],
    redFlags: [
      "You can't explain transformers without looking it up",
      "You're still unclear on the difference between training and inference",
      "Words like 'embedding', 'token', 'parameter' still feel fuzzy",
    ],
    stuckSignal: "If you've been in Phase 1 for more than 6 weeks, stop consuming and start doing. Confusion resolves faster through building than through re-reading.",
    skipRule: "If you already know all 7 green flags — skip Phase 1 entirely.",
  },
  {
    id: 2, icon: Settings, title: "Phase 2 – LLM Setup & Config",
    color: "from-slate-500 to-gray-600", border: "border-slate-600",
    duration: "2–3 weeks", nextPhase: "Phase 3 – Prompt Engineering",
    readinessThreshold: 80,
    greenFlags: [
      "You have Ollama installed and have run at least one local model",
      "You've called at least one LLM API (OpenAI or Anthropic) from Python",
      "You understand what temperature, top-p, and max_tokens do",
      "You can explain the difference between a 7B and 70B model in plain terms",
      "You understand what quantization means and why it matters for local use",
      "You know how to read a Hugging Face model card",
    ],
    redFlags: [
      "You've only used chat UIs (Claude.ai, ChatGPT) — never called an API",
      "You haven't run a model locally yet",
      "API keys, endpoints, and rate limits still feel foreign",
    ],
    stuckSignal: "Setup phases are often where people procrastinate. If you're stuck here more than 2 weeks, just use cloud APIs and skip local setup for now — come back to it in Phase 5.",
    skipRule: "Already calling APIs from code and running local models? Skip to Phase 3.",
  },
  {
    id: 3, icon: Wrench, title: "Phase 3 – Prompt Engineering",
    color: "from-blue-600 to-indigo-700", border: "border-blue-700",
    duration: "3–4 weeks", nextPhase: "Phase 4 – RAG",
    readinessThreshold: 80,
    greenFlags: [
      "You have a personal prompt template library with at least 10 reusable prompts",
      "You can write a system prompt that produces consistent output across 10 runs",
      "You know when to use zero-shot vs few-shot vs chain-of-thought",
      "You've used negative prompting to eliminate a specific failure mode",
      "You can diagnose why a prompt failed and fix it in one iteration",
      "You've built at least one small tool using an LLM API with structured JSON output",
      "You've compared Claude vs GPT-4 behavior on the same prompt",
    ],
    redFlags: [
      "Your prompts are still just questions — no role, format, or constraints",
      "You haven't built anything using an API yet (only used chat UIs)",
      "You're still surprised when the model ignores part of your prompt",
    ],
    stuckSignal: "Prompt engineering only truly sticks through daily use. If you're stuck, pick one real task you do every day and obsessively improve the prompt for it over 2 weeks.",
    skipRule: "Already an expert prompter with a template library? Skim Phase 3 and move on.",
  },
  {
    id: 4, icon: BookOpen, title: "Phase 4 – RAG Systems",
    color: "from-purple-600 to-violet-700", border: "border-purple-700",
    duration: "4–5 weeks", nextPhase: "Phase 5 – Agentic AI",
    readinessThreshold: 70,
    greenFlags: [
      "You can explain what a vector embedding is and why it enables semantic search",
      "You've built a working RAG pipeline end-to-end (even a simple one)",
      "You understand the difference between chunking strategies and why they matter",
      "You know what a vector database does and have used at least one (Chroma, Pinecone, etc.)",
      "You can evaluate a RAG system — know what context relevance and faithfulness mean",
      "You understand when RAG is better than fine-tuning and vice versa",
    ],
    redFlags: [
      "You haven't built a RAG pipeline — only read about it",
      "You can't explain why naive RAG fails on long documents",
      "Vector databases still feel like magic boxes",
    ],
    stuckSignal: "RAG is one of the hardest phases to get unstuck from because it requires real data. If stuck, build a RAG over a Wikipedia article — not your own data — to remove friction.",
    skipRule: "No skip recommended. RAG is foundational for Phases 5 and 6.",
  },
  {
    id: 5, icon: Bot, title: "Phase 5 – Agentic AI",
    color: "from-orange-600 to-amber-700", border: "border-orange-700",
    duration: "4–5 weeks", nextPhase: "Phase 6 – Fine-Tuning",
    readinessThreshold: 70,
    greenFlags: [
      "You've built a working agent that uses at least 2 tools (e.g. search + summarize)",
      "You understand the ReACT loop: Reason → Act → Observe → repeat",
      "You know the difference between a workflow and an agent",
      "You've debugged an agent that got stuck in a loop or took the wrong action",
      "You understand at least 3 agentic design patterns (routing, reflection, orchestrator-worker)",
      "You've used LangGraph or a similar framework for multi-step agent logic",
    ],
    redFlags: [
      "Your 'agent' is just prompt chaining without real tool use",
      "You haven't handled agent failure modes (infinite loops, wrong tool calls)",
      "MCP, A2A, and agentic frameworks still feel theoretical",
    ],
    stuckSignal: "Agents are complex. If stuck, reduce scope: build the simplest possible agent (1 tool, 1 goal) and get it working reliably before adding complexity.",
    skipRule: "No skip recommended. Agent patterns appear in nearly every advanced AI system.",
  },
  {
    id: 6, icon: Building2, title: "Phase 6 – Fine-Tuning & Training",
    color: "from-rose-600 to-pink-700", border: "border-rose-700",
    duration: "6–8 weeks", nextPhase: "Phase 7 – Ship & Specialize",
    readinessThreshold: 65,
    greenFlags: [
      "You understand when fine-tuning is worth it vs just prompting or RAG",
      "You've fine-tuned at least one model using LoRA/QLoRA (even on a toy dataset)",
      "You understand what SFT is and how it differs from RLHF",
      "You can read and roughly understand a model training paper (not full math — concepts)",
      "You've used Unsloth or PEFT and understand what they're doing",
      "You know what overfitting looks like in a fine-tuning run",
    ],
    redFlags: [
      "You haven't run a single fine-tuning job yet",
      "You don't know the difference between LoRA rank and alpha",
      "Training loss and validation loss curves look meaningless to you",
    ],
    stuckSignal: "Fine-tuning has the steepest setup curve. If stuck on infrastructure, use Unsloth's pre-built Colab notebooks — they remove 90% of the friction.",
    skipRule: "If your goals are purely app-building and not model internals, you can treat Phase 6 as optional and move to Phase 7 with awareness of what you're skipping.",
  },
  {
    id: 7, icon: Rocket, title: "Phase 7 – Build & Specialize",
    color: "from-teal-600 to-cyan-700", border: "border-teal-700",
    duration: "Ongoing", nextPhase: "Continuous mastery",
    readinessThreshold: 60,
    greenFlags: [
      "You've shipped at least one AI project publicly (GitHub, blog, deployed app)",
      "You have a specialization in mind — you know which domain excites you most",
      "You follow at least 3 people or newsletters that keep you current",
      "You can discuss AI trade-offs (RAG vs fine-tuning, agents vs workflows) with confidence",
      "You've read at least 3 AI research papers end to end (even if not every equation)",
      "You contribute back — blog post, open source PR, community answer",
    ],
    redFlags: [
      "You've finished Phase 6 but haven't shipped anything yet",
      "You're still waiting to feel 'ready' before building publicly",
      "You're consuming more than you're creating",
    ],
    stuckSignal: "Phase 7 is where most people stall — not from lack of knowledge but from lack of output. The antidote: commit to shipping something imperfect in the next 2 weeks.",
    skipRule: "There is no skipping Phase 7. It is the destination.",
  },
];

const generalRules = [
  {
    icon: Zap, title: "Time is a signal, not a deadline",
    desc: "The durations are estimates at 4–6 hrs/week. If you're spending more time, you'll move faster. If less, slower. Don't use time to judge readiness — use the green flags."
  },
  {
    icon: "🚦", title: "The 70% Rule",
    desc: "You don't need 100% of the green flags to move on. If you have 70%+ and you've built the phase project — move. The remaining 30% will fill in naturally in the next phase."
  },
  {
    icon: RotateCcw, title: "Phases aren't waterfall",
    desc: "You'll revisit earlier phases constantly. Moving to Phase 4 (RAG) doesn't mean you stop prompting. Think of it as 'primary focus' shifts, not 'completed and locked'."
  },
  {
    icon: "🏗️", title: "The project is the gate, not the content",
    desc: "Reading all the resources for a phase doesn't mean you're ready to move on. Building the phase project is the real readiness signal. No project = not done."
  },
  {
    icon: Zap, title: "Boredom is a green flag",
    desc: "If the current phase feels too easy or you're no longer learning anything new, that's a signal to move on — even if the checklist isn't 100% complete."
  },
  {
    icon: "😤", title: "Struggle is not a red flag",
    desc: "Confusion is normal and expected. The question isn't 'am I confused?' — it's 'am I making progress despite the confusion?' Stuck for 2+ weeks with zero progress = red flag."
  },
];

const CheckItem = ({ text, checked, onToggle }) => (
  <div className="flex items-start gap-2 cursor-pointer group" onClick={onToggle}>
    <div className={`w-4 h-4 rounded border mt-0.5 flex-shrink-0 flex items-center justify-center transition-colors ${checked ? "bg-green-600 border-green-600" : "border-gray-600 group-hover:border-gray-400"}`}>
      {checked && <span className="text-white text-xs">✓</span>}
    </div>
    <p className={`text-xs transition-colors ${checked ? "text-gray-400 line-through" : "text-gray-300"}`}>{text}</p>
  </div>
);

function ReadinessChecker() {
  useSeo(
    "AI Phase Readiness Checker – Know When to Move On | AI Learning Hub",
    "Check if you're ready to advance to the next AI learning phase. Green flags, red flags, move-on rules, and a progress overview for all 7 phases."
  );
  const [checks, setChecks] = useState({});
  const [openPhase, setOpenPhase] = useState(null);
  const [tab, setTab] = useState("checker");

  const toggle = (phaseId, idx) => {
    const key = `${phaseId}-${idx}`;
    setChecks(p => ({ ...p, [key]: !p[key] }));
  };

  const getScore = (phaseId, total) => {
    let count = 0;
    for (let i = 0; i < total; i++) if (checks[`${phaseId}-${i}`]) count++;
    return Math.round((count / total) * 100);
  };

  const getVerdict = (score, threshold) => {
    if (score >= threshold) return { label: "Ready to move on ✓", color: "text-green-400", bg: "bg-green-900 border-green-700" };
    if (score >= threshold - 20) return { label: "Almost there — keep going", color: "text-yellow-400", bg: "bg-yellow-900 border-yellow-800" };
    return { label: "Stay in this phase", color: "text-red-400", bg: "bg-red-950 border-red-900" };
  };

  return (
    <div className="min-h-screen text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">

        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-1"><CheckCircle size={20} className="inline mr-2 align-middle text-green-400"/>Phase Readiness Checker</h1>
          <p className="text-gray-400 text-sm max-w-lg mx-auto">Not sure if you're ready to move to the next phase? Use this to check — it tells you exactly what to verify before moving on.</p>
          <p className="text-xs text-gray-400 mt-2">Pick your current phase from the list, check the green flags, and review the move-on rules.</p>
        </div>

        {/* Tab */}
        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1 mb-6">
          {[["checker", "✅ Phase Checklist"], ["rules", "📐 Move-On Rules"], ["overview", "🗺 At a Glance"]].map(([k, v]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`flex-1 text-xs py-2 rounded-lg transition-colors ${tab === k ? "bg-gray-700 text-white font-semibold" : "text-gray-400 hover:text-gray-300"}`}>{v}</button>
          ))}
        </div>

        {/* CHECKLIST TAB */}
        {tab === "checker" && (
          <div className="space-y-3">
            <p className="text-gray-400 text-xs mb-4">Check off each green flag as you achieve it. The score tells you when you're ready to move on.</p>
            {readinessPhases.map(p => {
              const score = getScore(p.id, p.greenFlags.length);
              const verdict = getVerdict(score, p.readinessThreshold);
              const isOpen = openPhase === p.id;
              return (
                <div key={p.id} className={`rounded-xl border overflow-hidden transition-all ${isOpen ? "border-gray-500" : "border-gray-800"} bg-gray-900`}>
                  {/* Header */}
                  <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setOpenPhase(isOpen ? null : p.id)}>
                    <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${p.color} flex items-center justify-center text-base flex-shrink-0`}><p.icon size={20} className="text-white"/></div>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm">{p.title}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="flex-1 bg-gray-800 rounded-full h-1.5 max-w-24">
                          <div className={`h-1.5 rounded-full transition-all ${score >= p.readinessThreshold ? "bg-green-500" : score >= p.readinessThreshold - 20 ? "bg-yellow-500" : "bg-gray-600"}`}
                            style={{ width: `${score}%` }} />
                        </div>
                        <span className={`text-xs font-bold ${verdict.color}`}>{score}%</span>
                        {score >= p.readinessThreshold && <span className="text-xs bg-green-900 text-green-400 border border-green-800 px-2 py-0.5 rounded-full">Ready ✓</span>}
                      </div>
                    </div>
                    <span className="text-gray-400">{isOpen ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
                  </div>

                  {isOpen && (
                    <div className="border-t border-gray-800 p-4 space-y-4">
                      {/* Verdict */}
                      <div className={`border rounded-lg p-3 ${verdict.bg}`}>
                        <p className={`text-sm font-semibold ${verdict.color}`}>{verdict.label}</p>
                        <p className="text-xs text-gray-400 mt-0.5">Threshold: {p.readinessThreshold}% · You: {score}% · Need {Math.max(0, p.readinessThreshold - score)}% more</p>
                      </div>

                      {/* Green flags */}
                      <div>
                        <p className="text-xs font-semibold text-green-400 uppercase tracking-wider mb-2">✅ Readiness Signals</p>
                        <div className="space-y-2">
                          {p.greenFlags.map((f, i) => (
                            <CheckItem key={i} text={f} checked={!!checks[`${p.id}-${i}`]} onToggle={() => toggle(p.id, i)} />
                          ))}
                        </div>
                      </div>

                      {/* Red flags */}
                      <div className="bg-red-950 border border-red-900 rounded-lg p-3">
                        <p className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-2">🚩 Don't Move On If</p>
                        <div className="space-y-1">
                          {p.redFlags.map((f, i) => (
                            <div key={i} className="flex gap-2"><span className="text-red-500 flex-shrink-0 text-xs mt-0.5">✗</span><p className="text-xs text-gray-300">{f}</p></div>
                          ))}
                        </div>
                      </div>

                      {/* Stuck signal */}
                      <div className="bg-yellow-950 border border-yellow-900 rounded-lg p-3">
                        <p className="text-xs font-semibold text-yellow-400 mb-1">🔔 If You're Stuck</p>
                        <p className="text-xs text-gray-300">{p.stuckSignal}</p>
                      </div>

                      {/* Skip rule */}
                      <div className="bg-blue-950 border border-blue-900 rounded-lg p-3">
                        <p className="text-xs font-semibold text-blue-400 mb-1"><Zap size={11} className="inline mr-1 align-middle"/>Skip Rule</p>
                        <p className="text-xs text-gray-300">{p.skipRule}</p>
                      </div>

                      <div className="flex items-center justify-between text-xs text-gray-400 pt-1">
                        <span><Zap size={11} className="inline mr-1 align-middle"/>{p.duration}</span>
                        <span>Next: {p.nextPhase} →</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* RULES TAB */}
        {tab === "rules" && (
          <div className="space-y-3">
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-2">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">The Core Principle</p>
              <p className="text-gray-300 text-sm">You're ready to move on when you've <span className="text-white font-semibold">built the phase project</span> and hit <span className="text-white font-semibold">~70% of the green flags</span>. Time spent is a weak signal. Output is the strong signal.</p>
            </div>
            {generalRules.map((r, i) => (
              <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl p-4 flex gap-3">
                <span className="flex-shrink-0">{React.createElement(r.icon, {size:22})}</span>
                <div>
                  <p className="font-semibold text-sm text-white mb-1">{r.title}</p>
                  <p className="text-xs text-gray-400">{r.desc}</p>
                </div>
              </div>
            ))}

            {/* Decision tree */}
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">🌳 The Move-On Decision Tree</p>
              <div className="space-y-2">
                {[
                  { q: "Have you built the phase project?", yes: "→ Continue", no: "→ Build it first. No exceptions.", yColor: "text-gray-400", nColor: "text-red-400" },
                  { q: "Is your score ≥ 70%?", yes: "→ Move on", no: "→ Work on remaining green flags", yColor: "text-green-400", nColor: "text-yellow-400" },
                  { q: "Have you been here > 2× the estimated time?", yes: "→ Move on regardless — stagnation is worse than gaps", no: "→ Stay and keep building", yColor: "text-orange-400", nColor: "text-gray-400" },
                  { q: "Does the phase feel boring or too easy?", yes: "→ Move on now", no: "→ You're in the right place", yColor: "text-blue-400", nColor: "text-gray-400" },
                ].map((item, i) => (
                  <div key={i} className="bg-gray-800 rounded-lg p-3">
                    <p className="text-xs font-semibold text-white mb-2">{i + 1}. {item.q}</p>
                    <div className="flex gap-4">
                      <span className={`text-xs ${item.yColor}`}>Yes {item.yes}</span>
                      <span className={`text-xs ${item.nColor}`}>No {item.no}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* OVERVIEW TAB */}
        {tab === "overview" && (
          <div className="space-y-3">
            <p className="text-gray-400 text-xs mb-2">Quick view of all phases, thresholds, and your current progress.</p>
            <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400">
                    <th className="text-left p-3">Phase</th>
                    <th className="text-left p-3">Duration</th>
                    <th className="text-left p-3">Threshold</th>
                    <th className="text-left p-3">Your Score</th>
                    <th className="text-left p-3">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {readinessPhases.map(p => {
                    const score = getScore(p.id, p.greenFlags.length);
                    const ready = score >= p.readinessThreshold;
                    return (
                      <tr key={p.id} className="border-b border-gray-800 hover:bg-gray-800 cursor-pointer transition-colors"
                        onClick={() => { setTab("checker"); setOpenPhase(p.id); }}>
                        <td className="p-3">
                          <div className="flex items-center gap-2">
                            <span><p.icon size={20} className="text-white"/></span>
                            <span className="text-gray-300 font-medium">Phase {p.id}</span>
                          </div>
                        </td>
                        <td className="p-3 text-gray-400">{p.duration}</td>
                        <td className="p-3 text-gray-400">{p.readinessThreshold}%</td>
                        <td className="p-3">
                          <div className="flex items-center gap-2">
                            <div className="w-12 bg-gray-700 rounded-full h-1.5">
                              <div className={`h-1.5 rounded-full ${ready ? "bg-green-500" : "bg-gray-500"}`} style={{ width: `${score}%` }} />
                            </div>
                            <span className={`font-bold ${ready ? "text-green-400" : "text-gray-400"}`}>{score}%</span>
                          </div>
                        </td>
                        <td className="p-3">
                          {score === 0
                            ? <span className="text-gray-400">Not started</span>
                            : ready
                              ? <span className="text-green-400 font-semibold">Ready ✓</span>
                              : <span className="text-yellow-400">In progress</span>
                          }
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* total timeline */}
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">📅 Realistic Full Timeline (4–6 hrs/week)</p>
              <div className="space-y-2">
                {[
                  ["Months 1–2", "Phases 1–3", "Foundations, setup, prompt mastery", "bg-blue-900"],
                  ["Months 3–4", "Phases 4–5", "RAG systems + Agentic AI", "bg-purple-900"],
                  ["Months 5–7", "Phase 6", "Fine-tuning + model internals", "bg-rose-900"],
                  ["Month 8+", "Phase 7", "Building, shipping, specializing", "bg-teal-900"],
                ].map(([period, phase, desc, bg], i) => (
                  <div key={i} className={`${bg} rounded-lg p-3 flex gap-3`}>
                    <div className="flex-shrink-0">
                      <p className="text-xs font-bold text-white">{period}</p>
                      <p className="text-xs text-gray-400">{phase}</p>
                    </div>
                    <p className="text-xs text-gray-300 self-center">{desc}</p>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-400 mt-3">💡 These are estimates. Moving faster is great. Moving slower is fine. What matters is building at every phase.</p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}


    // ─── WHATS LEFT COMPONENT ───
const covered = [
  { icon: MapIcon, label: "AI Zero→Hero Roadmap" },
  { icon: BookOpen, label: "Books & Video Courses by Phase" },
  { icon: Zap, label: "Prompt Engineering Mastery" },
  { icon: BrainCircuit, label: "Generative AI (all 4 domains)" },
  { icon: Target, label: "Phase Readiness Checker" },
  { icon: Settings, label: "LLM Setup & Configuration" },
  { icon: Bot, label: "Agentic AI Deep Dive" },
  { icon: Building2, label: "Building & Training LLMs" },
];

const topics = [
  {
    id: "mlops", icon: Wrench, title: "MLOps & AI in Production",
    tag: "Critical for builders", tagColor: "bg-red-900 text-red-300 border-red-800",
    color: "from-red-600 to-rose-700",
    why: "Knowing how to build an AI app is only half the story. Deploying, monitoring, and maintaining it in production is a completely different skill — and where most projects fail.",
    covers: [
      "Model serving & inference optimization (vLLM, TGI, Triton)",
      "Latency vs cost vs quality trade-offs in production",
      "LLM observability & tracing (LangSmith, Helicone, Arize)",
      "Evaluation pipelines & regression testing for prompts",
      "Rate limiting, caching, fallback strategies",
      "Cost management at scale (token budgets, batching)",
      "CI/CD for AI systems — how to ship prompt changes safely",
    ],
    depth: "deep",
    tools: ["LangSmith", "Helicone", "vLLM", "Arize AI", "PromptFoo", "Weights & Biases"],
    resource: { label: "Chip Huyen – Designing ML Systems (O'Reilly)", url: "https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/" },
  },
  {
    id: "evals", icon: BarChart, title: "AI Evaluation & Evals Engineering",
    tag: "Most underrated skill", tagColor: "bg-yellow-900 text-yellow-300 border-yellow-800",
    color: "from-yellow-600 to-amber-700",
    why: "How do you know if your AI app is actually good? Evals is the discipline of measuring AI quality systematically — it's the testing and QA of the AI world, and almost nobody does it well.",
    covers: [
      "Building eval datasets — golden sets, adversarial cases, edge cases",
      "LLM-as-a-Judge: using AI to evaluate AI output",
      "RAG-specific evals: RAGAS framework (context relevance, faithfulness, answer correctness)",
      "Agent evals: trajectory evaluation, success rate, tool use accuracy",
      "A/B testing prompts and models in production",
      "Human evaluation workflows and inter-rater reliability",
      "Benchmark leaderboards — what they measure and how to game them",
    ],
    depth: "deep",
    tools: ["RAGAS", "PromptFoo", "Evidently AI", "LangSmith Evals", "Braintrust"],
    resource: { label: "DeepLearning.AI – Evaluating & Debugging GenAI (free)", url: "https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/" },
  },
  {
    id: "security", icon: Lock, title: "AI Security & Red Teaming",
    tag: "Essential for apps", tagColor: "bg-orange-900 text-orange-300 border-orange-800",
    color: "from-orange-600 to-red-700",
    why: "AI apps have a completely new attack surface. Prompt injection, jailbreaking, data exfiltration, and model inversion are real threats that traditional security doesn't cover.",
    covers: [
      "Prompt injection attacks — direct and indirect",
      "Jailbreaking techniques and why they work",
      "Data poisoning and training-time attacks",
      "PII leakage through model memorization",
      "Red teaming LLM applications systematically",
      "Guardrails and content filtering (NeMo, Guardrails AI, Llama Guard)",
      "OWASP Top 10 for LLM Applications",
    ],
    depth: "medium",
    tools: ["Garak", "Guardrails AI", "Llama Guard", "NeMo Guardrails", "Rebuff"],
    resource: { label: "OWASP LLM Top 10 (free)", url: "https://owasp.org/www-project-top-10-for-large-language-model-applications/" },
  },
  {
    id: "multimodal", icon: Eye, title: "Multimodal AI (Vision + Language)",
    tag: "Fast growing", tagColor: "bg-purple-900 text-purple-300 border-purple-800",
    color: "from-purple-600 to-violet-700",
    why: "The frontier is models that see, hear, and read simultaneously. Vision-language models (VLMs) are already in production — GPT-4V, Claude 3.5, Gemini. Understanding them opens a new class of applications.",
    covers: [
      "How Vision-Language Models (VLMs) work — CLIP, visual encoders",
      "Image + text prompting techniques",
      "Document understanding (PDFs, charts, screenshots)",
      "Video understanding models",
      "Multimodal RAG — embedding images and text together",
      "Building apps with vision: receipt scanning, diagram analysis, UI testing",
      "On-device multimodal models (Apple MLX, llama.cpp multimodal)",
    ],
    depth: "medium",
    tools: ["GPT-4V", "Claude 3.5 Vision", "Gemini Vision", "LLaVA (open)", "Moondream (open, tiny)"],
    resource: { label: "DeepLearning.AI – Prompt Engineering with Llama 2 (free)", url: "https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/" },
  },
  {
    id: "reasoning", icon: Brain, title: "Reasoning Models & Thinking AI",
    tag: "Frontier topic", tagColor: "bg-blue-900 text-blue-300 border-blue-800",
    color: "from-blue-600 to-indigo-700",
    why: "OpenAI's o1/o3, DeepSeek R1, and Claude's extended thinking represent a new paradigm — AI that thinks before it answers. Understanding when and how to use reasoning models is a new skill.",
    covers: [
      "How chain-of-thought training works vs inference-time CoT",
      "Test-time compute scaling — why thinking longer improves accuracy",
      "When to use reasoning models vs standard models (cost vs quality)",
      "Prompting reasoning models differently — they respond poorly to CoT prompts",
      "DeepSeek R1 architecture — RL-based reasoning without supervised data",
      "Tree of Thoughts, Self-Consistency, and other reasoning techniques",
      "Reasoning for code, math, and multi-step planning tasks",
    ],
    depth: "medium",
    tools: ["OpenAI o3", "Claude Extended Thinking", "DeepSeek R1", "Gemini 2.0 Flash Thinking"],
    resource: { label: "DeepLearning.AI – Reasoning with o1 (free)", url: "https://www.deeplearning.ai/short-courses/reasoning-with-o1/" },
  },
  {
    id: "data", icon: Database, title: "Data Engineering for AI",
    tag: "Often overlooked", tagColor: "bg-teal-900 text-teal-300 border-teal-800",
    color: "from-teal-600 to-cyan-700",
    why: "AI is only as good as the data it's built on. Data collection, cleaning, labeling, and pipeline engineering determine model quality more than architecture choices — yet most AI courses skip this.",
    covers: [
      "Data collection strategies — web scraping, Common Crawl, synthetic data",
      "Data cleaning pipelines — deduplication, quality filtering, PII removal",
      "Annotation and labeling workflows — human + AI hybrid labeling",
      "Dataset versioning and lineage (DVC, Weights & Biases Artifacts)",
      "Synthetic data generation for fine-tuning",
      "Instruction dataset construction for SFT",
      "Data flywheels — how production apps improve model training over time",
    ],
    depth: "medium",
    tools: ["Argilla", "Label Studio", "DVC", "Great Expectations", "dbt for ML"],
    resource: { label: "Hugging Face – Datasets library docs (free)", url: "https://huggingface.co/docs/datasets/index" },
  },
  {
    id: "ondevice", icon: Smartphone, title: "On-Device & Edge AI",
    tag: "Growing fast", tagColor: "bg-green-900 text-green-300 border-green-800",
    color: "from-green-600 to-emerald-700",
    why: "Not all AI runs in the cloud. Running models on phones, laptops, and edge hardware enables privacy-first, offline, and ultra-low latency applications — a massive underserved market.",
    covers: [
      "Model quantization for edge (4-bit, 8-bit, GGUF format)",
      "Apple MLX — optimized ML framework for Apple Silicon",
      "llama.cpp — run LLMs on CPU efficiently",
      "Ollama for desktop apps",
      "WebLLM — run models in the browser (WebGPU)",
      "Small language models (SLMs): Phi-3, Gemma 2B, Llama 3.2 1B",
      "On-device use cases: privacy apps, offline assistants, IoT",
    ],
    depth: "medium",
    tools: ["Ollama", "llama.cpp", "Apple MLX", "WebLLM", "MLC LLM", "ONNX Runtime"],
    resource: { label: "DeepLearning.AI – Intro to On-Device AI (free)", url: "https://www.deeplearning.ai/short-courses/introduction-to-on-device-ai/" },
  },
  {
    id: "ai_safety", icon: Shield, title: "AI Safety & Alignment",
    tag: "Domain knowledge", tagColor: "bg-gray-700 text-gray-300 border-gray-600",
    color: "from-gray-600 to-slate-700",
    why: "Understanding AI safety isn't just for researchers — it shapes how you build responsible AI products, how you understand model behavior, and how you think about the long-term trajectory of the field.",
    covers: [
      "Alignment problem — why aligning AI to human values is hard",
      "Constitutional AI — how Anthropic trains Claude to be safe",
      "RLHF trade-offs — reward hacking, Goodhart's Law in AI",
      "Interpretability — mechanistic understanding of what's inside models",
      "Bias, fairness, and harm in AI systems",
      "AI governance and regulation landscape (EU AI Act, Executive Orders)",
      "Responsible AI development principles",
    ],
    depth: "awareness",
    tools: ["Anthropic Responsible Scaling Policy", "EU AI Act", "AI Safety course (BlueDot)"],
    resource: { label: "BlueDot – AI Safety Fundamentals (free)", url: "https://aisafetyfundamentals.com/" },
  },
  {
    id: "career", icon: Database, title: "AI Career Paths & Roles",
    tag: "For future planning", tagColor: "bg-indigo-900 text-indigo-300 border-indigo-800",
    color: "from-indigo-600 to-blue-700",
    why: "The AI job market is fragmented and confusing. Understanding the different roles — and what each actually requires — helps you aim your learning at the right target.",
    covers: [
      "AI Engineer vs ML Engineer vs Data Scientist vs Research Scientist",
      "What AI Engineers actually do day-to-day",
      "Portfolio projects that signal genuine competence",
      "AI freelancing & consulting opportunities",
      "Open source contributions as a career signal",
      "How to evaluate AI job descriptions (what's real vs buzzword)",
      "Salary ranges and market dynamics in 2025–2026",
    ],
    depth: "awareness",
    tools: [],
    resource: { label: "Latent Space – The AI Engineer Job (podcast)", url: "https://www.latent.space/p/ai-engineer" },
  },
  {
    id: "research", icon: FlaskConical, title: "Reading & Understanding AI Research",
    tag: "Level up fast", tagColor: "bg-pink-900 text-pink-300 border-pink-800",
    color: "from-pink-600 to-rose-700",
    why: "AI moves at paper speed. Practitioners who can read research papers are 6–12 months ahead of those who wait for blog post summaries. It's a learnable skill — not a PhD requirement.",
    covers: [
      "How to read a paper in 3 passes (abstract → intro/results → details)",
      "Key landmark papers everyone should read: Attention is All You Need, GPT-3, LoRA, InstructGPT",
      "How to find relevant papers: Arxiv, Papers With Code, Semantic Scholar",
      "Annotating and taking notes on papers effectively",
      "Understanding ablation studies and why they matter",
      "Following the research frontier: NeurIPS, ICML, ICLR, ACL",
      "Translating paper ideas into code",
    ],
    depth: "skill",
    tools: ["Papers With Code", "Arxiv Sanity", "Connected Papers", "Semantic Scholar", "Elicit"],
    resource: { label: "Yannic Kilcher – Paper walkthroughs (YouTube, free)", url: "https://www.youtube.com/@YannicKilcher" },
  },
];

const depthColors = {
  deep: "bg-red-900 text-red-300 border-red-800",
  medium: "bg-yellow-900 text-yellow-300 border-yellow-800",
  awareness: "bg-blue-900 text-blue-300 border-blue-800",
  skill: "bg-purple-900 text-purple-300 border-purple-800",
};
const depthLabels = {
  deep: "Deep dive needed",
  medium: "Medium depth",
  awareness: "Awareness level fine",
  skill: "Learnable skill",
};

function WhatsLeft() {
  const [open, setOpen] = useState(null);
  const [filter, setFilter] = useState("all");

  const filters = [
    { key: "all", label: "All Topics" },
    { key: "deep", label: "🔴 Deep Dives" },
    { key: "medium", label: "🟡 Medium" },
    { key: "awareness", label: "🔵 Awareness" },
    { key: "skill", label: "🟣 Skills" },
  ];

  const filtered = filter === "all" ? topics : topics.filter(t => t.depth === filter);

  return (
    <div className="min-h-screen text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">

        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-1">🗺️ What's Still Ahead</h1>
          <p className="text-gray-400 text-sm">Topics we haven't covered yet — and why each matters</p>
        </div>

        {/* What we covered */}
        <div className="bg-gray-900 border border-green-900 rounded-xl p-4 mb-6">
          <p className="text-xs font-semibold text-green-400 uppercase tracking-wider mb-3">✅ Already Covered in Our Conversations</p>
          <div className="grid grid-cols-2 gap-1.5">
            {covered.map((c, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-gray-400">
                <span className="text-green-400">✓</span>
                <span>{c.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Filter */}
        <div className="flex gap-1 flex-wrap mb-4">
          {filters.map(f => (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${filter === f.key ? "bg-gray-700 text-white" : "text-gray-400 hover:text-gray-300 bg-gray-900 border border-gray-800"}`}>
              {f.label}
            </button>
          ))}
        </div>

        {/* Topics */}
        <div className="space-y-3 mb-6">
          {filtered.map(t => (
            <div key={t.id}
              className={`rounded-xl border overflow-hidden transition-all cursor-pointer ${open === t.id ? "border-gray-500" : "border-gray-800 hover:border-gray-600"} bg-gray-900`}
              onClick={() => setOpen(open === t.id ? null : t.id)}>
              <div className="flex items-center gap-3 p-4">
                <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${t.color} flex items-center justify-center text-base flex-shrink-0`}>{React.createElement(t.icon, {size:20, className:"text-white"})}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <p className="font-semibold text-sm">{t.title}</p>
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${t.tagColor}`}>{t.tag}</span>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${depthColors[t.depth]}`}>{depthLabels[t.depth]}</span>
                </div>
                <span className="text-gray-400 flex-shrink-0">{open === t.id ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
              </div>

              {open === t.id && (
                <div className="border-t border-gray-800 p-4 space-y-4">
                  <div className="bg-gray-800 rounded-lg p-3">
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Why It Matters</p>
                    <p className="text-gray-300 text-sm">{t.why}</p>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">What It Covers</p>
                    <div className="space-y-1">
                      {t.covers.map((c, i) => (
                        <div key={i} className="flex gap-2">
                          <span className="text-gray-400 flex-shrink-0 mt-0.5">•</span>
                          <p className="text-xs text-gray-300">{c}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {t.tools.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Key Tools</p>
                      <div className="flex flex-wrap gap-1.5">
                        {t.tools.map((tool, i) => (
                          <span key={i} className="text-xs bg-gray-800 border border-gray-700 text-gray-300 px-2 py-1 rounded-lg">{tool}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  <a href={t.resource.url} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 bg-blue-950 border border-blue-900 hover:border-blue-700 rounded-lg p-3 transition-colors"
                    onClick={e => e.stopPropagation()}>
                    <span className="text-blue-400 text-xs">📖 Start here: {t.resource.label}</span>
                  </a>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Priority guide */}
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">🎯 What to Prioritize Next (For Your Goals)</p>
          <div className="space-y-2">
            {[
              { priority: "1st", topic: "AI Evaluation & Evals Engineering", reason: "You're building apps — this is what separates reliable AI from lucky AI.", color: "text-red-400" },
              { priority: "2nd", topic: "MLOps & AI in Production", reason: "The moment you want to share or deploy what you build, this becomes essential.", color: "text-orange-400" },
              { priority: "3rd", topic: "Multimodal AI (Vision)", reason: "Claude and GPT-4 already support images — you're leaving capability on the table.", color: "text-yellow-400" },
              { priority: "4th", topic: "Reading AI Research Papers", reason: "As a developer with your knowledge level, this is the highest ROI next skill.", color: "text-blue-400" },
              { priority: "5th", topic: "Reasoning Models", reason: "Already available in Claude and GPT — knowing when to use them saves cost and improves quality.", color: "text-purple-400" },
            ].map((r, i) => (
              <div key={i} className="flex gap-3 bg-gray-800 rounded-lg p-3">
                <span className={`text-xs font-bold ${r.color} w-6 flex-shrink-0`}>{r.priority}</span>
                <div>
                  <p className="text-sm font-semibold text-white">{r.topic}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{r.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
          <p className="text-gray-400 text-sm">Just ask <span className="text-white font-semibold">"tell me about [any topic above]"</span> and I'll build a full deep-dive guide for it.</p>
        </div>

      </div>
    </div>
  );
}


    // ─── KNOWLEDGE GAPS COMPONENT ───
const areas = [
  {
    id: "llm", icon: Brain, title: "LLM Concepts & Internals", current: 80, target: 90,
    color: "from-blue-600 to-indigo-700", border: "border-blue-800",
    gap: "You know the what. The 10% gap is the deep why — mathematical intuition behind attention, positional encodings, and scaling laws.",
    mathNeeded: "light",
    mathDetail: "You don't need to derive backpropagation. You need enough linear algebra to understand matrix multiplication in attention (Q×K×V), and enough calculus to understand what 'gradient descent minimizes a loss function' actually means geometrically.",
    mathTopics: [
      { topic: "Linear Algebra", need: "Vectors, matrices, dot products, matrix multiply. That's it.", depth: "Conceptual" },
      { topic: "Calculus", need: "What a derivative is. Chain rule intuitively. No need to compute by hand.", depth: "Conceptual" },
      { topic: "Probability", need: "Softmax, probability distributions, cross-entropy loss. Why we maximize log-likelihood.", depth: "Light" },
    ],
    exploreTopics: [
      { name: "Attention mechanism math", why: "The core operation. Q, K, V matrices. Scaled dot-product. You can read the Attention paper now.", resource: "The Illustrated Transformer – Jay Alammar", url: "https://jalammar.github.io/illustrated-transformer/" },
      { name: "Scaling laws", why: "Why bigger models are smarter. Chinchilla laws. Compute-optimal training. Explains the entire LLM arms race.", resource: "Chinchilla paper (skim)", url: "https://arxiv.org/abs/2203.15556" },
      { name: "KV Cache", why: "How inference is made fast. Explains context window limits and latency.", resource: "Karpathy's LLM internals talk", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g" },
      { name: "MoE (Mixture of Experts)", why: "How GPT-4, Mixtral, and DeepSeek scale without proportional compute cost.", resource: "Hugging Face MoE blog", url: "https://huggingface.co/blog/moe" },
      { name: "Positional encodings (RoPE, ALiBi)", why: "How models understand word order and handle long contexts.", resource: "EleutherAI blog", url: "https://blog.eleuther.ai/rotary-embeddings/" },
    ],
  },
  {
    id: "prompt", icon: Zap, title: "Prompt Engineering", current: 90, target: 95,
    color: "from-green-600 to-teal-700", border: "border-green-800",
    gap: "Near expert. The 5% gap is systematic evaluation of prompts and knowledge of model-specific quirks (Claude vs GPT-4 vs Gemini behave differently at edge cases).",
    mathNeeded: "none",
    mathDetail: "Zero math needed. Prompt engineering is pure empirical experimentation and pattern recognition.",
    mathTopics: [],
    exploreTopics: [
      { name: "Prompt evaluation frameworks", why: "Treat prompts like code — version, test, measure. Most people never do this.", resource: "PromptFoo docs", url: "https://promptfoo.dev/docs/intro" },
      { name: "Model-specific quirks", why: "Claude responds differently to authority framing vs GPT-4. Gemini handles long docs differently. Knowing these edges saves hours.", resource: "Anthropic model card", url: "https://www.anthropic.com/claude" },
      { name: "DSPy – Programmatic prompting", why: "Next evolution: optimizing prompts automatically using algorithms instead of manual tuning.", resource: "DSPy GitHub", url: "https://github.com/stanfordnlp/dspy" },
    ],
  },
  {
    id: "rag", icon: BookOpen, title: "RAG Systems", current: 80, target: 90,
    color: "from-purple-600 to-violet-700", border: "border-purple-800",
    gap: "You can build RAG. The gap is advanced retrieval techniques, hybrid search, and production-grade evaluation.",
    mathNeeded: "light",
    mathDetail: "Cosine similarity (dot product of normalized vectors) is the core math behind semantic search. You need to understand it conceptually — why similar vectors have high cosine similarity — not derive it.",
    mathTopics: [
      { topic: "Cosine Similarity", need: "Why it measures semantic closeness. Intuition: angle between vectors.", depth: "Conceptual" },
      { topic: "Vector spaces", need: "Why embeddings cluster semantically similar content.", depth: "Conceptual" },
      { topic: "BM25 (sparse retrieval)", need: "How keyword search works as complement to semantic search.", depth: "Light" },
    ],
    exploreTopics: [
      { name: "Hybrid search (sparse + dense)", why: "BM25 + vector search together beats either alone. Production RAG almost always uses hybrid.", resource: "Weaviate hybrid search docs", url: "https://weaviate.io/blog/hybrid-search-explained" },
      { name: "HyDE (Hypothetical Document Embeddings)", why: "Generate a hypothetical answer, embed it, then search. Dramatically improves retrieval quality.", resource: "HyDE paper", url: "https://arxiv.org/abs/2212.10496" },
      { name: "GraphRAG", why: "Microsoft's technique — build a knowledge graph from documents for multi-hop reasoning. Outperforms naive RAG on complex questions.", resource: "Microsoft GraphRAG", url: "https://microsoft.github.io/graphrag/" },
      { name: "RAGAS evaluation", why: "Systematic RAG evaluation. Context relevance, faithfulness, answer correctness. Essential for production.", resource: "RAGAS docs", url: "https://docs.ragas.io" },
      { name: "Reranking", why: "Two-stage retrieval: broad semantic search → reranker (Cohere, cross-encoder) picks best results. Huge quality boost.", resource: "Cohere reranking guide", url: "https://docs.cohere.com/docs/reranking" },
    ],
  },
  {
    id: "agents", icon: Bot, title: "Agentic AI / Tool Use", current: 75, target: 88,
    color: "from-orange-600 to-amber-700", border: "border-orange-800",
    gap: "You understand patterns. The gap is reliability engineering for agents — making them robust, observable, and recoverable when they fail.",
    mathNeeded: "none",
    mathDetail: "No math needed for agentic AI. It's software architecture, prompt design, and systems thinking.",
    mathTopics: [],
    exploreTopics: [
      { name: "Agent failure modes & recovery", why: "Infinite loops, wrong tool calls, hallucinated tool inputs. How to detect and recover gracefully.", resource: "Anthropic agent guide", url: "https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview" },
      { name: "Long-horizon task planning", why: "Agents that need 10+ steps to complete a task. How to maintain coherence over long runs.", resource: "LangGraph docs", url: "https://langchain-ai.github.io/langgraph/" },
      { name: "Agent memory systems", why: "Short-term (context), long-term (vector store), episodic (past runs), semantic (facts). Knowing which to use when.", resource: "MemGPT paper", url: "https://arxiv.org/abs/2310.08560" },
      { name: "MCP deep dive", why: "Model Context Protocol is becoming the USB-C of AI tool integration. Already supported by Claude, Cursor, and dozens of services.", resource: "MCP docs", url: "https://docs.anthropic.com/en/docs/agents-and-tools/mcp" },
      { name: "Agent observability", why: "Tracing every LLM call, tool invocation, and decision in a multi-step agent. Essential for debugging.", resource: "LangSmith tracing", url: "https://docs.smith.langchain.com" },
    ],
  },
  {
    id: "finetune", icon: Building2, title: "Fine-Tuning / Training", current: 60, target: 80,
    color: "from-rose-600 to-pink-700", border: "border-rose-800",
    gap: "This is your biggest gap between where you are and where a solid AI engineer should be. The gap is hands-on training experience and understanding the math behind optimization.",
    mathNeeded: "medium",
    mathDetail: "This is where math actually matters. You need enough to interpret training runs, understand why your model is overfitting, and make informed hyperparameter decisions. You don't need PhD-level proofs — you need applied intuition.",
    mathTopics: [
      { topic: "Calculus — Gradients & Loss", need: "What a loss curve means. Why high loss = bad. What overfitting looks like on a val loss graph.", depth: "Applied" },
      { topic: "Linear Algebra — Matrix ops", need: "LoRA adds low-rank matrices to weight matrices. You need to understand rank and why low-rank approximations work.", depth: "Applied" },
      { topic: "Statistics — Distributions", need: "Cross-entropy loss, KL divergence (used in RLHF), perplexity. Not derivations — intuitions.", depth: "Applied" },
      { topic: "Optimization", need: "Adam optimizer, learning rate schedules, warmup. Why these choices matter practically.", depth: "Applied" },
    ],
    exploreTopics: [
      { name: "LoRA mechanics (the math)", why: "Understand WHY low-rank adaptation works. W = W₀ + AB where rank(AB) << rank(W). This intuition helps you tune rank and alpha.", resource: "LoRA paper (original)", url: "https://arxiv.org/abs/2106.09685" },
      { name: "DPO (Direct Preference Optimization)", why: "The modern RLHF replacement. Simpler than PPO, no reward model needed. Most new models use DPO now.", resource: "DPO paper", url: "https://arxiv.org/abs/2305.18290" },
      { name: "Unsloth fine-tuning notebooks", why: "Hands-on: fine-tune Llama 3 on Colab in 30 minutes. The fastest path from 60% to 75%.", resource: "Unsloth GitHub", url: "https://github.com/unslothai/unsloth" },
      { name: "Hyperparameter intuition", why: "Learning rate, batch size, warmup steps, LoRA rank — knowing what each does to training dynamics.", resource: "Sebastian Raschka's blog", url: "https://sebastianraschka.com/blog/" },
      { name: "Evaluation during training", why: "Train/val loss curves, perplexity, MMLU benchmarks. How to know if your fine-tune is working.", resource: "Weights & Biases LLM tutorial", url: "https://docs.wandb.ai/tutorials/llm_finetuning" },
    ],
  },
  {
    id: "multimodal", icon: Eye, title: "Multimodal AI", current: 40, target: 75,
    color: "from-violet-600 to-purple-700", border: "border-violet-800",
    gap: "Large gap. You understand diffusion conceptually but haven't built multimodal apps. This is the most immediately useful area to improve given Claude and GPT-4V are already in your hands.",
    mathNeeded: "light",
    mathDetail: "For practical multimodal AI (building apps with vision), almost no math is needed. If you want to understand diffusion models deeply, you need probability theory and some calculus.",
    mathTopics: [
      { topic: "Probability — Gaussian distributions", need: "Diffusion adds/removes Gaussian noise. Understanding normal distributions explains the forward/reverse process.", depth: "Conceptual" },
      { topic: "Linear Algebra — CLIP embeddings", need: "How text and images are projected into the same vector space. Same cosine similarity intuition as RAG.", depth: "Conceptual" },
    ],
    exploreTopics: [
      { name: "Vision prompting techniques", why: "How to prompt Claude/GPT-4V effectively with images. Different from text prompting. Very high immediate ROI.", resource: "Anthropic vision docs", url: "https://docs.anthropic.com/en/docs/build-with-claude/vision" },
      { name: "Document understanding apps", why: "Extract structured data from PDFs, receipts, invoices, contracts using vision models. Immediately buildable.", resource: "Claude vision cookbook", url: "https://github.com/anthropics/anthropic-cookbook/tree/main/multimodal" },
      { name: "CLIP — how text and images share a space", why: "The foundational idea behind image generation, image search, and vision-language models.", resource: "OpenAI CLIP blog", url: "https://openai.com/research/clip" },
      { name: "Multimodal RAG", why: "Embed both text and images in the same vector store. Search across PDFs with charts, diagrams, and text together.", resource: "LlamaIndex multimodal RAG", url: "https://docs.llamaindex.ai/en/stable/examples/multi_modal/" },
      { name: "Video understanding", why: "Gemini 1.5 can process entire videos. New class of applications: video summarization, meeting analysis, video search.", resource: "Gemini video docs", url: "https://ai.google.dev/gemini-api/docs/vision" },
    ],
  },
  {
    id: "math", icon: BookMarked, title: "ML Research / Math", current: 30, target: 60,
    color: "from-cyan-600 to-blue-700", border: "border-cyan-800",
    gap: "30% is actually fine for a developer with your goals. You don't need 90% here. A target of 55–60% — enough to read papers fluently and understand training dynamics — is realistic and sufficient.",
    mathNeeded: "structured",
    mathDetail: "Here's the honest breakdown: you need 4 specific math areas, each to a specific depth. NOT a full university course in each. Targeted, applied study of each concept as it relates to AI.",
    mathTopics: [
      { topic: "Linear Algebra", need: "Vectors, matrices, dot product, matrix multiply, eigenvalues (basic intuition). 3Blue1Brown Essence of Linear Algebra is all you need.", depth: "Applied — 10 hrs" },
      { topic: "Calculus", need: "Derivatives, chain rule, gradients, partial derivatives. Enough to understand backprop directionally. NOT integration, NOT limits proofs.", depth: "Applied — 6 hrs" },
      { topic: "Probability & Statistics", need: "Probability distributions, Bayes theorem, expectation, variance, KL divergence, cross-entropy. YES — this is the most important one for AI.", depth: "Applied — 12 hrs" },
      { topic: "Information Theory", need: "Entropy, cross-entropy loss, mutual information. Directly used in LLM training objectives.", depth: "Light — 4 hrs" },
    ],
    exploreTopics: [
      { name: "3Blue1Brown – Essence of Linear Algebra", why: "The best visual introduction to linear algebra for AI. 15 videos, ~3hrs total. After this, attention mechanism math will click.", resource: "YouTube playlist (free)", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab" },
      { name: "3Blue1Brown – Essence of Calculus", why: "11 videos. Gradients, derivatives, chain rule — all you need for backpropagation intuition.", resource: "YouTube playlist (free)", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr" },
      { name: "StatQuest – Statistics & ML", why: "Josh Starmer makes probability and stats genuinely easy. Covers everything from Bayes to KL divergence with visuals.", resource: "YouTube channel (free)", url: "https://www.youtube.com/@statquest" },
      { name: "Mathematics for Machine Learning (book)", why: "Free PDF from Cambridge. Covers linear algebra, calculus, and probability specifically for ML. Reference, not cover-to-cover.", resource: "Free PDF (Cambridge)", url: "https://mml-book.github.io/" },
      { name: "Fast.ai – Practical Deep Learning", why: "Code-first approach to ML math. Learn the math by implementing it. Best for developers who hate pure theory.", resource: "fast.ai (free)", url: "https://course.fast.ai" },
    ],
  },
  {
    id: "mlops", icon: Wrench, title: "Production / MLOps", current: 50, target: 75,
    color: "from-red-600 to-rose-700", border: "border-red-800",
    gap: "You know the concepts but lack hands-on deployment experience. The gap is practical: deploy something, break it, observe it, fix it.",
    mathNeeded: "none",
    mathDetail: "Zero math needed. MLOps is software engineering applied to AI — observability, reliability, cost optimization, and DevOps patterns.",
    mathTopics: [],
    exploreTopics: [
      { name: "LLM observability with LangSmith", why: "Trace every LLM call in your app. See latency, cost, inputs, outputs, errors. Transforms how you debug.", resource: "LangSmith quickstart", url: "https://docs.smith.langchain.com/tutorials/Developers/traceable" },
      { name: "Prompt versioning & regression testing", why: "When you change a prompt, how do you know it didn't break something else? Treat prompts like code.", resource: "PromptFoo CI/CD guide", url: "https://promptfoo.dev/docs/integrations/ci-cd/" },
      { name: "vLLM for inference", why: "Serve open-source models with 10–20× better throughput than naive HuggingFace inference. Continuous batching, PagedAttention.", resource: "vLLM docs", url: "https://docs.vllm.ai" },
      { name: "Cost optimization patterns", why: "Token caching, model routing (cheap model first, expensive only if needed), batching async requests. Cuts bills by 60–80%.", resource: "Anthropic prompt caching docs", url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching" },
      { name: "Designing ML Systems – Chip Huyen", why: "The definitive book on production ML. Covers data pipelines, deployment, monitoring, and failure modes.", resource: "O'Reilly book", url: "https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/" },
    ],
  },
  {
    id: "safety", icon: Shield, title: "AI Safety & Ethics", current: 40, target: 60,
    color: "from-gray-600 to-slate-700", border: "border-gray-700",
    gap: "For your goals (building + domain knowledge), 60% awareness is sufficient. You don't need to become an alignment researcher. Focus on practical safety: how to build responsible AI products.",
    mathNeeded: "none",
    mathDetail: "No math needed for practical AI safety. Alignment research uses heavy math, but responsible AI product development is policy, systems thinking, and ethics.",
    mathTopics: [],
    exploreTopics: [
      { name: "Constitutional AI (Anthropic)", why: "Understanding how Claude is trained to be safe makes you a better Claude user and prompt engineer. Directly applicable.", resource: "Constitutional AI paper", url: "https://arxiv.org/abs/2212.08073" },
      { name: "OWASP Top 10 for LLMs", why: "Prompt injection, insecure output handling, model denial of service. Practical security for AI app builders.", resource: "OWASP LLM Top 10", url: "https://owasp.org/www-project-top-10-for-large-language-model-applications/" },
      { name: "Responsible AI product checklist", why: "Before shipping: bias audit, adversarial testing, data privacy review, transparency to users. Practical framework.", resource: "Google Responsible AI practices", url: "https://ai.google/responsibility/responsible-ai-practices/" },
      { name: "AI Safety Fundamentals course", why: "6-week structured intro to alignment, interpretability, and governance. Free, well-designed.", resource: "BlueDot AI Safety Fundamentals", url: "https://aisafetyfundamentals.com/" },
    ],
  },
];

const mathColors = {
  none: "bg-green-900 text-green-300 border-green-800",
  light: "bg-blue-900 text-blue-300 border-blue-800",
  medium: "bg-yellow-900 text-yellow-300 border-yellow-800",
  structured: "bg-red-900 text-red-300 border-red-800",
};
const mathLabels = {
  none: "No math needed",
  light: "Light math (conceptual)",
  medium: "Medium math (applied)",
  structured: "Structured math study needed",
};

const GapBar = ({ current, target }) => (
  <div className="relative w-full bg-gray-800 rounded-full h-2 mt-2">
    <div className="h-2 rounded-full bg-gray-600" style={{ width: `${target}%` }} />
    <div className="absolute top-0 h-2 rounded-full bg-blue-500" style={{ width: `${current}%` }} />
    <div className="absolute top-0 h-2 rounded-l-none rounded-r-full opacity-60"
      style={{ left: `${current}%`, width: `${target - current}%`, background: "repeating-linear-gradient(45deg, #f59e0b, #f59e0b 2px, transparent 2px, transparent 6px)" }} />
  </div>
);

function KnowledgeGaps() {
  const [open, setOpen] = useState(null);
  const [openTech, setOpenTech] = useState(null);

  return (
    <div className="min-h-screen text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">

        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-1">📐 What to Study Per Area</h1>
          <p className="text-gray-400 text-sm">Exact gaps, required math, and what to explore next — per knowledge domain</p>
        </div>

        {/* Legend */}
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-3 mb-5 flex flex-wrap gap-3 items-center">
          <span className="text-xs text-gray-400">Math needed:</span>
          {Object.entries(mathLabels).map(([k, v]) => (
            <span key={k} className={`text-xs px-2 py-0.5 rounded-full border ${mathColors[k]}`}>{v}</span>
          ))}
        </div>

        {/* Answer the stats question directly */}
        <div className="bg-gray-900 border border-yellow-800 rounded-xl p-4 mb-5">
          <p className="text-yellow-400 font-semibold text-sm mb-2">📊 Do You Need Statistics?</p>
          <p className="text-gray-300 text-sm mb-2">
            <span className="text-white font-semibold">Yes — but selectively.</span> Statistics is the most important math topic for AI, but you only need 4 specific concepts:
          </p>
          <div className="grid grid-cols-2 gap-2">
            {[
              ["Probability distributions", "Understand how models sample outputs (softmax, temperature)"],
              ["Cross-entropy loss", "THE loss function for LLMs. You'll see it everywhere."],
              ["KL divergence", "Used in RLHF, VAEs, and fine-tuning. Measures 'distance' between distributions."],
              ["Bayes theorem", "Conceptual understanding of how evidence updates beliefs — useful for RAG intuition."],
            ].map(([k, v], i) => (
              <div key={i} className="bg-gray-800 rounded-lg p-2">
                <p className="text-xs font-semibold text-yellow-300">{k}</p>
                <p className="text-xs text-gray-400 mt-0.5">{v}</p>
              </div>
            ))}
          </div>
          <p className="text-gray-400 text-xs mt-2">You do NOT need: hypothesis testing, regression analysis, ANOVA, or most classical statistics. Those are for data science, not AI engineering.</p>
        </div>

        {/* Areas */}
        <div className="space-y-3">
          {areas.map(a => (
            <div key={a.id}
              className={`rounded-xl border overflow-hidden cursor-pointer transition-all ${open === a.id ? "border-gray-500" : "border-gray-800 hover:border-gray-600"} bg-gray-900`}
              onClick={() => setOpen(open === a.id ? null : a.id)}>

              <div className="flex items-center gap-3 p-4">
                <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${a.color} flex items-center justify-center text-base flex-shrink-0`}>{React.createElement(a.icon, {size:20, className:"text-white"})}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <p className="font-semibold text-sm">{a.title}</p>
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${mathColors[a.mathNeeded]}`}>{mathLabels[a.mathNeeded]}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <GapBar current={a.current} target={a.target} />
                    <span className="text-xs text-gray-400 flex-shrink-0">{a.current}% → {a.target}%</span>
                  </div>
                </div>
                <span className="text-gray-400 flex-shrink-0">{open === a.id ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</span>
              </div>

              {open === a.id && (
                <div className="border-t border-gray-800 p-4 space-y-4">

                  {/* Gap description */}
                  <div className="bg-gray-800 rounded-lg p-3">
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">The Gap</p>
                    <p className="text-gray-300 text-sm">{a.gap}</p>
                  </div>

                  {/* Math breakdown */}
                  <div className={`rounded-lg p-3 border ${mathColors[a.mathNeeded].replace("text-", "border-").replace("bg-", "bg-")}`}>
                    <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{color: a.mathNeeded === "none" ? "#86efac" : a.mathNeeded === "light" ? "#93c5fd" : a.mathNeeded === "medium" ? "#fbbf24" : "#fca5a5"}}>
                      📐 Math Reality Check
                    </p>
                    <p className="text-gray-300 text-sm mb-3">{a.mathDetail}</p>
                    {a.mathTopics.length > 0 && (
                      <div className="space-y-2">
                        {a.mathTopics.map((m, i) => (
                          <div key={i} className="bg-gray-900 rounded-lg p-2">
                            <div className="flex items-center justify-between mb-0.5">
                              <p className="text-xs font-semibold text-white">{m.topic}</p>
                              <span className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded">{m.depth}</span>
                            </div>
                            <p className="text-xs text-gray-400">{m.need}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Explore topics */}
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">What to Explore Next</p>
                    <div className="space-y-2">
                      {a.exploreTopics.map((t, i) => (
                        <div key={i} className={`rounded-lg border overflow-hidden ${openTech === `${a.id}-${i}` ? "border-gray-500" : "border-gray-700"}`}>
                          <div className="flex items-center justify-between p-3 bg-gray-800 cursor-pointer"
                            onClick={e => { e.stopPropagation(); setOpenTech(openTech === `${a.id}-${i}` ? null : `${a.id}-${i}`); }}>
                            <p className="text-sm font-medium text-white">{t.name}</p>
                            <span className="text-gray-400 text-xs ml-2">{openTech === `${a.id}-${i}` ? "▲" : "▼"}</span>
                          </div>
                          {openTech === `${a.id}-${i}` && (
                            <div className="p-3 bg-gray-900 space-y-2">
                              <p className="text-xs text-gray-300">{t.why}</p>
                              <a href={t.url} target="_blank" rel="noopener noreferrer"
                                className="flex items-center gap-2 text-blue-400 text-xs hover:text-blue-300 transition-colors"
                                onClick={e => e.stopPropagation()}>
                                📖 {t.resource} →
                              </a>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Priority summary */}
        <div className="mt-6 bg-gray-900 border border-gray-700 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">🎯 Recommended Study Order (For Your Profile)</p>
          <div className="space-y-2">
            {[
              { n: "1", label: "Statistics (4 concepts only)", sub: "Probability distributions, cross-entropy, KL divergence, Bayes. Use StatQuest.", color: "text-red-400" },
              { n: "2", label: "Linear Algebra (visually)", sub: "3Blue1Brown Essence of Linear Algebra. 3 hrs. Unlocks LLM math intuition.", color: "text-orange-400" },
              { n: "3", label: "LoRA mechanics", sub: "Read the original LoRA paper after linear algebra. It will make sense now.", color: "text-yellow-400" },
              { n: "4", label: "Multimodal — vision prompting", sub: "Immediate ROI. Claude and GPT-4 vision are already in your hands.", color: "text-green-400" },
              { n: "5", label: "LangSmith / observability", sub: "Install it in your next project. Transforms how you build and debug.", color: "text-blue-400" },
              { n: "6", label: "Calculus (chain rule only)", sub: "3Blue1Brown Essence of Calculus. 2 hrs. Makes backprop click.", color: "text-purple-400" },
            ].map((r, i) => (
              <div key={i} className="flex gap-3 bg-gray-800 rounded-lg p-3 items-start">
                <span className={`text-sm font-bold ${r.color} flex-shrink-0 w-5`}>{r.n}</span>
                <div>
                  <p className="text-sm font-semibold text-white">{r.label}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{r.sub}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}

    // ─── BEYOND ROADMAP (merged What's Left + Knowledge Gaps) ───
    function BeyondRoadmap() {
      useSeo(
        "Beyond the AI Roadmap – Knowledge Gaps & What's Next | AI Learning Hub",
        "Finished the AI roadmap? See your knowledge gaps by area, explore advanced topics not covered in the core roadmap, and plan your specialization."
      );
      const [subTab, setSubTab] = React.useState(0);
      return (
        <div className="min-h-screen text-gray-100 font-sans">
          <div className="max-w-3xl mx-auto px-4 pt-6 pb-2">
            <div className="text-center mb-5">
              <div className="inline-flex items-center gap-2 mb-3">
                <Compass size={20} className="text-blue-400"/>
                <h1 className="text-2xl md:text-3xl font-bold">Beyond the Roadmap</h1>
              </div>
              <p className="text-gray-400 text-sm max-w-lg mx-auto">Finished the roadmap? This section shows where you stand as an AI engineer, what gaps remain, and what to build next.</p>
              <p className="text-xs text-gray-400 mt-2">Start with Knowledge Gaps to see what areas need more depth, then check What's Left to plan next steps.</p>
            </div>
            <div className="flex gap-1 bg-gray-900/60 border border-white/8 rounded-xl p-1 mb-4">
              <button onClick={() => setSubTab(0)}
                className={`flex-1 text-xs py-2 rounded-lg transition-colors ${subTab === 0 ? "bg-blue-600 text-white font-semibold" : "text-gray-400 hover:text-gray-300"}`}>
                Knowledge Gaps
              </button>
              <button onClick={() => setSubTab(1)}
                className={`flex-1 text-xs py-2 rounded-lg transition-colors ${subTab === 1 ? "bg-blue-600 text-white font-semibold" : "text-gray-400 hover:text-gray-300"}`}>
                What's Left
              </button>
            </div>
          </div>
          {subTab === 0 ? <KnowledgeGaps /> : <WhatsLeft />}
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // AI ROADMAP LANDING PAGE
    // ═══════════════════════════════════════════════════════════
    function AiRoadmapPage() {
      useSeo(
        "AI Roadmap 2026: Complete Guide for Developers",
        "Follow the complete AI roadmap for developers in 2026. Our step-by-step guide covers 7 phases — from AI foundations to LLMs, RAG, Prompt Engineering, and Agentic AI."
      );
      const phases = [
        { n: 1, title: "AI Foundations", time: "4–6 wks", color: "from-green-500 to-emerald-600", outcome: "Understand how neural networks, LLMs, and GenAI work — no math required." },
        { n: 2, title: "LLM Setup & APIs", time: "2–3 wks", color: "from-slate-500 to-gray-600", outcome: "Run local models with Ollama, call OpenAI / Anthropic / Gemini APIs from code." },
        { n: 3, title: "Prompt Engineering", time: "3–4 wks", color: "from-blue-500 to-indigo-600", outcome: "Ship your first AI-powered app using zero-shot, few-shot, and chain-of-thought." },
        { n: 4, title: "RAG & Your Data", time: "4–5 wks", color: "from-purple-500 to-violet-600", outcome: "Build a chatbot over your own documents with vector databases and LangChain." },
        { n: 5, title: "Agentic AI", time: "4–5 wks", color: "from-orange-500 to-amber-600", outcome: "Build agents that plan, call tools, and execute multi-step tasks autonomously." },
        { n: 6, title: "Fine-tuning LLMs", time: "6–8 wks", color: "from-rose-500 to-pink-600", outcome: "Fine-tune Llama 3 with QLoRA on free Colab GPUs. Know when to train vs prompt vs RAG." },
        { n: 7, title: "Ship Real Projects", time: "Ongoing", color: "from-teal-500 to-cyan-600", outcome: "Launch 2–3 public AI projects. Real mastery comes from building things people use." },
      ];
      return (
        <div className="max-w-3xl mx-auto px-4 py-10 text-gray-100">
          <div className="mb-8">
            <span className="inline-block bg-blue-600/20 text-blue-400 text-xs font-semibold px-3 py-1 rounded-full mb-4">AI Roadmap 2026</span>
            <h1 className="text-3xl font-bold text-white mb-4">AI Roadmap 2026: Complete Guide for Developers</h1>
            <p className="text-gray-400 text-lg leading-relaxed">A structured, opinionated AI learning roadmap for software developers. 7 phases, each with a clear outcome and project milestone — so you always know what to learn next and when you've actually learned it.</p>
          </div>

          {/* CTA to interactive tool */}
          <div className="bg-blue-600/10 border border-blue-500/30 rounded-xl p-5 mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <p className="text-white font-semibold text-sm mb-1">Want progress tracking + curated resources?</p>
              <p className="text-gray-400 text-xs">The interactive roadmap has per-phase resources, topic checklists, and a progress bar.</p>
            </div>
            <a href="/" className="flex-shrink-0 inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold px-4 py-2 rounded-lg transition-colors whitespace-nowrap">
              Open Interactive Roadmap →
            </a>
          </div>

          <h2 className="text-2xl font-bold text-white mb-5">The 7 Phases at a Glance</h2>
          <div className="space-y-3 mb-10">
            {phases.map(({ n, title, time, color, outcome }) => (
              <div key={n} className="flex items-start gap-4 bg-gray-800/40 rounded-xl p-4 border border-white/8">
                <div className={`w-9 h-9 rounded-lg bg-gradient-to-br ${color} flex items-center justify-center text-white font-bold text-sm flex-shrink-0`}>{n}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-0.5">
                    <span className="text-white font-semibold text-sm">{title}</span>
                    <span className="text-xs text-gray-500">{time}</span>
                  </div>
                  <p className="text-gray-400 text-sm">{outcome}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Timeline */}
          <div className="bg-gray-800/50 rounded-xl p-5 mb-8 border border-white/8">
            <h2 className="text-lg font-semibold text-white mb-4">Timeline at 4–6 hrs / week</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[["Phases 1–3", "~3 months", "Foundations → First app"],["Phases 4–5", "~2 months", "RAG + Agents"],["Phase 6", "~2 months", "Fine-tuning"],["Phase 7", "Ongoing", "Ship & iterate"]].map(([label, time, sub]) => (
                <div key={label} className="bg-gray-900/50 rounded-lg p-3 text-center">
                  <div className="text-blue-400 font-semibold text-sm">{label}</div>
                  <div className="text-white text-xs font-medium mt-1">{time}</div>
                  <div className="text-gray-500 text-xs mt-0.5">{sub}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Frequently asked */}
          <div className="space-y-4 mb-8">
            <h2 className="text-xl font-semibold text-white">Common Questions</h2>
            {[
              ["Do I need a math background?", "No. Phases 1–5 are entirely practical — APIs, RAG, agents. Phase 6 benefits from linear algebra intuition but you can fine-tune with QLoRA without deep math."],
              ["I already know Python. Where do I start?", "Jump to Phase 2 (LLM Setup) or even Phase 3 (Prompt Engineering) if you've already experimented with LLM APIs. Use the readiness checklist to calibrate."],
              ["How is this different from a machine learning roadmap?", "This roadmap is LLM-application focused — building products with pre-trained models. A classical ML roadmap (scikit-learn, PyTorch, training from scratch) has more overlap with data science. See our ML roadmap for that path."],
            ].map(([q, a]) => (
              <div key={q} className="bg-gray-800/40 rounded-xl p-5 border border-white/8">
                <p className="text-white font-semibold text-sm mb-2">{q}</p>
                <p className="text-gray-400 text-sm">{a}</p>
              </div>
            ))}
          </div>

          {/* Related links */}
          <div className="bg-gray-800/40 rounded-xl p-5 border border-white/8">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Go Deeper</h2>
            <div className="grid sm:grid-cols-2 gap-3">
              {[
                ["/", "Interactive Roadmap", "Phase-by-phase with resources, checklists, and progress tracking"],
                ["/llm-roadmap/", "LLM Roadmap", "Focused path for learning large language models specifically"],
                ["/rag-tutorial/", "RAG Tutorial", "Step-by-step guide to building your first RAG pipeline"],
                ["/ai-projects/", "AI Project Ideas", "10 hands-on projects from beginner to advanced"],
              ].map(([href, title, desc]) => (
                <a key={href} href={href} className="block bg-gray-900/50 rounded-lg p-3 hover:bg-gray-700/50 transition-colors">
                  <p className="text-blue-400 text-sm font-medium mb-0.5">{title} →</p>
                  <p className="text-gray-500 text-xs">{desc}</p>
                </a>
              ))}
            </div>
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // LLM ROADMAP LANDING PAGE
    // ═══════════════════════════════════════════════════════════
    function LlmRoadmapPage() {
      useSeo(
        "LLM Roadmap 2026: Learn Large Language Models from Scratch",
        "The complete LLM roadmap for developers in 2026. Learn how large language models work, use them via APIs, fine-tune with LoRA, and build RAG and agentic AI applications."
      );
      return (
        <div className="max-w-3xl mx-auto px-4 py-10 text-gray-100">
          <div className="mb-8">
            <span className="inline-block bg-purple-600/20 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full mb-4">LLM Roadmap 2026</span>
            <h1 className="text-3xl font-bold text-white mb-4">LLM Roadmap 2026: Learn Large Language Models from Scratch</h1>
            <p className="text-gray-400 text-lg leading-relaxed">Large language models are the core technology behind every major AI product in 2026. This LLM roadmap takes you from understanding how LLMs work internally to building production applications, fine-tuning models, and deploying them at scale.</p>
          </div>

          <div className="bg-gray-800/50 rounded-xl p-6 mb-8 border border-white/8">
            <h2 className="text-xl font-semibold text-white mb-3">What Are Large Language Models?</h2>
            <p className="text-gray-300 mb-3">Large language models (LLMs) are neural networks trained on massive text datasets to understand and generate human language. Models like GPT-4, Claude, and Llama 3 are trained on trillions of tokens and have billions to hundreds of billions of parameters.</p>
            <p className="text-gray-300">LLMs work by predicting the next token in a sequence. Despite this simple objective, the ability to predict text at scale results in emergent capabilities: reasoning, coding, math, translation, and complex instruction following.</p>
          </div>

          <h2 className="text-2xl font-bold text-white mb-6">LLM Learning Roadmap: 6 Stages</h2>

          <div className="space-y-4 mb-10">
            {[
              { n: 1, title: "How LLMs Work Internally", color: "bg-blue-600", items: ["Tokenization and vocabulary (BPE, SentencePiece)", "Transformer architecture: attention, MLP, positional encoding", "Pre-training objective: next token prediction", "Watch: Karpathy's 'Let's build GPT' (free, YouTube)"] },
              { n: 2, title: "Using LLM APIs in Code", color: "bg-indigo-600", items: ["OpenAI, Anthropic, and Gemini Python SDKs", "Chat completions vs completions API", "Streaming responses, function calling, JSON mode", "Context window management and token counting"] },
              { n: 3, title: "Prompt Engineering for LLMs", color: "bg-violet-600", items: ["System prompts and conversation structure", "Zero-shot, few-shot, chain-of-thought techniques", "Structured output and format control", "Production prompt versioning and testing"] },
              { n: 4, title: "RAG: LLMs + Your Own Data", color: "bg-purple-600", items: ["Vector embeddings and semantic search", "LangChain document loaders and text splitters", "ChromaDB, Pinecone vector database setup", "Build a document Q&A chatbot from scratch"] },
              { n: 5, title: "Fine-tuning LLMs", color: "bg-pink-600", items: ["When to fine-tune vs prompt engineer vs RAG", "Supervised Fine-Tuning (SFT) dataset format", "LoRA and QLoRA: parameter-efficient fine-tuning", "Fine-tune Llama 3 on free Google Colab GPUs"] },
              { n: 6, title: "Deploying LLM Applications", color: "bg-rose-600", items: ["FastAPI server for LLM endpoints", "Streaming responses and server-sent events", "LLM observability with LangSmith / Langfuse", "Cost optimization: caching, batching, model selection"] },
            ].map(({ n, title, color, items }) => (
              <div key={n} className="bg-gray-800/40 rounded-xl p-5 border border-white/8">
                <div className="flex items-center gap-3 mb-3">
                  <span className={`${color} text-white text-xs font-bold w-7 h-7 rounded-full flex items-center justify-center`}>{n}</span>
                  <h3 className="text-white font-semibold">{title}</h3>
                </div>
                <ul className="space-y-1.5 ml-10">
                  {items.map((item, i) => <li key={i} className="text-gray-300 text-sm flex items-start gap-2"><span className="text-purple-400 mt-0.5">▸</span>{item}</li>)}
                </ul>
              </div>
            ))}
          </div>

          <div className="bg-gray-800/50 rounded-xl p-6 mb-8 border border-white/8">
            <h2 className="text-xl font-semibold text-white mb-4">Open Source vs Closed Source LLMs</h2>
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="bg-gray-900/50 rounded-lg p-4">
                <div className="text-blue-400 font-semibold mb-2">Closed Source (API)</div>
                <ul className="space-y-1 text-sm text-gray-300">
                  <li>• GPT-4o, GPT-4 (OpenAI)</li>
                  <li>• Claude 3.5 Sonnet (Anthropic)</li>
                  <li>• Gemini 1.5 Pro (Google)</li>
                  <li>• Best quality, paid per token</li>
                </ul>
              </div>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <div className="text-green-400 font-semibold mb-2">Open Source (Run Locally)</div>
                <ul className="space-y-1 text-sm text-gray-300">
                  <li>• Llama 3 8B/70B (Meta)</li>
                  <li>• Mistral 7B / Mixtral (Mistral AI)</li>
                  <li>• Gemma 2 (Google)</li>
                  <li>• Free, private, customizable</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-8 p-5 bg-gray-800/40 rounded-xl border border-white/8">
            <h2 className="text-lg font-semibold text-white mb-3">Start Your LLM Journey</h2>
            <p className="text-gray-400 text-sm mb-4">Our full 7-phase AI roadmap includes the complete LLM learning path with curated resources, project milestones, and an interactive progress tracker.</p>
            <div className="flex flex-wrap gap-3">
              <a href="/" className="inline-flex items-center gap-2 bg-purple-600 hover:bg-purple-500 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors">View Full Roadmap →</a>
              <a href="/prompt-eng/" className="inline-flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors">Prompt Engineering Guide →</a>
            </div>
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // RAG TUTORIAL LANDING PAGE
    // ═══════════════════════════════════════════════════════════
    function RagTutorialPage() {
      useSeo(
        "RAG Tutorial 2026: Build Retrieval-Augmented Generation Step by Step",
        "Complete RAG tutorial for developers in 2026. Learn to build a Retrieval-Augmented Generation pipeline from scratch — document ingestion, vector embeddings, semantic search, and LLM generation."
      );
      return (
        <div className="max-w-3xl mx-auto px-4 py-10 text-gray-100">
          <div className="mb-8">
            <span className="inline-block bg-violet-600/20 text-violet-400 text-xs font-semibold px-3 py-1 rounded-full mb-4">RAG Tutorial 2026</span>
            <h1 className="text-3xl font-bold text-white mb-4">RAG Tutorial 2026: Build Retrieval-Augmented Generation Step by Step</h1>
            <p className="text-gray-400 text-lg leading-relaxed">Retrieval-Augmented Generation (RAG) is the most important AI architecture for building real-world applications. This tutorial walks you through building a complete RAG pipeline from scratch — document loading, vector embeddings, semantic search, and LLM-powered answer generation.</p>
          </div>

          <div className="bg-gray-800/50 rounded-xl p-6 mb-8 border border-white/8">
            <h2 className="text-xl font-semibold text-white mb-3">What is RAG and Why Does It Matter?</h2>
            <p className="text-gray-300 mb-3">LLMs have a fundamental problem: their knowledge is frozen at training time. They can't answer questions about your private documents, company data, or recent events. RAG solves this by retrieving relevant documents at query time and including them in the LLM's context.</p>
            <div className="bg-gray-900/60 rounded-lg p-4 mt-4">
              <p className="text-blue-400 font-mono text-sm mb-2">RAG Pipeline Flow:</p>
              <p className="text-gray-300 text-sm font-mono">User Query → Embed Query → Vector Search → Retrieve Top-k Chunks → Augment Prompt → LLM Generate → Answer</p>
            </div>
          </div>

          <h2 className="text-2xl font-bold text-white mb-6">Step-by-Step RAG Tutorial</h2>

          <div className="space-y-4 mb-10">
            {[
              { step: 1, title: "Load and Parse Documents", color: "bg-blue-600", code: 'from langchain.document_loaders import PyPDFLoader\nloader = PyPDFLoader("document.pdf")\ndocs = loader.load()', desc: "Use LangChain document loaders to ingest PDFs, web pages, Notion exports, and text files. Each document becomes a list of Document objects with content and metadata." },
              { step: 2, title: "Chunk Documents", color: "bg-indigo-600", code: 'from langchain.text_splitter import RecursiveCharacterTextSplitter\nsplitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)\nchunks = splitter.split_documents(docs)', desc: "Split documents into overlapping chunks. Smaller chunks (256–512 tokens) give precise retrieval. Larger chunks (1024+) give more context. The overlap preserves continuity across chunk boundaries." },
              { step: 3, title: "Generate Embeddings and Store", color: "bg-violet-600", code: 'from langchain_openai import OpenAIEmbeddings\nfrom langchain.vectorstores import Chroma\n\nembeddings = OpenAIEmbeddings()\ndb = Chroma.from_documents(chunks, embeddings)', desc: "Convert text chunks to embedding vectors and store in a vector database. ChromaDB is free and runs locally. Pinecone and Weaviate are managed cloud options for production." },
              { step: 4, title: "Build the Retriever", color: "bg-purple-600", code: 'retriever = db.as_retriever(\n  search_type="similarity",\n  search_kwargs={"k": 5}\n)', desc: "Create a retriever that returns the top-k most semantically similar chunks for a given query. The k value (3–10) controls how much context the LLM receives." },
              { step: 5, title: "Create the RAG Chain", color: "bg-pink-600", code: 'from langchain_anthropic import ChatAnthropic\nfrom langchain.chains import RetrievalQA\n\nllm = ChatAnthropic(model="claude-3-5-sonnet-20241022")\nrag_chain = RetrievalQA.from_chain_type(\n  llm=llm, retriever=retriever\n)', desc: "Combine the retriever with an LLM using LangChain's RetrievalQA chain or LCEL. The chain retrieves relevant context and passes it to the LLM with the user's question." },
              { step: 6, title: "Evaluate RAG Quality", color: "bg-rose-600", code: 'from ragas import evaluate\nfrom ragas.metrics import faithfulness, answer_relevancy\n\nresult = evaluate(dataset, metrics=[faithfulness, answer_relevancy])\nprint(result)', desc: "Measure RAG quality with RAGAS metrics: Faithfulness (is the answer grounded in retrieved context?), Answer Relevancy (does the answer address the question?), Context Precision (are retrieved chunks relevant?)." },
            ].map(({ step, title, color, code, desc }) => (
              <div key={step} className="bg-gray-800/40 rounded-xl p-5 border border-white/8">
                <div className="flex items-center gap-3 mb-3">
                  <span className={`${color} text-white text-xs font-bold w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0`}>{step}</span>
                  <h3 className="text-white font-semibold">{title}</h3>
                </div>
                <p className="text-gray-400 text-sm mb-3">{desc}</p>
                <pre className="bg-gray-950/70 rounded-lg p-3 text-xs text-green-300 overflow-x-auto font-mono">{code}</pre>
              </div>
            ))}
          </div>

          <div className="bg-gray-800/50 rounded-xl p-6 mb-8 border border-white/8">
            <h2 className="text-xl font-semibold text-white mb-4">RAG vs Fine-tuning: When to Use Each</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b border-white/10"><th className="text-left text-gray-400 pb-2 pr-4">Approach</th><th className="text-left text-gray-400 pb-2 pr-4">Best For</th><th className="text-left text-gray-400 pb-2">When to Use</th></tr></thead>
                <tbody className="divide-y divide-white/5">
                  {[["RAG", "Private / dynamic data", "Data changes often, need citations"], ["Fine-tuning", "Consistent style/format", "Lots of labeled examples, consistent task"], ["Prompting", "Simple tasks", "Start here — works 80% of the time"]].map(([a, b, c]) => (
                    <tr key={a}><td className="text-white py-2 pr-4 font-medium">{a}</td><td className="text-gray-300 py-2 pr-4">{b}</td><td className="text-gray-300 py-2">{c}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-8 p-5 bg-gray-800/40 rounded-xl border border-white/8">
            <h2 className="text-lg font-semibold text-white mb-3">Learn RAG in Our Full Roadmap</h2>
            <p className="text-gray-400 text-sm mb-4">Phase 4 of the AI engineer roadmap covers RAG in detail — with curated courses, project milestones, and the best free resources.</p>
            <div className="flex flex-wrap gap-3">
              <a href="/" className="inline-flex items-center gap-2 bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors">View AI Roadmap →</a>
              <a href="/resources/" className="inline-flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors">Free Resources →</a>
            </div>
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // MACHINE LEARNING ROADMAP LANDING PAGE
    // ═══════════════════════════════════════════════════════════
    function MachineLearningRoadmapPage() {
      useSeo(
        "Machine Learning Roadmap 2026: From Beginner to AI Engineer",
        "The complete machine learning roadmap for 2026. Go from absolute beginner to AI engineer — covering Python, math fundamentals, classical ML, deep learning, LLMs, and real-world ML projects."
      );
      return (
        <div className="max-w-3xl mx-auto px-4 py-10 text-gray-100">
          <div className="mb-8">
            <span className="inline-block bg-green-600/20 text-green-400 text-xs font-semibold px-3 py-1 rounded-full mb-4">ML Roadmap 2026</span>
            <h1 className="text-3xl font-bold text-white mb-4">Machine Learning Roadmap 2026: From Beginner to AI Engineer</h1>
            <p className="text-gray-400 text-lg leading-relaxed">The bottom-up machine learning path — starting from Python and math, through classical ML and deep learning, up to LLMs and production deployment. Best for people who want to understand <em>how</em> models work, not just how to call them.</p>
          </div>

          {/* ML vs AI Engineer distinction — upfront */}
          <div className="bg-gray-800/50 rounded-xl p-5 mb-8 border border-white/8">
            <h2 className="text-base font-semibold text-white mb-4">ML Roadmap vs AI Engineer Roadmap — Which one?</h2>
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="bg-green-900/20 border border-green-500/20 rounded-lg p-4">
                <p className="text-green-400 font-semibold text-sm mb-2">This page — ML Roadmap</p>
                <ul className="space-y-1 text-xs text-gray-300">
                  <li>▸ Starts from Python + math basics</li>
                  <li>▸ Classical ML: sklearn, XGBoost, SVMs</li>
                  <li>▸ Deep learning: PyTorch, CNNs, transformers</li>
                  <li>▸ Training and fine-tuning from scratch</li>
                  <li>▸ MLOps: W&B, MLflow, drift monitoring</li>
                  <li className="text-green-400 mt-2">→ Goal: ML Engineer or Research Engineer</li>
                </ul>
              </div>
              <div className="bg-blue-900/20 border border-blue-500/20 rounded-lg p-4">
                <p className="text-blue-400 font-semibold text-sm mb-2"><a href="/ai-roadmap/" className="hover:underline">AI Engineer Roadmap →</a></p>
                <ul className="space-y-1 text-xs text-gray-300">
                  <li>▸ Starts from LLM APIs (faster ramp-up)</li>
                  <li>▸ Prompt engineering, RAG, agents</li>
                  <li>▸ Building products with pre-trained models</li>
                  <li>▸ LoRA fine-tuning (not from scratch)</li>
                  <li>▸ LangChain, vector databases, deployment</li>
                  <li className="text-blue-400 mt-2">→ Goal: AI Engineer, LLM App Developer</li>
                </ul>
              </div>
            </div>
            <p className="text-gray-500 text-xs mt-3">Not sure? If you want to build AI products quickly, start with the AI Engineer Roadmap. If you want to understand the internals and can invest 12+ months, the ML Roadmap gives you a stronger foundation.</p>
          </div>

          <h2 className="text-2xl font-bold text-white mb-5">The ML Roadmap: 6 Stages</h2>
          <div className="space-y-4 mb-10">
            {[
              { n: 1, title: "Python & Math Prerequisites", time: "4–6 weeks", color: "from-green-500 to-emerald-600", items: ["NumPy, Pandas, Matplotlib — the daily toolkit", "Linear algebra: vectors, matrices, dot products (intuition-first)", "Probability: distributions, Bayes' theorem, conditional probability", "Calculus: derivatives and gradients — what 'learning' actually is", "Start: fast.ai Part 1 or Google ML Crash Course (both free)"] },
              { n: 2, title: "Classical Machine Learning", time: "6–8 weeks", color: "from-teal-500 to-cyan-600", items: ["Linear & logistic regression — the building blocks", "Decision trees, random forests, gradient boosting (XGBoost)", "Unsupervised: k-means clustering, PCA for dimensionality reduction", "Evaluation: cross-validation, ROC-AUC, precision/recall, confusion matrix", "Project: enter a Kaggle structured data competition (Titanic, House Prices)"] },
              { n: 3, title: "Deep Learning & Neural Networks", time: "6–8 weeks", color: "from-blue-500 to-indigo-600", items: ["Forward pass, backpropagation, gradient descent — implemented from scratch", "CNNs for images, RNNs/LSTMs for sequences", "Transformer architecture: attention, positional encoding, MLP layers", "PyTorch training loop: dataset, dataloader, optimizer, scheduler", "Resource: Karpathy's Neural Networks: Zero to Hero (free, YouTube)"] },
              { n: 4, title: "Large Language Models", time: "4–6 weeks", color: "from-purple-500 to-violet-600", items: ["Pre-training: data pipeline, tokenization (BPE), training objective", "Supervised Fine-Tuning (SFT) and RLHF at conceptual + practical level", "LoRA / QLoRA fine-tuning with Unsloth on free Google Colab GPUs", "Evaluation: perplexity, BLEU, benchmarks, human eval frameworks", "Connect to the AI Engineer skills: APIs, RAG, agents"] },
              { n: 5, title: "MLOps & Production", time: "4–6 weeks", color: "from-orange-500 to-amber-600", items: ["Experiment tracking: Weights & Biases (free tier — use it from day one)", "Model serving: FastAPI + Docker, or BentoML for higher-level abstractions", "Data versioning with DVC — treat data like code", "Monitoring: Evidently AI for data drift and concept drift detection", "CI/CD for ML with GitHub Actions"] },
              { n: 6, title: "Portfolio Projects", time: "Ongoing", color: "from-rose-500 to-pink-600", items: ["Kaggle competition: submit, iterate, read top notebooks, improve", "End-to-end project: collect data → model → API → deployed app with monitoring", "LLM fine-tune: domain adaptation on a dataset you care about", "Write one technical blog post per project — forces clarity and signals depth"] },
            ].map(({ n, title, time, color, items }) => (
              <div key={n} className="bg-gray-800/40 rounded-xl p-5 border border-white/8">
                <div className="flex items-start gap-4">
                  <div className={`w-9 h-9 rounded-lg bg-gradient-to-br ${color} flex items-center justify-center text-white font-bold text-sm flex-shrink-0`}>{n}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-white font-semibold text-sm">{title}</h3>
                      <span className="text-xs text-gray-500">{time}</span>
                    </div>
                    <ul className="space-y-1">
                      {items.map((item, i) => <li key={i} className="text-gray-300 text-sm flex items-start gap-2"><span className="text-green-400 mt-0.5 flex-shrink-0">▸</span>{item}</li>)}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Timeline */}
          <div className="bg-gray-800/50 rounded-xl p-5 mb-8 border border-white/8">
            <h2 className="text-lg font-semibold text-white mb-3">Timeline</h2>
            <div className="flex items-center gap-2 text-sm text-gray-300 mb-4">
              <span className="text-green-400 font-semibold">~12 months</span>
              <span className="text-gray-500">at 6–8 hrs / week</span>
              <span className="text-gray-600 mx-1">·</span>
              <span className="text-blue-400 font-semibold">~18 months</span>
              <span className="text-gray-500">at 4 hrs / week</span>
            </div>
            <p className="text-gray-400 text-sm">This is the longer but more thorough path. The ML roadmap builds understanding from first principles — slower than jumping straight to LLM APIs, but the foundation pays off when debugging hard problems and designing novel systems.</p>
          </div>

          {/* Go deeper */}
          <div className="bg-gray-800/40 rounded-xl p-5 border border-white/8">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Related</h2>
            <div className="grid sm:grid-cols-2 gap-3">
              {[
                ["/ai-roadmap/", "AI Engineer Roadmap", "Faster path focused on LLM apps, RAG, and agents"],
                ["/llm-roadmap/", "LLM Roadmap", "Deep dive into large language models specifically"],
                ["/resources/", "Learning Resources", "Best free books and courses by phase"],
                ["/ai-projects/", "AI Project Ideas", "10 projects to build your ML/AI portfolio"],
              ].map(([href, title, desc]) => (
                <a key={href} href={href} className="block bg-gray-900/50 rounded-lg p-3 hover:bg-gray-700/50 transition-colors">
                  <p className="text-green-400 text-sm font-medium mb-0.5">{title} →</p>
                  <p className="text-gray-500 text-xs">{desc}</p>
                </a>
              ))}
            </div>
          </div>
        </div>
      );
    }

    // ═══════════════════════════════════════════════════════════
    // AI PROJECTS LANDING PAGE
    // ═══════════════════════════════════════════════════════════
    function AiProjectsPage() {
      useSeo(
        "AI Projects for Developers 2026: 10 Hands-On Ideas with Code",
        "10 hands-on AI project ideas for developers in 2026 — from beginner to advanced. Build a RAG chatbot, AI agent, fine-tuned LLM, and more with free resources and tech stack guidance."
      );
      const projects = [
        { n: 1, level: "Beginner", levelColor: "bg-green-600/20 text-green-400", title: "PDF Q&A Chatbot with RAG", time: "4–8 hours", tech: "LangChain · ChromaDB · Claude API", desc: "Build a chatbot that answers questions from your own PDF documents. This is the classic starter RAG project — it teaches document loading, chunking, vector embeddings, and LLM-powered retrieval in one build.", skills: ["LangChain document loaders", "Text chunking and vector embeddings", "ChromaDB vector store", "RetrievalQA chain"] },
        { n: 2, level: "Beginner", levelColor: "bg-green-600/20 text-green-400", title: "AI Code Review CLI Tool", time: "2–4 hours", tech: "OpenAI API · Python · Few-shot prompting", desc: "Build a CLI tool that accepts a code diff or file and returns a structured code review. Uses few-shot prompting to define the review format. Great for learning structured output and system prompts.", skills: ["Few-shot prompting", "Structured JSON output", "System prompt design", "Python CLI with argparse"] },
        { n: 3, level: "Intermediate", levelColor: "bg-blue-600/20 text-blue-400", title: "Web Research Agent", time: "1–2 days", tech: "LangChain · Tavily Search · Claude API", desc: "Build a ReACT agent that searches the web, fetches and reads pages, and writes structured research summaries. This project teaches tool use, the ReACT loop, and multi-step agent patterns.", skills: ["ReACT (Reason + Act + Observe) loop", "Tool calling and function calling", "LangGraph agent implementation", "Output structured reports"] },
        { n: 4, level: "Intermediate", levelColor: "bg-blue-600/20 text-blue-400", title: "Voice-to-Text Meeting Summarizer", time: "4–8 hours", tech: "OpenAI Whisper · Claude API · Gradio", desc: "Combine Whisper (speech recognition) with Claude to transcribe audio files and generate structured meeting summaries with action items. Teaches multimodal pipelines.", skills: ["OpenAI Whisper transcription", "Prompt engineering for summaries", "Gradio web interface", "Audio file processing"] },
        { n: 5, level: "Intermediate", levelColor: "bg-blue-600/20 text-blue-400", title: "Personal Knowledge Base AI", time: "1–2 days", tech: "LlamaIndex · Obsidian / Markdown · Local LLM", desc: "Build a RAG system over your own Obsidian notes or markdown files. Chat with your personal knowledge base using a local Llama model via Ollama — fully private, no API costs.", skills: ["LlamaIndex ingestion pipeline", "Markdown document processing", "Ollama local LLM integration", "Incremental document updates"] },
        { n: 6, level: "Advanced", levelColor: "bg-orange-600/20 text-orange-400", title: "Fine-tuned Domain LLM", time: "2–3 days", tech: "Unsloth · QLoRA · Google Colab · Ollama", desc: "Fine-tune Llama 3 8B on a custom domain dataset (legal docs, medical text, code, etc.) using QLoRA on free Google Colab GPUs. Deploy locally with Ollama.", skills: ["Dataset preparation in Alpaca/ShareGPT format", "QLoRA fine-tuning with Unsloth", "LoRA adapter merging", "Evaluation with perplexity and human evals"] },
        { n: 7, level: "Advanced", levelColor: "bg-orange-600/20 text-orange-400", title: "Multi-Agent Research System", time: "3–5 days", tech: "LangGraph · Claude API · Custom Tools", desc: "Build an orchestrator + worker multi-agent system. The planner agent breaks a complex research task into subtasks and delegates to specialized worker agents (web search, data analysis, summarization).", skills: ["LangGraph multi-agent orchestration", "Shared state and message passing", "Agent-to-agent delegation patterns", "Parallelization of agent tasks"] },
        { n: 8, level: "Intermediate", levelColor: "bg-blue-600/20 text-blue-400", title: "Natural Language SQL Generator", time: "4–8 hours", tech: "OpenAI API · SQLite · LangChain · Gradio", desc: "Build a tool that converts natural language questions into SQL queries for a database. The LLM reads the schema and generates accurate SQL. Teaches structured output and database integration.", skills: ["Schema-aware prompting", "SQL generation and validation", "LangChain SQL toolkit", "Error recovery and query retrying"] },
        { n: 9, level: "Intermediate", levelColor: "bg-blue-600/20 text-blue-400", title: "LLM Evaluation Dashboard", time: "1–2 days", tech: "RAGAS · Streamlit · LangChain · Any LLM", desc: "Build a dashboard to evaluate RAG pipeline quality using RAGAS metrics: faithfulness, answer relevancy, and context precision. Compare different chunking strategies and LLMs.", skills: ["RAGAS evaluation framework", "Faithfulness and relevancy metrics", "Streamlit dashboard", "A/B testing RAG configurations"] },
        { n: 10, level: "Advanced", levelColor: "bg-orange-600/20 text-orange-400", title: "AI Coding Assistant (Cursor Clone)", time: "1–2 weeks", tech: "Claude API · AST parsing · LSP · VS Code extension", desc: "Build a VS Code extension powered by Claude that reads your codebase, understands your code with AST parsing, and suggests context-aware completions and refactors.", skills: ["VS Code extension development", "Codebase indexing and RAG over code", "AST parsing for code understanding", "Claude API streaming completions"] },
      ];
      return (
        <div className="max-w-3xl mx-auto px-4 py-10 text-gray-100">
          <div className="mb-8">
            <span className="inline-block bg-orange-600/20 text-orange-400 text-xs font-semibold px-3 py-1 rounded-full mb-4">AI Projects 2026</span>
            <h1 className="text-3xl font-bold text-white mb-4">AI Projects for Developers 2026: 10 Hands-On Ideas with Code</h1>
            <p className="text-gray-400 text-lg leading-relaxed">The fastest way to learn AI engineering is to build real projects. These 10 AI project ideas range from beginner-friendly (4 hours) to advanced (1–2 weeks), with tech stack guidance, key skills, and estimated build time for each.</p>
          </div>

          <div className="flex gap-3 mb-8 flex-wrap">
            {[["3 Beginner", "bg-green-600/20 text-green-400"], ["4 Intermediate", "bg-blue-600/20 text-blue-400"], ["3 Advanced", "bg-orange-600/20 text-orange-400"]].map(([label, cls]) => (
              <span key={label} className={`text-xs font-semibold px-3 py-1.5 rounded-full ${cls}`}>{label}</span>
            ))}
          </div>

          <div className="space-y-5 mb-10">
            {projects.map(({ n, level, levelColor, title, time, tech, desc, skills }) => (
              <div key={n} className="bg-gray-800/40 rounded-xl p-5 border border-white/8">
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="flex items-center gap-3">
                    <span className="text-gray-500 font-bold text-sm w-6">{n}.</span>
                    <h3 className="text-white font-semibold">{title}</h3>
                  </div>
                  <span className={`text-xs font-semibold px-2 py-1 rounded-full flex-shrink-0 ${levelColor}`}>{level}</span>
                </div>
                <div className="ml-9">
                  <div className="flex gap-4 mb-2 text-xs">
                    <span className="text-gray-500">⏱ {time}</span>
                    <span className="text-gray-500">🔧 {tech}</span>
                  </div>
                  <p className="text-gray-400 text-sm mb-3">{desc}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {skills.map((s, i) => <span key={i} className="bg-gray-700/50 text-gray-300 text-xs px-2 py-0.5 rounded">{s}</span>)}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="bg-gray-800/50 rounded-xl p-6 mb-8 border border-white/8">
            <h2 className="text-xl font-semibold text-white mb-4">How to Build AI Projects Effectively</h2>
            <ul className="space-y-3">
              {["Start with the simplest version that works. Add complexity iteratively.", "Document as you go — write a README and commit history that tells the story.", "Evaluate your AI outputs systematically, not just via vibe checks.", "Deploy publicly (Hugging Face Spaces, Vercel, Railway) — real users catch real bugs.", "Write a short blog post or X/LinkedIn thread about what you built and what you learned."].map((tip, i) => (
                <li key={i} className="text-gray-300 text-sm flex items-start gap-2"><span className="text-orange-400 mt-0.5">▸</span>{tip}</li>
              ))}
            </ul>
          </div>

          <div className="mt-8 p-5 bg-gray-800/40 rounded-xl border border-white/8">
            <h2 className="text-lg font-semibold text-white mb-3">Learn the Skills to Build These Projects</h2>
            <p className="text-gray-400 text-sm mb-4">Our AI engineer roadmap covers all the skills you need to build these projects — with curated free resources and milestone projects for each phase.</p>
            <div className="flex flex-wrap gap-3">
              <a href="/" className="inline-flex items-center gap-2 bg-orange-600 hover:bg-orange-500 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors">View AI Roadmap →</a>
              <a href="/rag-tutorial/" className="inline-flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors">RAG Tutorial →</a>
            </div>
          </div>
        </div>
      );
    }

    // ─── MASTER APP ───
    const TABS = [
      { label: "Roadmap",         slug: "roadmap",               icon: BrainCircuit, Component: Roadmap },
      { label: "Prep Plan",       slug: "prep-plan",             icon: Calendar,     Component: PrepPlan },
      { label: "GenAI Guide",     slug: "genai-guide",           icon: Cpu,          Component: GenAIGuide },
      { label: "Prompt Eng",      slug: "prompt-eng",            icon: Zap,          Component: PromptEngineering },
      { label: "Resources",       slug: "resources",             icon: BookOpen,     Component: AltResources },
      { label: "Readiness",       slug: "readiness",             icon: CheckCircle,  Component: ReadinessChecker },
      { label: "Beyond Roadmap",  slug: "beyond-roadmap",        icon: Compass,      Component: BeyondRoadmap },
      { label: "Assessment",      slug: "assessment",            icon: BarChart2,    Component: KnowledgeAssessment },
      { label: "AI Roadmap",      slug: "ai-roadmap",            icon: BrainCircuit, Component: AiRoadmapPage },
      { label: "LLM Roadmap",     slug: "llm-roadmap",           icon: Cpu,          Component: LlmRoadmapPage },
      { label: "RAG Tutorial",    slug: "rag-tutorial",          icon: BookOpen,     Component: RagTutorialPage },
      { label: "ML Roadmap",      slug: "machine-learning-roadmap", icon: BarChart2, Component: MachineLearningRoadmapPage },
      { label: "AI Projects",     slug: "ai-projects",           icon: Rocket,       Component: AiProjectsPage },
      { label: "Blog",            slug: "blog",                  icon: BookOpen,     Component: BlogIndexPage },
    ];

    function Footer() {
      return (
        <footer className="border-t border-white/8 bg-gray-950/80 backdrop-blur-sm mt-8">
          <div className="max-w-3xl mx-auto px-4 py-8">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
              <div>
                <p className="text-white font-semibold text-sm mb-1">ailearnings.in</p>
                <p className="text-gray-400 text-xs">Free AI roadmap for software developers</p>
                <div className="flex items-center gap-4 mt-3">
                  <a href="https://github.com/amit352" target="_blank" rel="noopener noreferrer"
                    className="text-gray-400 hover:text-white transition-colors flex items-center gap-1.5" aria-label="GitHub">
                    <Github size={16}/>
                    <span className="text-xs">@amit352</span>
                  </a>
                  <a href="https://github.com/amit352/ailearnings/discussions" target="_blank" rel="noopener noreferrer"
                    className="text-gray-400 hover:text-white transition-colors flex items-center gap-1.5">
                    <MessageSquare size={14}/>
                    <span className="text-xs">Discussions</span>
                  </a>
                  <a href="https://github.com/sponsors/amit352" target="_blank" rel="noopener noreferrer"
                    className="text-pink-400 hover:text-pink-300 transition-colors flex items-center gap-1.5">
                    <Heart size={14}/>
                    <span className="text-xs">Support</span>
                  </a>
                </div>
              </div>
              <div className="flex flex-col items-start md:items-end gap-3">
                <div className="flex flex-wrap gap-3">
                  {["Roadmap","Prep Plan","Resources","Readiness"].map((label) => {
                    const tab = TABS.find(t => t.label === label);
                    return tab ? (
                      <a key={label} href={tabPath(tab.slug)}
                        className="text-xs text-gray-400 hover:text-white transition-colors">
                        {label}
                      </a>
                    ) : null;
                  })}
                </div>
                <p className="text-gray-400 text-xs">Last updated: March 2026 · © 2026 ailearnings.in</p>
                <p className="text-gray-400 text-xs">Built for the dev community · No ads · Always free</p>
              </div>
            </div>
          </div>
        </footer>
      );
    }


    function ScrollToTop() {
      const [visible, setVisible] = React.useState(false);
      React.useEffect(() => {
        const onScroll = () => setVisible(window.scrollY > 400);
        window.addEventListener("scroll", onScroll, { passive: true });
        return () => window.removeEventListener("scroll", onScroll);
      }, []);
      if (!visible) return null;
      return (
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          aria-label="Scroll to top"
          className="fixed bottom-6 right-6 z-50 bg-gray-800 hover:bg-gray-700 text-white rounded-full p-2.5 shadow-lg transition-colors">
          <ArrowUp size={16}/>
        </button>
      );
    }

    function tabPath(slug) {
      return slug === "roadmap" ? "/" : "/" + slug + "/";
    }

    function App() {
      const getInitialTab = () => {
        const p = window.location.pathname.replace(/\/+$/, "") || "";
        const slug = p.replace(/^\//, "") || "roadmap";
        const idx = TABS.findIndex(t => t.slug === slug);
        return idx >= 0 ? idx : 0;
      };

      const [activeTab, setActiveTab] = React.useState(getInitialTab);
      const [menuOpen, setMenuOpen] = React.useState(false);
      const { Component } = TABS[activeTab];

      const handleTabClick = (i, e) => {
        if (e) e.preventDefault();
        setActiveTab(i);
        setMenuOpen(false);
        history.pushState(null, "", tabPath(TABS[i].slug));
        window.scrollTo({ top: 0, behavior: "smooth" });
      };

      React.useEffect(() => {
        const onPopState = () => {
          const p = window.location.pathname.replace(/\/+$/, "") || "";
          const slug = p.replace(/^\//, "") || "roadmap";
          const idx = TABS.findIndex(t => t.slug === slug);
          if (idx >= 0) { setActiveTab(idx); window.scrollTo({ top: 0, behavior: "smooth" }); }
        };
        window.addEventListener("popstate", onPopState);
        return () => window.removeEventListener("popstate", onPopState);
      }, []);

      return (
        <div>
          {/* Navbar */}
          <nav className="sticky top-0 z-50 bg-gray-900/70 backdrop-blur-md border-b border-white/8">

            {/* Desktop: horizontal tabs */}
            <div className="hidden md:flex max-w-full mx-auto gap-1 px-3 py-2 overflow-x-auto">
              {TABS.map((t, i) => (
                <a key={i} href={tabPath(t.slug)} onClick={(e) => handleTabClick(i, e)}
                  aria-current={activeTab === i ? "page" : undefined}
                  className={`text-xs px-3 py-1.5 rounded-lg whitespace-nowrap transition-colors flex-shrink-0 flex items-center gap-1.5 ${
                    activeTab === i
                      ? "bg-blue-600 text-white font-semibold shadow-[0_0_12px_rgba(59,130,246,0.4)]"
                      : "text-gray-400 hover:text-white hover:bg-gray-800"
                  }`}>
                  <t.icon size={13} strokeWidth={2} />
                  {t.label}
                </a>
              ))}
            </div>

            {/* Mobile: current tab + hamburger */}
            <div className="md:hidden flex items-center justify-between px-4 py-3">
              <span className="text-sm font-semibold text-white flex items-center gap-1.5">
                {React.createElement(TABS[activeTab].icon, {size: 14})}
                {TABS[activeTab].label}
              </span>
              <button onClick={() => setMenuOpen(!menuOpen)}
                aria-label={menuOpen ? "Close menu" : "Open menu"}
                aria-expanded={menuOpen}
                className="text-gray-400 hover:text-white p-1 rounded-lg hover:bg-gray-800 transition-colors">
                {menuOpen ? <X size={20} /> : <Menu size={20} />}
              </button>
            </div>

            {/* Mobile dropdown menu */}
            {menuOpen && (
              <div className="md:hidden border-t border-white/8 bg-gray-900/95 backdrop-blur-md">
                {TABS.map((t, i) => (
                  <a key={i} href={tabPath(t.slug)} onClick={(e) => handleTabClick(i, e)}
                    aria-current={activeTab === i ? "page" : undefined}
                    className={`w-full text-left px-4 py-3 text-sm transition-colors border-b border-gray-800/50 flex items-center gap-2.5 ${
                      activeTab === i
                        ? "bg-blue-600/20 text-blue-400 font-semibold"
                        : "text-gray-400 hover:text-white hover:bg-gray-800"
                    }`}>
                    <t.icon size={15} />
                    {t.label}
                  </a>
                ))}
              </div>
            )}
          </nav>

          <main><div key={activeTab}><Component /></div></main>
          <Footer />
          <ScrollToTop />
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById("root")).render(<App />);
