import { useState } from "react";

const FREE = "free";
const PAID = "paid";
const OREILLY = "oreilly";

const Badge = ({ type }) => {
  const map = {
    free: "bg-green-900 text-green-300 border border-green-700",
    paid: "bg-gray-800 text-gray-400 border border-gray-600",
    oreilly: "bg-orange-900 text-orange-300 border border-orange-700",
  };
  const label = { free: "Free", paid: "~$40–60", oreilly: "O'Reilly" };
  return <span className={`text-xs px-2 py-0.5 rounded-full ${map[type]}`}>{label[type]}</span>;
};

const phases = [
  {
    id: 1,
    emoji: "🌱",
    title: "Phase 1 – AI Foundations",
    color: "from-green-500 to-emerald-600",
    books: [
      {
        title: "Hands-On Large Language Models",
        authors: "Jay Alammar & Maarten Grootendorst",
        publisher: "O'Reilly, 2024",
        type: OREILLY,
        why: "The #1 beginner book for devs. Visual, intuitive, covers LLM internals without heavy math. Perfect Phase 1 read.",
        url: "https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/",
        rating: "⭐ 4.7",
      },
      {
        title: "The Hundred-Page Language Models Book",
        authors: "Andriy Burkov",
        publisher: "True Positive Inc., 2025",
        type: PAID,
        why: "Dense but accessible. Great conceptual coverage of transformers, training, and inference in under 200 pages.",
        url: "https://www.amazon.com/Hundred-Page-Language-Models-Book/dp/B0F2BKFQSS",
        rating: "⭐ 4.8",
      },
    ],
    videos: [
      {
        title: "Neural Networks: Zero to Hero",
        author: "Andrej Karpathy (YouTube)",
        type: FREE,
        why: "The gold standard. Builds intuition from scratch. Watch this before anything else.",
        url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ",
        duration: "~10 hrs total",
      },
      {
        title: "Intro to Large Language Models",
        author: "Andrej Karpathy (YouTube)",
        type: FREE,
        why: "1-hour masterclass on how LLMs work. Best single video for beginners.",
        url: "https://www.youtube.com/watch?v=zjkBMFhNj_g",
        duration: "1 hr",
      },
      {
        title: "Generative AI for Beginners",
        author: "Microsoft (GitHub + YouTube)",
        type: FREE,
        why: "18 structured lessons covering GenAI fundamentals, prompting, RAG, agents. Beginner friendly.",
        url: "https://github.com/microsoft/generative-ai-for-beginners",
        duration: "Self-paced",
      },
    ],
  },
  {
    id: 2,
    emoji: "⚙️",
    title: "Phase 2 – LLM Setup & Configuration",
    color: "from-slate-500 to-gray-600",
    books: [
      {
        title: "Designing Large Language Model Applications",
        authors: "Suhas Pai",
        publisher: "O'Reilly, 2024",
        type: OREILLY,
        why: "Covers loading LLMs, decoding strategies (top-k, top-p, beam search), quantization, Ollama, and HF Accelerate. Exactly what Phase 2 needs.",
        url: "https://www.oreilly.com/library/view/designing-large-language/9781098150495/",
        rating: "⭐ 4.5",
      },
    ],
    videos: [
      {
        title: "Open Source Models with Hugging Face",
        author: "DeepLearning.AI (free short course)",
        type: FREE,
        why: "Covers loading, configuring, and running open-source models. Directly addresses setup needs.",
        url: "https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/",
        duration: "1.5 hrs",
      },
      {
        title: "Running LLMs Locally with Ollama (freeCodeCamp)",
        author: "freeCodeCamp (YouTube)",
        type: FREE,
        why: "Practical walkthrough on setting up local models, quantization, and config params.",
        url: "https://www.youtube.com/results?search_query=ollama+local+llm+freecodecamp",
        duration: "~2 hrs",
      },
    ],
  },
  {
    id: 3,
    emoji: "🔧",
    title: "Phase 3 – Prompt Engineering & LLM APIs",
    color: "from-blue-500 to-indigo-600",
    books: [
      {
        title: "Prompt Engineering for Generative AI",
        authors: "James Phoenix & Mike Taylor",
        publisher: "O'Reilly, 2024",
        type: OREILLY,
        why: "The definitive O'Reilly book on prompting. Covers zero-shot, few-shot, CoT, role prompting, and advanced patterns. Rated 4.5★.",
        url: "https://www.oreilly.com/library/view/prompt-engineering-for/9781098153427/",
        rating: "⭐ 4.5",
      },
      {
        title: "AI Engineering",
        authors: "Chip Huyen",
        publisher: "O'Reilly, 2025",
        type: OREILLY,
        why: "The most-read book on O'Reilly since launch. Covers the full AI app stack — evaluation, RAG, agents, fine-tuning. A must-read across Phases 3–6.",
        url: "https://www.oreilly.com/library/view/ai-engineering/9781098166298/",
        rating: "⭐ 4.7",
      },
    ],
    videos: [
      {
        title: "ChatGPT Prompt Engineering for Developers",
        author: "DeepLearning.AI (free short course)",
        type: FREE,
        why: "The canonical prompt engineering course. Covers all techniques with hands-on notebooks.",
        url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/",
        duration: "1.5 hrs",
      },
      {
        title: "Building Systems with the ChatGPT API",
        author: "DeepLearning.AI (free short course)",
        type: FREE,
        why: "Goes from prompting to building multi-step LLM-powered apps. Great bridge into real development.",
        url: "https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/",
        duration: "1.5 hrs",
      },
    ],
  },
  {
    id: 4,
    emoji: "📚",
    title: "Phase 4 – RAG & Working with Data",
    color: "from-purple-500 to-violet-600",
    books: [
      {
        title: "Designing Large Language Model Applications",
        authors: "Suhas Pai",
        publisher: "O'Reilly, 2024",
        type: OREILLY,
        why: "Has dedicated chapters on RAG pipelines, chunking, embedding strategies, vector DBs, and RAG vs fine-tuning comparisons.",
        url: "https://www.oreilly.com/library/view/designing-large-language/9781098150495/",
        rating: "⭐ 4.5",
      },
      {
        title: "Building AI Agents with LLMs, RAG, and Knowledge Graphs",
        authors: "Multiple Authors",
        publisher: "O'Reilly / Packt, 2024",
        type: OREILLY,
        why: "Covers naive RAG, advanced RAG (chunking, embedding strategies, evaluation), and how RAG compares to fine-tuning.",
        url: "https://www.oreilly.com/library/view/building-ai-agents/9781835087060/",
        rating: "⭐ 4.4",
      },
    ],
    videos: [
      {
        title: "LangChain: Chat with Your Data",
        author: "DeepLearning.AI (free short course)",
        type: FREE,
        why: "Hands-on RAG from document loading to retrieval and generation. The best intro to RAG in practice.",
        url: "https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/",
        duration: "1.5 hrs",
      },
      {
        title: "Building & Evaluating Advanced RAG",
        author: "DeepLearning.AI (free short course)",
        type: FREE,
        why: "Goes deeper — sentence-window retrieval, auto-merging, evaluation metrics. Great follow-up.",
        url: "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/",
        duration: "1.5 hrs",
      },
      {
        title: "Generative AI with LLMs (Coursera/AWS)",
        author: "DeepLearning.AI + AWS",
        type: FREE,
        why: "3-week course covering transformers, RAG, fine-tuning, and deployment. Audit for free.",
        url: "https://www.coursera.org/learn/generative-ai-with-llms",
        duration: "~16 hrs",
      },
    ],
  },
  {
    id: 5,
    emoji: "🤖",
    title: "Phase 5 – Agentic AI",
    color: "from-orange-500 to-amber-600",
    books: [
      {
        title: "Building Applications with AI Agents",
        authors: "Multiple Authors",
        publisher: "O'Reilly, 2025",
        type: OREILLY,
        why: "Most up-to-date O'Reilly agent book. Covers LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, multi-agent coordination, and real-world use cases.",
        url: "https://www.oreilly.com/library/view/building-applications-with/9781098176495/",
        rating: "⭐ 4.5",
      },
      {
        title: "Building AI Agents with LLMs, RAG, and Knowledge Graphs",
        authors: "Multiple Authors",
        publisher: "O'Reilly / Packt, 2024",
        type: OREILLY,
        why: "Covers agent frameworks (LangChain, LlamaIndex, AutoGen), tool usage, planning, and multi-agent systems end-to-end.",
        url: "https://www.oreilly.com/library/view/building-ai-agents/9781835087060/",
        rating: "⭐ 4.4",
      },
    ],
    videos: [
      {
        title: "AI Agents in LangGraph",
        author: "DeepLearning.AI (free short course)",
        type: FREE,
        why: "Builds ReACT agents, tool-calling agents, and multi-agent systems using LangGraph. Hands-on and practical.",
        url: "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/",
        duration: "2 hrs",
      },
      {
        title: "AI Agentic Design Patterns with AutoGen",
        author: "Microsoft / DeepLearning.AI (free)",
        type: FREE,
        why: "Covers orchestrator-worker, reflection, and multi-agent patterns using Microsoft's AutoGen framework.",
        url: "https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/",
        duration: "1 hr",
      },
      {
        title: "Full Stack LLM Bootcamp – Agents & Tools",
        author: "Full Stack Deep Learning (YouTube, free)",
        type: FREE,
        why: "University-quality lectures covering agents, tool use, and production patterns. Free on YouTube.",
        url: "https://www.youtube.com/@FullStackDeepLearning",
        duration: "Self-paced",
      },
    ],
  },
  {
    id: 6,
    emoji: "🏗️",
    title: "Phase 6 – Building & Training LLMs",
    color: "from-rose-500 to-pink-600",
    books: [
      {
        title: "Build a Large Language Model (From Scratch)",
        authors: "Sebastian Raschka",
        publisher: "Manning, 2024",
        type: PAID,
        why: "THE book for understanding LLM internals by building one from scratch in PyTorch. Covers tokenization, attention, pre-training, SFT, RLHF. Rated 4.6★ — best in class for this topic.",
        url: "https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167",
        rating: "⭐ 4.6",
      },
      {
        title: "LLM Engineer's Handbook",
        authors: "Paul Iusztin & Maxime Labonne",
        publisher: "Packt, 2024",
        type: PAID,
        why: "Covers LLM engineering end-to-end: fine-tuning with LoRA/QLoRA, RAG, evaluation, and production deployment. Highly practical.",
        url: "https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072",
        rating: "⭐ 4.6",
      },
      {
        title: "Natural Language Processing with Transformers",
        authors: "Lewis Tunstall et al. (Hugging Face team)",
        publisher: "O'Reilly, 2022",
        type: OREILLY,
        why: "Still the best deep dive into transformer architecture and fine-tuning with Hugging Face. Foundational for anyone who wants to understand the mechanics.",
        url: "https://www.oreilly.com/library/view/natural-language-processing/9781098136789/",
        rating: "⭐ 4.6",
      },
    ],
    videos: [
      {
        title: "Let's Build GPT from Scratch",
        author: "Andrej Karpathy (YouTube)",
        type: FREE,
        why: "Legendary 2-hour video. Builds a GPT model from scratch in Python. Best hands-on transformer tutorial that exists.",
        url: "https://www.youtube.com/watch?v=kCc8FmEb1nY",
        duration: "2 hrs",
      },
      {
        title: "Finetuning Large Language Models",
        author: "DeepLearning.AI (free short course)",
        type: FREE,
        why: "Practical fine-tuning walkthrough covering SFT, LoRA, and evaluation.",
        url: "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/",
        duration: "1 hr",
      },
      {
        title: "Stanford CS336: Language Modeling from Scratch",
        author: "Stanford University (YouTube, free)",
        type: FREE,
        why: "Full university course on building LLMs from scratch. One of the most rigorous free resources available.",
        url: "https://www.youtube.com/@stanfordonline",
        duration: "Full course",
      },
    ],
  },
  {
    id: 7,
    emoji: "🚀",
    title: "Phase 7 – Production & Staying Current",
    color: "from-teal-500 to-cyan-600",
    books: [
      {
        title: "AI Engineering",
        authors: "Chip Huyen",
        publisher: "O'Reilly, 2025",
        type: OREILLY,
        why: "If you read one book across this entire roadmap, make it this. Covers evaluation, RAG, agents, fine-tuning, and production — all from a systems perspective. Most read book on O'Reilly in 2025.",
        url: "https://www.oreilly.com/library/view/ai-engineering/9781098166298/",
        rating: "⭐ 4.7",
      },
      {
        title: "Building LLMs for Production",
        authors: "Louis-François Bouchard & Louie Peters",
        publisher: "Independent, 2024",
        type: PAID,
        why: "Focuses on prompting, fine-tuning, RAG, and reliability in production. Practical and concise.",
        url: "https://www.amazon.com/Building-LLMs-Production-Reliability-Fine-Tuning/dp/B0D4FFPFW5",
        rating: "⭐ 4.4",
      },
    ],
    videos: [
      {
        title: "Latent Space Podcast",
        author: "swyx & Alessio (YouTube + Podcast)",
        type: FREE,
        why: "Best podcast to stay current on AI engineering. Interviews with researchers and engineers building real AI systems.",
        url: "https://www.latent.space/podcast",
        duration: "Ongoing",
      },
      {
        title: "Two Minute Papers",
        author: "Károly Zsolnai-Fehér (YouTube)",
        type: FREE,
        why: "Bite-sized breakdowns of the latest AI research papers. Great for keeping up with the field without reading papers yourself.",
        url: "https://www.youtube.com/@TwoMinutePapers",
        duration: "Ongoing",
      },
      {
        title: "Yannic Kilcher",
        author: "YouTube",
        type: FREE,
        why: "Deep technical paper walkthroughs. Best for understanding what's happening at the research frontier (transformers, reasoning models, etc.).",
        url: "https://www.youtube.com/@YannicKilcher",
        duration: "Ongoing",
      },
    ],
  },
];

const oreillyCost = {
  title: "💡 O'Reilly Subscription – Is It Worth It?",
  points: [
    ["Monthly plan", "~$50/month — cancellable anytime. Read 2–3 books in a month, cancel."],
    ["Annual plan", "~$500/year — better value if you're serious across all phases."],
    ["Free alternative", "Many O'Reilly books are available at local/national public libraries via apps like Libby or O'Reilly for Public Libraries."],
    ["Trial", "O'Reilly often offers 10-day free trials — enough to read 1 full book."],
    ["University access", "If you have any university affiliation, check if they provide O'Reilly access (many do, for free)."],
  ],
};

export default function AltResources() {
  const [open, setOpen] = useState(null);
  const [view, setView] = useState({});

  const toggle = (id) => setOpen(open === id ? null : id);
  const setV = (id, v, e) => { e.stopPropagation(); setView(p => ({ ...p, [id]: v })); };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-1">📖 ByteByteAI Alternative</h1>
          <p className="text-gray-400 text-sm">Best Books & Video Courses — Mapped by Phase</p>
          <div className="flex flex-wrap justify-center gap-2 mt-3">
            <span className="bg-green-900 text-green-300 border border-green-700 text-xs px-3 py-1 rounded-full">Free</span>
            <span className="bg-orange-900 text-orange-300 border border-orange-700 text-xs px-3 py-1 rounded-full">O'Reilly</span>
            <span className="bg-gray-800 text-gray-400 border border-gray-600 text-xs px-3 py-1 rounded-full">~$40–60 Book</span>
          </div>
        </div>

        {/* O'Reilly cost box */}
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

        {/* Phases */}
        <div className="space-y-3 mb-8">
          {phases.map((p) => {
            const v = view[p.id] || "books";
            return (
              <div key={p.id} className={`rounded-xl border cursor-pointer transition-all ${open === p.id ? "border-gray-500 bg-gray-900" : "border-gray-800 bg-gray-900 hover:border-gray-600"}`}
                onClick={() => toggle(p.id)}>
                <div className="flex items-center gap-3 p-4">
                  <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${p.color} flex items-center justify-center text-base flex-shrink-0`}>{p.emoji}</div>
                  <div className="flex-1">
                    <p className="font-semibold text-sm">{p.title}</p>
                    <p className="text-gray-500 text-xs mt-0.5">{p.books.length} book{p.books.length > 1 ? "s" : ""} · {p.videos.length} video course{p.videos.length > 1 ? "s" : ""}</p>
                  </div>
                  <span className="text-gray-600">{open === p.id ? "▲" : "▼"}</span>
                </div>

                {open === p.id && (
                  <div className="border-t border-gray-800">
                    {/* Tab switcher */}
                    <div className="flex gap-1 px-4 pt-3">
                      {["books", "videos"].map(t => (
                        <button key={t} onClick={e => setV(p.id, t, e)}
                          className={`text-xs px-4 py-1.5 rounded-lg capitalize transition-colors ${v === t ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"}`}>
                          {t === "books" ? "📚 Books" : "🎬 Video Courses"}
                        </button>
                      ))}
                    </div>

                    <div className="p-4 space-y-3">
                      {v === "books" && p.books.map((b, i) => (
                        <a key={i} href={b.url} target="_blank" rel="noopener noreferrer"
                          className="block bg-gray-800 hover:bg-gray-750 rounded-xl p-4 transition-colors border border-gray-700 hover:border-gray-500"
                          onClick={e => e.stopPropagation()}>
                          <div className="flex items-start justify-between gap-2 mb-1">
                            <p className="text-blue-400 font-semibold text-sm leading-snug">{b.title}</p>
                            <Badge type={b.type} />
                          </div>
                          <p className="text-gray-500 text-xs mb-2">{b.authors} · {b.publisher} · {b.rating}</p>
                          <p className="text-gray-300 text-xs leading-relaxed">{b.why}</p>
                        </a>
                      ))}

                      {v === "videos" && p.videos.map((vid, i) => (
                        <a key={i} href={vid.url} target="_blank" rel="noopener noreferrer"
                          className="block bg-gray-800 hover:bg-gray-750 rounded-xl p-4 transition-colors border border-gray-700 hover:border-gray-500"
                          onClick={e => e.stopPropagation()}>
                          <div className="flex items-start justify-between gap-2 mb-1">
                            <p className="text-blue-400 font-semibold text-sm leading-snug">{vid.title}</p>
                            <Badge type={vid.type} />
                          </div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-gray-500 text-xs">{vid.author}</p>
                            <span className="text-gray-700">·</span>
                            <p className="text-gray-500 text-xs">{vid.duration}</p>
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

        {/* Top 3 picks */}
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-6">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">🏆 If You Only Pick 3 Resources Total</p>
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
                  <p className="text-xs text-gray-500 mb-0.5">{r.label}</p>
                  <p className="text-blue-400 text-sm font-semibold">{r.rec}</p>
                  <p className="text-gray-400 text-xs mt-1">{r.why}</p>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* vs ByteByteAI */}
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">⚖️ This Path vs ByteByteAI</p>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "ByteByteAI", pros: ["Live cohort + community", "Instructor feedback", "Structured 6 weeks", "Certificate"], cons: ["Expensive", "Fixed schedule", "One-time coverage"] },
              { label: "This Path", pros: ["Mostly free or ~$100 total", "Self-paced", "Go deeper on any topic", "Referenceable forever"], cons: ["Self-discipline needed", "No live feedback", "No certificate"] },
            ].map((col, i) => (
              <div key={i} className="bg-gray-800 rounded-lg p-3">
                <p className="font-semibold text-sm text-white mb-2">{col.label}</p>
                {col.pros.map((p, j) => <p key={j} className="text-xs text-green-400 mb-1">✓ {p}</p>)}
                {col.cons.map((c, j) => <p key={j} className="text-xs text-gray-500 mb-1">✗ {c}</p>)}
              </div>
            ))}
          </div>
        </div>

        <p className="text-center text-gray-600 text-xs mt-6">Click each phase → toggle Books / Video Courses</p>
      </div>
    </div>
  );
}
