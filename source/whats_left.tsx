import { useState } from "react";

const covered = [
  { emoji: "🗺️", label: "AI Zero→Hero Roadmap" },
  { emoji: "📚", label: "Books & Video Courses by Phase" },
  { emoji: "⚡", label: "Prompt Engineering Mastery" },
  { emoji: "🧬", label: "Generative AI (all 4 domains)" },
  { emoji: "🚦", label: "Phase Readiness Checker" },
  { emoji: "⚙️", label: "LLM Setup & Configuration" },
  { emoji: "🤖", label: "Agentic AI Deep Dive" },
  { emoji: "🏗️", label: "Building & Training LLMs" },
];

const topics = [
  {
    id: "mlops", emoji: "🔧", title: "MLOps & AI in Production",
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
    id: "evals", emoji: "📊", title: "AI Evaluation & Evals Engineering",
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
    id: "security", emoji: "🔒", title: "AI Security & Red Teaming",
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
    id: "multimodal", emoji: "👁️", title: "Multimodal AI (Vision + Language)",
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
    id: "reasoning", emoji: "🧠", title: "Reasoning Models & Thinking AI",
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
    id: "data", emoji: "🗄️", title: "Data Engineering for AI",
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
    id: "ondevice", emoji: "📱", title: "On-Device & Edge AI",
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
    id: "ai_safety", emoji: "🛡️", title: "AI Safety & Alignment",
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
    id: "career", emoji: "💼", title: "AI Career Paths & Roles",
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
    id: "research", emoji: "🔬", title: "Reading & Understanding AI Research",
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

export default function WhatsLeft() {
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
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
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
              <div key={i} className="flex items-center gap-2 text-xs text-gray-500">
                <span className="text-green-700">✓</span>
                <span>{c.emoji} {c.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Filter */}
        <div className="flex gap-1 flex-wrap mb-4">
          {filters.map(f => (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${filter === f.key ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300 bg-gray-900 border border-gray-800"}`}>
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
                <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${t.color} flex items-center justify-center text-base flex-shrink-0`}>{t.emoji}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <p className="font-semibold text-sm">{t.title}</p>
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${t.tagColor}`}>{t.tag}</span>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${depthColors[t.depth]}`}>{depthLabels[t.depth]}</span>
                </div>
                <span className="text-gray-600 flex-shrink-0">{open === t.id ? "▲" : "▼"}</span>
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
                          <span className="text-gray-600 flex-shrink-0 mt-0.5">•</span>
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
