import { useState } from "react";

const levels = [
  { label: "Hobbyist", color: "bg-gray-600", pct: 0 },
  { label: "Beginner", color: "bg-blue-600", pct: 20 },
  { label: "Practitioner", color: "bg-indigo-500", pct: 40 },
  { label: "Engineer", color: "bg-purple-500", pct: 60 },
  { label: "Specialist", color: "bg-orange-500", pct: 80 },
  { label: "Expert / Researcher", color: "bg-red-500", pct: 100 },
];

const afterRoadmap = {
  overall: 65,
  label: "AI Engineer (Solid)",
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
  { icon: "✅", title: "What you CAN do confidently", color: "text-green-400", items: [
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
  { icon: "⚠️", title: "What you still can't do (yet)", color: "text-yellow-400", items: [
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
    id: 1,
    emoji: "🏗️",
    title: "Deepen by Building Publicly",
    when: "Immediately & ongoing",
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
    id: 2,
    emoji: "📐",
    title: "Fill the Math Gap (Selectively)",
    when: "After Phase 4–5, if curious",
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
    id: 3,
    emoji: "🔬",
    title: "Start Reading Papers",
    when: "After Phase 5",
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
    id: 4,
    emoji: "🎯",
    title: "Pick a Specialization",
    when: "After completing the roadmap",
    color: "from-orange-500 to-amber-600",
    desc: "At this point you're a generalist AI engineer. Real depth — and real opportunity — comes from going deep in one area.",
    specializations: [
      { name: "AI Application Builder", icon: "🛠️", desc: "Go deep on LangChain/LangGraph, evals, production, observability. Build AI products.", fit: "Best fit for your goals" },
      { name: "RAG Specialist", icon: "📚", desc: "Advanced chunking, hybrid search, knowledge graphs, multi-hop RAG.", fit: "High demand in enterprises" },
      { name: "Agent Systems", icon: "🤖", desc: "Multi-agent coordination, planning, MCP integrations, autonomous workflows.", fit: "Fastest growing area" },
      { name: "Fine-Tuning / Alignment", icon: "⚙️", desc: "SFT, RLHF, DPO, preference optimization, model behavior.", fit: "More ML depth needed" },
      { name: "Multimodal AI", icon: "🎨", desc: "Image/video generation, vision-language models, diffusion systems.", fit: "Creative + technical" },
      { name: "AI Infra / MLOps", icon: "🏗️", desc: "Serving, inference optimization, monitoring, scaling AI in production.", fit: "Strong eng background helps" },
    ],
  },
  {
    id: 5,
    emoji: "🌐",
    title: "Engage with the AI Community",
    when: "Throughout the journey",
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
    id: 6,
    emoji: "🔄",
    title: "Keep a 'Learning Radar'",
    when: "Ongoing forever",
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

export default function KnowledgeAssessment() {
  const [openStep, setOpenStep] = useState(null);
  const [showSpec, setShowSpec] = useState(false);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold mb-1">🧠 After the Roadmap</h1>
          <p className="text-gray-400 text-sm">Honest assessment of where you'll stand — and where to go next</p>
        </div>

        {/* Overall level */}
        <div className="bg-gray-900 border border-gray-700 rounded-2xl p-5 mb-6">
          <div className="flex items-center justify-between mb-1">
            <p className="text-lg font-bold text-white">{afterRoadmap.label}</p>
            <span className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-orange-400 bg-clip-text text-transparent">{afterRoadmap.overall}%</span>
          </div>
          <p className="text-gray-400 text-xs mb-4">{afterRoadmap.sublabel}</p>
          <div className="w-full bg-gray-800 rounded-full h-3 mb-1">
            <div className={`h-3 rounded-full bg-gradient-to-r ${afterRoadmap.color} transition-all`} style={{ width: `${afterRoadmap.overall}%` }} />
          </div>
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>Hobbyist</span><span>Beginner</span><span>Practitioner</span><span>Engineer</span><span>Specialist</span><span>Researcher</span>
          </div>
        </div>

        {/* Comparison bar */}
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-6">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">How You Compare</p>
          <div className="space-y-3">
            {comparisons.map((c, i) => (
              <div key={i} className={`${c.highlight ? "bg-purple-950 border border-purple-700 rounded-lg p-2 -mx-2" : ""}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-xs font-medium ${c.highlight ? "text-purple-300" : "text-gray-400"}`}>{c.role}</span>
                  <span className={`text-xs ${c.highlight ? "text-purple-300 font-bold" : "text-gray-500"}`}>{c.pct}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-1.5">
                  <div className={`h-1.5 rounded-full ${c.highlight ? "bg-purple-500" : "bg-gray-600"}`} style={{ width: `${c.pct}%` }} />
                </div>
                <p className="text-xs text-gray-600 mt-0.5">{c.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Area breakdown */}
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
                  <div className="h-1.5 rounded-full" style={{
                    width: `${a.pct}%`,
                    background: a.pct >= 75 ? "#22c55e" : a.pct >= 55 ? "#f59e0b" : "#ef4444"
                  }} />
                </div>
                <p className="text-xs text-gray-600">{a.note}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Can / Can't */}
        <div className="grid grid-cols-1 gap-4 mb-6">
          {honest.map((h, i) => (
            <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className={`text-xs font-semibold uppercase tracking-wider mb-3 ${h.color}`}>{h.icon} {h.title}</p>
              <div className="space-y-1.5">
                {h.items.map((item, j) => (
                  <div key={j} className="flex gap-2 text-sm">
                    <span className={`flex-shrink-0 ${h.color}`}>{h.icon === "✅" ? "✓" : "✗"}</span>
                    <span className="text-gray-300 text-xs">{item}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Next Steps */}
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">🚀 What's Next</p>
        <div className="space-y-3 mb-8">
          {nextSteps.map((s) => (
            <div key={s.id}
              className={`rounded-xl border cursor-pointer transition-all ${openStep === s.id ? "border-gray-500 bg-gray-900" : "border-gray-800 bg-gray-900 hover:border-gray-600"}`}
              onClick={() => setOpenStep(openStep === s.id ? null : s.id)}>
              <div className="flex items-center gap-3 p-4">
                <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${s.color} flex items-center justify-center text-base flex-shrink-0`}>{s.emoji}</div>
                <div className="flex-1">
                  <p className="font-semibold text-sm">{s.title}</p>
                  <p className="text-gray-500 text-xs mt-0.5">{s.when}</p>
                </div>
                <span className="text-gray-600">{openStep === s.id ? "▲" : "▼"}</span>
              </div>

              {openStep === s.id && (
                <div className="border-t border-gray-800 p-4 space-y-3">
                  <p className="text-gray-300 text-sm">{s.desc}</p>

                  {s.actions && s.actions.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Actions</p>
                      <div className="space-y-1.5">
                        {s.actions.map((a, i) => (
                          <div key={i} className="flex gap-2">
                            <span className="text-gray-600 flex-shrink-0">→</span>
                            <span className="text-xs text-gray-300">{a}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {s.examples && s.examples.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Examples</p>
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
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Choose Your Path</p>
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
                              <p className="text-xs text-gray-600 mt-0.5">{sp.fit}</p>
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

        {/* The honest reality */}
        <div className="bg-gray-900 border border-yellow-800 rounded-xl p-4">
          <p className="text-yellow-400 font-semibold text-sm mb-3">💬 The Honest Reality</p>
          <div className="space-y-2 text-sm text-gray-300">
            <p>After this roadmap, you'll know more about AI than <span className="text-white font-semibold">95% of software developers</span> and more than the majority of people who call themselves "AI engineers" today.</p>
            <p className="text-gray-400">But AI is a <span className="text-white">field</span>, not a course. The best practitioners treat it as a continuous practice — building things, reading papers, and re-learning as the landscape shifts every 6 months.</p>
            <p className="text-gray-400">The gap between a <span className="text-white">65% AI Engineer</span> and a <span className="text-white">90% Researcher</span> isn't really about courses. It's about <span className="text-white">years of building real systems</span> and going deep on one specific problem that matters to you.</p>
            <p className="mt-2 text-gray-500 text-xs">Your developer instincts are your biggest asset. Most AI courses are taught to people who can't code. You can. That alone puts you 2 years ahead.</p>
          </div>
        </div>

      </div>
    </div>
  );
}
