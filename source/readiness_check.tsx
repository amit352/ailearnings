import { useState } from "react";

const phases = [
  {
    id: 1, emoji: "🌱", title: "Phase 1 – AI Foundations",
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
    id: 2, emoji: "⚙️", title: "Phase 2 – LLM Setup & Config",
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
    id: 3, emoji: "🔧", title: "Phase 3 – Prompt Engineering",
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
    id: 4, emoji: "📚", title: "Phase 4 – RAG Systems",
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
    id: 5, emoji: "🤖", title: "Phase 5 – Agentic AI",
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
    id: 6, emoji: "🏗️", title: "Phase 6 – Fine-Tuning & Training",
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
    id: 7, emoji: "🚀", title: "Phase 7 – Build & Specialize",
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
    icon: "⏱️", title: "Time is a signal, not a deadline",
    desc: "The durations are estimates at 4–6 hrs/week. If you're spending more time, you'll move faster. If less, slower. Don't use time to judge readiness — use the green flags."
  },
  {
    icon: "🚦", title: "The 70% Rule",
    desc: "You don't need 100% of the green flags to move on. If you have 70%+ and you've built the phase project — move. The remaining 30% will fill in naturally in the next phase."
  },
  {
    icon: "🔁", title: "Phases aren't waterfall",
    desc: "You'll revisit earlier phases constantly. Moving to Phase 4 (RAG) doesn't mean you stop prompting. Think of it as 'primary focus' shifts, not 'completed and locked'."
  },
  {
    icon: "🏗️", title: "The project is the gate, not the content",
    desc: "Reading all the resources for a phase doesn't mean you're ready to move on. Building the phase project is the real readiness signal. No project = not done."
  },
  {
    icon: "⚡", title: "Boredom is a green flag",
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

export default function ReadinessChecker() {
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
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">

        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-1">🚦 Phase Readiness Checker</h1>
          <p className="text-gray-400 text-sm">Know exactly when to move on — not too early, not too late</p>
        </div>

        {/* Tab */}
        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1 mb-6">
          {[["checker", "✅ Phase Checklist"], ["rules", "📐 Move-On Rules"], ["overview", "🗺 At a Glance"]].map(([k, v]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`flex-1 text-xs py-2 rounded-lg transition-colors ${tab === k ? "bg-gray-700 text-white font-semibold" : "text-gray-500 hover:text-gray-300"}`}>{v}</button>
          ))}
        </div>

        {/* CHECKLIST TAB */}
        {tab === "checker" && (
          <div className="space-y-3">
            <p className="text-gray-500 text-xs mb-4">Check off each green flag as you achieve it. The score tells you when you're ready to move on.</p>
            {phases.map(p => {
              const score = getScore(p.id, p.greenFlags.length);
              const verdict = getVerdict(score, p.readinessThreshold);
              const isOpen = openPhase === p.id;
              return (
                <div key={p.id} className={`rounded-xl border overflow-hidden transition-all ${isOpen ? "border-gray-500" : "border-gray-800"} bg-gray-900`}>
                  {/* Header */}
                  <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setOpenPhase(isOpen ? null : p.id)}>
                    <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${p.color} flex items-center justify-center text-base flex-shrink-0`}>{p.emoji}</div>
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
                    <span className="text-gray-600">{isOpen ? "▲" : "▼"}</span>
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
                        <p className="text-xs font-semibold text-blue-400 mb-1">⚡ Skip Rule</p>
                        <p className="text-xs text-gray-300">{p.skipRule}</p>
                      </div>

                      <div className="flex items-center justify-between text-xs text-gray-600 pt-1">
                        <span>⏱ Estimated: {p.duration}</span>
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
                <span className="text-2xl flex-shrink-0">{r.icon}</span>
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
            <p className="text-gray-500 text-xs mb-2">Quick view of all phases, thresholds, and your current progress.</p>
            <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-500">
                    <th className="text-left p-3">Phase</th>
                    <th className="text-left p-3">Duration</th>
                    <th className="text-left p-3">Threshold</th>
                    <th className="text-left p-3">Your Score</th>
                    <th className="text-left p-3">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {phases.map(p => {
                    const score = getScore(p.id, p.greenFlags.length);
                    const ready = score >= p.readinessThreshold;
                    return (
                      <tr key={p.id} className="border-b border-gray-800 hover:bg-gray-800 cursor-pointer transition-colors"
                        onClick={() => { setTab("checker"); setOpenPhase(p.id); }}>
                        <td className="p-3">
                          <div className="flex items-center gap-2">
                            <span>{p.emoji}</span>
                            <span className="text-gray-300 font-medium">Phase {p.id}</span>
                          </div>
                        </td>
                        <td className="p-3 text-gray-500">{p.duration}</td>
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
                            ? <span className="text-gray-600">Not started</span>
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
              <p className="text-xs text-gray-600 mt-3">💡 These are estimates. Moving faster is great. Moving slower is fine. What matters is building at every phase.</p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
