import { useState } from "react";

const areas = [
  {
    id: "llm", emoji: "🧠", title: "LLM Concepts & Internals", current: 80, target: 90,
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
    id: "prompt", emoji: "⚡", title: "Prompt Engineering", current: 90, target: 95,
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
    id: "rag", emoji: "📚", title: "RAG Systems", current: 80, target: 90,
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
    id: "agents", emoji: "🤖", title: "Agentic AI / Tool Use", current: 75, target: 88,
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
    id: "finetune", emoji: "🏗️", title: "Fine-Tuning / Training", current: 60, target: 80,
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
    id: "multimodal", emoji: "👁️", title: "Multimodal AI", current: 40, target: 75,
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
    id: "math", emoji: "📐", title: "ML Research / Math", current: 30, target: 60,
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
    id: "mlops", emoji: "🔧", title: "Production / MLOps", current: 50, target: 75,
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
    id: "safety", emoji: "🛡️", title: "AI Safety & Ethics", current: 40, target: 60,
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

export default function KnowledgeGaps() {
  const [open, setOpen] = useState(null);
  const [openTech, setOpenTech] = useState(null);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">

        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-1">📐 What to Study Per Area</h1>
          <p className="text-gray-400 text-sm">Exact gaps, required math, and what to explore next — per knowledge domain</p>
        </div>

        {/* Legend */}
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-3 mb-5 flex flex-wrap gap-3 items-center">
          <span className="text-xs text-gray-500">Math needed:</span>
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
          <p className="text-gray-500 text-xs mt-2">You do NOT need: hypothesis testing, regression analysis, ANOVA, or most classical statistics. Those are for data science, not AI engineering.</p>
        </div>

        {/* Areas */}
        <div className="space-y-3">
          {areas.map(a => (
            <div key={a.id}
              className={`rounded-xl border overflow-hidden cursor-pointer transition-all ${open === a.id ? "border-gray-500" : "border-gray-800 hover:border-gray-600"} bg-gray-900`}
              onClick={() => setOpen(open === a.id ? null : a.id)}>

              <div className="flex items-center gap-3 p-4">
                <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${a.color} flex items-center justify-center text-base flex-shrink-0`}>{a.emoji}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <p className="font-semibold text-sm">{a.title}</p>
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${mathColors[a.mathNeeded]}`}>{mathLabels[a.mathNeeded]}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <GapBar current={a.current} target={a.target} />
                    <span className="text-xs text-gray-500 flex-shrink-0">{a.current}% → {a.target}%</span>
                  </div>
                </div>
                <span className="text-gray-600 flex-shrink-0">{open === a.id ? "▲" : "▼"}</span>
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
                            <span className="text-gray-600 text-xs ml-2">{openTech === `${a.id}-${i}` ? "▲" : "▼"}</span>
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
