import { useState } from "react";

const Badge = ({ label, color }) => (
  <span className={`text-xs px-2 py-0.5 rounded-full border ${color}`}>{label}</span>
);

const techniques = [
  {
    id: 1, tier: "Foundation", emoji: "🧱", color: "from-blue-600 to-blue-800", border: "border-blue-700",
    title: "Core Prompting Techniques",
    desc: "These are non-negotiable. Master these before anything else. Most people stop here — experts start here.",
    techniques: [
      {
        name: "Zero-Shot Prompting",
        difficulty: "Beginner",
        use: "All use cases",
        what: "Ask directly with no examples. The baseline of everything.",
        bad: "Fix this code.",
        good: "You are a senior Python engineer. Review this function for bugs, performance issues, and readability. Output a list of issues with explanations and a corrected version.\n\n[paste code]",
        insight: "Adding a role, goal, and output format to a plain ask typically improves output quality by 40–60%. Most people never do this."
      },
      {
        name: "Few-Shot Prompting",
        difficulty: "Beginner",
        use: "Writing, classification, formatting",
        what: "Give the model 2–5 examples of input→output before your real request. Teaches format and tone without fine-tuning.",
        bad: "Write a commit message for this diff.",
        good: "Write a commit message for this diff. Follow this format:\n\nExample 1:\nDiff: Added null check to user login\nMessage: fix(auth): handle null user in login flow\n\nExample 2:\nDiff: Refactored payment module into separate service\nMessage: refactor(payments): extract payment logic to PaymentService\n\nNow write one for:\n[paste your diff]",
        insight: "Few-shot is the single most powerful technique for getting consistent output format — far easier than fine-tuning for most tasks."
      },
      {
        name: "Chain-of-Thought (CoT)",
        difficulty: "Beginner",
        use: "Reasoning, debugging, research, decisions",
        what: "Tell the model to think step by step before answering. Dramatically improves accuracy on complex tasks.",
        bad: "Which database should I use for this app?",
        good: "I'm building a real-time chat app expecting 100k concurrent users. Think step by step:\n1. Analyze the requirements (read/write patterns, latency needs, scale)\n2. Compare relevant database types\n3. Evaluate tradeoffs for my specific use case\n4. Give a final recommendation with reasoning",
        insight: "Adding 'think step by step' or breaking into numbered reasoning steps has been shown to improve LLM accuracy by 30–50% on complex tasks. Works even without examples."
      },
      {
        name: "Role / Persona Prompting",
        difficulty: "Beginner",
        use: "All use cases",
        what: "Assign the model an expert identity. Changes vocabulary, depth, assumptions, and communication style.",
        bad: "Review my resume.",
        good: "You are a senior engineering hiring manager at a top-tier tech company (FAANG level). You've reviewed 500+ resumes. Review this resume ruthlessly — identify what would get it rejected in under 10 seconds, what's missing, and what's weak. Be direct and specific.",
        insight: "The role you assign determines the frame of reference. 'Senior engineer' vs 'engineering manager' gives completely different feedback on the same resume."
      },
      {
        name: "Output Format Control",
        difficulty: "Beginner",
        use: "Coding, automation, structured data",
        what: "Explicitly specify the output format — JSON, markdown table, bullet points, specific length, etc.",
        bad: "Extract the key points from this article.",
        good: "Extract key points from this article. Return ONLY a JSON array with this structure:\n[\n  {\n    \"point\": \"...\",\n    \"importance\": \"high|medium|low\",\n    \"category\": \"...\"\n  }\n]\nNo preamble. No explanation. JSON only.",
        insight: "For app development, always specify JSON output + 'no preamble' to get parseable responses without extra parsing logic."
      },
    ]
  },
  {
    id: 2, tier: "Intermediate", emoji: "⚙️", color: "from-purple-600 to-purple-800", border: "border-purple-700",
    title: "Power Techniques",
    desc: "Where good prompt engineers separate from great ones. These unlock reliability, depth, and consistency.",
    techniques: [
      {
        name: "System Prompt Design",
        difficulty: "Intermediate",
        use: "Apps, automation, consistent assistants",
        what: "The system prompt defines persistent identity, constraints, tone, and behavior. It's the most leveraged prompt you'll write.",
        bad: "You are a helpful assistant.",
        good: "## Identity\nYou are CodeReviewer, a senior software engineer specializing in Python and system design.\n\n## Behavior\n- Always identify bugs before suggesting improvements\n- Cite specific line numbers\n- Distinguish between 'must fix' and 'nice to have'\n- Ask clarifying questions if requirements are unclear\n\n## Output Format\nStructure every review as:\n1. Critical Issues\n2. Suggested Improvements\n3. Positive Observations\n\n## Constraints\n- Never rewrite entire files — give targeted diffs\n- Don't suggest changes outside the stated scope",
        insight: "A well-designed system prompt can replace 80% of per-request prompting. Build one for every recurring use case you have."
      },
      {
        name: "Prompt Chaining",
        difficulty: "Intermediate",
        use: "Research, complex writing, automation",
        what: "Break a complex task into sequential prompts where output of one feeds into the next. More reliable than one giant prompt.",
        bad: "Research this topic, write a summary, then create a blog post and add SEO metadata.",
        good: "Step 1 prompt: 'Research [topic]. Output: 5 key insights with source context.'\n→ feed output into →\nStep 2 prompt: 'Given these insights: [output], write a 600-word blog post for a technical audience.'\n→ feed output into →\nStep 3 prompt: 'Given this blog post: [output], generate: title, meta description, 5 tags, and a tweet.'",
        insight: "Chaining with validation between steps (asking the model to check its own step before proceeding) dramatically reduces compounding errors."
      },
      {
        name: "Negative Prompting",
        difficulty: "Intermediate",
        use: "Writing, coding, content quality",
        what: "Explicitly tell the model what NOT to do. Eliminates common failure modes before they happen.",
        bad: "Summarize this article.",
        good: "Summarize this article in 3 bullet points.\n\nDo NOT:\n- Use filler phrases like 'The article discusses...'\n- Add your own opinions or analysis\n- Include information not in the article\n- Use bullet points longer than 2 sentences\n- Start with 'Sure!' or any affirmation",
        insight: "Negative constraints are often more powerful than positive instructions. They eliminate the model's default bad habits directly."
      },
      {
        name: "Self-Consistency & Verification",
        difficulty: "Intermediate",
        use: "Research, critical decisions, debugging",
        what: "Ask the model to verify, critique, or challenge its own output in a follow-up prompt.",
        bad: "(accept first answer)",
        good: "First prompt: 'Explain why [X approach] is the best solution for [problem].'\n\nSecond prompt: 'Now steelman the opposing view. What are the strongest arguments AGAINST the approach you just recommended? What have you missed?'\n\nThird prompt: 'Given both perspectives, what is your updated final recommendation?'",
        insight: "LLMs are sycophantic by default — they agree with themselves. Forcing adversarial self-review surfaces blind spots that a single prompt never catches."
      },
      {
        name: "Contextual Priming",
        difficulty: "Intermediate",
        use: "Research, writing, code review",
        what: "Front-load rich context before your ask. The quality of context determines the quality of output more than prompt wording.",
        bad: "Write a technical spec for my feature.",
        good: "## Context\nApp: B2B SaaS project management tool\nUsers: Engineering teams at mid-sized companies\nStack: Next.js, PostgreSQL, REST API\nTeam: 3 engineers, 1 designer\nTimeline: 2-week sprint\n\n## Feature\nReal-time notifications when task status changes\n\n## Audience for this spec\nJunior engineers who will implement it\n\nWrite a technical specification covering: overview, user stories, API design, database schema changes, edge cases, and acceptance criteria.",
        insight: "The GIGO principle (garbage in, garbage out) applies more to context than prompt wording. Rich context → dramatically better output."
      },
    ]
  },
  {
    id: 3, tier: "Advanced", emoji: "🔬", color: "from-orange-600 to-red-700", border: "border-orange-700",
    title: "Expert Techniques",
    desc: "What separates prompt engineers who build reliable systems from those who get lucky sometimes.",
    techniques: [
      {
        name: "Tree of Thoughts (ToT)",
        difficulty: "Advanced",
        use: "Complex decisions, architecture, research",
        what: "Ask the model to explore multiple reasoning paths simultaneously, then select the best one.",
        bad: "How should I architect this system?",
        good: "I need to architect [system]. Generate 3 completely different high-level architectural approaches. For each:\n1. Name and brief description\n2. Core assumptions\n3. Key advantages\n4. Fatal weaknesses\n5. When this is the right choice\n\nThen evaluate all 3 against my constraints: [list constraints] and recommend one with justification.",
        insight: "ToT prevents the model from anchoring on its first idea. Exploring multiple paths before committing surfaces options you'd never think to ask about."
      },
      {
        name: "Meta-Prompting",
        difficulty: "Advanced",
        use: "Building prompt systems, optimization",
        what: "Use the model to write, improve, or evaluate prompts. Ask the model to be the prompt engineer.",
        bad: "(struggle to write prompts yourself)",
        good: "I need a prompt that makes Claude act as a code reviewer for Python. The prompt should:\n- Produce consistent structured output every time\n- Handle both small functions and large modules\n- Distinguish bugs from style issues\n- Be reusable across projects\n\nWrite me the best possible system prompt for this use case. Then explain why you made each design decision.",
        insight: "Claude and GPT-4 are excellent prompt engineers. Using them to design prompts for themselves produces better results than most humans can write manually."
      },
      {
        name: "Constitutional Prompting",
        difficulty: "Advanced",
        use: "Content moderation, consistent AI behavior, apps",
        what: "Define a set of principles or rules the model must follow, then ask it to self-evaluate against those principles.",
        bad: "Write a response to this customer complaint.",
        good: "## Principles\n1. Never blame the customer\n2. Always offer a concrete next step\n3. Acknowledge emotion before explaining\n4. Avoid corporate jargon\n5. Response must be under 100 words\n\nWrite a response to this complaint: [complaint]\n\nThen score your response against each principle (1–5). If any score is below 4, rewrite and re-score until all principles are met.",
        insight: "This is how Anthropic trains Claude — having it evaluate its own outputs against a constitution. You can apply the same technique at prompt time."
      },
      {
        name: "Retrieval-Augmented Prompting",
        difficulty: "Advanced",
        use: "Research, knowledge-grounded tasks",
        what: "Inject specific reference material into the prompt context and instruct the model to ground its answer strictly in that material.",
        bad: "What does our API documentation say about rate limits?",
        good: "## Reference Material\n[paste exact docs section]\n\n## Instructions\nAnswer the following question using ONLY the reference material above. If the answer is not in the reference material, say 'Not covered in provided documentation' — do not use your general knowledge.\n\n## Question\nWhat are the rate limits for the /v2/messages endpoint?",
        insight: "This is manual RAG. By grounding the model in specific text, you eliminate hallucinations and get citable, reliable answers."
      },
      {
        name: "Prompt Evaluation & Testing",
        difficulty: "Advanced",
        use: "App development, reliability engineering",
        what: "Treat prompts like code — version control them, test against edge cases, measure consistency across runs.",
        bad: "(change prompts based on gut feel)",
        good: "1. Define success criteria before writing the prompt ('output must be valid JSON, < 200 words, contain 3 action items')\n2. Build a test set of 10–20 representative inputs including edge cases\n3. Run all inputs through your prompt, score against criteria\n4. A/B test prompt variants — change one variable at a time\n5. Track prompt versions in Git with changelogs\n6. Use tools: PromptFoo, LangSmith, or simple Python scripts",
        insight: "The #1 difference between amateur and professional prompt engineering is systematic testing. One-shot prompts that 'seem to work' fail unpredictably in production."
      },
    ]
  },
];

const useCases = [
  {
    id: "coding",
    emoji: "💻",
    title: "Coding & Debugging",
    color: "from-blue-600 to-cyan-700",
    tips: [
      { title: "The Debug Template", prompt: "You are a senior [language] engineer debugging a production issue.\n\nCode:\n```\n[paste code]\n```\n\nError:\n```\n[paste error]\n```\n\nContext: [what you expected vs what happened]\n\nThink step by step:\n1. What is this code trying to do?\n2. What does the error indicate?\n3. What are the 3 most likely root causes?\n4. What is the fix for each?\n5. Which fix do you recommend and why?" },
      { title: "Code Review Template", prompt: "Review this [language] code as a senior engineer. Grade each dimension 1–5 and explain:\n- Correctness: Does it do what it intends?\n- Performance: Any O(n²) issues or unnecessary operations?\n- Security: Any vulnerabilities?\n- Readability: Is it self-documenting?\n- Edge cases: What inputs would break this?\n\nThen provide a corrected version with comments explaining each change.\n\n```\n[paste code]\n```" },
      { title: "Architecture Advisor", prompt: "I'm building [describe system]. Current scale: [users/requests]. Expected growth: [timeline].\n\nI'm considering [approach A] vs [approach B].\n\nEvaluate both approaches against:\n1. Scalability to [target scale]\n2. Developer complexity\n3. Operational overhead\n4. Cost at scale\n5. Migration difficulty from current state\n\nRecommend one and justify your choice." },
    ]
  },
  {
    id: "writing",
    emoji: "✍️",
    title: "Writing & Content",
    color: "from-purple-600 to-violet-700",
    tips: [
      { title: "Tone Calibration", prompt: "Rewrite the following in my voice. My writing style:\n- Direct and confident, never hedging\n- Short sentences. No fluff.\n- Technical but accessible\n- I use dry humor occasionally\n- I never use: 'leverage', 'synergy', 'delve', 'it's worth noting'\n\nOriginal:\n[paste text]\n\nRewrite it maintaining my style. Don't change the meaning." },
      { title: "Content Reviewer", prompt: "Review this [blog post/email/doc] as a critical editor.\n\nEvaluate:\n1. Opening — does it hook immediately or does it warm up too slowly?\n2. Clarity — any sentences that require re-reading?\n3. Flow — where does the reader's momentum break?\n4. Fluff — which sentences add no value?\n5. Ending — does it close strongly or fade out?\n\nProvide: (a) your overall verdict, (b) 3 specific changes that would most improve it, (c) a rewritten opening if the current one is weak." },
      { title: "Structured Research Summary", prompt: "Summarize [topic or paste article] for a technical audience who is time-constrained.\n\nFormat:\n**TL;DR** (2 sentences max)\n\n**Key Points** (5 bullets, each 1 sentence)\n\n**Why It Matters** (2–3 sentences connecting to [my context])\n\n**Open Questions** (2–3 things this doesn't answer)\n\n**Recommended Action** (1 sentence — what should I do with this information?)" },
    ]
  },
  {
    id: "research",
    emoji: "🔍",
    title: "Research & Summarizing",
    color: "from-green-600 to-teal-700",
    tips: [
      { title: "The Socratic Researcher", prompt: "I want to deeply understand [topic]. Don't give me an overview.\n\nInstead:\n1. What is the single most important thing to understand about this topic?\n2. What is the most common misconception?\n3. What do experts disagree about?\n4. What question should I be asking that I'm not asking?\n5. What would change my mind about the conventional view?\n\nThen ask me 3 questions to understand my specific context before going deeper." },
      { title: "Second-Order Thinking", prompt: "I'm considering [decision/action].\n\nMap out:\n1. First-order effects (immediate, obvious consequences)\n2. Second-order effects (what happens as a result of the first-order effects?)\n3. Third-order effects (longer term ripple effects)\n4. What assumptions am I making that could be wrong?\n5. What would make this decision clearly wrong in hindsight?" },
      { title: "Comparative Analysis", prompt: "Compare [A] vs [B] for my use case: [describe use case].\n\nEvaluate on:\n| Dimension | A | B | Winner |\n|---|---|---|---|\n[fill table]\n\nAfter the table:\n- When would you clearly choose A over B?\n- When would you clearly choose B over A?\n- Is there a C option I haven't considered?\n- Your final recommendation for my specific use case." },
    ]
  },
  {
    id: "automation",
    emoji: "⚙️",
    title: "Work Automation",
    color: "from-orange-600 to-amber-700",
    tips: [
      { title: "The Reusable System Prompt", prompt: "Build me a reusable system prompt for an AI assistant that helps with [specific task].\n\nThe assistant should:\n- [behavior 1]\n- [behavior 2]\n- [behavior 3]\n\nAlways output: [format]\nNever: [anti-behaviors]\n\nThe prompt should be robust enough that I can use it every day without modification." },
      { title: "Document Processor", prompt: "Process the following document and extract structured information.\n\nDocument type: [invoice/contract/email/report]\n\nExtract and return as JSON:\n{\n  \"summary\": \"2-sentence summary\",\n  \"key_entities\": [...],\n  \"action_items\": [...],\n  \"deadlines\": [...],\n  \"flags\": [\"anything unusual or requiring attention\"]\n}\n\nDocument:\n[paste document]" },
      { title: "Meeting/Notes Processor", prompt: "Transform these raw meeting notes into a structured document.\n\nNotes:\n[paste notes]\n\nOutput format:\n## Meeting Summary (3 sentences)\n\n## Decisions Made\n- [decision] — Owner: [name] \n\n## Action Items\n- [ ] [task] — Owner: [name] — Due: [date]\n\n## Open Questions\n- [question] — Who resolves: [name]\n\n## Next Meeting\nDate: / Agenda:" },
    ]
  },
];

const resources = [
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

const milestones = [
  { week: "Week 1–2", title: "Foundation Mastery", tasks: ["Master all 5 core techniques", "Build a personal template library for your top 3 daily tasks", "Compare zero-shot vs few-shot on the same task 10 times"] },
  { week: "Week 3–4", title: "Power Techniques", tasks: ["Build 1 reusable system prompt per use case (coding, writing, research)", "Practice prompt chaining on a real research task", "Start a 'prompt journal' — log what works and why"] },
  { week: "Week 5–6", title: "Advanced + Evaluation", tasks: ["Use meta-prompting to improve your existing prompts", "Set up PromptFoo and test 3 prompts systematically", "Build a 'prompt library' in Notion or GitHub for reuse"] },
  { week: "Month 2–3", title: "Expert Territory", tasks: ["Build a small AI-powered tool using your prompt library", "Write about your learnings — blog post or LinkedIn article", "Start reading prompt engineering research papers (CoT, ToT papers)"] },
];

export default function PromptEngineering() {
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
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-1">⚡ Prompt Engineering Guide</h1>
          <p className="text-gray-400 text-sm">For Claude + ChatGPT · All use cases · Power user → Expert</p>
        </div>

        {/* Main tabs */}
        <div className="flex gap-1 bg-gray-900 rounded-xl p-1 mb-6 border border-gray-800">
          {[["learn", "🎓 Techniques"], ["templates", "📋 Templates"], ["milestones", "🗓️ Plan"], ["resources", "📚 Resources"]].map(([k, v]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`flex-1 text-xs py-2 rounded-lg transition-colors ${tab === k ? "bg-gray-700 text-white font-semibold" : "text-gray-500 hover:text-gray-300"}`}>
              {v}
            </button>
          ))}
        </div>

        {/* TECHNIQUES TAB */}
        {tab === "learn" && (
          <div className="space-y-4">
            <div className="bg-gray-900 border border-yellow-800 rounded-xl p-4 mb-4">
              <p className="text-yellow-400 font-semibold text-sm mb-1">💡 The Expert's Mental Model</p>
              <p className="text-gray-300 text-sm">A prompt is a <span className="text-white font-semibold">specification</span>, not a question. The more precisely you specify Role + Context + Task + Format + Constraints, the more reliably you get expert-level output. Every technique below is just a way to add one of these dimensions.</p>
            </div>

            {techniques.map(tier => (
              <div key={tier.id} className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden">
                <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setOpenTier(openTier === tier.id ? null : tier.id)}>
                  <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${tier.color} flex items-center justify-center text-base flex-shrink-0`}>{tier.emoji}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-sm">{tier.title}</p>
                      <span className={`text-xs px-2 py-0.5 rounded-full border ${tier.border} text-gray-300`}>{tier.tier}</span>
                    </div>
                    <p className="text-gray-500 text-xs mt-0.5">{tier.techniques.length} techniques</p>
                  </div>
                  <span className="text-gray-600">{openTier === tier.id ? "▲" : "▼"}</span>
                </div>

                {openTier === tier.id && (
                  <div className="border-t border-gray-800 p-4 space-y-3">
                    <p className="text-gray-400 text-xs">{tier.desc}</p>
                    {tier.techniques.map((t, i) => (
                      <div key={i} className={`rounded-lg border overflow-hidden ${openTech === `${tier.id}-${i}` ? "border-gray-500" : "border-gray-700"}`}>
                        <div className="flex items-center gap-3 p-3 cursor-pointer bg-gray-800 hover:bg-gray-750"
                          onClick={() => setOpenTech(openTech === `${tier.id}-${i}` ? null : `${tier.id}-${i}`)}>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <p className="text-sm font-semibold text-white">{t.name}</p>
                              <span className="text-xs text-gray-500 bg-gray-700 px-2 py-0.5 rounded">{t.use}</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{t.what}</p>
                          </div>
                          <span className="text-gray-600 text-sm">{openTech === `${tier.id}-${i}` ? "▲" : "▼"}</span>
                        </div>

                        {openTech === `${tier.id}-${i}` && (
                          <div className="p-3 space-y-3 bg-gray-900">
                            <div className="grid grid-cols-2 gap-2">
                              <div className="bg-red-950 border border-red-900 rounded-lg p-3">
                                <p className="text-xs text-red-400 font-semibold mb-1">❌ Weak Prompt</p>
                                <p className="text-xs text-gray-300 font-mono whitespace-pre-wrap">{t.bad}</p>
                              </div>
                              <div className="bg-green-950 border border-green-900 rounded-lg p-3">
                                <p className="text-xs text-green-400 font-semibold mb-1">✅ Strong Prompt</p>
                                <p className="text-xs text-gray-300 font-mono whitespace-pre-wrap">{t.good}</p>
                              </div>
                            </div>
                            <div className="bg-yellow-950 border border-yellow-900 rounded-lg p-3">
                              <p className="text-xs text-yellow-400 font-semibold mb-1">💡 Key Insight</p>
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

        {/* TEMPLATES TAB */}
        {tab === "templates" && (
          <div className="space-y-4">
            <p className="text-gray-400 text-xs mb-2">Ready-to-use templates for your specific daily use cases. Click to expand, copy to use.</p>
            {useCases.map(uc => (
              <div key={uc.id} className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden">
                <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setOpenUC(openUC === uc.id ? null : uc.id)}>
                  <div className={`w-9 h-9 rounded-full bg-gradient-to-br ${uc.color} flex items-center justify-center text-base flex-shrink-0`}>{uc.emoji}</div>
                  <p className="font-semibold text-sm flex-1">{uc.title}</p>
                  <span className="text-gray-600">{openUC === uc.id ? "▲" : "▼"}</span>
                </div>
                {openUC === uc.id && (
                  <div className="border-t border-gray-800 p-4 space-y-3">
                    {uc.tips.map((tip, i) => (
                      <div key={i} className={`rounded-lg border overflow-hidden ${openTemplate === `${uc.id}-${i}` ? "border-gray-500" : "border-gray-700"}`}>
                        <div className="flex items-center justify-between p-3 bg-gray-800 cursor-pointer"
                          onClick={() => setOpenTemplate(openTemplate === `${uc.id}-${i}` ? null : `${uc.id}-${i}`)}>
                          <p className="text-sm font-semibold text-white">{tip.title}</p>
                          <span className="text-gray-600 text-sm">{openTemplate === `${uc.id}-${i}` ? "▲" : "▼"}</span>
                        </div>
                        {openTemplate === `${uc.id}-${i}` && (
                          <div className="p-3 bg-gray-900">
                            <div className="bg-gray-800 rounded-lg p-3 font-mono text-xs text-gray-300 whitespace-pre-wrap mb-2">{tip.prompt}</div>
                            <button onClick={() => copy(tip.prompt, `${uc.id}-${i}`)}
                              className="text-xs bg-blue-700 hover:bg-blue-600 text-white px-3 py-1.5 rounded-lg transition-colors">
                              {copied === `${uc.id}-${i}` ? "✓ Copied!" : "Copy Template"}
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
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">🔑 The Universal Prompt Formula</p>
              <div className="bg-gray-800 rounded-lg p-3 font-mono text-xs text-gray-200 whitespace-pre-wrap mb-2">{`[ROLE] You are a [expert identity] with [relevant experience].

[CONTEXT] I am [your situation]. The goal is [what you're trying to achieve].

[TASK] [Clear, specific instruction verb — Write / Review / Analyze / Compare / Extract]

[CONSTRAINTS]
- Do NOT: [anti-behavior 1], [anti-behavior 2]
- Always: [required behavior]

[OUTPUT FORMAT]
Return your response as: [format — JSON / bullet list / table / prose]
Length: [word count or structure]

[INPUT]
[paste your actual content here]`}</div>
              <button onClick={() => copy(`[ROLE] You are a [expert identity] with [relevant experience].\n\n[CONTEXT] I am [your situation]. The goal is [what you're trying to achieve].\n\n[TASK] [Clear, specific instruction verb — Write / Review / Analyze / Compare / Extract]\n\n[CONSTRAINTS]\n- Do NOT: [anti-behavior 1], [anti-behavior 2]\n- Always: [required behavior]\n\n[OUTPUT FORMAT]\nReturn your response as: [format — JSON / bullet list / table / prose]\nLength: [word count or structure]\n\n[INPUT]\n[paste your actual content here]`, "universal")}
                className="text-xs bg-blue-700 hover:bg-blue-600 text-white px-3 py-1.5 rounded-lg transition-colors">
                {copied === "universal" ? "✓ Copied!" : "Copy Universal Template"}
              </button>
            </div>
          </div>
        )}

        {/* MILESTONES TAB */}
        {tab === "milestones" && (
          <div className="space-y-4">
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 mb-2">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">⏱ Time to Expert</p>
              <p className="text-gray-300 text-sm">At your current level (daily AI user, software dev), expect <span className="text-white font-semibold">6–8 weeks</span> of deliberate practice to reach expert level. The key word is <span className="text-white font-semibold">deliberate</span> — random usage doesn't build expertise. Intentional practice does.</p>
            </div>
            {milestones.map((m, i) => (
              <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl p-4">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center text-xs font-bold flex-shrink-0">{i + 1}</div>
                  <div>
                    <p className="font-semibold text-sm text-white">{m.title}</p>
                    <p className="text-xs text-gray-500">{m.week}</p>
                  </div>
                </div>
                <div className="space-y-1.5">
                  {m.tasks.map((t, j) => (
                    <div key={j} className="flex gap-2">
                      <span className="text-gray-600 flex-shrink-0 mt-0.5">→</span>
                      <p className="text-xs text-gray-300">{t}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            <div className="bg-gray-900 border border-green-800 rounded-xl p-4">
              <p className="text-green-400 font-semibold text-sm mb-2">🏆 How You Know You're Expert Level</p>
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
                    <span className="text-green-400 flex-shrink-0">✓</span>
                    <p className="text-xs text-gray-300">{s}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* RESOURCES TAB */}
        {tab === "resources" && (
          <div className="space-y-3">
            {["Free", "Book", "Tool", "Practice"].map(type => (
              <div key={type}>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">{type === "Free" ? "🎓 Free Courses & Guides" : type === "Book" ? "📖 Books" : type === "Tool" ? "🛠 Tools" : "💬 Community"}</p>
                <div className="space-y-2">
                  {resources.filter(r => r.type === type).map((r, i) => (
                    <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
                      className="flex items-start gap-3 bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-lg p-3 transition-colors">
                      <div className="flex-1">
                        <p className="text-blue-400 text-sm font-medium">{r.label}</p>
                        <p className="text-gray-500 text-xs mt-0.5">{r.note}</p>
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
