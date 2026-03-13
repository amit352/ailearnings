import { useState } from "react";

const tabs = ["overview", "text", "code", "image", "audio", "tools", "roadmap"];

const domainData = {
  text: {
    emoji: "💬", title: "Text & LLMs", color: "from-blue-600 to-indigo-700", border: "border-blue-700",
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
    emoji: "💻", title: "Code Generation", color: "from-green-600 to-teal-700", border: "border-green-700",
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
    emoji: "🎨", title: "Image & Video Generation", color: "from-purple-600 to-pink-700", border: "border-purple-700",
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
    emoji: "🎵", title: "Audio & Music Generation", color: "from-orange-600 to-red-700", border: "border-orange-700",
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
    category: "Text & Chat", emoji: "💬",
    tools: [
      { name: "Claude.ai", maker: "Anthropic", free: "Free tier", best: "Long docs, analysis, coding, nuanced writing", url: "https://claude.ai" },
      { name: "ChatGPT", maker: "OpenAI", free: "Free tier", best: "General use, image gen (DALL-E), browsing", url: "https://chat.openai.com" },
      { name: "Gemini Advanced", maker: "Google", free: "Free tier", best: "Google Workspace integration, 1M context", url: "https://gemini.google.com" },
      { name: "Perplexity", maker: "Perplexity AI", free: "Free tier", best: "Research with citations, web search", url: "https://perplexity.ai" },
    ]
  },
  {
    category: "Code", emoji: "💻",
    tools: [
      { name: "Cursor", maker: "Cursor", free: "Free tier", best: "Best AI code editor, codebase-aware", url: "https://cursor.sh" },
      { name: "GitHub Copilot", maker: "GitHub", free: "$10/mo", best: "Inline autocomplete, VS Code native", url: "https://github.com/features/copilot" },
      { name: "Claude Code", maker: "Anthropic", free: "Usage-based", best: "Agentic coding CLI, file operations", url: "https://claude.ai/code" },
      { name: "Replit AI", maker: "Replit", free: "Free tier", best: "Browser-based, deploy instantly", url: "https://replit.com" },
    ]
  },
  {
    category: "Image Generation", emoji: "🎨",
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
    category: "Video Generation", emoji: "🎬",
    tools: [
      { name: "Runway Gen-3", maker: "Runway", free: "Free credits", best: "Best quality, professional use", url: "https://runwayml.com" },
      { name: "Kling AI", maker: "Kuaishou", free: "Free tier", best: "Realistic motion, free generous tier", url: "https://klingai.com" },
      { name: "Sora", maker: "OpenAI", free: "In ChatGPT Plus", best: "Longest clips, good physics", url: "https://sora.com" },
      { name: "Pika Labs", maker: "Pika", free: "Free tier", best: "Animate images, easy to use", url: "https://pika.art" },
    ]
  },
  {
    category: "Audio & Voice", emoji: "🎵",
    tools: [
      { name: "ElevenLabs", maker: "ElevenLabs", free: "Free tier", best: "Best TTS + voice cloning", url: "https://elevenlabs.io" },
      { name: "Suno", maker: "Suno", free: "Free tier", best: "Full song generation with lyrics", url: "https://suno.com" },
      { name: "Whisper", maker: "OpenAI", free: "Free (local)", best: "Best open-source transcription", url: "https://github.com/openai/whisper" },
      { name: "Udio", maker: "Udio", free: "Free tier", best: "High quality music, genre control", url: "https://udio.com" },
    ]
  },
  {
    category: "Productivity & Workflow", emoji: "⚡",
    tools: [
      { name: "NotebookLM", maker: "Google", free: "Free", best: "Chat with documents, podcast generation", url: "https://notebooklm.google.com" },
      { name: "Notion AI", maker: "Notion", free: "Add-on", best: "AI inside your notes/docs/wiki", url: "https://notion.so/product/ai" },
      { name: "v0 by Vercel", maker: "Vercel", free: "Free tier", best: "Generate UI components from descriptions", url: "https://v0.dev" },
      { name: "Gamma", maker: "Gamma", free: "Free tier", best: "AI-generated presentations/slides", url: "https://gamma.app" },
    ]
  },
];

const roadmapPhases = [
  {
    phase: "Phase 1", title: "Foundations Across All Domains", duration: "3–4 weeks",
    color: "from-blue-600 to-indigo-700",
    goal: "Build mental models for how each GenAI domain works. Not hands-on yet — concepts first.",
    tasks: [
      { domain: "Text", icon: "💬", task: "Watch Karpathy's Intro to LLMs (1hr). Understand: tokens, attention, next-token prediction, temperature." },
      { domain: "Image", icon: "🎨", task: "Watch Computerphile's Stable Diffusion video (20min). Understand: latent space, diffusion process, text conditioning." },
      { domain: "Audio", icon: "🎵", task: "Read ElevenLabs blog on how TTS works. Try Whisper for transcription. Understand: spectrograms, codec models." },
      { domain: "Code", icon: "💻", task: "Install Cursor or Copilot. Use it for a real coding task. Notice what it does well and where it fails." },
    ]
  },
  {
    phase: "Phase 2", title: "Hands-On With Every Domain", duration: "4–5 weeks",
    color: "from-purple-600 to-violet-700",
    goal: "Get practical experience with the best tools in each domain. Build something small in each.",
    tasks: [
      { domain: "Text", icon: "💬", task: "Build a prompt library for your top 5 use cases. Experiment with system prompts. Compare Claude vs GPT-4 on the same tasks." },
      { domain: "Image", icon: "🎨", task: "Use Midjourney or DALL-E 3 for 2 weeks daily. Learn CFG scale, negative prompts, style references. Try Flux locally via ComfyUI." },
      { domain: "Audio", icon: "🎵", task: "Clone your own voice with ElevenLabs. Generate a full song with Suno. Run Whisper locally on a long audio file." },
      { domain: "Code", icon: "💻", task: "Build a small full-stack feature using only AI assistance. Use Claude Code for a refactoring task. Measure time saved vs unassisted." },
    ]
  },
  {
    phase: "Phase 3", title: "Build Across Modalities", duration: "4–6 weeks",
    color: "from-orange-600 to-amber-700",
    goal: "Combine domains. The real power of GenAI comes from chaining modalities together.",
    tasks: [
      { domain: "Text + Code", icon: "💬💻", task: "Build an AI app using LLM APIs. Add structured output parsing. Deploy it." },
      { domain: "Text + Image", icon: "💬🎨", task: "Build an automated image generation pipeline: text prompt → LLM refines prompt → image model generates → auto-saved." },
      { domain: "Text + Audio", icon: "💬🎵", task: "Build a document-to-podcast pipeline: PDF → LLM summarizes → ElevenLabs narrates → audio file output." },
      { domain: "Full Pipeline", icon: "🔗", task: "Pick one ambitious project that combines 3+ modalities. E.g.: video script generator → voiceover → auto image/video selection." },
    ]
  },
  {
    phase: "Phase 4", title: "Go Deep on One Domain", duration: "6–8 weeks",
    color: "from-teal-600 to-cyan-700",
    goal: "Generalist foundation → specialize in the domain most relevant to your work and goals.",
    tasks: [
      { domain: "If Text/LLMs", icon: "💬", task: "Study RAG, fine-tuning (LoRA), and agent systems. Build a production-grade LLM app with evals." },
      { domain: "If Image/Video", icon: "🎨", task: "Learn ComfyUI workflows, LoRA training, ControlNet. Take HuggingFace Diffusion Course." },
      { domain: "If Audio", icon: "🎵", task: "Build a voice agent with Vapi or ElevenLabs Conversational AI. Study real-time speech models." },
      { domain: "If Code", icon: "💻", task: "Build an agentic coding pipeline. Study how Claude Code/Devin work. Contribute to an open source AI coding tool." },
    ]
  },
];

const overview = {
  domains: [
    { emoji: "💬", name: "Text & LLMs", status: "Mature", desc: "Transformers → next token prediction. GPT, Claude, Gemini. Powers everything.", color: "blue" },
    { emoji: "💻", name: "Code Generation", status: "Mature", desc: "Specialized LLMs trained on code. Copilot, Cursor, Claude Code. Already transforming dev.", color: "green" },
    { emoji: "🎨", name: "Image & Video", status: "Rapidly evolving", desc: "Diffusion models. DALL-E, Midjourney, Sora. Going from images → films.", color: "purple" },
    { emoji: "🎵", name: "Audio & Music", status: "Fast growing", desc: "Codec + transformer models. ElevenLabs, Suno, Whisper. Voice is nearly perfect now.", color: "orange" },
  ],
  convergence: "All four domains are converging. GPT-4o processes text, image, and audio in one model. Gemini 1.5 is natively multimodal. The future is a single model that sees, hears, speaks, reads, and writes — and the components you learn today are the building blocks of that future.",
};

const TabBtn = ({ id, label, active, onClick }) => (
  <button onClick={() => onClick(id)}
    className={`text-xs px-3 py-2 rounded-lg whitespace-nowrap transition-colors ${active ? "bg-gray-700 text-white font-semibold" : "text-gray-500 hover:text-gray-300"}`}>
    {label}
  </button>
);

const Resource = ({ r }) => (
  <a href={r.url} target="_blank" rel="noopener noreferrer"
    className="flex items-center justify-between bg-gray-800 hover:bg-gray-700 rounded-lg px-3 py-2 transition-colors">
    <span className="text-blue-400 text-xs">{r.label}</span>
    <div className="flex items-center gap-2 flex-shrink-0 ml-2">
      <span className="text-xs text-gray-500">{r.time}</span>
      <span className={`text-xs px-1.5 py-0.5 rounded ${r.free ? "bg-green-900 text-green-400" : "bg-gray-700 text-gray-400"}`}>{r.free ? "Free" : "Paid"}</span>
    </div>
  </a>
);

const DomainDetail = ({ d }) => {
  const [sec, setSec] = useState("how");
  return (
    <div className="space-y-4">
      <div className={`bg-gradient-to-r ${d.color} rounded-xl p-4`}>
        <p className="text-lg font-bold">{d.emoji} {d.title}</p>
        <p className="text-white text-opacity-80 text-sm mt-1">{d.tagline}</p>
      </div>
      <div className="flex gap-1 flex-wrap">
        {[["how", "How It Works"], ["concepts", d.keyConceptsTitle], ["models", "Models"], ["resources", "Resources"]].map(([k, v]) => (
          <button key={k} onClick={() => setSec(k)}
            className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${sec === k ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300 bg-gray-900"}`}>{v}</button>
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
            <thead><tr className="text-gray-500 border-b border-gray-700">
              <th className="text-left pb-2 pr-3">Model</th><th className="text-left pb-2 pr-3">Maker</th><th className="text-left pb-2 pr-3">Best For</th><th className="text-left pb-2">Access</th>
            </tr></thead>
            <tbody>{d.models.map((m, i) => (
              <tr key={i} className="border-b border-gray-800">
                <td className="py-2 pr-3 font-semibold text-blue-400">{m.name}</td>
                <td className="py-2 pr-3 text-gray-400">{m.maker}</td>
                <td className="py-2 pr-3 text-gray-300">{m.best}</td>
                <td className="py-2 text-gray-500">{m.access}</td>
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
            <div><p className="text-blue-400 text-sm font-medium">{t.name}</p><p className="text-gray-500 text-xs">{t.desc}</p></div>
          </a>
        ))}</div>
      )}
    </div>
  );
};

export default function GenAIGuide() {
  const [tab, setTab] = useState("overview");

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-5">
          <h1 className="text-2xl font-bold mb-1">🧬 Generative AI — Complete Guide</h1>
          <p className="text-gray-400 text-sm">Text · Code · Image · Audio · Roadmap · Tools</p>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 overflow-x-auto bg-gray-900 rounded-xl p-1 mb-6 border border-gray-800">
          {[["overview","🗺 Overview"],["text","💬 Text"],["code","💻 Code"],["image","🎨 Image/Video"],["audio","🎵 Audio"],["tools","🛠 All Tools"],["roadmap","🚀 Roadmap"]].map(([k,v]) => (
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
                  <p className="text-2xl mb-1">{d.emoji}</p>
                  <p className="font-semibold text-sm text-white">{d.name}</p>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${d.status === "Mature" ? "bg-green-900 text-green-400" : "bg-yellow-900 text-yellow-400"}`}>{d.status}</span>
                  <p className="text-xs text-gray-500 mt-1">{d.desc}</p>
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
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">{cat.emoji} {cat.category}</p>
                <div className="space-y-2">
                  {cat.tools.map((t, j) => (
                    <a key={j} href={t.url} target="_blank" rel="noopener noreferrer"
                      className="flex items-start gap-3 bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-xl p-3 transition-colors">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 flex-wrap mb-0.5">
                          <p className="text-blue-400 font-semibold text-sm">{t.name}</p>
                          <span className="text-xs text-gray-600">by {t.maker}</span>
                          <span className={`text-xs px-2 py-0.5 rounded-full ${t.free === "Free" ? "bg-green-900 text-green-400" : "bg-gray-800 text-gray-500"}`}>{t.free}</span>
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
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">⏱ Timeline</p>
              <p className="text-gray-300 text-sm">At 4–6 hrs/week: <span className="text-white font-semibold">4–5 months</span> to strong generalist across all 4 domains. This roadmap is designed to run <span className="text-white">in parallel with</span> the main AI learning roadmap — not after it.</p>
            </div>
            {roadmapPhases.map((p, i) => (
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
                  <div key={i} className="flex gap-2"><span className="text-yellow-500 flex-shrink-0">→</span><p className="text-xs text-gray-300">{s}</p></div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
