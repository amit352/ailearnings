#!/usr/bin/env node
/**
 * generate-static.js
 * Serves the built index.html locally, visits each route with Puppeteer,
 * waits for React to render, then saves the pre-rendered HTML to the
 * correct directory so GitHub Pages serves each route as a real URL.
 *
 * Usage:  node scripts/generate-static.js
 */

const puppeteer = require('puppeteer');
const http      = require('http');
const fs        = require('fs');
const path      = require('path');

const ROOT = path.resolve(__dirname, '../..');

// ── SEO metadata per route ──────────────────────────────────────────────────
const PAGES = [
  {
    slug:        '',
    outDir:      '.',
    url:         'http://localhost:3131/',
    title:       'AI Learning Roadmap 2026: Become an AI Engineer (Free Guide)',
    description: 'Free AI engineer roadmap for software developers. Follow our 7-phase LLM roadmap — from machine learning basics to RAG, Prompt Engineering, and Agentic AI. Learn how to learn AI with curated free resources.',
    canonical:   'https://ailearnings.in/',
    ogUrl:       'https://ailearnings.in/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'WebSite',
        name: 'AI Learning Hub',
        url: 'https://ailearnings.in/',
        description: 'Free 7-phase AI engineer roadmap for software developers. Master LLMs, Prompt Engineering, RAG, and Agentic AI with curated free resources and hands-on projects.',
      },
      {
        '@context': 'https://schema.org',
        '@type': 'EducationalOrganization',
        name: 'AI Learning Hub',
        url: 'https://ailearnings.in/',
        description: 'A free platform guiding software developers through a structured AI engineer roadmap — from machine learning fundamentals to advanced LLM and agentic AI techniques.',
        educationalCredentialAwarded: 'AI Engineer Skills',
        hasOfferCatalog: {
          '@type': 'OfferCatalog',
          name: 'AI Learning Roadmap Phases',
          itemListElement: [
            { '@type': 'Course', name: 'Phase 1: AI Foundations' },
            { '@type': 'Course', name: 'Phase 2: Machine Learning Fundamentals' },
            { '@type': 'Course', name: 'Phase 3: Deep Learning & Neural Networks' },
            { '@type': 'Course', name: 'Phase 4: LLMs & Language Models' },
            { '@type': 'Course', name: 'Phase 5: Prompt Engineering' },
            { '@type': 'Course', name: 'Phase 6: RAG & Retrieval Systems' },
            { '@type': 'Course', name: 'Phase 7: Agentic AI & Deployment' },
          ],
        },
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What is the best AI roadmap for developers?', acceptedAnswer: { '@type': 'Answer', text: 'The best AI roadmap for developers covers 7 phases: AI Foundations, LLM Setup, Prompt Engineering, RAG, Agentic AI, Building & Training LLMs, and Real Projects. Our free guide at ailearnings.in walks through each phase with curated free resources and hands-on projects.' } },
          { '@type': 'Question', name: 'How long does it take to become an AI engineer?', acceptedAnswer: { '@type': 'Answer', text: 'Most software developers can become AI engineers in 6–12 months of focused study (4–6 hours per week). Our 7-phase AI learning roadmap covers the full journey from AI foundations to building and shipping real AI projects.' } },
          { '@type': 'Question', name: 'Do I need a math background to learn AI?', acceptedAnswer: { '@type': 'Answer', text: 'No. While linear algebra and probability help for Phase 6 (building LLMs), the first 5 phases of our AI roadmap focus on practical skills — using APIs, building RAG pipelines, and deploying agents — that require no advanced math.' } },
          { '@type': 'Question', name: 'What is the difference between ML, deep learning, and generative AI?', acceptedAnswer: { '@type': 'Answer', text: 'Machine learning is the broad field of training models from data. Deep learning uses neural networks with multiple layers. Generative AI (GenAI) is a subset of deep learning focused on creating new content — text, images, audio, and code. LLMs are the most powerful form of GenAI today.' } },
          { '@type': 'Question', name: 'What should I learn first to get into AI?', acceptedAnswer: { '@type': 'Answer', text: 'Start with Phase 1 of our AI roadmap: AI Foundations. Watch Karpathy\'s "Neural Networks: Zero to Hero" and Andrej Karpathy\'s "Intro to LLMs" on YouTube. These give you the intuition and vocabulary to understand the entire field in just a few hours.' } },
          { '@type': 'Question', name: 'Is it possible to learn AI for free?', acceptedAnswer: { '@type': 'Answer', text: 'Yes — the entire AI engineer roadmap on ailearnings.in uses free resources: Karpathy\'s YouTube series, DeepLearning.AI short courses (free), Hugging Face courses, fast.ai, and free cloud GPUs on Google Colab and Kaggle.' } },
          { '@type': 'Question', name: 'What is an LLM and how do I start using one?', acceptedAnswer: { '@type': 'Answer', text: 'An LLM (Large Language Model) is a neural network trained on massive text data to generate and understand language. To start using one, sign up for a free API key from OpenAI, Anthropic, or Google Gemini, or run a local model like Llama 3 using Ollama (free, no GPU required for 7B models).' } },
        ],
      },
    ],
  },
  {
    slug:        'prep-plan',
    outDir:      'prep-plan',
    url:         'http://localhost:3131/prep-plan/',
    title:       'AI Interview Prep Plan 2026 – 6-Week Fast Track for Developers',
    description: 'Structured 6-week AI interview prep plan for software developers. Follow this machine learning roadmap to cover LLMs, Prompt Engineering, RAG, and Agentic AI — 4–6 hours per week with free resources.',
    canonical:   'https://ailearnings.in/prep-plan/',
    ogUrl:       'https://ailearnings.in/prep-plan/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'HowTo',
        name: 'AI Interview Prep Plan – 6-Week Fast Track for Developers',
        description: 'A structured 6-week plan to prepare for AI engineer interviews. Cover the full AI roadmap including LLMs, Prompt Engineering, RAG, and Agentic AI — 4–6 hours per week.',
        totalTime: 'PT6W',
        step: [
          { '@type': 'HowToStep', position: 1, name: 'Week 1: AI Foundations & Math Essentials', text: 'Learn the core concepts of AI, linear algebra, probability, and Python basics needed for the AI engineer roadmap.' },
          { '@type': 'HowToStep', position: 2, name: 'Week 2: Machine Learning Fundamentals', text: 'Cover supervised and unsupervised learning, key algorithms, and hands-on ML projects using scikit-learn.' },
          { '@type': 'HowToStep', position: 3, name: 'Week 3: Deep Learning & Neural Networks', text: 'Study neural networks, CNNs, RNNs, and transformers. Build foundational models with PyTorch or TensorFlow.' },
          { '@type': 'HowToStep', position: 4, name: 'Week 4: LLMs & Language Models', text: 'Understand how large language models work, fine-tuning strategies, and how to use LLM APIs effectively.' },
          { '@type': 'HowToStep', position: 5, name: 'Week 5: Prompt Engineering & RAG', text: 'Master prompt engineering techniques and build retrieval-augmented generation (RAG) pipelines with vector databases.' },
          { '@type': 'HowToStep', position: 6, name: 'Week 6: Agentic AI & Mock Interviews', text: 'Explore agentic AI frameworks, tool use, and multi-agent systems. Practice with mock AI engineer interview questions.' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'AI Interview Prep Plan', item: 'https://ailearnings.in/prep-plan/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'How do I prepare for an AI engineer interview?', acceptedAnswer: { '@type': 'Answer', text: 'Follow a structured 6-week prep plan covering: Week 1 (AI Foundations), Week 2 (Machine Learning), Week 3 (Deep Learning), Week 4 (LLMs), Week 5 (Prompt Engineering & RAG), and Week 6 (Agentic AI & Mock Interviews). Use free resources from DeepLearning.AI and Hugging Face.' } },
          { '@type': 'Question', name: 'How long does AI interview preparation take?', acceptedAnswer: { '@type': 'Answer', text: 'With 4–6 hours per week, you can prepare for an AI engineer interview in 6 weeks. If you already have software engineering experience, you may focus more time on LLMs, RAG, and agentic AI (Weeks 4–6) and spend less on foundations.' } },
          { '@type': 'Question', name: 'What topics are asked in AI engineer interviews in 2026?', acceptedAnswer: { '@type': 'Answer', text: 'AI engineer interviews in 2026 cover: LLM concepts (attention, transformers, tokenization), prompt engineering techniques, RAG pipeline design, vector databases, agentic AI patterns (ReACT, tool use), fine-tuning strategies (LoRA, QLoRA), and system design for AI applications.' } },
          { '@type': 'Question', name: 'Should I study LLMs or traditional ML for AI engineer interviews?', acceptedAnswer: { '@type': 'Answer', text: 'Both matter, but weight your study toward LLMs, RAG, and agentic AI for 2026 AI engineer roles. Most modern AI engineering jobs focus on building applications with LLMs rather than training models from scratch. Traditional ML (linear regression, clustering) may appear in ML engineer roles.' } },
          { '@type': 'Question', name: 'What AI projects should I build for an engineering portfolio?', acceptedAnswer: { '@type': 'Answer', text: 'Build these 3 core projects for an AI engineering portfolio: (1) A RAG chatbot that answers questions from your own PDF documents using LangChain and a vector database, (2) A ReACT agent that can search the web and summarize results, and (3) A fine-tuned LLM on a custom dataset using QLoRA on free Google Colab GPUs.' } },
          { '@type': 'Question', name: 'How is an AI engineer interview different from a software engineer interview?', acceptedAnswer: { '@type': 'Answer', text: 'AI engineer interviews add domain-specific rounds: LLM system design (building RAG pipelines, agent architectures), ML concepts (when to fine-tune vs prompt vs RAG), prompt engineering challenges, and practical coding with LLM APIs. Standard SWE skills (algorithms, data structures) may still be tested at larger companies.' } },
        ],
      },
    ],
  },
  {
    slug:        'genai-guide',
    outDir:      'genai-guide',
    url:         'http://localhost:3131/genai-guide/',
    title:       'Generative AI Guide 2026 – LLMs, Image, Audio & Code for Developers',
    description: 'Complete Generative AI guide for developers. Learn how LLMs, image generation, audio synthesis, and code generation work — with top models, tools, and an AI roadmap to master each domain.',
    canonical:   'https://ailearnings.in/genai-guide/',
    ogUrl:       'https://ailearnings.in/genai-guide/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: 'Generative AI Guide 2026 – LLMs, Image, Audio & Code for Developers',
        description: 'A comprehensive guide to Generative AI covering LLMs, image generation, audio synthesis, and code generation. Includes top models, practical tools, and a learning roadmap for AI engineers.',
        url: 'https://ailearnings.in/genai-guide/',
        author: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        dateModified: '2026-01-01',
        mainEntityOfPage: { '@type': 'WebPage', '@id': 'https://ailearnings.in/genai-guide/' },
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'Generative AI Guide', item: 'https://ailearnings.in/genai-guide/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What is generative AI and how does it work?', acceptedAnswer: { '@type': 'Answer', text: 'Generative AI is a category of AI that creates new content — text, images, audio, video, and code — by learning statistical patterns from training data. LLMs like GPT-4 and Claude generate text by predicting the most likely next token. Image models like Stable Diffusion use diffusion processes to generate images from text prompts.' } },
          { '@type': 'Question', name: 'What is the difference between generative AI and discriminative AI?', acceptedAnswer: { '@type': 'Answer', text: 'Discriminative AI classifies or predicts labels for existing data (e.g., "is this email spam?"). Generative AI creates new data samples that resemble training data (e.g., "write a professional email about X"). LLMs, image generators, and voice synthesis tools are all examples of generative AI.' } },
          { '@type': 'Question', name: 'What are the best generative AI tools for developers in 2026?', acceptedAnswer: { '@type': 'Answer', text: 'The best generative AI tools for developers in 2026 are: (1) Text: Claude API, OpenAI API, Gemini API; (2) Code: GitHub Copilot, Cursor; (3) Images: Stable Diffusion, Midjourney, DALL-E 3; (4) Audio: ElevenLabs, Whisper; (5) Local models: Ollama + Llama 3 or Mistral.' } },
          { '@type': 'Question', name: 'How do large language models generate text?', acceptedAnswer: { '@type': 'Answer', text: 'LLMs generate text by predicting the next token (word piece) one at a time. The model converts input text into numerical tokens, processes them through transformer layers with attention mechanisms, and outputs a probability distribution over the vocabulary. The highest-probability token is selected (or sampled), appended, and the process repeats.' } },
          { '@type': 'Question', name: 'What is multimodal AI?', acceptedAnswer: { '@type': 'Answer', text: 'Multimodal AI processes and generates multiple data types — text, images, audio, and video — in the same model. Examples include GPT-4o (text + images + audio), Gemini 1.5 Pro (text, images, video, audio), and Claude 3.5 Sonnet (text + images). Multimodal models can analyze images, answer questions about documents, and generate audio from text.' } },
          { '@type': 'Question', name: 'How do I build my first generative AI application?', acceptedAnswer: { '@type': 'Answer', text: 'Build your first GenAI app in 4 steps: (1) Sign up for a free API key from Anthropic or OpenAI, (2) Install the Python SDK (pip install anthropic), (3) Call the API with a prompt from your code, (4) Add a simple web interface with Gradio or Streamlit. The DeepLearning.AI "Prompt Engineering for Developers" course (free) is the best starting point.' } },
        ],
      },
    ],
  },
  {
    slug:        'prompt-eng',
    outDir:      'prompt-eng',
    url:         'http://localhost:3131/prompt-eng/',
    title:       'Prompt Engineering Guide 2026 – 15 Techniques & Templates',
    description: 'Master prompt engineering with 15 techniques: zero-shot, few-shot, chain-of-thought, tree-of-thoughts, and more. Copy-paste templates for coding, writing, and research — a key skill on every AI engineer roadmap.',
    canonical:   'https://ailearnings.in/prompt-eng/',
    ogUrl:       'https://ailearnings.in/prompt-eng/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'HowTo',
        name: 'Prompt Engineering Guide – 15 Techniques & Templates',
        description: 'Learn 15 prompt engineering techniques from zero-shot to tree-of-thoughts. Includes copy-paste templates for coding, writing, and research tasks.',
        step: [
          { '@type': 'HowToStep', position: 1, name: 'Zero-Shot Prompting', text: 'Ask the model to perform a task without any examples. Best for simple, well-defined tasks.' },
          { '@type': 'HowToStep', position: 2, name: 'Few-Shot Prompting', text: 'Provide 2–5 input/output examples before your actual task to guide model behavior.' },
          { '@type': 'HowToStep', position: 3, name: 'Chain-of-Thought (CoT)', text: 'Ask the model to reason step-by-step before giving a final answer to improve accuracy.' },
          { '@type': 'HowToStep', position: 4, name: 'Self-Consistency', text: 'Generate multiple reasoning paths and select the most consistent answer.' },
          { '@type': 'HowToStep', position: 5, name: 'Tree of Thoughts (ToT)', text: 'Explore multiple reasoning branches and evaluate intermediate steps for complex problems.' },
          { '@type': 'HowToStep', position: 6, name: 'Role Prompting', text: 'Assign a persona or role to the model to influence its tone and expertise level.' },
          { '@type': 'HowToStep', position: 7, name: 'ReAct (Reason + Act)', text: 'Interleave reasoning and tool-use actions for agentic AI tasks.' },
          { '@type': 'HowToStep', position: 8, name: 'Retrieval-Augmented Prompting', text: 'Inject retrieved context from a knowledge base to ground model responses in facts.' },
          { '@type': 'HowToStep', position: 9, name: 'Instruction Tuning Prompts', text: 'Structure prompts as explicit instructions for models fine-tuned on instruction data.' },
          { '@type': 'HowToStep', position: 10, name: 'Constrained Output Prompting', text: 'Specify exact output format (JSON, bullet list, table) to control structure.' },
          { '@type': 'HowToStep', position: 11, name: 'Negative Prompting', text: 'Explicitly state what the model should NOT do to reduce unwanted outputs.' },
          { '@type': 'HowToStep', position: 12, name: 'Prompt Chaining', text: 'Break complex tasks into sequential prompts where each output feeds the next.' },
          { '@type': 'HowToStep', position: 13, name: 'Meta-Prompting', text: 'Ask the model to generate or refine prompts for a given task.' },
          { '@type': 'HowToStep', position: 14, name: 'Contrastive Prompting', text: 'Show good vs. bad examples to sharpen the model\'s understanding of quality.' },
          { '@type': 'HowToStep', position: 15, name: 'Socratic Prompting', text: 'Use a question-and-answer dialogue structure to guide the model toward a correct answer.' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'Prompt Engineering Guide', item: 'https://ailearnings.in/prompt-eng/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What is prompt engineering and why does it matter?', acceptedAnswer: { '@type': 'Answer', text: 'Prompt engineering is the practice of crafting inputs to LLMs to get more accurate, relevant, and useful outputs. It matters because the same model can produce vastly different results depending on how the question is framed. Key techniques include zero-shot prompting, few-shot prompting, chain-of-thought, and role prompting.' } },
          { '@type': 'Question', name: 'What is chain-of-thought prompting?', acceptedAnswer: { '@type': 'Answer', text: 'Chain-of-thought (CoT) prompting asks the model to reason step-by-step before giving a final answer. Adding "Let\'s think step by step" or showing examples of multi-step reasoning significantly improves performance on math, logic, and complex reasoning tasks. It\'s one of the most reliable prompt engineering techniques.' } },
          { '@type': 'Question', name: 'What is the difference between zero-shot and few-shot prompting?', acceptedAnswer: { '@type': 'Answer', text: 'Zero-shot prompting gives the model a task with no examples — you rely on its pretrained knowledge. Few-shot prompting provides 2–5 input/output examples before the actual task, showing the model the expected format and style. Few-shot generally outperforms zero-shot for structured or domain-specific tasks.' } },
          { '@type': 'Question', name: 'How do I write better prompts for Claude or ChatGPT?', acceptedAnswer: { '@type': 'Answer', text: 'Write better LLM prompts by: (1) Being specific about the output format you want, (2) Providing context and constraints, (3) Using a system prompt to set the role and persona, (4) Giving 1–3 examples for complex tasks (few-shot), (5) Breaking complex tasks into steps with chain-of-thought, and (6) Testing and iterating on your prompts.' } },
          { '@type': 'Question', name: 'What is ReAct prompting?', acceptedAnswer: { '@type': 'Answer', text: 'ReAct (Reason + Act) is a prompting pattern for agentic AI where the model interleaves reasoning steps (Thought) with actions (Act) and observations (Observe). It enables the model to use tools like web search, calculators, and code execution while explaining its reasoning at each step. ReAct is the foundation of most AI agent frameworks.' } },
          { '@type': 'Question', name: 'Is prompt engineering still relevant with newer AI models?', acceptedAnswer: { '@type': 'Answer', text: 'Yes. While newer models (Claude 3.5, GPT-4o) are more instruction-following, prompt engineering is still essential for production applications. Structured output formats, system prompts, few-shot examples, and chain-of-thought techniques consistently improve results across all frontier models.' } },
        ],
      },
    ],
  },
  {
    slug:        'resources',
    outDir:      'resources',
    url:         'http://localhost:3131/resources/',
    title:       'Best Free AI Learning Resources 2026 – Books & Courses by Phase',
    description: 'Best free AI learning resources for developers following an AI engineer roadmap. Curated books, video courses, and references mapped to all 7 phases — from machine learning basics to LLMs and agentic AI.',
    canonical:   'https://ailearnings.in/resources/',
    ogUrl:       'https://ailearnings.in/resources/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'ItemList',
        name: 'Best Free AI Learning Resources 2026 – Books & Courses by Phase',
        description: 'Curated list of the best free AI learning resources for each phase of the AI engineer roadmap, including books, courses, and references.',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Phase 1 Resources: AI Foundations', url: 'https://ailearnings.in/resources/' },
          { '@type': 'ListItem', position: 2, name: 'Phase 2 Resources: Machine Learning', url: 'https://ailearnings.in/resources/' },
          { '@type': 'ListItem', position: 3, name: 'Phase 3 Resources: Deep Learning', url: 'https://ailearnings.in/resources/' },
          { '@type': 'ListItem', position: 4, name: 'Phase 4 Resources: Large Language Models', url: 'https://ailearnings.in/resources/' },
          { '@type': 'ListItem', position: 5, name: 'Phase 5 Resources: Prompt Engineering', url: 'https://ailearnings.in/resources/' },
          { '@type': 'ListItem', position: 6, name: 'Phase 6 Resources: RAG & Retrieval', url: 'https://ailearnings.in/resources/' },
          { '@type': 'ListItem', position: 7, name: 'Phase 7 Resources: Agentic AI', url: 'https://ailearnings.in/resources/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'AI Learning Resources', item: 'https://ailearnings.in/resources/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What are the best free resources to learn AI in 2026?', acceptedAnswer: { '@type': 'Answer', text: 'The best free AI learning resources in 2026 are: (1) Karpathy\'s "Neural Networks: Zero to Hero" on YouTube, (2) DeepLearning.AI short courses (all free), (3) fast.ai Practical Deep Learning course, (4) Hugging Face NLP Course, (5) Google ML Crash Course, and (6) Anthropic and OpenAI official documentation.' } },
          { '@type': 'Question', name: 'Is the fast.ai course good for AI beginners?', acceptedAnswer: { '@type': 'Answer', text: 'Yes. fast.ai\'s "Practical Deep Learning for Coders" is excellent for software developers because it takes a top-down approach — you build working models first, then learn the theory. It covers computer vision, NLP, and tabular data using PyTorch. Best for developers who learn by doing.' } },
          { '@type': 'Question', name: 'What is the best free course for learning LLMs?', acceptedAnswer: { '@type': 'Answer', text: 'The best free LLM learning resources are: Andrej Karpathy\'s "Intro to LLMs" (1-hour YouTube masterclass), DeepLearning.AI "Building Systems with the ChatGPT API" (free), and Hugging Face\'s NLP Course (Chapters 1–4). For building with LLMs, DeepLearning.AI\'s "LangChain: Chat with Your Data" is essential.' } },
          { '@type': 'Question', name: 'Are DeepLearning.AI courses worth it for developers?', acceptedAnswer: { '@type': 'Answer', text: 'Yes — and they\'re all free (audit mode). DeepLearning.AI short courses are 1–2 hours each, taught by Andrew Ng and leading AI researchers. The most valuable for developers: Prompt Engineering for Devs, Building Systems with LLM API, LangChain: Chat with Your Data, and AI Agents in LangGraph.' } },
          { '@type': 'Question', name: 'What Python libraries do I need to learn AI?', acceptedAnswer: { '@type': 'Answer', text: 'Essential Python libraries for AI engineers: NumPy and Pandas (data manipulation), scikit-learn (classical ML), PyTorch (deep learning), Hugging Face Transformers (LLMs), LangChain or LlamaIndex (RAG & agents), ChromaDB or Pinecone (vector databases), and FastAPI or Gradio (building AI apps).' } },
          { '@type': 'Question', name: 'How can I learn AI without a computer science degree?', acceptedAnswer: { '@type': 'Answer', text: 'You can learn AI without a CS degree by focusing on practical skills: start with Python basics (3–4 weeks), follow our free AI roadmap at ailearnings.in, use DeepLearning.AI short courses, and build 2–3 portfolio projects (a RAG chatbot, an agent, and a fine-tuned model). Many successful AI engineers are self-taught developers.' } },
        ],
      },
    ],
  },
  {
    slug:        'readiness',
    outDir:      'readiness',
    url:         'http://localhost:3131/readiness/',
    title:       'AI Learning Readiness Checker – Know When to Level Up (2026)',
    description: 'Know when to advance on your AI engineer roadmap. Check green flags, red flags, and move-on rules for all 7 phases — from machine learning basics to agentic AI. Track how to learn AI systematically.',
    canonical:   'https://ailearnings.in/readiness/',
    ogUrl:       'https://ailearnings.in/readiness/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: 'AI Learning Readiness Checker – Know When to Level Up (2026)',
        description: 'A readiness checker for each phase of the AI engineer roadmap. Identify green flags, red flags, and move-on rules to progress confidently through the AI learning roadmap.',
        url: 'https://ailearnings.in/readiness/',
        author: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        dateModified: '2026-01-01',
        mainEntityOfPage: { '@type': 'WebPage', '@id': 'https://ailearnings.in/readiness/' },
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'AI Readiness Checker', item: 'https://ailearnings.in/readiness/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'How do I know when to advance to the next phase of the AI roadmap?', acceptedAnswer: { '@type': 'Answer', text: 'You\'re ready to advance when you can explain the current phase\'s concepts to someone else, complete the phase\'s milestone project, and identify 3+ green flag indicators on the readiness checklist. Don\'t aim for 100% mastery — move forward when you hit 70–80% confidence and can build something that works.' } },
          { '@type': 'Question', name: 'How long should I spend on each phase of the AI learning roadmap?', acceptedAnswer: { '@type': 'Answer', text: 'Suggested time per phase at 4–6 hours/week: Phase 1 (4–6 weeks), Phase 2 (2–3 weeks), Phase 3 (3–4 weeks), Phase 4 (4–5 weeks), Phase 5 (4–5 weeks), Phase 6 (6–8 weeks), Phase 7 (ongoing). Total: 6–9 months to complete the full AI engineer roadmap.' } },
          { '@type': 'Question', name: 'What are signs I\'m ready to start building AI applications?', acceptedAnswer: { '@type': 'Answer', text: 'You\'re ready to build AI applications when you can: call an LLM API from Python code, understand what system prompts do, know the difference between temperature and top-p, can debug why an LLM gives bad outputs, and have completed a basic prompt engineering project. These are Phase 3 milestones in our AI roadmap.' } },
          { '@type': 'Question', name: 'How do I know if I\'ve mastered the basics of machine learning?', acceptedAnswer: { '@type': 'Answer', text: 'You\'ve mastered ML basics when you can: explain the difference between supervised and unsupervised learning, implement a linear regression and a random forest from scratch using scikit-learn, understand overfitting and how to prevent it, and interpret a confusion matrix and ROC curve. These map to Phase 2 milestones.' } },
          { '@type': 'Question', name: 'What should I be able to build before moving to RAG?', acceptedAnswer: { '@type': 'Answer', text: 'Before moving to RAG (Phase 4), you should be able to build: a working LLM-powered app using an API (a chatbot, code reviewer, or text summarizer), implement at least 3 different prompting techniques, and handle context window limits and token counting. These are Phase 3 milestones in our AI engineer roadmap.' } },
        ],
      },
    ],
  },
  {
    slug:        'beyond-roadmap',
    outDir:      'beyond-roadmap',
    url:         'http://localhost:3131/beyond-roadmap/',
    title:       'Beyond the AI Roadmap – Advanced AI Topics & Knowledge Gaps',
    description: 'Finished the AI engineer roadmap? Discover advanced AI topics beyond the core LLM roadmap — knowledge gaps by specialization, cutting-edge research areas, and next steps for senior AI engineers.',
    canonical:   'https://ailearnings.in/beyond-roadmap/',
    ogUrl:       'https://ailearnings.in/beyond-roadmap/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: 'Beyond the AI Roadmap – Advanced AI Topics & Knowledge Gaps',
        description: 'A guide for developers who have completed the AI engineer roadmap. Explore advanced AI topics, specialization paths, and knowledge gaps in LLMs, multimodal AI, and AI systems research.',
        url: 'https://ailearnings.in/beyond-roadmap/',
        author: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        dateModified: '2026-01-01',
        mainEntityOfPage: { '@type': 'WebPage', '@id': 'https://ailearnings.in/beyond-roadmap/' },
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'Beyond the AI Roadmap', item: 'https://ailearnings.in/beyond-roadmap/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What should I learn after completing the AI engineer roadmap?', acceptedAnswer: { '@type': 'Answer', text: 'After the core AI roadmap, focus on specialization: (1) Applied AI engineering — production RAG systems, LLM observability, evaluation frameworks; (2) AI research — reading papers, contributing to open-source; (3) Specialized domains — multimodal AI, AI for healthcare/legal/finance; or (4) Systems AI — inference optimization, distributed training, MLOps.' } },
          { '@type': 'Question', name: 'What are the most in-demand AI specializations in 2026?', acceptedAnswer: { '@type': 'Answer', text: 'The most in-demand AI specializations in 2026 are: (1) AI engineering (LLM apps, RAG, agents), (2) ML engineering (training pipelines, MLOps), (3) Applied research (model evaluation, red-teaming), (4) AI product management, and (5) Domain-specific AI (AI for law, medicine, finance). AI engineering has the highest job growth.' } },
          { '@type': 'Question', name: 'Should I focus on LLM research or applied AI engineering?', acceptedAnswer: { '@type': 'Answer', text: 'For most developers, applied AI engineering offers more job opportunities and faster career growth in 2026. LLM research requires deep math background and PhD-level work for frontier models. Applied AI engineering focuses on building reliable applications with existing models — RAG systems, agents, and fine-tuned models for specific domains.' } },
          { '@type': 'Question', name: 'What are the emerging AI topics beyond LLMs?', acceptedAnswer: { '@type': 'Answer', text: 'Emerging AI topics in 2026 beyond LLMs include: (1) Multimodal AI (vision-language models), (2) AI reasoning (o1-style chain-of-thought models), (3) World models for robotics, (4) AI agents and multi-agent systems, (5) Neuromorphic computing, (6) AI for drug discovery and biology, and (7) Efficient inference (quantization, speculative decoding).' } },
          { '@type': 'Question', name: 'How do AI engineers stay current with rapid AI advances?', acceptedAnswer: { '@type': 'Answer', text: 'Stay current with AI by: (1) Reading Hugging Face papers weekly (huggingface.co/papers), (2) Following the Latent Space and Lex Fridman podcasts, (3) Subscribing to The Batch newsletter by DeepLearning.AI, (4) Following key researchers on Twitter/X, (5) Contributing to open-source AI projects on GitHub, and (6) Building side projects with new models as they release.' } },
        ],
      },
    ],
  },
  {
    slug:        'assessment',
    outDir:      'assessment',
    url:         'http://localhost:3131/assessment/',
    title:       'AI Engineer Skill Assessment – What You\'ll Know After the Roadmap',
    description: 'Honest AI engineer skill assessment after completing the full AI roadmap. See exactly what you can build, which knowledge gaps remain, and the best next steps to grow as an AI engineer in 2026.',
    canonical:   'https://ailearnings.in/assessment/',
    ogUrl:       'https://ailearnings.in/assessment/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: 'AI Engineer Skill Assessment – What You\'ll Know After the Roadmap',
        description: 'An honest assessment of the AI engineer skills you gain after completing the full AI learning roadmap — covering LLMs, machine learning, RAG, prompt engineering, and agentic AI.',
        url: 'https://ailearnings.in/assessment/',
        author: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        dateModified: '2026-01-01',
        mainEntityOfPage: { '@type': 'WebPage', '@id': 'https://ailearnings.in/assessment/' },
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'AI Skill Assessment', item: 'https://ailearnings.in/assessment/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What skills does an AI engineer need in 2026?', acceptedAnswer: { '@type': 'Answer', text: 'An AI engineer in 2026 needs: Python proficiency, LLM API integration (OpenAI, Anthropic, Gemini), prompt engineering techniques, RAG pipeline design (vector databases, chunking, retrieval), agentic AI frameworks (LangChain, LangGraph), basic fine-tuning (LoRA/QLoRA), and system design skills for production AI applications.' } },
          { '@type': 'Question', name: 'What can you build after completing an AI learning roadmap?', acceptedAnswer: { '@type': 'Answer', text: 'After completing the AI engineer roadmap, you can build: RAG chatbots that answer from private documents, AI agents that search the web and take actions, fine-tuned LLMs for domain-specific tasks, code review and generation tools, AI-powered data analysis pipelines, and multimodal apps that process images and audio.' } },
          { '@type': 'Question', name: 'What is the salary of an AI engineer in 2026?', acceptedAnswer: { '@type': 'Answer', text: 'AI engineer salaries in 2026 range from $120k–$200k+ in the US for mid-to-senior roles, with top companies paying $200k–$400k+ in total compensation. Globally, AI engineering commands a significant premium over standard software engineering roles, with high demand and limited supply of skilled engineers.' } },
          { '@type': 'Question', name: 'What knowledge gaps do most AI engineers have?', acceptedAnswer: { '@type': 'Answer', text: 'Common knowledge gaps for AI engineers: (1) Evaluation frameworks — most skip rigorous evals in favor of vibe checks; (2) Production reliability — handling LLM failures, latency, and cost at scale; (3) Security — prompt injection, data leakage, and adversarial attacks; (4) Mathematical foundations — attention mechanics and why certain architectures work.' } },
          { '@type': 'Question', name: 'How do I demonstrate AI engineering skills to employers?', acceptedAnswer: { '@type': 'Answer', text: 'Demonstrate AI skills by: (1) Building and publishing 2–3 AI projects on GitHub with clean READMEs, (2) Writing technical blog posts about what you built and what you learned, (3) Contributing to open-source AI projects, (4) Earning relevant certifications (DeepLearning.AI, Hugging Face), and (5) Sharing your work on LinkedIn and in AI communities.' } },
          { '@type': 'Question', name: 'What distinguishes a senior AI engineer from a junior one?', acceptedAnswer: { '@type': 'Answer', text: 'Senior AI engineers can: design end-to-end AI systems (not just write prompts), evaluate and improve RAG pipelines with quantitative metrics, make fine-tuning vs RAG vs prompting tradeoff decisions, handle production issues (latency, cost, reliability), contribute to AI safety and alignment discussions, and mentor others through the AI learning roadmap.' } },
        ],
      },
    ],
  },
  {
    slug:        'ai-roadmap',
    outDir:      'ai-roadmap',
    url:         'http://localhost:3131/ai-roadmap/',
    title:       'AI Roadmap 2026: Complete Guide for Developers',
    description: 'Follow the complete AI roadmap for developers in 2026. Our step-by-step guide covers 7 phases — from AI foundations to LLMs, RAG, Prompt Engineering, and Agentic AI — with free resources and project milestones.',
    canonical:   'https://ailearnings.in/ai-roadmap/',
    ogUrl:       'https://ailearnings.in/ai-roadmap/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'HowTo',
        name: 'AI Roadmap 2026: Complete Guide for Developers',
        description: 'Step-by-step AI engineer roadmap covering 7 phases from AI foundations to building and shipping real AI projects.',
        totalTime: 'P9M',
        step: [
          { '@type': 'HowToStep', position: 1, name: 'Phase 1: AI Foundations', text: 'Learn how AI, ML, and LLMs work conceptually. Watch Karpathy\'s Neural Networks: Zero to Hero and 3Blue1Brown\'s neural network series.' },
          { '@type': 'HowToStep', position: 2, name: 'Phase 2: LLM Setup & Configuration', text: 'Set up your AI environment. Run local models with Ollama, use cloud LLM APIs, and learn key parameters like temperature, top-p, and context window.' },
          { '@type': 'HowToStep', position: 3, name: 'Phase 3: Prompt Engineering & LLM APIs', text: 'Master zero-shot, few-shot, and chain-of-thought prompting. Build your first AI-powered application using OpenAI, Anthropic, or Gemini APIs.' },
          { '@type': 'HowToStep', position: 4, name: 'Phase 4: RAG & Your Own Data', text: 'Build retrieval-augmented generation pipelines. Learn vector databases, document chunking, semantic search, and RAG evaluation.' },
          { '@type': 'HowToStep', position: 5, name: 'Phase 5: Agentic AI', text: 'Build AI agents that plan, use tools, and execute multi-step tasks. Implement ReACT pattern and multi-agent systems.' },
          { '@type': 'HowToStep', position: 6, name: 'Phase 6: Building & Training LLMs', text: 'Understand transformer architecture, fine-tuning with LoRA/QLoRA, and when to fine-tune vs use RAG vs prompt engineering.' },
          { '@type': 'HowToStep', position: 7, name: 'Phase 7: Ship Real Projects', text: 'Build and launch 2–3 real AI projects. Work with multimodal AI, reasoning models, and deploy AI into your existing dev stack.' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'AI Roadmap', item: 'https://ailearnings.in/ai-roadmap/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What is the best AI roadmap for 2026?', acceptedAnswer: { '@type': 'Answer', text: 'The best AI roadmap for 2026 covers 7 phases: AI Foundations, LLM Setup, Prompt Engineering, RAG, Agentic AI, LLM Training, and Real Projects. Start with AI foundations (Karpathy\'s YouTube series), progress through practical LLM APIs, and finish by building and shipping your own AI projects.' } },
          { '@type': 'Question', name: 'How long does the AI roadmap take to complete?', acceptedAnswer: { '@type': 'Answer', text: 'The complete AI roadmap takes 6–9 months at 4–6 hours per week. Experienced software developers may complete it faster since they already know Python and can focus on AI-specific skills. The most time-intensive phases are Phase 6 (Building LLMs, 6–8 weeks) and Phase 7 (ongoing projects).' } },
          { '@type': 'Question', name: 'Can I follow the AI roadmap without a machine learning background?', acceptedAnswer: { '@type': 'Answer', text: 'Yes. The AI roadmap on ailearnings.in is designed for software developers without an ML background. Phase 1 builds conceptual understanding without heavy math. You can start building real AI apps by Phase 3 using LLM APIs — no training or math required at that stage.' } },
          { '@type': 'Question', name: 'What is the most important phase in the AI roadmap?', acceptedAnswer: { '@type': 'Answer', text: 'Phase 3 (Prompt Engineering & LLM APIs) is the most high-leverage phase for most developers. It enables you to build real AI-powered applications immediately using cloud LLM APIs. Phases 4–5 (RAG and Agents) are the most in-demand skills for AI engineer jobs in 2026.' } },
          { '@type': 'Question', name: 'Do I need a GPU to follow the AI roadmap?', acceptedAnswer: { '@type': 'Answer', text: 'No GPU is required for the first 5 phases of the AI roadmap. You can use free cloud LLM APIs (OpenAI, Anthropic free tiers) and free cloud GPUs (Google Colab, Kaggle) for Phase 6 fine-tuning experiments. Running models locally with Ollama only requires a modern CPU for 7B models.' } },
        ],
      },
    ],
  },
  {
    slug:        'llm-roadmap',
    outDir:      'llm-roadmap',
    url:         'http://localhost:3131/llm-roadmap/',
    title:       'LLM Roadmap 2026: Learn Large Language Models from Scratch',
    description: 'The complete LLM roadmap for developers in 2026. Learn how large language models work, how to use them via APIs, fine-tune them with LoRA, and build RAG and agentic AI applications from scratch.',
    canonical:   'https://ailearnings.in/llm-roadmap/',
    ogUrl:       'https://ailearnings.in/llm-roadmap/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'HowTo',
        name: 'LLM Roadmap 2026: Learn Large Language Models from Scratch',
        description: 'Step-by-step roadmap to learn large language models — from transformer architecture to building production LLM applications.',
        step: [
          { '@type': 'HowToStep', position: 1, name: 'Understand How LLMs Work', text: 'Learn tokenization, attention mechanisms, transformer architecture, and how LLMs generate text. Watch Karpathy\'s "Let\'s build GPT" on YouTube.' },
          { '@type': 'HowToStep', position: 2, name: 'Use LLM APIs in Code', text: 'Call OpenAI, Anthropic, and Gemini APIs from Python. Understand context windows, temperature, and structured outputs.' },
          { '@type': 'HowToStep', position: 3, name: 'Master Prompt Engineering for LLMs', text: 'Learn zero-shot, few-shot, chain-of-thought, and ReAct prompting. Build a production-ready system prompt library.' },
          { '@type': 'HowToStep', position: 4, name: 'Build RAG Applications', text: 'Combine LLMs with your own data using vector databases and retrieval-augmented generation. Build a document Q&A system.' },
          { '@type': 'HowToStep', position: 5, name: 'Fine-tune an LLM', text: 'Fine-tune Llama 3 or Mistral using LoRA/QLoRA on free Google Colab GPUs. Learn SFT, RLHF, and parameter-efficient fine-tuning.' },
          { '@type': 'HowToStep', position: 6, name: 'Deploy LLM Applications', text: 'Deploy LLM apps to production with FastAPI, handle latency and cost, implement streaming, and add observability with LangSmith.' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'LLM Roadmap', item: 'https://ailearnings.in/llm-roadmap/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'How do I start learning large language models?', acceptedAnswer: { '@type': 'Answer', text: 'Start by understanding how LLMs work conceptually (Karpathy\'s YouTube series), then learn to use them via APIs (OpenAI, Anthropic). Progress to prompt engineering, RAG applications, and finally fine-tuning. Our LLM roadmap at ailearnings.in provides the exact sequence with free resources.' } },
          { '@type': 'Question', name: 'What programming language is best for learning LLMs?', acceptedAnswer: { '@type': 'Answer', text: 'Python is the standard language for LLM development. The entire ecosystem — PyTorch, Hugging Face Transformers, LangChain, LlamaIndex — is Python-first. JavaScript/TypeScript developers can use the Vercel AI SDK or LangChain.js, but the best resources and tools are in Python.' } },
          { '@type': 'Question', name: 'What is the difference between GPT-4, Claude, and Llama?', acceptedAnswer: { '@type': 'Answer', text: 'GPT-4 (OpenAI) and Claude (Anthropic) are closed-source frontier LLMs available via paid API. Llama 3 (Meta) is an open-source LLM you can run locally for free via Ollama. For learning, start with free API tiers from OpenAI/Anthropic, then explore local Llama models once you understand the basics.' } },
          { '@type': 'Question', name: 'What is fine-tuning an LLM and when should I do it?', acceptedAnswer: { '@type': 'Answer', text: 'Fine-tuning adapts a pre-trained LLM to a specific task or domain using a custom dataset. Use fine-tuning when: prompt engineering can\'t achieve consistent output format, you need domain-specific knowledge not in the base model, or you need to reduce prompt length for cost/latency. For most use cases, RAG + prompting works better than fine-tuning.' } },
          { '@type': 'Question', name: 'How much compute do I need to work with LLMs?', acceptedAnswer: { '@type': 'Answer', text: 'For using LLM APIs: any laptop with internet. For running local 7B models: 8GB RAM, no GPU needed (CPU inference with Ollama). For fine-tuning: free Google Colab T4 GPU handles QLoRA fine-tuning of 7B–13B models. For serious training: cloud GPUs (Lambda Labs, RunPod) at $1–3/hour.' } },
        ],
      },
    ],
  },
  {
    slug:        'rag-tutorial',
    outDir:      'rag-tutorial',
    url:         'http://localhost:3131/rag-tutorial/',
    title:       'RAG Tutorial 2026: Build Retrieval-Augmented Generation Step by Step',
    description: 'Complete RAG tutorial for developers in 2026. Learn to build a Retrieval-Augmented Generation pipeline from scratch — document ingestion, vector embeddings, semantic search, and LLM generation with LangChain and ChromaDB.',
    canonical:   'https://ailearnings.in/rag-tutorial/',
    ogUrl:       'https://ailearnings.in/rag-tutorial/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'HowTo',
        name: 'RAG Tutorial 2026: Build Retrieval-Augmented Generation Step by Step',
        description: 'Step-by-step tutorial to build a RAG (Retrieval-Augmented Generation) pipeline from scratch using LangChain, ChromaDB, and an LLM API.',
        step: [
          { '@type': 'HowToStep', position: 1, name: 'Understand RAG Architecture', text: 'Learn how RAG combines a retriever (vector search) with a generator (LLM) to ground AI responses in your own documents. Understand when to use RAG vs fine-tuning.' },
          { '@type': 'HowToStep', position: 2, name: 'Load and Parse Documents', text: 'Use LangChain document loaders to ingest PDFs, web pages, and text files. Choose the right loader for your data source.' },
          { '@type': 'HowToStep', position: 3, name: 'Chunk Documents Effectively', text: 'Split documents into chunks using recursive character splitting, semantic chunking, or parent-document retrieval. Chunk size affects retrieval quality.' },
          { '@type': 'HowToStep', position: 4, name: 'Generate Vector Embeddings', text: 'Convert text chunks to embedding vectors using OpenAI embeddings, HuggingFace sentence-transformers, or Cohere. Store in ChromaDB, Pinecone, or Weaviate.' },
          { '@type': 'HowToStep', position: 5, name: 'Build the Retrieval Pipeline', text: 'Implement semantic search to retrieve the most relevant chunks for a user query. Combine with metadata filtering for precision.' },
          { '@type': 'HowToStep', position: 6, name: 'Generate Answers with an LLM', text: 'Pass retrieved context and user query to an LLM to generate grounded answers. Implement the full RAG chain with LangChain\'s LCEL.' },
          { '@type': 'HowToStep', position: 7, name: 'Evaluate RAG Quality', text: 'Measure faithfulness, answer relevancy, and context precision with RAGAS. Iterate on chunking strategy and retrieval parameters to improve quality.' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'RAG Tutorial', item: 'https://ailearnings.in/rag-tutorial/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What is RAG (Retrieval-Augmented Generation)?', acceptedAnswer: { '@type': 'Answer', text: 'RAG is an AI architecture that enhances LLM responses by retrieving relevant documents from a knowledge base before generating an answer. Instead of relying only on the model\'s training data, RAG retrieves up-to-date, relevant context and includes it in the prompt. This reduces hallucinations and enables LLMs to answer questions about your private data.' } },
          { '@type': 'Question', name: 'What is the difference between RAG and fine-tuning?', acceptedAnswer: { '@type': 'Answer', text: 'RAG retrieves documents at inference time and provides them as context — no training required. Fine-tuning modifies the model\'s weights using a custom dataset. Use RAG when your data updates frequently or you need citations. Use fine-tuning when you need consistent output format, domain-specific style, or have large amounts of labeled examples.' } },
          { '@type': 'Question', name: 'What tools do I need to build a RAG pipeline?', acceptedAnswer: { '@type': 'Answer', text: 'To build a RAG pipeline, you need: (1) LangChain or LlamaIndex for orchestration, (2) A vector database (ChromaDB for local dev, Pinecone for production), (3) An embedding model (OpenAI text-embedding-3 or HuggingFace sentence-transformers), and (4) An LLM API (Claude, OpenAI, or Gemini). All have free tiers to start.' } },
          { '@type': 'Question', name: 'How do I improve the quality of my RAG pipeline?', acceptedAnswer: { '@type': 'Answer', text: 'Improve RAG quality by: (1) Experimenting with chunk size (512–1024 tokens typically work well), (2) Using sentence-window or parent-document retrieval instead of naive chunking, (3) Adding metadata filtering, (4) Using hybrid search (semantic + keyword), (5) Reranking retrieved chunks with a cross-encoder, and (6) Measuring with RAGAS evaluation framework.' } },
          { '@type': 'Question', name: 'What is a vector database and why does RAG need one?', acceptedAnswer: { '@type': 'Answer', text: 'A vector database stores text as high-dimensional numerical vectors (embeddings) and enables fast similarity search. RAG needs a vector database to find the most semantically relevant document chunks for a user query — traditional SQL databases can\'t do semantic search. Popular options: ChromaDB (free, local), Pinecone (managed), Weaviate, and pgvector (Postgres extension).' } },
        ],
      },
    ],
  },
  {
    slug:        'machine-learning-roadmap',
    outDir:      'machine-learning-roadmap',
    url:         'http://localhost:3131/machine-learning-roadmap/',
    title:       'Machine Learning Roadmap 2026: From Beginner to AI Engineer',
    description: 'The complete machine learning roadmap for 2026. Go from absolute beginner to AI engineer — covering Python, math fundamentals, classical ML, deep learning, LLMs, and real-world ML projects with free resources.',
    canonical:   'https://ailearnings.in/machine-learning-roadmap/',
    ogUrl:       'https://ailearnings.in/machine-learning-roadmap/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'HowTo',
        name: 'Machine Learning Roadmap 2026: From Beginner to AI Engineer',
        description: 'Complete machine learning learning path from beginner to AI engineer covering Python, math, classical ML, deep learning, and LLMs.',
        step: [
          { '@type': 'HowToStep', position: 1, name: 'Python & Math Prerequisites', text: 'Learn Python (NumPy, Pandas, Matplotlib), linear algebra (vectors, matrices), probability, and basic calculus. Khan Academy and fast.ai are great free resources.' },
          { '@type': 'HowToStep', position: 2, name: 'Classical Machine Learning', text: 'Master supervised learning (linear regression, decision trees, SVMs, random forests) and unsupervised learning (k-means, PCA). Use scikit-learn with hands-on projects.' },
          { '@type': 'HowToStep', position: 3, name: 'Deep Learning & Neural Networks', text: 'Learn neural networks, backpropagation, CNNs, RNNs, and transformers. Use PyTorch and follow Karpathy\'s "Neural Networks: Zero to Hero" series.' },
          { '@type': 'HowToStep', position: 4, name: 'Large Language Models', text: 'Understand how LLMs work (tokenization, attention, pre-training). Use Hugging Face Transformers to load and fine-tune models.' },
          { '@type': 'HowToStep', position: 5, name: 'ML in Production', text: 'Learn MLOps: experiment tracking with W&B, model deployment with FastAPI, data versioning with DVC, and monitoring production ML models.' },
          { '@type': 'HowToStep', position: 6, name: 'Build ML Projects', text: 'Build 3 portfolio projects: a classical ML project (Kaggle competition), a deep learning project (image classification or NLP), and an LLM application (RAG or agent).' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'Machine Learning Roadmap', item: 'https://ailearnings.in/machine-learning-roadmap/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'How long does it take to learn machine learning?', acceptedAnswer: { '@type': 'Answer', text: 'Learning machine learning to a job-ready level takes 6–12 months with consistent study (1–2 hours daily). The timeline depends on your starting point: developers with Python experience can move faster through the prerequisites. Classical ML takes 2–3 months, deep learning 2–3 months, and LLMs/applied AI another 3–4 months.' } },
          { '@type': 'Question', name: 'Do I need to know math to learn machine learning?', acceptedAnswer: { '@type': 'Answer', text: 'Basic math helps but isn\'t required to start. You can build working ML models with scikit-learn and PyTorch before mastering the math. As you progress, understanding linear algebra (matrices, dot products), probability (Bayes theorem, distributions), and calculus (gradients, derivatives) will deepen your intuition and help you debug model behavior.' } },
          { '@type': 'Question', name: 'What is the best machine learning course for beginners?', acceptedAnswer: { '@type': 'Answer', text: 'The best free ML courses for beginners: (1) Google ML Crash Course (structured, hands-on), (2) fast.ai Practical Deep Learning (top-down, code-first), (3) Andrew Ng\'s Machine Learning Specialization on Coursera (math-focused, free to audit), and (4) Karpathy\'s Neural Networks: Zero to Hero (builds intuition from scratch). Start with fast.ai if you\'re a developer.' } },
          { '@type': 'Question', name: 'Is machine learning still relevant now that we have LLMs?', acceptedAnswer: { '@type': 'Answer', text: 'Yes. Machine learning fundamentals (feature engineering, model evaluation, overfitting, cross-validation) are essential for understanding why LLMs behave the way they do and for production ML beyond just calling APIs. Classical ML is still widely used for tabular data, recommendation systems, fraud detection, and other structured data problems.' } },
          { '@type': 'Question', name: 'What is the difference between machine learning and deep learning?', acceptedAnswer: { '@type': 'Answer', text: 'Machine learning is the broad field of training models from data using algorithms like linear regression, decision trees, and random forests. Deep learning is a subset of ML using neural networks with many layers. Deep learning excels at unstructured data (images, text, audio) and powers LLMs. Classical ML often outperforms deep learning on tabular/structured data.' } },
        ],
      },
    ],
  },
  {
    slug:        'ai-projects',
    outDir:      'ai-projects',
    url:         'http://localhost:3131/ai-projects/',
    title:       'AI Projects for Developers 2026: 10 Hands-On Ideas with Code',
    description: '10 hands-on AI project ideas for developers in 2026 — from beginner to advanced. Build a RAG chatbot, AI agent, fine-tuned LLM, and more. Each project includes tech stack, learning outcomes, and free resources.',
    canonical:   'https://ailearnings.in/ai-projects/',
    ogUrl:       'https://ailearnings.in/ai-projects/',
    schema: [
      {
        '@context': 'https://schema.org',
        '@type': 'ItemList',
        name: 'AI Projects for Developers 2026: 10 Hands-On Ideas with Code',
        description: '10 hands-on AI project ideas for developers in 2026, from beginner RAG chatbots to advanced fine-tuned LLMs and multi-agent systems.',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'PDF Q&A Chatbot with RAG', description: 'Build a chatbot that answers questions from your own PDF documents using LangChain, ChromaDB, and an LLM API.' },
          { '@type': 'ListItem', position: 2, name: 'AI Code Review Tool', description: 'Build a CLI tool that reviews code diffs and suggests improvements using an LLM API with few-shot prompting.' },
          { '@type': 'ListItem', position: 3, name: 'Web Research Agent', description: 'Build a ReACT agent that searches the web, reads pages, and writes research summaries using LangChain and a search tool.' },
          { '@type': 'ListItem', position: 4, name: 'Fine-tuned LLM for a Domain', description: 'Fine-tune Llama 3 8B on a custom dataset using QLoRA on Google Colab. Deploy as a local API with Ollama.' },
          { '@type': 'ListItem', position: 5, name: 'AI Writing Assistant', description: 'Build a context-aware writing assistant with a system prompt, memory, and Gradio UI.' },
          { '@type': 'ListItem', position: 6, name: 'Voice-to-Text Summarizer', description: 'Combine OpenAI Whisper (speech recognition) with an LLM to transcribe and summarize audio files or meetings.' },
          { '@type': 'ListItem', position: 7, name: 'Personal Knowledge Base', description: 'Build a RAG system over your own notes, bookmarks, and documents with automatic embedding updates.' },
          { '@type': 'ListItem', position: 8, name: 'AI SQL Query Generator', description: 'Build a natural language to SQL tool that generates queries from plain English questions about a database.' },
          { '@type': 'ListItem', position: 9, name: 'Multi-Agent Research System', description: 'Build an orchestrator + worker agent system where a planner agent delegates research tasks to specialized worker agents.' },
          { '@type': 'ListItem', position: 10, name: 'LLM Evaluation Dashboard', description: 'Build a dashboard to evaluate LLM outputs with RAGAS metrics — faithfulness, answer relevancy, and context precision.' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'AI Projects', item: 'https://ailearnings.in/ai-projects/' },
        ],
      },
      {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: [
          { '@type': 'Question', name: 'What are good AI projects for beginners?', acceptedAnswer: { '@type': 'Answer', text: 'Good beginner AI projects include: (1) A PDF Q&A chatbot using LangChain and an LLM API, (2) A CLI code review tool using few-shot prompting, (3) A web summarizer using the Anthropic or OpenAI API, and (4) A simple Gradio chat interface. These projects can be built in 1–4 hours and cover the most important AI engineering skills.' } },
          { '@type': 'Question', name: 'What AI projects look good on a resume?', acceptedAnswer: { '@type': 'Answer', text: 'AI projects that stand out on resumes: (1) A production-quality RAG chatbot with evaluation metrics, (2) A fine-tuned LLM on a custom domain dataset, (3) A multi-agent system with tool use, (4) An AI application with > 100 real users, and (5) An open-source AI tool with GitHub stars. Focus on projects that demonstrate end-to-end engineering, not just API calls.' } },
          { '@type': 'Question', name: 'How long does it take to build an AI project?', acceptedAnswer: { '@type': 'Answer', text: 'A basic LLM-powered project (chatbot, summarizer) takes 4–8 hours. A RAG pipeline with evaluation takes 2–3 days. Fine-tuning an LLM on custom data takes 1–2 days of prep + a few hours of training on Google Colab. A production multi-agent system takes 1–2 weeks. Start simple and iterate.' } },
          { '@type': 'Question', name: 'What is the easiest AI project to build?', acceptedAnswer: { '@type': 'Answer', text: 'The easiest AI project is a web summarizer: paste a URL, the script fetches the page content, sends it to an LLM API, and returns a structured summary. It requires just 20–30 lines of Python and one API key. From there, add a Gradio UI, then a chat interface, then document ingestion — each step teaches a new skill.' } },
          { '@type': 'Question', name: 'Should I use LangChain or LlamaIndex for AI projects?', acceptedAnswer: { '@type': 'Answer', text: 'Use LangChain for agent-heavy and RAG applications — it has the most integrations and community support. Use LlamaIndex for complex document retrieval and multi-document RAG — it\'s optimized for ingestion pipelines. For learning, start with LangChain. For production, evaluate both based on your use case. Both are free and open-source.' } },
        ],
      },
    ],
  },
  {
    slug:        'blog',
    outDir:      'blog',
    url:         'http://localhost:3131/blog/',
    title:       'AI Engineering Blog – Guides & Tutorials | AI Learning Hub',
    description: 'AI engineering articles, tutorials, and roadmaps for software developers. Learn LLMs, RAG, prompt engineering, machine learning, and agentic AI with practical guides.',
    canonical:   'https://ailearnings.in/blog/',
    ogUrl:       'https://ailearnings.in/blog/',
    schema: [
      { '@context': 'https://schema.org', '@type': 'Blog', name: 'AI Learning Hub Blog', url: 'https://ailearnings.in/blog/', description: 'AI engineering articles and tutorials for software developers.', publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' } },
    ],
  },
  {
    slug:        'projects',
    outDir:      'projects',
    url:         'http://localhost:3131/projects/',
    title:       'AI Projects for Developers 2026 – 20 Hands-On Builds | AI Learning Hub',
    description: '20 hands-on AI project guides for developers — beginner to advanced. Build chatbots, RAG systems, AI agents, and real-world AI applications with step-by-step instructions.',
    canonical:   'https://ailearnings.in/projects/',
    ogUrl:       'https://ailearnings.in/projects/',
    schema: [
      { '@context': 'https://schema.org', '@type': 'CollectionPage', name: 'AI Projects for Developers', url: 'https://ailearnings.in/projects/', description: 'Hands-on AI project guides from beginner to advanced for software developers.', publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' } },
    ],
  },
  {
    slug:        'paths',
    outDir:      'paths',
    url:         'http://localhost:3131/paths/',
    title:       'AI Engineering Learning Paths | AI Learning Hub',
    description: 'Structured learning paths for AI engineers, ML engineers, LLM engineers, and AI researchers. Find the right roadmap for your career goal.',
    canonical:   'https://ailearnings.in/paths/',
    ogUrl:       'https://ailearnings.in/paths/',
    schema: [
      { '@context': 'https://schema.org', '@type': 'CollectionPage', name: 'AI Engineering Learning Paths', url: 'https://ailearnings.in/paths/', description: 'Structured learning paths for AI engineers, ML engineers, and LLM engineers.' },
    ],
  },
];

// ── Static file server with SPA fallback ────────────────────────────────────
const MIME = {
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.html': 'text/html; charset=utf-8',
  '.json': 'application/json',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.svg':  'image/svg+xml',
  '.ico':  'image/x-icon',
};

function startServer() {
  // Use absolute path for app.js so it works from any sub-route
  const indexHtml = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8')
    .replace('src="dist/app.js"', 'src="/dist/app.js"');

  const server = http.createServer((req, res) => {
    // Try to serve the exact file first (for dist/app.js, etc.)
    const filePath = path.join(ROOT, req.url.split('?')[0]);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext  = path.extname(filePath);
      const mime = MIME[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime });
      fs.createReadStream(filePath).pipe(res);
      return;
    }
    // Fallback: serve index.html (SPA routing)
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(indexHtml);
  });

  return new Promise((resolve) => {
    server.listen(3131, () => {
      console.log('🌐 Server listening on http://localhost:3131');
      resolve(server);
    });
  });
}

// ── Patch SEO tags into the saved HTML ──────────────────────────────────────
function patchSeo(html, page) {
  // title
  html = html.replace(/<title>[^<]*<\/title>/,
    `<title>${esc(page.title)}</title>`);

  // meta description — update existing or insert after <title>
  if (/<meta\s+name="description"/.test(html)) {
    html = html.replace(/<meta\s+name="description"[^>]*>/,
      `<meta name="description" content="${esc(page.description)}" />`);
  } else {
    html = html.replace('</title>', `</title>\n  <meta name="description" content="${esc(page.description)}" />`);
  }

  // canonical
  if (/<link\s+rel="canonical"/.test(html)) {
    html = html.replace(/<link\s+rel="canonical"[^>]*>/,
      `<link rel="canonical" href="${page.canonical}" />`);
  } else {
    html = html.replace('</title>', `</title>\n  <link rel="canonical" href="${page.canonical}" />`);
  }

  // og:url
  html = html.replace(/(<meta\s+property="og:url"[^>]*content=")[^"]*(")/,
    `$1${page.ogUrl}$2`);

  // og:title
  html = html.replace(/(<meta\s+property="og:title"[^>]*content=")[^"]*(")/,
    `$1${esc(page.title)}$2`);

  // og:description
  html = html.replace(/(<meta\s+property="og:description"[^>]*content=")[^"]*(")/,
    `$1${esc(page.description)}$2`);

  // twitter:url
  html = html.replace(/(<meta\s+name="twitter:url"[^>]*content=")[^"]*(")/,
    `$1${page.canonical}$2`);

  // twitter:title
  html = html.replace(/(<meta\s+name="twitter:title"[^>]*content=")[^"]*(")/,
    `$1${esc(page.title)}$2`);

  // twitter:description
  html = html.replace(/(<meta\s+name="twitter:description"[^>]*content=")[^"]*(")/,
    `$1${esc(page.description)}$2`);

  // Restore non-blocking font loading — Puppeteer fires onload which flips media="print" → media="all"
  html = html.replace(
    /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
    '$1media="print"$2'
  );

  return html;
}

function esc(s) {
  return s.replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Inject JSON-LD schema markup into <head> ─────────────────────────────────
function patchSchema(html, page) {
  if (!page.schema || page.schema.length === 0) return html;
  const blocks = page.schema
    .map(s => `<script type="application/ld+json">\n${JSON.stringify(s, null, 2)}\n</script>`)
    .join('\n  ');
  return html.replace('</head>', `  ${blocks}\n</head>`);
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  const server = await startServer();

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    for (const page of PAGES) {
      console.log(`\n📄 Generating ${page.url}`);
      const tab = await browser.newPage();

      // Block fonts/images to speed up capture (they don't affect content)
      await tab.setRequestInterception(true);
      tab.on('request', (req) => {
        const type = req.resourceType();
        if (type === 'font') {
          req.abort();
        } else {
          req.continue();
        }
      });

      await tab.goto(page.url, { waitUntil: 'networkidle0', timeout: 30000 });

      // Wait for React to render real content into #root
      await tab.waitForFunction(
        () => {
          const root = document.getElementById('root');
          return root && root.children.length > 0 &&
                 root.querySelector('nav') !== null;
        },
        { timeout: 30000 }
      );

      // Small extra wait for any lazy rendering
      await new Promise(r => setTimeout(r, 500));

      const html = await tab.evaluate(() => '<!DOCTYPE html>\n' + document.documentElement.outerHTML);
      await tab.close();

      const patched = patchSchema(patchSeo(html, page), page);

      // Write output
      const outDir  = path.join(ROOT, page.outDir);
      const outFile = path.join(outDir, 'index.html');
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(outFile, patched, 'utf8');

      const kb = Math.round(Buffer.byteLength(patched, 'utf8') / 1024);
      console.log(`   ✓ Wrote ${path.relative(ROOT, outFile)} (${kb} KB)`);
    }
  } finally {
    await browser.close();
    server.close();
    console.log('\n✅ Static generation complete');
  }
}

main().catch(err => { console.error(err); process.exit(1); });
