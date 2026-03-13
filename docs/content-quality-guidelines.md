# Content Quality Guidelines — AI Learning Hub

> **Purpose:** Define and enforce consistent standards across all content on ailearnings.in.
> These guidelines apply to blog posts, roadmap guides, project guides, and learning path pages.

---

## 1. Tone and Voice

### Core principles
- **Developer-first.** Write for working developers, not students or academics. Assume the reader can code but may be new to AI.
- **Direct and specific.** Avoid generic advice ("it depends", "there are many ways"). Give concrete recommendations with rationale.
- **Confidence without condescension.** Write authoritatively but never talk down. Never use phrases like "simply" or "just" for non-trivial steps.
- **Actionable throughout.** Every section should leave the reader with something to do, try, or understand more concretely.

### Anti-patterns to avoid
| Anti-pattern | Instead |
|---|---|
| "There are many approaches to consider" | "Use approach X for Y, approach Z for W" |
| "As you may already know..." | Start directly with the content |
| "In this article, we will..." | Start with the first piece of content |
| "Hopefully, by now you understand..." | State what was just covered factually |
| Generic encouragement ("You can do it!") | Specific milestone confirmation |

---

## 2. Formatting Standards

### Headings
- `##` — major sections (What Is, Why Use, How It Works, Implementation, etc.)
- `###` — subsections within a major section
- `####` — avoid where possible; flatten the hierarchy
- Headings must be sentence-style caps: "What is a vector database" not "What Is A Vector Database"
- Every guide must start with a concrete description of what the reader will learn/build

### Code blocks
- All code examples must be complete and runnable (no `# ... rest of code here`)
- Use language tags: ` ```python `, ` ```bash `, ` ```json `
- Include import statements; readers should be able to copy-paste and execute
- Maximum 40 lines per code block for readability; break into named steps if longer
- Add a comment line before each logical step in multi-step examples

### Tables
- Use for comparison, tool reference, or structured data — not for prose
- Keep columns to 3–4 max
- All cells must have content; no empty cells

### Lists
- Unordered lists: use for non-sequential items (skills, tools, considerations)
- Ordered lists: use only for sequential steps
- Maximum 7 items before splitting into subsections
- Each list item should be a complete thought (avoid single-word items)

---

## 3. Content Structure by Type

### Blog posts (`/blog/`)

Required sections (in order):
1. **Opening paragraph** — what this is, who it's for, what they'll learn (2–3 sentences)
2. **Prerequisites** — what the reader needs to know first (inline or bullet list)
3. **Core concept sections** — 3–6 `##` sections covering the topic
4. **Implementation / Code** — at least one working code example
5. **Common mistakes or troubleshooting** — real failure modes with fixes
6. **Summary** — 3–5 bullet points capturing the key takeaways
7. **Next steps** — 2–3 links to related guides or projects

Minimum word count: 1,200 words
Target word count: 1,500–2,500 words

### Roadmap guides (`/blog/roadmap-guides/`)

Required sections (in order):
1. **What you'll learn** — explicit learning objectives as a bullet list
2. **Concept explanation** — the "why" before the "how"
3. **Step-by-step walkthrough** — with code at each step
4. **Mental model** — a concise analogy or diagram description
5. **Common pitfalls** — 3–5 real mistakes beginners make
6. **Practice exercises** — 2–3 specific tasks to reinforce learning
7. **Further reading** — curated links, not a generic list

Minimum word count: 1,500 words
Must include at least 2 code examples

### Project guides (`/projects/`)

Required sections (in order):
1. **Project overview** — what it does, screenshot or architecture diagram (text)
2. **What you'll learn** — explicit skill bullets
3. **Prerequisites** — tools, libraries, accounts needed
4. **Architecture** — component diagram in text/table form
5. **Implementation** — numbered steps with code for each
6. **Testing it** — how to verify it works
7. **Deployment** — at least one deployment path (Railway, Vercel, Docker, etc.)
8. **Extensions** — 3–5 ideas to take the project further

Minimum word count: 1,200 words
Must include at least 3 code blocks with real, runnable snippets

### Learning path pages (`/paths/`)

Required sections (in order):
1. **Role description** — what the job is, who hires for it (2–3 paragraphs)
2. **Skills required** — Must-Have / Important / Nice to Have tiers
3. **Learning phases** — 4–6 phases, each with: Learn / Build / Milestone
4. **Recommended projects table** — project name, skills, level
5. **Key tools table** — category, tools
6. **Interview topics** — 5–8 specific questions with implied answers
7. **Next paths** — 2 links to adjacent career paths

---

## 4. SEO Standards

### Title format
- Blog posts: `{Topic}: {Benefit or Outcome}` — e.g., "RAG Systems: How to Build Retrieval-Augmented Generation"
- Project guides: `Build {Project Name}: {Key Skill}` — e.g., "Build an AI Support Bot: RAG + FastAPI"
- Learning paths: `{Role} Learning Path` — e.g., "AI Engineer Learning Path"
- Roadmap guides: `{Topic} for AI Developers` or `{Topic} Explained`

### Meta description
- 140–160 characters
- Must include the primary keyword naturally
- Must describe what the reader gets (not just what the article is about)
- No clickbait; factual and specific

### Internal linking
- Every piece of content must link to at least 2 other pages on the site
- Blog posts should link to related projects
- Project guides should link to the prerequisite blog guides
- Learning paths should link to both guides and projects for each phase

### Keyword placement
- Primary keyword in: `<h1>`, first paragraph, one `<h2>`, meta description
- Do not repeat the exact primary keyword more than 3 times in the first 300 words
- Use semantic variants naturally throughout

---

## 5. Technical Accuracy Standards

### Code examples
- All Python code must be compatible with Python 3.9+
- All OpenAI API calls must use the current SDK (v1.x, `from openai import OpenAI`)
- All Anthropic API calls must use the current SDK (`from anthropic import Anthropic`)
- Package versions must be specified in `requirements.txt` snippets
- Avoid deprecated patterns: no `openai.Completion.create()`, no `openai.ChatCompletion.create()`

### Concept explanations
- Define every technical term the first time it appears
- Use concrete numbers when comparing approaches (e.g., "reduces memory by ~4x" not "reduces memory significantly")
- Distinguish between what something is vs. how to use it vs. when to use it
- Acknowledge tradeoffs honestly; do not oversell any single approach

### Validation checklist before publishing
- [ ] All code blocks execute without errors
- [ ] All internal links resolve to real pages
- [ ] Meta description is 140–160 characters
- [ ] H1 contains the primary keyword
- [ ] At least 2 internal links included
- [ ] JSON-LD schema is present and valid
- [ ] Canonical URL matches the page URL

---

## 6. Content Inventory (as of 2026-03-10)

### Phase 1 — Developer Guides (`/blog/`) — 28 posts

**Machine Learning**
- Machine Learning Basics for Developers
- Supervised Learning Guide
- Model Evaluation and Metrics
- Feature Engineering Guide
- Python for Machine Learning
- Machine Learning Roadmap

**LLM Engineering**
- Transformer Architecture Explained
- Fine-Tuning LLMs Guide
- LLM Inference and Serving
- OpenAI API Complete Guide
- Open Source LLMs Guide

**Prompt Engineering**
- Prompt Engineering Techniques
- Chain-of-Thought Prompting

**RAG Systems**
- RAG Tutorial Step by Step
- RAG System Architecture
- Vector Database Guide
- LangChain RAG Tutorial
- Embeddings Explained
- Document Chunking Strategies

**AI Agents**
- AI Agent Fundamentals
- Tool Use and Function Calling
- Multi-Agent Systems

**AI Application Development**
- LangChain Tutorial Complete
- Deploying AI Applications
- Building AI Chatbots
- AI Application Architecture

**Other**
- AI Roadmap for Developers
- How to Become an AI Engineer

### Phase 2 — Project Guides (`/projects/`) — 20 projects

**Beginner (7)**
- AI Chatbot (Python)
- Document Summarizer
- AI Quiz Generator
- AI Resume Analyzer
- AI Code Explainer
- AI Email Writer
- Sentiment Analyzer

**Intermediate (8)**
- RAG Document Assistant
- AI Meeting Summarizer
- AI Research Assistant
- AI Data Analyst
- AI Support Bot
- AI Content Moderator
- Voice AI Assistant
- AI Study Planner (reclassified)

**Advanced (6)**
- AI Personal Knowledge Base
- AI Code Review Assistant
- Multi-Agent Research System
- AI Sales Agent
- AI Security Analyzer
- AI Study Planner

### Phase 3 — Roadmap Guides (`/blog/roadmap-guides/`) — 14 guides

- AI Foundations for Developers
- Python for AI Complete Guide
- Linear Algebra for AI
- Statistics for Machine Learning
- Neural Networks from Scratch
- ML Project Workflow
- Deep Learning Fundamentals
- PyTorch for AI Developers
- How LLMs Work
- Working with Local LLMs
- Semantic Search Explained
- Production RAG Best Practices
- Building AI Agents Guide
- AI Agent Evaluation

### Phase 4 — Learning Paths (`/paths/`) — 5 paths

- AI Engineer (6–12 months, $130k–$220k, Demand: Very High)
- ML Engineer (8–14 months, $140k–$230k, Demand: High)
- LLM Engineer (9–15 months, $150k–$260k, Demand: Very High)
- AI Product Engineer (6–10 months, $130k–$210k, Demand: High)
- AI Research Engineer (12–24 months, $160k–$350k, Demand: Moderate)

---

## 7. Content Gap Analysis

### High-priority gaps to fill in future phases

| Topic | Target Keyword | Estimated Volume | Type |
|---|---|---|---|
| Kubernetes for ML | kubernetes ml deployment | 2k/mo | Blog |
| LangGraph tutorial | langgraph tutorial | 3k/mo | Blog |
| Ollama guide | ollama setup | 8k/mo | Blog |
| Claude API guide | anthropic claude api | 4k/mo | Blog |
| Mistral fine-tuning | mistral fine-tuning | 2k/mo | Blog |
| FastAPI AI backend | fastapi ml api | 5k/mo | Blog |
| AI interview prep | ai engineer interview | 10k/mo | Landing page |
| Data Engineer path | data engineer roadmap | 15k/mo | Path |

### Structural improvements
- Add "Updated on" dates to all content
- Add estimated reading time to all articles
- Add "Related guides" sidebar to blog posts
- Add difficulty tags to all project guides

---

## 8. Build System Reference

| Content type | Source directory | Generator script | Output directory |
|---|---|---|---|
| Blog posts | `source/blog/posts/` | `generate-blog.js` | `blog/` |
| Roadmap guides | `source/blog/roadmap-guides/` | `generate-blog.js` | `blog/roadmap-guides/` |
| Project guides | `source/projects/posts/` | `generate-projects.js` | `projects/` |
| Learning paths | `source/paths/` | `generate-paths.js` | `paths/` |
| Core pages | `source/app.jsx` | `generate-static.js` (Puppeteer) | root |

### Run all generators
```bash
cd source
npm run generate:all
```

### Run individual generators
```bash
npm run generate:blog       # blog posts + roadmap guides
npm run generate:projects   # project guides
npm run generate:paths      # learning paths
npm run generate            # core pages (Puppeteer, slow)
```

### Adding new content
1. Create a `.md` file in the appropriate source directory
2. Add frontmatter with `title`, `description`, `slug`, and type-specific fields
3. Run the relevant generator script
4. Add the new URL to `sitemap.xml`
5. Add internal links from related existing pages
