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
