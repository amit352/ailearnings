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
    title:       'AI Learning Hub – Zero to Hero Roadmap for Developers',
    description: 'Free 7-phase AI roadmap for software developers. Master LLMs, Prompt Engineering, RAG, and Agentic AI with curated free resources and hands-on projects.',
    canonical:   'https://ailearnings.in/',
    ogUrl:       'https://ailearnings.in/',
  },
  {
    slug:        'prep-plan',
    outDir:      'prep-plan',
    url:         'http://localhost:3131/prep-plan/',
    title:       'AI Interview Prep Plan – 6-Week Fast Track | AI Learning Hub',
    description: 'Structured 6-week AI prep plan for software developers. Cover LLMs, Prompt Engineering, RAG, and Agentic AI with free resources — 4–6 hours per week.',
    canonical:   'https://ailearnings.in/prep-plan/',
    ogUrl:       'https://ailearnings.in/prep-plan/',
  },
  {
    slug:        'genai-guide',
    outDir:      'genai-guide',
    url:         'http://localhost:3131/genai-guide/',
    title:       'Generative AI Guide – Text, Code, Image & Audio | AI Learning Hub',
    description: 'Complete overview of Generative AI domains — text, code, image, audio. How each works, top models, tools, and a learning roadmap for developers.',
    canonical:   'https://ailearnings.in/genai-guide/',
    ogUrl:       'https://ailearnings.in/genai-guide/',
  },
  {
    slug:        'prompt-eng',
    outDir:      'prompt-eng',
    url:         'http://localhost:3131/prompt-eng/',
    title:       'Prompt Engineering Guide – Techniques & Templates | AI Learning Hub',
    description: 'Master prompt engineering with 15 techniques from zero-shot to tree-of-thoughts, copy-paste templates for coding, writing, and research, plus a 6-week practice plan.',
    canonical:   'https://ailearnings.in/prompt-eng/',
    ogUrl:       'https://ailearnings.in/prompt-eng/',
  },
  {
    slug:        'resources',
    outDir:      'resources',
    url:         'http://localhost:3131/resources/',
    title:       'AI Learning Resources – Books & Courses by Phase | AI Learning Hub',
    description: 'Curated books, video courses, and references mapped to each phase of the AI roadmap. Includes free and O\'Reilly resources for every level.',
    canonical:   'https://ailearnings.in/resources/',
    ogUrl:       'https://ailearnings.in/resources/',
  },
  {
    slug:        'readiness',
    outDir:      'readiness',
    url:         'http://localhost:3131/readiness/',
    title:       'AI Phase Readiness Checker – Know When to Move On | AI Learning Hub',
    description: 'Check if you\'re ready to advance to the next AI learning phase. Green flags, red flags, move-on rules, and an at-a-glance progress overview for all 7 phases.',
    canonical:   'https://ailearnings.in/readiness/',
    ogUrl:       'https://ailearnings.in/readiness/',
  },
  {
    slug:        'beyond-roadmap',
    outDir:      'beyond-roadmap',
    url:         'http://localhost:3131/beyond-roadmap/',
    title:       'Beyond the AI Roadmap – Knowledge Gaps & What\'s Next | AI Learning Hub',
    description: 'Finished the AI roadmap? See your knowledge gaps by area, explore advanced topics not covered in the core roadmap, and plan your specialization.',
    canonical:   'https://ailearnings.in/beyond-roadmap/',
    ogUrl:       'https://ailearnings.in/beyond-roadmap/',
  },
  {
    slug:        'assessment',
    outDir:      'assessment',
    url:         'http://localhost:3131/assessment/',
    title:       'AI Knowledge Assessment – Where You\'ll Stand After the Roadmap | AI Learning Hub',
    description: 'An honest assessment of your AI engineer skill level after completing all 7 phases — what you can do, what gaps remain, and the best next steps.',
    canonical:   'https://ailearnings.in/assessment/',
    ogUrl:       'https://ailearnings.in/assessment/',
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
        { timeout: 15000 }
      );

      // Small extra wait for any lazy rendering
      await new Promise(r => setTimeout(r, 500));

      const html = await tab.evaluate(() => '<!DOCTYPE html>\n' + document.documentElement.outerHTML);
      await tab.close();

      const patched = patchSeo(html, page);

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
