#!/usr/bin/env node
/**
 * generate-paths.js
 * Reads .md files from source/paths/, converts to static HTML using
 * the path-page.html template, and outputs to /paths/{slug}/index.html.
 * Also generates /paths/index.html from aggregated path metadata.
 *
 * Usage: node scripts/generate-paths.js
 */

const fs   = require('fs');
const path = require('path');
const { marked } = require('marked');
const hljs = require('highlight.js');

const LANG_META = {
  python: { label: 'Python', dot: '#3b82f6' }, py: { label: 'Python', dot: '#3b82f6' },
  javascript: { label: 'JavaScript', dot: '#f59e0b' }, js: { label: 'JavaScript', dot: '#f59e0b' },
  typescript: { label: 'TypeScript', dot: '#60a5fa' }, ts: { label: 'TypeScript', dot: '#60a5fa' },
  bash: { label: 'Bash', dot: '#10b981' }, sh: { label: 'Shell', dot: '#10b981' }, shell: { label: 'Shell', dot: '#10b981' },
  json: { label: 'JSON', dot: '#a78bfa' }, yaml: { label: 'YAML', dot: '#fb923c' },
  html: { label: 'HTML', dot: '#f87171' }, css: { label: 'CSS', dot: '#38bdf8' },
  sql: { label: 'SQL', dot: '#34d399' }, rust: { label: 'Rust', dot: '#f97316' },
  go: { label: 'Go', dot: '#22d3ee' }, java: { label: 'Java', dot: '#f59e0b' },
};

// Phase color palette (Phase 0–5)
const PHASE_COLORS = ['#10b981','#6366f1','#3b82f6','#8b5cf6','#f59e0b','#ec4899'];

marked.use({
  renderer: {
    heading(token) {
      const text = token.text;
      const level = token.depth;
      const tag = `h${level}`;
      // Detect "Phase N:" headings and color them
      const phaseMatch = text.match(/^Phase\s+(\d+)\s*:/);
      if (phaseMatch) {
        const phaseNum = parseInt(phaseMatch[1], 10);
        const color = PHASE_COLORS[phaseNum % PHASE_COLORS.length] || '#6366f1';
        return `<${tag} class="phase-heading" style="color:${color}"><span class="phase-badge" style="background:${color}20;border:1px solid ${color}40;color:${color}">Phase ${phaseMatch[1]}</span>${text.replace(/^Phase\s+\d+\s*:\s*/, '')}</${tag}>\n`;
      }
      return `<${tag}>${text}</${tag}>\n`;
    },
    code(token) {
      const lang = (token.lang || '').toLowerCase();
      const code = token.text;
      let highlighted;
      if (lang && hljs.getLanguage(lang)) {
        highlighted = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
      } else {
        highlighted = hljs.highlightAuto(code).value;
      }
      const meta = LANG_META[lang];
      const label = meta ? meta.label : (lang || 'plaintext');
      const dot = meta ? meta.dot : '#6b7280';
      return `<div class="code-viewer"><div class="code-viewer-header"><span class="code-viewer-dot" style="background:${dot}"></span><span class="code-viewer-lang">${label}</span><button class="code-copy-btn" onclick="(function(b){var p=b.closest('.code-viewer').querySelector('code');navigator.clipboard.writeText(p.innerText).then(function(){b.textContent='Copied!';b.classList.add('copied');setTimeout(function(){b.textContent='Copy';b.classList.remove('copied');},2000);});})(this)">Copy</button></div><pre><code class="hljs language-${lang || 'plaintext'}">${highlighted}</code></pre></div>`;
    }
  }
});

const ROOT          = path.resolve(__dirname, '../..');
const PATHS_DIR     = path.join(__dirname, '../paths');
const TEMPLATES_DIR = path.join(__dirname, '../templates');
const OUT_DIR       = path.join(ROOT, 'paths');
const CSS_FILE      = path.join(ROOT, 'dist/app.css');

// ── Helpers ───────────────────────────────────────────────────────────────────

function parseFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { meta: {}, content: raw };

  const meta = {};
  match[1].split('\n').forEach(line => {
    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) return;
    const key = line.slice(0, colonIdx).trim();
    let val   = line.slice(colonIdx + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    meta[key] = val;
  });

  return { meta, content: match[2] };
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Main ──────────────────────────────────────────────────────────────────────

function main() {
  const postTemplate  = fs.readFileSync(path.join(TEMPLATES_DIR, 'path-page.html'), 'utf8');
  const indexTemplate = fs.readFileSync(path.join(TEMPLATES_DIR, 'paths-index.html'), 'utf8');
  const inlineCss     = fs.readFileSync(CSS_FILE, 'utf8');

  const mdFiles = fs.readdirSync(PATHS_DIR)
    .filter(f => f.endsWith('.md'))
    .sort();

  if (mdFiles.length === 0) {
    console.log('No .md files found in', PATHS_DIR);
    return;
  }

  const pathsMeta = [];

  for (const file of mdFiles) {
    const raw  = fs.readFileSync(path.join(PATHS_DIR, file), 'utf8');
    const { meta, content } = parseFrontmatter(raw);

    const slug        = meta.slug || file.replace('.md', '');
    const title       = meta.title || slug;
    const description = meta.description || '';
    const timeline    = meta.timeline || '';
    const salary      = meta.salary || '';
    const demand      = meta.demand || '';
    const canonical   = `https://ailearnings.in/paths/${slug}/`;
    const date        = '2026-03-10';
    const htmlContent = marked.parse(content);

    const articleSchema = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'Article',
      headline: title,
      description: description,
      image: 'https://ailearnings.in/og-image.jpg',
      url: canonical,
      datePublished: date,
      dateModified: date,
      author: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
      publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
    }, null, 2);

    const breadcrumbSchema = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home',           item: 'https://ailearnings.in/' },
        { '@type': 'ListItem', position: 2, name: 'Learning Paths', item: 'https://ailearnings.in/paths/' },
        { '@type': 'ListItem', position: 3, name: title,            item: canonical },
      ],
    }, null, 2);

    let html = postTemplate
      .replace(/\{\{TITLE\}\}/g,           esc(title))
      .replace(/\{\{TITLE_SHORT\}\}/g,     esc(title.length > 50 ? title.slice(0, 47) + '…' : title))
      .replace(/\{\{DESCRIPTION\}\}/g,     esc(description))
      .replace(/\{\{CANONICAL\}\}/g,       canonical)
      .replace(/\{\{TIMELINE\}\}/g,        esc(timeline))
      .replace(/\{\{SALARY\}\}/g,          esc(salary))
      .replace(/\{\{CONTENT\}\}/g,         htmlContent)
      .replace(/\{\{INLINE_CSS\}\}/g,      inlineCss)
      .replace(/\{\{ARTICLE_SCHEMA\}\}/g,  articleSchema)
      .replace(/\{\{BREADCRUMB_SCHEMA\}\}/, breadcrumbSchema);

    html = html.replace(
      /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
      '$1media="print"$2'
    );

    const outDir  = path.join(OUT_DIR, slug);
    const outFile = path.join(outDir, 'index.html');
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(outFile, html, 'utf8');

    const kb = Math.round(Buffer.byteLength(html, 'utf8') / 1024);
    console.log(`   ✓ Wrote paths/${slug}/index.html (${kb} KB)`);

    pathsMeta.push({ slug, title, description, timeline, salary, demand });
  }

  // Demand order for sorting
  const demandOrder = { 'Very High': 0, 'High': 1, 'Moderate': 2 };
  pathsMeta.sort((a, b) => (demandOrder[a.demand] ?? 9) - (demandOrder[b.demand] ?? 9));

  const demandColors = {
    'Very High': 'color:#34d399;background:rgba(16,185,129,0.1)',
    'High':      'color:#fbbf24;background:rgba(245,158,11,0.1)',
    'Moderate':  'color:#60a5fa;background:rgba(59,130,246,0.1)',
  };

  const pathList = pathsMeta.map(({ slug, title, description, timeline, salary, demand }) => {
    const demandStyle = demandColors[demand] || 'color:#9ca3af;background:rgba(156,163,175,0.1)';
    return `
    <a href="/paths/${esc(slug)}/" style="display:block;background:rgba(31,41,55,0.4);border-radius:0.75rem;padding:1.25rem 1.5rem;border:1px solid rgba(255,255,255,0.08);text-decoration:none;margin-bottom:1rem;transition:border-color 0.15s;">
      <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;flex-wrap:wrap;">
        ${timeline ? `<span style="font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:9999px;color:#34d399;background:rgba(16,185,129,0.1);">${esc(timeline)}</span>` : ''}
        ${salary   ? `<span style="font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:9999px;color:#fbbf24;background:rgba(245,158,11,0.1);">${esc(salary)}</span>` : ''}
        ${demand   ? `<span style="font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:9999px;${demandStyle};">Demand: ${esc(demand)}</span>` : ''}
      </div>
      <h2 style="color:#f9fafb;font-size:1rem;font-weight:700;margin:0 0 0.375rem;line-height:1.4;">${esc(title)}</h2>
      <p style="color:#9ca3af;font-size:0.875rem;line-height:1.6;margin:0;">${esc(description)}</p>
    </a>`;
  }).join('\n');

  let indexHtml = indexTemplate
    .replace('{{PATH_LIST}}', pathList)
    .replace('{{INLINE_CSS}}', inlineCss);

  indexHtml = indexHtml.replace(
    /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
    '$1media="print"$2'
  );

  // paths/index.html is generated by generate-static.js (Puppeteer) to match the React layout.
  // Do not write it here to avoid overwriting with a mismatched template layout.
  console.log(`   ℹ paths/index.html skipped — regenerate with: node scripts/generate-static.js`);

  console.log('\n✅ Path generation complete');
}

main();
