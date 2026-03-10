#!/usr/bin/env node
/**
 * generate-projects.js
 * Reads .md files from source/projects/posts/, converts to static HTML using
 * the project-post.html template, and outputs to /projects/{slug}/index.html.
 * Also generates /projects/index.html from aggregated project metadata.
 *
 * Usage: node scripts/generate-projects.js
 */

const fs   = require('fs');
const path = require('path');
const { marked } = require('marked');

const ROOT          = path.resolve(__dirname, '../..');
const POSTS_DIR     = path.join(__dirname, '../projects/posts');
const TEMPLATES_DIR = path.join(__dirname, '../templates');
const OUT_DIR       = path.join(ROOT, 'projects');
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
    if (val.startsWith('[')) {
      try { meta[key] = JSON.parse(val.replace(/'/g, '"')); }
      catch { meta[key] = val; }
    } else {
      meta[key] = val;
    }
  });

  return { meta, content: match[2] };
}

function readTime(markdown) {
  const words = markdown.replace(/[#*`>\[\]()-]/g, '').split(/\s+/).length;
  return Math.max(1, Math.round(words / 220));
}

function formatDate(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' });
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Main ──────────────────────────────────────────────────────────────────────

function main() {
  const postTemplate  = fs.readFileSync(path.join(TEMPLATES_DIR, 'project-post.html'), 'utf8');
  const indexTemplate = fs.readFileSync(path.join(TEMPLATES_DIR, 'projects-index.html'), 'utf8');
  const inlineCss     = fs.readFileSync(CSS_FILE, 'utf8');

  const mdFiles = fs.readdirSync(POSTS_DIR)
    .filter(f => f.endsWith('.md'))
    .sort();

  if (mdFiles.length === 0) {
    console.log('No .md files found in', POSTS_DIR);
    return;
  }

  const projects = [];

  for (const file of mdFiles) {
    const raw  = fs.readFileSync(path.join(POSTS_DIR, file), 'utf8');
    const { meta, content } = parseFrontmatter(raw);

    const slug        = meta.slug || file.replace('.md', '');
    const title       = meta.title || slug;
    const description = meta.description || '';
    const date        = meta.date || '2026-03-10';
    const level       = meta.level || 'Beginner';
    const time        = meta.time || '2–4 hours';
    const stack       = meta.stack || 'Python';
    const canonical   = `https://ailearnings.in/projects/${slug}/`;
    const mins        = readTime(content);
    const htmlContent = marked.parse(content);

    const articleSchema = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'Article',
      headline: title,
      description: description,
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
        { '@type': 'ListItem', position: 1, name: 'Home',     item: 'https://ailearnings.in/' },
        { '@type': 'ListItem', position: 2, name: 'Projects', item: 'https://ailearnings.in/projects/' },
        { '@type': 'ListItem', position: 3, name: title,      item: canonical },
      ],
    }, null, 2);

    let html = postTemplate
      .replace(/\{\{TITLE\}\}/g,           esc(title))
      .replace(/\{\{TITLE_SHORT\}\}/g,     esc(title.length > 50 ? title.slice(0, 47) + '…' : title))
      .replace(/\{\{DESCRIPTION\}\}/g,     esc(description))
      .replace(/\{\{CANONICAL\}\}/g,       canonical)
      .replace(/\{\{DATE\}\}/g,            date)
      .replace(/\{\{DATE_DISPLAY\}\}/g,    formatDate(date))
      .replace(/\{\{READ_TIME\}\}/g,       String(mins))
      .replace(/\{\{LEVEL\}\}/g,           esc(level))
      .replace(/\{\{TIME\}\}/g,            esc(time))
      .replace(/\{\{STACK\}\}/g,           esc(stack))
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
    console.log(`   ✓ Wrote projects/${slug}/index.html (${kb} KB)`);

    projects.push({ slug, title, description, date, level, time, stack });
  }

  // Sort by level then date
  const levelOrder = { 'Beginner': 0, 'Intermediate': 1, 'Advanced': 2 };
  projects.sort((a, b) =>
    (levelOrder[a.level] || 0) - (levelOrder[b.level] || 0) ||
    b.date.localeCompare(a.date)
  );

  // Group by level for the index
  const byLevel = {};
  for (const p of projects) {
    if (!byLevel[p.level]) byLevel[p.level] = [];
    byLevel[p.level].push(p);
  }

  const levelColors = {
    'Beginner':     { bg: 'bg-blue-600/20',  text: 'text-blue-400' },
    'Intermediate': { bg: 'bg-yellow-600/20', text: 'text-yellow-400' },
    'Advanced':     { bg: 'bg-red-600/20',    text: 'text-red-400' },
  };

  const levelSections = Object.entries(byLevel).map(([level, items]) => {
    const color = levelColors[level] || { bg: 'bg-gray-600/20', text: 'text-gray-400' };
    const cards = items.map(({ slug, title, description, time, stack }) => `
    <a href="/projects/${slug}/" class="block bg-gray-800/40 rounded-xl p-5 border border-white/8 hover:border-green-500/30 hover:bg-gray-800/60 transition-all" style="text-decoration:none;">
      <div class="flex items-center gap-2 mb-2">
        <span class="${color.bg} ${color.text} text-xs font-semibold px-2 py-0.5 rounded-full">${esc(level)}</span>
        <span class="text-gray-500 text-xs">${esc(time)}</span>
      </div>
      <h2 class="text-white font-semibold text-base mb-2 leading-snug">${esc(title)}</h2>
      <p class="text-gray-400 text-sm leading-relaxed mb-3">${esc(description)}</p>
      <p class="text-xs text-gray-500">Stack: ${esc(stack)}</p>
    </a>`).join('\n');

    return `
  <div class="mb-10">
    <h2 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
      <span class="${color.bg} ${color.text} text-xs font-semibold px-3 py-1 rounded-full">${esc(level)}</span>
      <span class="text-gray-500 text-sm font-normal">${items.length} projects</span>
    </h2>
    <div class="space-y-4">
      ${cards}
    </div>
  </div>`;
  }).join('\n');

  let indexHtml = indexTemplate
    .replace('{{LEVEL_SECTIONS}}', levelSections)
    .replace('{{INLINE_CSS}}', inlineCss);

  indexHtml = indexHtml.replace(
    /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
    '$1media="print"$2'
  );

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUT_DIR, 'index.html'), indexHtml, 'utf8');
  const idxKb = Math.round(Buffer.byteLength(indexHtml, 'utf8') / 1024);
  console.log(`   ✓ Wrote projects/index.html (${idxKb} KB) — ${projects.length} projects`);

  console.log('\n✅ Project generation complete');
}

main();
