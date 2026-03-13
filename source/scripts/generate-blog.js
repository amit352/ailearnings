#!/usr/bin/env node
/**
 * generate-blog.js
 * Reads .md files from source/blog/posts/, converts to static HTML using
 * the blog-post.html template, and outputs to /blog/{slug}/index.html.
 * Also generates /blog/index.html from aggregated post metadata.
 *
 * Usage: node scripts/generate-blog.js
 */

const fs   = require('fs');
const path = require('path');
const { marked } = require('marked');

const ROOT              = path.resolve(__dirname, '../..');
const POSTS_DIR         = path.join(__dirname, '../blog/posts');
const ROADMAP_GUIDES_DIR = path.join(__dirname, '../blog/roadmap-guides');
const TEMPLATES_DIR     = path.join(__dirname, '../templates');
const BLOG_OUT          = path.join(ROOT, 'blog');
const CSS_FILE          = path.join(ROOT, 'dist/app.css');

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { meta: {}, content: raw };

  const meta = {};
  match[1].split('\n').forEach(line => {
    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) return;
    const key = line.slice(0, colonIdx).trim();
    let val   = line.slice(colonIdx + 1).trim();
    // Strip surrounding quotes
    if ((val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    // Parse YAML arrays: ["a", "b"]
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

// ── Main ─────────────────────────────────────────────────────────────────────

function main() {
  const postTemplate  = fs.readFileSync(path.join(TEMPLATES_DIR, 'blog-post.html'), 'utf8');
  const indexTemplate = fs.readFileSync(path.join(TEMPLATES_DIR, 'blog-index.html'), 'utf8');
  const inlineCss     = fs.readFileSync(CSS_FILE, 'utf8');

  const mdFiles = fs.readdirSync(POSTS_DIR)
    .filter(f => f.endsWith('.md'))
    .sort();

  if (mdFiles.length === 0) {
    console.log('No .md files found in', POSTS_DIR);
    return;
  }

  const posts = [];

  for (const file of mdFiles) {
    const raw  = fs.readFileSync(path.join(POSTS_DIR, file), 'utf8');
    const { meta, content } = parseFrontmatter(raw);

    const slug        = meta.slug || file.replace('.md', '');
    const title       = meta.title || slug;
    const description = meta.description || '';
    const date        = meta.date || '2026-03-09';
    const canonical   = `https://ailearnings.in/blog/${slug}/`;
    const mins        = readTime(content);
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
      mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
    }, null, 2);

    const breadcrumbSchema = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
        { '@type': 'ListItem', position: 2, name: 'Blog', item: 'https://ailearnings.in/blog/' },
        { '@type': 'ListItem', position: 3, name: title, item: canonical },
      ],
    }, null, 2);

    // Fill template
    let html = postTemplate
      .replace(/\{\{TITLE\}\}/g,           esc(title))
      .replace(/\{\{TITLE_SHORT\}\}/g,      esc(title.length > 50 ? title.slice(0, 47) + '…' : title))
      .replace(/\{\{DESCRIPTION\}\}/g,      esc(description))
      .replace(/\{\{CANONICAL\}\}/g,        canonical)
      .replace(/\{\{DATE\}\}/g,             date)
      .replace(/\{\{DATE_DISPLAY\}\}/g,     formatDate(date))
      .replace(/\{\{READ_TIME\}\}/g,        String(mins))
      .replace(/\{\{CONTENT\}\}/g,          htmlContent)
      .replace(/\{\{INLINE_CSS\}\}/g,       inlineCss)
      .replace(/\{\{ARTICLE_SCHEMA\}\}/g,   articleSchema)
      .replace(/\{\{BREADCRUMB_SCHEMA\}\}/, breadcrumbSchema);

    // Restore non-blocking font (same fix as generate-static.js)
    html = html.replace(
      /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
      '$1media="print"$2'
    );

    const outDir  = path.join(BLOG_OUT, slug);
    const outFile = path.join(outDir, 'index.html');
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(outFile, html, 'utf8');

    const kb = Math.round(Buffer.byteLength(html, 'utf8') / 1024);
    console.log(`   ✓ Wrote blog/${slug}/index.html (${kb} KB)`);

    posts.push({ slug, title, description, date, mins });
  }

  // Sort posts newest-first
  posts.sort((a, b) => b.date.localeCompare(a.date));

  // Generate blog/index.html from template
  const postListHtml = posts.map(({ slug, title, description, date, mins }) => `
    <a href="/blog/${slug}/" class="group block bg-gray-800/40 rounded-xl p-5 border border-white/8 hover:border-blue-500/30 hover:bg-gray-800/60 transition-all" style="text-decoration:none;">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500">${formatDate(date)}</span>
        <span class="text-xs text-gray-600">${mins} min read</span>
      </div>
      <h2 class="text-white font-semibold text-base mb-2 leading-snug group-hover:text-blue-300 transition-colors">${esc(title)}</h2>
      <p class="text-gray-400 text-sm leading-relaxed mb-3">${esc(description)}</p>
      <div class="flex items-center gap-1 text-xs text-blue-400">Read article <span>→</span></div>
    </a>`).join('\n');

  let indexHtml = indexTemplate
    .replace('{{POST_LIST}}', postListHtml)
    .replace('{{INLINE_CSS}}', inlineCss);

  indexHtml = indexHtml.replace(
    /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
    '$1media="print"$2'
  );

  fs.mkdirSync(BLOG_OUT, { recursive: true });
  fs.writeFileSync(path.join(BLOG_OUT, 'index.html'), indexHtml, 'utf8');
  const idxKb = Math.round(Buffer.byteLength(indexHtml, 'utf8') / 1024);
  console.log(`   ✓ Wrote blog/index.html (${idxKb} KB) — ${posts.length} posts`);

  // ── Roadmap Guides ───────────────────────────────────────────────────────────
  if (fs.existsSync(ROADMAP_GUIDES_DIR)) {
    const rgFiles = fs.readdirSync(ROADMAP_GUIDES_DIR).filter(f => f.endsWith('.md')).sort();
    const rgOut   = path.join(BLOG_OUT, 'roadmap-guides');
    const rgPosts = [];

    for (const file of rgFiles) {
      const raw = fs.readFileSync(path.join(ROADMAP_GUIDES_DIR, file), 'utf8');
      const { meta, content } = parseFrontmatter(raw);

      const slug        = meta.slug || file.replace('.md', '');
      const title       = meta.title || slug;
      const description = meta.description || '';
      const date        = meta.date || '2026-03-10';
      const canonical   = `https://ailearnings.in/blog/roadmap-guides/${slug}/`;
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
        mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
      }, null, 2);

      const breadcrumbSchema = JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home',           item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'Blog',           item: 'https://ailearnings.in/blog/' },
          { '@type': 'ListItem', position: 3, name: 'Roadmap Guides', item: 'https://ailearnings.in/blog/roadmap-guides/' },
          { '@type': 'ListItem', position: 4, name: title,            item: canonical },
        ],
      }, null, 2);

      let html = postTemplate
        .replace(/\{\{TITLE\}\}/g,           esc(title))
        .replace(/\{\{TITLE_SHORT\}\}/g,      esc(title.length > 50 ? title.slice(0, 47) + '…' : title))
        .replace(/\{\{DESCRIPTION\}\}/g,      esc(description))
        .replace(/\{\{CANONICAL\}\}/g,        canonical)
        .replace(/\{\{DATE\}\}/g,             date)
        .replace(/\{\{DATE_DISPLAY\}\}/g,     formatDate(date))
        .replace(/\{\{READ_TIME\}\}/g,        String(mins))
        .replace(/\{\{CONTENT\}\}/g,          htmlContent)
        .replace(/\{\{INLINE_CSS\}\}/g,       inlineCss)
        .replace(/\{\{ARTICLE_SCHEMA\}\}/g,   articleSchema)
        .replace(/\{\{BREADCRUMB_SCHEMA\}\}/, breadcrumbSchema);

      html = html.replace(
        /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
        '$1media="print"$2'
      );

      const outDir = path.join(rgOut, slug);
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, 'index.html'), html, 'utf8');
      const kb = Math.round(Buffer.byteLength(html, 'utf8') / 1024);
      console.log(`   ✓ Wrote blog/roadmap-guides/${slug}/index.html (${kb} KB)`);

      rgPosts.push({ slug, title, description, date, mins });
    }

    // Generate blog/roadmap-guides/index.html
    rgPosts.sort((a, b) => b.date.localeCompare(a.date));
    const rgListHtml = rgPosts.map(({ slug, title, description, date, mins }) => `
    <a href="/blog/roadmap-guides/${slug}/" class="group block bg-gray-800/40 rounded-xl p-5 border border-white/8 hover:border-blue-500/30 hover:bg-gray-800/60 transition-all" style="text-decoration:none;">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500">${formatDate(date)}</span>
        <span class="text-xs text-gray-600">${mins} min read</span>
      </div>
      <h2 class="text-white font-semibold text-base mb-2 leading-snug group-hover:text-blue-300 transition-colors">${esc(title)}</h2>
      <p class="text-gray-400 text-sm leading-relaxed mb-3">${esc(description)}</p>
      <div class="flex items-center gap-1 text-xs text-blue-400">Read guide <span>→</span></div>
    </a>`).join('\n');

    let rgIndexHtml = indexTemplate
      .replace('{{POST_LIST}}', rgListHtml)
      .replace('{{INLINE_CSS}}', inlineCss);
    rgIndexHtml = rgIndexHtml.replace(
      /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
      '$1media="print"$2'
    );

    fs.mkdirSync(rgOut, { recursive: true });
    fs.writeFileSync(path.join(rgOut, 'index.html'), rgIndexHtml, 'utf8');
    const rgIdxKb = Math.round(Buffer.byteLength(rgIndexHtml, 'utf8') / 1024);
    console.log(`   ✓ Wrote blog/roadmap-guides/index.html (${rgIdxKb} KB) — ${rgPosts.length} guides`);
  }

  console.log('\n✅ Blog generation complete');
}

main();
