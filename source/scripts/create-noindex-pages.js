#!/usr/bin/env node
/**
 * create-noindex-pages.js
 * Creates noindex + redirect HTML pages at legacy URL paths so Google
 * permanently deindexes them instead of serving the generic 404 page.
 *
 * Usage:
 *   node scripts/create-noindex-pages.js
 *
 * Output: files written to repo root (one directory up from source/).
 */

const fs   = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');

// Legacy paths to suppress.
// Each entry creates ROOT/<dir>/index.html with noindex + redirect to /.
const LEGACY_PATHS = [
  'lander',
  'comments/feed',
  'search',
  'category',
  'author',
  'feed',
  'wp-admin',
  'wp-login.php',
];

function noindexPage(redirectTo = '/') {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="robots" content="noindex, nofollow">
  <meta http-equiv="refresh" content="0; url=${redirectTo}">
  <link rel="canonical" href="https://ailearnings.in/">
  <title>Redirecting…</title>
</head>
<body>
  <script>window.location.replace('${redirectTo}');</script>
</body>
</html>`;
}

let created = 0;
let skipped = 0;

for (const legacyPath of LEGACY_PATHS) {
  const dir      = path.join(ROOT, legacyPath);
  const filePath = path.join(dir, 'index.html');

  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // Don't overwrite real pages (e.g. if slug coincidentally matches)
  if (fs.existsSync(filePath)) {
    const existing = fs.readFileSync(filePath, 'utf8');
    if (!existing.includes('noindex')) {
      console.log(`  ⏭  Skipping /${legacyPath}/ — real page exists`);
      skipped++;
      continue;
    }
  }

  fs.writeFileSync(filePath, noindexPage('/'));
  console.log(`  ✓  Created /${legacyPath}/index.html`);
  created++;
}

console.log(`\nDone — ${created} created, ${skipped} skipped.`);
console.log('Commit and push to deploy. Google will deindex these paths on next crawl.');
