#!/usr/bin/env node
/**
 * request-indexing.js
 * Submits all site URLs to the Google Indexing API to trigger fast crawling.
 *
 * ── One-time setup ────────────────────────────────────────────────────────
 *  1. Google Cloud Console  →  create a project (or use existing)
 *  2. APIs & Services       →  Enable "Web Search Indexing API"
 *  3. IAM & Admin           →  Service Accounts → Create service account
 *                              → Add role: "Owner" (or no role needed)
 *                              → Keys → Add Key → JSON  →  download
 *  4. Save the downloaded JSON as:
 *       source/service-account.json   (already in .gitignore)
 *  5. Google Search Console →  Settings → Users and permissions
 *                              → Add user → paste the service account email
 *                              → Permission: Owner
 *  6. npm install google-auth-library   (in source/)
 *
 * ── Usage ─────────────────────────────────────────────────────────────────
 *  node scripts/request-indexing.js              # all sitemaps
 *  node scripts/request-indexing.js --blog       # blog URLs only
 *  node scripts/request-indexing.js --dry-run    # preview without submitting
 */

const { GoogleAuth }  = require('google-auth-library');
const https           = require('https');
const fs              = require('fs');
const path            = require('path');

const ROOT         = path.resolve(__dirname, '../..');
const CREDS_FILE   = path.join(__dirname, '../service-account.json');
const INDEXING_URL = 'https://indexing.googleapis.com/v3/urlNotifications:publish';

// ── Sitemaps to read URLs from ─────────────────────────────────────────────
const SITEMAPS = [
  path.join(ROOT, 'sitemap-blog.xml'),
  path.join(ROOT, 'sitemap-pages.xml'),
  path.join(ROOT, 'sitemap-projects.xml'),
  path.join(ROOT, 'sitemap-guides.xml'),
];

const BLOG_SITEMAPS = [
  path.join(ROOT, 'sitemap-blog.xml'),
];

// ── Helpers ────────────────────────────────────────────────────────────────

function parseUrlsFromSitemap(filePath) {
  if (!fs.existsSync(filePath)) {
    console.warn(`  ⚠ Sitemap not found: ${filePath}`);
    return [];
  }
  const xml = fs.readFileSync(filePath, 'utf8');
  const matches = [...xml.matchAll(/<loc>(https?:\/\/[^<]+)<\/loc>/g)];
  return matches.map(m => m[1].trim());
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function submitUrl(authClient, url) {
  const token = await authClient.getAccessToken();
  const body  = JSON.stringify({ url, type: 'URL_UPDATED' });

  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname: 'indexing.googleapis.com',
        path:     '/v3/urlNotifications:publish',
        method:   'POST',
        headers:  {
          'Authorization':  `Bearer ${token.token}`,
          'Content-Type':   'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      res => {
        let data = '';
        res.on('data', chunk => (data += chunk));
        res.on('end', () => resolve({ status: res.statusCode, body: data }));
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── Main ───────────────────────────────────────────────────────────────────

async function main() {
  const args   = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const blogOnly = args.includes('--blog');

  // ── Collect URLs ───────────────────────────────────────────────────────
  const sitemaps = blogOnly ? BLOG_SITEMAPS : SITEMAPS;
  const allUrls  = [];

  for (const sitemap of sitemaps) {
    const urls = parseUrlsFromSitemap(sitemap);
    console.log(`  📄 ${path.basename(sitemap)} → ${urls.length} URLs`);
    allUrls.push(...urls);
  }

  // Deduplicate
  const urls = [...new Set(allUrls)];
  console.log(`\nTotal unique URLs: ${urls.length}`);

  if (dryRun) {
    console.log('\n── Dry run — URLs that would be submitted ──────────────────');
    urls.forEach((u, i) => console.log(`  ${String(i + 1).padStart(3)}. ${u}`));
    console.log('\nRun without --dry-run to submit.');
    return;
  }

  // ── Validate credentials file ──────────────────────────────────────────
  if (!fs.existsSync(CREDS_FILE)) {
    console.error(`\n❌  Credentials file not found: ${CREDS_FILE}`);
    console.error(`\nSetup steps:`);
    console.error(`  1. Go to console.cloud.google.com`);
    console.error(`  2. APIs & Services → Enable "Web Search Indexing API"`);
    console.error(`  3. IAM → Service Accounts → Create → Keys → JSON`);
    console.error(`  4. Save the JSON as: source/service-account.json`);
    console.error(`  5. In Google Search Console → Settings → Users`);
    console.error(`     → Add the service account email as Owner\n`);
    process.exit(1);
  }

  // ── Auth ───────────────────────────────────────────────────────────────
  const auth = new GoogleAuth({
    keyFile: CREDS_FILE,
    scopes:  ['https://www.googleapis.com/auth/indexing'],
  });
  const client = await auth.getClient();

  // ── Submit ─────────────────────────────────────────────────────────────
  // Google Indexing API limit: 200 requests/day per property
  const DAILY_LIMIT  = 200;
  const BATCH_DELAY  = 500; // ms between requests (avoid bursting)

  console.log(`\nSubmitting to Google Indexing API (limit: ${DAILY_LIMIT}/day)...\n`);

  let ok = 0, failed = 0, skipped = 0;
  const errors = [];

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];

    if (i >= DAILY_LIMIT) {
      console.log(`\n⚠  Daily limit of ${DAILY_LIMIT} reached. Remaining URLs skipped.`);
      skipped = urls.length - i;
      break;
    }

    try {
      const result = await submitUrl(client, url);

      if (result.status === 200) {
        console.log(`  ✓ [${String(i + 1).padStart(3)}/${urls.length}] ${url}`);
        ok++;
      } else if (result.status === 429) {
        console.log(`  ⏳ [${String(i + 1).padStart(3)}/${urls.length}] Rate limited — waiting 10s...`);
        await sleep(10000);
        i--; // retry same URL
        continue;
      } else {
        const parsed = JSON.parse(result.body);
        const msg    = parsed?.error?.message || result.body;
        console.log(`  ✗ [${String(i + 1).padStart(3)}/${urls.length}] ${url}`);
        console.log(`      → ${result.status}: ${msg}`);
        errors.push({ url, status: result.status, msg });
        failed++;
      }
    } catch (err) {
      console.log(`  ✗ [${String(i + 1).padStart(3)}/${urls.length}] ${url}`);
      console.log(`      → ${err.message}`);
      errors.push({ url, msg: err.message });
      failed++;
    }

    if (i < urls.length - 1) {
      await sleep(BATCH_DELAY);
    }
  }

  // ── Summary ────────────────────────────────────────────────────────────
  console.log('\n─────────────────────────────────────────────');
  console.log(`✅  Submitted:  ${ok}`);
  if (failed  > 0) console.log(`❌  Failed:     ${failed}`);
  if (skipped > 0) console.log(`⏭  Skipped:    ${skipped} (daily limit — run again tomorrow)`);

  if (errors.length > 0) {
    console.log('\nFailed URLs:');
    errors.forEach(e => console.log(`  • ${e.url} (${e.status || 'network'}: ${e.msg})`));
  }

  console.log('\nGoogle typically processes indexing requests within 24–72 hours.');
  console.log('Check status in Search Console → URL Inspection.\n');
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
