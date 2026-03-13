#!/usr/bin/env node
/**
 * generate-og-image.js
 * Renders og-image.html at 1200×630 via Puppeteer and saves to /assets/og-image.png
 */

const puppeteer = require('puppeteer');
const path      = require('path');
const fs        = require('fs');

const HTML_FILE = path.resolve(__dirname, '../templates/og-image.html');
const OUT_FILE  = path.resolve(__dirname, '../../assets/og-image.png');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  const page    = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630 });
  await page.goto('file://' + HTML_FILE, { waitUntil: 'networkidle0' });
  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  await page.screenshot({ path: OUT_FILE, type: 'png' });
  await browser.close();
  console.log('✓ og-image.png — 1200×630 saved to /assets/');
})();
