#!/usr/bin/env node
// Inlines dist/app.css and vendor scripts into src/index.html
// app.js is kept as an external deferred script for better browser parse performance
// Output: index.html

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '../..');

let html = fs.readFileSync(path.join(root, 'source/src/index.html'), 'utf8');

const css    = fs.readFileSync(path.join(root, 'dist/app.css'), 'utf8');
const react  = fs.readFileSync(path.join(root, 'dist/vendor/react.production.min.js'), 'utf8');
const rdom   = fs.readFileSync(path.join(root, 'dist/vendor/react-dom.production.min.js'), 'utf8');
const marked = fs.readFileSync(path.join(root, 'dist/vendor/marked.min.js'), 'utf8');

// Use a function replacer to avoid $ special characters in replacement strings
// (e.g. react-dom contains $& $1 etc which String.replace interprets as patterns)
const r = (str, search, replacement) => str.replace(search, () => replacement);

html = r(html, '<link rel="stylesheet" href="dist/app.css">',                               `<style>${css}</style>`);
html = r(html, '<script src="dist/vendor/react.production.min.js" crossorigin></script>',   `<script>${react}</script>`);
html = r(html, '<script src="dist/vendor/react-dom.production.min.js" crossorigin></script>', `<script>${rdom}</script>`);
html = r(html, '<script src="dist/vendor/marked.min.js"></script>',                         `<script>${marked}</script>`);

fs.writeFileSync(path.join(root, 'index.html'), html);

const kb = Math.round(fs.statSync(path.join(root, 'index.html')).size / 1024);
console.log(`✓ index.html inlined — ${kb} KB`);
