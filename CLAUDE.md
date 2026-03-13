# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a static AI learning hub website (React SPA + SSG) deployed to GitHub Pages. Content is authored as markdown files with frontmatter, compiled into a JavaScript data layer, then pre-rendered to static HTML via Puppeteer.

## Development Commands

All commands run from the `source/` directory:

```bash
cd source/
npm install          # Install dependencies
npm run dev          # Watch mode: auto-rebuild on file changes (500ms debounce)
npm run build        # Build JS (Babel+Terser) + CSS (Tailwind)
npm run build:inline # Inline CSS into HTML
```

**Content pipeline:**
```bash
npm run generate:content-data  # Regenerate content-data.js from markdown frontmatter
npm run generate:blog          # Pre-render blog pages with Puppeteer
npm run generate:projects      # Pre-render project pages
npm run generate:paths         # Pre-render learning path pages
npm run generate:all           # Full site generation (slow)
npm run generate:og            # Generate OG social media images
```

**Deploy:**
```bash
npm run deploy  # git commit + push (auto-deploys to GitHub Pages)
```

There is no test suite. Validation is done via static generation + visual inspection.

## Architecture

### Data Flow

`Markdown frontmatter` → `scripts/generate-content-data.js` → `source/src/content-data.js` → `source/app.jsx` → `Puppeteer SSG` → static HTML in `docs/` → GitHub Pages

**Important:** `source/src/content-data.js` is auto-generated. Never edit it manually; run `npm run generate:content-data` after adding/modifying content.

### Content Directories

- `/blog/posts/` and `/blog/roadmap-guides/` — Blog guides (30+ markdown files)
- `/projects/posts/` — Project tutorials (22 markdown files)
- `/paths/` — Career learning paths (5 markdown files)

Frontmatter fields: `slug`, `title`, `description`, `date`, `level`, `time`, `stack`

### Source Code (`source/`)

- `app.jsx` (~8,000 LOC) — Main React app with all routing, navigation, and core page components
- `src/icons.jsx` — Custom React icon components
- `src/content-data.js` — Auto-generated content index (do not edit)
- `ai_roadmap.tsx`, `prompt_eng.tsx`, `genai_guide.tsx`, etc. — Large standalone feature components
- `knowledge_assessment.tsx`, `knowledge_gaps.tsx`, `readiness_check.tsx` — Interactive tool components
- `scripts/generate-static.js` — Puppeteer SSG script
- `scripts/generate-content-data.js` — Markdown frontmatter parser
- `tailwind.config.js` — Tailwind config with extensive safelist for dynamic gradient classes

### Static Output

Pre-rendered HTML files are written to `docs/[slug]/index.html` (served as `/slug` by GitHub Pages). The root `index.html` and `404.html` live at the repo root.

### Styling

Tailwind CSS v3 with a large safelist of dynamic color classes (gradient phases use colors like green, slate, blue, purple, orange, rose, teal). Dark mode uses explicit dark-variant classes. Each roadmap phase has a unique `from-X to-Y` gradient defined in `tailwind.config.js`.

### Deployment

Commits auto-deploy via GitHub Pages. The `npm run deploy` script commits with message `"build: auto-deploy [skip ci]"` and pushes. The `.github/workflows/build.yml` handles CI.
