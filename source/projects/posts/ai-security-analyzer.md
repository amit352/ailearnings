---
title: "Build an AI Security Analyzer: Automated Vulnerability Detection"
description: "Build a static analysis tool that scans codebases for security vulnerabilities using GPT-4o-mini, detecting OWASP Top 10 issues, secrets in code, and insecure patterns with severity scoring."
date: "2026-03-10"
slug: "ai-security-analyzer"
level: "Advanced"
time: "6–8 hours"
stack: "Python, OpenAI API, GitPython, Click"
keywords: ["AI security analyzer", "automated vulnerability detection Python", "LLM SAST tool"]
---

## Project Overview

An AI-powered static application security testing (SAST) tool that scans Python, JavaScript, and other codebases for OWASP Top 10 vulnerabilities, hardcoded secrets, and insecure patterns. Generates prioritized remediation reports.

---

## Learning Goals

- Design security-focused LLM prompts with false positive control
- Scan codebases efficiently with batching
- Combine regex pattern matching with LLM analysis
- Generate actionable security reports with CVSS-style scoring

---

## Architecture

```
Codebase (directory / git repo / single file)
        ↓
File scanner: collect relevant source files
        ↓
Layer 1: Regex patterns (fast, free) — obvious secrets/anti-patterns
        ↓
Layer 2: LLM analysis per file (thorough, contextual)
        ↓
Deduplicate + prioritize findings
        ↓
HTML/Markdown security report
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai click gitpython rich
```

### Step 2: Pattern Scanner (Fast Layer)

```python
# patterns.py
import re
from pathlib import Path

# High-confidence regex patterns for obvious issues
PATTERNS = [
    # Secrets
    {"id": "SEC-001", "name": "Hardcoded API Key", "severity": "critical",
     "regex": r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']([A-Za-z0-9_\-]{20,})["\']',
     "description": "API key hardcoded in source code"},
    {"id": "SEC-002", "name": "AWS Secret Key", "severity": "critical",
     "regex": r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']([A-Za-z0-9/+]{40})["\']',
     "description": "AWS secret key in source code"},
    {"id": "SEC-003", "name": "Hardcoded Password", "severity": "high",
     "regex": r'(?i)password\s*[=:]\s*["\'](?!.*\{)[A-Za-z0-9!@#$%^&*]{8,}["\']',
     "description": "Password hardcoded in source code"},
    {"id": "SEC-004", "name": "Private Key", "severity": "critical",
     "regex": r'-----BEGIN (RSA |EC )?PRIVATE KEY-----',
     "description": "Private key embedded in source file"},
    # Python-specific
    {"id": "SEC-010", "name": "SQL Injection Risk", "severity": "high",
     "regex": r'execute\s*\(\s*["\'].*%[s|d].*["\']|f["\'].*SELECT.*{.*}',
     "description": "Potential SQL injection via string formatting"},
    {"id": "SEC-011", "name": "Shell Injection Risk", "severity": "high",
     "regex": r'subprocess\.(call|run|Popen)\s*\(.*shell\s*=\s*True',
     "description": "Shell injection risk with shell=True"},
    {"id": "SEC-012", "name": "Pickle Deserialization", "severity": "medium",
     "regex": r'pickle\.(loads|load)\s*\(',
     "description": "Unsafe pickle deserialization"},
    {"id": "SEC-013", "name": "Eval Usage", "severity": "high",
     "regex": r'\beval\s*\(',
     "description": "Dangerous eval() usage"},
    # JavaScript
    {"id": "SEC-020", "name": "XSS Risk (innerHTML)", "severity": "high",
     "regex": r'\.innerHTML\s*=',
     "description": "Potential XSS via innerHTML assignment"},
    {"id": "SEC-021", "name": "dangerouslySetInnerHTML", "severity": "medium",
     "regex": r'dangerouslySetInnerHTML',
     "description": "React XSS risk via dangerouslySetInnerHTML"},
]

SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rb", ".php", ".java"}


def scan_file_patterns(filepath: str) -> list[dict]:
    """Run regex patterns against a single file."""
    path = Path(filepath)
    if path.suffix not in SOURCE_EXTENSIONS:
        return []

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    findings = []
    for pattern in PATTERNS:
        for match in re.finditer(pattern["regex"], content, re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1
            findings.append({
                "id": pattern["id"],
                "name": pattern["name"],
                "severity": pattern["severity"],
                "description": pattern["description"],
                "file": str(path),
                "line": line_num,
                "match": match.group(0)[:100],
                "source": "pattern",
            })
    return findings


def collect_source_files(directory: str, max_files: int = 200) -> list[str]:
    """Recursively collect source files, skipping common noise."""
    skip_dirs = {"node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build", ".next"}
    files = []
    for path in Path(directory).rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in SOURCE_EXTENSIONS:
            files.append(str(path))
    return files[:max_files]
```

### Step 3: LLM Security Analyzer

```python
# llm_analyzer.py
import json
from openai import OpenAI

client = OpenAI()

SECURITY_PROMPT = """You are an expert security engineer. Analyze this code for security vulnerabilities.

Focus on OWASP Top 10 and common security issues:
- Injection (SQL, command, LDAP)
- Authentication/authorization flaws
- Sensitive data exposure
- Security misconfigurations
- Cryptographic failures
- Insecure deserialization
- Missing input validation

Return JSON:
{{
  "findings": [
    {{
      "id": "LLM-001",
      "name": "vulnerability name",
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "cwe": "CWE-xxx if applicable",
      "line_range": "approximate line range",
      "description": "clear description of the vulnerability",
      "exploit_scenario": "brief attack scenario",
      "remediation": "specific fix with code example if applicable"
    }}
  ],
  "overall_risk": "critical" | "high" | "medium" | "low" | "none",
  "summary": "2-3 sentence security assessment"
}}

If no issues found, return empty findings array.

File: {filename}
Language: {language}
Code:
{code}"""


def analyze_file_security(filepath: str) -> dict:
    """LLM security analysis of a single file."""
    from pathlib import Path
    from patterns import SOURCE_EXTENSIONS

    path = Path(filepath)
    if path.suffix not in SOURCE_EXTENSIONS:
        return {"findings": [], "overall_risk": "none"}

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"findings": [], "error": str(e)}

    if len(content) > 8000:
        content = content[:8000] + "\n# ... (truncated)"

    lang_map = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
                ".go": "Go", ".java": "Java", ".rb": "Ruby", ".php": "PHP"}
    language = lang_map.get(path.suffix, "code")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SECURITY_PROMPT.format(
            filename=path.name,
            language=language,
            code=content,
        )}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    result = json.loads(response.choices[0].message.content)
    result["file"] = filepath
    return result
```

### Step 4: Scanner + Report

```python
# scanner.py
from patterns import scan_file_patterns, collect_source_files
from llm_analyzer import analyze_file_security
from pathlib import Path


def scan_directory(directory: str, use_llm: bool = True, max_llm_files: int = 20) -> dict:
    """Full security scan of a directory."""
    print(f"Scanning: {directory}")
    files = collect_source_files(directory)
    print(f"Found {len(files)} source files")

    all_findings = []

    # Layer 1: Pattern scan all files
    print("Running pattern scan...")
    for filepath in files:
        findings = scan_file_patterns(filepath)
        all_findings.extend(findings)

    print(f"Pattern scan: {len(all_findings)} findings")

    # Layer 2: LLM scan (limited to top files by size/importance)
    if use_llm:
        # Prioritize files with pattern findings, then by recency
        files_with_findings = {f["file"] for f in all_findings}
        priority_files = sorted(files, key=lambda f: (f not in files_with_findings, -Path(f).stat().st_size))
        llm_files = priority_files[:max_llm_files]

        print(f"Running LLM analysis on {len(llm_files)} files...")
        for i, filepath in enumerate(llm_files):
            print(f"  [{i+1}/{len(llm_files)}] {Path(filepath).name}")
            result = analyze_file_security(filepath)
            for finding in result.get("findings", []):
                finding["file"] = filepath
                finding["source"] = "llm"
                all_findings.append(finding)

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    all_findings.sort(key=lambda f: severity_order.get(f.get("severity", "info"), 5))

    # Stats
    from collections import Counter
    severity_counts = Counter(f.get("severity", "info") for f in all_findings)

    return {
        "directory": directory,
        "files_scanned": len(files),
        "total_findings": len(all_findings),
        "severity_counts": dict(severity_counts),
        "findings": all_findings,
    }


def generate_report(scan_result: dict, output_path: str = None) -> str:
    """Generate a Markdown security report."""
    r = scan_result
    counts = r["severity_counts"]

    lines = [
        f"# Security Analysis Report",
        f"\n**Target:** `{r['directory']}`",
        f"**Files Scanned:** {r['files_scanned']}",
        f"**Total Findings:** {r['total_findings']}",
        f"\n## Severity Summary",
        f"| Severity | Count |",
        f"|----------|-------|",
    ]
    for sev in ["critical", "high", "medium", "low", "info"]:
        count = counts.get(sev, 0)
        if count:
            emoji = {"critical": "🚨", "high": "⚠️", "medium": "🟡", "low": "🔵", "info": "ℹ️"}[sev]
            lines.append(f"| {emoji} {sev.capitalize()} | {count} |")

    if r["findings"]:
        lines.append("\n## Findings")
        for f in r["findings"]:
            sev = f.get("severity", "info").upper()
            lines.append(f"\n### [{sev}] {f.get('name', 'Finding')}")
            lines.append(f"**File:** `{f.get('file', 'unknown')}`")
            if f.get("line"):
                lines.append(f"**Line:** {f['line']}")
            lines.append(f"\n{f.get('description', '')}")
            if f.get("remediation"):
                lines.append(f"\n**Fix:** {f['remediation']}")
    else:
        lines.append("\n✅ No security issues found!")

    report = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"Report saved to {output_path}")
    return report
```

### Step 5: CLI

```python
# main.py
import click
from scanner import scan_directory, generate_report


@click.command()
@click.argument("path")
@click.option("--output", "-o", help="Save report to file")
@click.option("--no-llm", is_flag=True, help="Skip LLM analysis (faster, less thorough)")
@click.option("--max-files", default=20, help="Max files for LLM analysis")
def main(path, output, no_llm, max_files):
    """AI Security Analyzer — scan code for vulnerabilities."""
    result = scan_directory(path, use_llm=not no_llm, max_llm_files=max_files)
    report = generate_report(result, output)
    print("\n" + report[:3000])
    if result["severity_counts"].get("critical", 0) > 0:
        import sys
        sys.exit(1)  # Non-zero exit for CI integration


if __name__ == "__main__":
    main()
```

### Step 6: Run

```bash
# Full scan
python main.py ./my_project --output security_report.md

# Fast pattern-only scan
python main.py ./my_project --no-llm

# Single file
python main.py ./app.py --no-llm
```

---

## Extension Ideas

1. **CI/CD integration** — GitHub Action that fails PRs with critical findings
2. **Dependency scanning** — check requirements.txt for known vulnerable packages
3. **SARIF output** — export in SARIF format for GitHub Security tab integration
4. **Custom rules** — YAML-based custom pattern definitions
5. **Trend tracking** — compare scan results over time to track security debt

---

## What to Learn Next

- **Code review automation** → [AI Code Review Assistant](/projects/ai-code-review-assistant/)
- **Multi-agent systems** → [Multi-Agent Research System](/projects/multi-agent-research-system/)
