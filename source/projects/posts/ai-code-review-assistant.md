---
title: "Build an AI Code Review Assistant: Automated PR Reviews"
description: "Build a tool that reviews Git diffs and pull requests, identifies bugs, security issues, style violations, and suggests improvements — integrated with GitHub Actions."
date: "2026-03-10"
slug: "ai-code-review-assistant"
level: "Advanced"
time: "6–8 hours"
stack: "Python, OpenAI API, GitHub API, FastAPI"
keywords: ["AI code review", "automated PR review Python", "LLM code analysis GitHub"]
---

## Project Overview

An automated code review assistant that analyzes Git diffs, identifies real bugs, security vulnerabilities, and design issues, then posts inline comments on GitHub PRs. Configurable rules and severity levels.

---

## Learning Goals

- Parse and understand Git diffs programmatically
- Design effective code review prompts with security focus
- Integrate with the GitHub API for inline PR comments
- Build a GitHub Actions workflow for automated reviews

---

## Architecture

```
GitHub PR opened/updated
        ↓
Webhook → fetch diff via GitHub API
        ↓
Parse diff into file hunks
        ↓
LLM analyzes each hunk (bugs, security, style)
        ↓
Post inline comments on the PR
        ↓
Summary comment with overall assessment
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai fastapi uvicorn PyGithub python-dotenv
```

### Step 2: Diff Parser

```python
# diff_parser.py
import re
from dataclasses import dataclass, field


@dataclass
class FileDiff:
    filename: str
    old_filename: str
    change_type: str  # added, modified, deleted, renamed
    hunks: list[dict] = field(default_factory=list)
    language: str = ""


def detect_language(filename: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
        ".php": "php", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
        ".swift": "swift", ".kt": "kotlin", ".sh": "bash", ".sql": "sql",
    }
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    return ext_map.get(ext, "code")


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff into structured FileDiff objects."""
    files = []
    current_file = None
    current_hunk = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            if current_file and current_hunk:
                current_file.hunks.append(current_hunk)
            current_file = None
            current_hunk = None

        elif line.startswith("--- "):
            old_name = line[4:].strip()
            if old_name.startswith("a/"):
                old_name = old_name[2:]

        elif line.startswith("+++ "):
            new_name = line[4:].strip()
            if new_name.startswith("b/"):
                new_name = new_name[2:]
            old_name_ref = old_name if 'old_name' in dir() else new_name
            change_type = "added" if old_name_ref == "/dev/null" else "modified"
            current_file = FileDiff(
                filename=new_name,
                old_filename=old_name_ref,
                change_type=change_type,
                language=detect_language(new_name),
            )
            files.append(current_file)

        elif line.startswith("@@") and current_file:
            if current_hunk:
                current_file.hunks.append(current_hunk)
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            start_line = int(match.group(1)) if match else 1
            current_hunk = {"header": line, "lines": [], "start_line": start_line}

        elif current_hunk is not None:
            current_hunk["lines"].append(line)

    if current_file and current_hunk:
        current_file.hunks.append(current_hunk)

    return files


def hunk_to_text(hunk: dict) -> str:
    return hunk["header"] + "\n" + "\n".join(hunk["lines"])
```

### Step 3: Code Reviewer

```python
# reviewer.py
import json
from openai import OpenAI
from diff_parser import FileDiff, hunk_to_text

client = OpenAI()

REVIEW_PROMPT = """You are a senior software engineer conducting a thorough code review.

Review this {language} code diff and identify issues. Focus on:
1. Bugs (logic errors, edge cases, off-by-one errors)
2. Security vulnerabilities (injection, auth bypasses, secrets in code, unsafe operations)
3. Performance issues (N+1 queries, unnecessary computation, memory leaks)
4. Error handling gaps (unhandled exceptions, missing validation)
5. Design issues (SOLID violations, unclear naming, high complexity)

Return JSON:
{{
  "issues": [
    {{
      "type": "bug" | "security" | "performance" | "error_handling" | "style",
      "severity": "critical" | "high" | "medium" | "low",
      "line": <approximate line number in the diff, or null>,
      "description": "clear description of the issue",
      "suggestion": "specific fix or improvement"
    }}
  ],
  "summary": "2-3 sentence assessment of the changes",
  "approved": true | false
}}

File: {filename}
Diff:
{diff}"""


def review_file(file: FileDiff) -> dict:
    """Review a single changed file."""
    # Combine all hunks for context
    diff_text = "\n\n".join(hunk_to_text(h) for h in file.hunks)
    if len(diff_text) > 6000:
        diff_text = diff_text[:6000] + "\n... (truncated)"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": REVIEW_PROMPT.format(
            language=file.language,
            filename=file.filename,
            diff=diff_text,
        )}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    result = json.loads(response.choices[0].message.content)
    result["filename"] = file.filename
    return result


def review_diff(diff_text: str, max_files: int = 10) -> dict:
    """Review a complete PR diff."""
    from diff_parser import parse_diff
    files = parse_diff(diff_text)

    # Skip binary files, generated files
    skip_patterns = {".lock", ".min.js", ".map", ".png", ".jpg", ".ico"}
    files_to_review = [
        f for f in files
        if not any(f.filename.endswith(p) for p in skip_patterns)
           and f.change_type != "deleted"
    ][:max_files]

    results = []
    for file in files_to_review:
        print(f"  Reviewing {file.filename}...")
        result = review_file(file)
        results.append(result)

    # Generate overall summary
    all_issues = [i for r in results for i in r.get("issues", [])]
    critical = [i for i in all_issues if i["severity"] == "critical"]
    high = [i for i in all_issues if i["severity"] == "high"]

    overall_approved = not critical and len(high) < 3

    return {
        "files_reviewed": len(results),
        "total_issues": len(all_issues),
        "critical_issues": len(critical),
        "high_issues": len(high),
        "approved": overall_approved,
        "file_reviews": results,
    }
```

### Step 4: GitHub Integration

```python
# github_integration.py
import os
from github import Github
from reviewer import review_diff

g = Github(os.environ["GITHUB_TOKEN"])


def review_pr(repo_name: str, pr_number: int):
    """Fetch PR diff, review it, and post comments."""
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Get the diff
    import requests
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github.diff"}
    diff_resp = requests.get(pr.diff_url, headers=headers)
    diff_text = diff_resp.text

    print(f"Reviewing PR #{pr_number}: {pr.title}")
    review_result = review_diff(diff_text)

    # Post inline comments for each issue
    comments = []
    for file_review in review_result["file_reviews"]:
        for issue in file_review.get("issues", []):
            if issue["severity"] in ("critical", "high") and issue.get("line"):
                severity_emoji = {"critical": "🚨", "high": "⚠️"}.get(issue["severity"], "💡")
                comment_body = f"""{severity_emoji} **{issue['type'].upper()} ({issue['severity']})**: {issue['description']}

**Suggestion:** {issue['suggestion']}"""
                try:
                    pr.create_review_comment(
                        body=comment_body,
                        commit=list(pr.get_commits())[-1],
                        path=file_review["filename"],
                        line=issue["line"],
                    )
                except Exception:
                    comments.append(f"**{file_review['filename']}** line ~{issue['line']}: {comment_body}")

    # Post summary comment
    status = "✅ Approved" if review_result["approved"] else "🔴 Changes Requested"
    summary = f"""## AI Code Review Summary — {status}

| Metric | Count |
|--------|-------|
| Files reviewed | {review_result['files_reviewed']} |
| Total issues | {review_result['total_issues']} |
| Critical | {review_result['critical_issues']} |
| High | {review_result['high_issues']} |

{"Additional inline comments below." if comments else ""}
{"".join(comments)}

*Reviewed by AI Code Review Assistant*"""

    pr.create_issue_comment(summary)
    print(f"Review complete. Approved: {review_result['approved']}")
    return review_result
```

### Step 5: GitHub Actions Workflow

```yaml
# .github/workflows/ai-review.yml
name: AI Code Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install openai PyGithub
      - run: python -c "
          from github_integration import review_pr
          import os
          review_pr(
            os.environ['GITHUB_REPOSITORY'],
            int(os.environ['PR_NUMBER'])
          )
          "
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          PR_NUMBER: ${{ github.event.number }}
```

### Step 6: Run Locally

```bash
export OPENAI_API_KEY=your-key
export GITHUB_TOKEN=your-token

# Review a specific PR
python -c "from github_integration import review_pr; review_pr('owner/repo', 42)"

# Review a local diff
git diff main HEAD > diff.patch
python -c "
from reviewer import review_diff
from pathlib import Path
diff = Path('diff.patch').read_text()
result = review_diff(diff)
print(f'Issues: {result[\"total_issues\"]} ({result[\"critical_issues\"]} critical)')
"
```

---

## Extension Ideas

1. **Custom ruleset** — YAML config for project-specific rules and severity thresholds
2. **Incremental re-review** — only review new commits, not the whole PR again
3. **Test coverage check** — flag new code without test coverage
4. **Complexity scoring** — calculate cyclomatic complexity and flag high-complexity functions
5. **Team style guide** — ingest your CONTRIBUTING.md as context for reviews

---

## What to Learn Next

- **AI agents with tools** → [Tool Use and Function Calling](/blog/tool-use-and-function-calling/)
- **Multi-agent systems** → [Multi-Agent Research System](/projects/multi-agent-research-system/)
