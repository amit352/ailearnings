---
title: "Build an AI Sentiment Analyzer: Classify Text Sentiment and Emotions"
description: "Build a sentiment analysis tool using GPT-4o-mini that classifies text as positive, negative, or neutral, detects emotions, and processes batches of reviews or tweets."
date: "2026-03-10"
slug: "sentiment-analyzer"
level: "Beginner"
time: "2–3 hours"
stack: "Python, OpenAI API, pandas"
keywords: ["sentiment analysis Python", "text classification AI", "LLM sentiment analyzer"]
---

## Project Overview

A sentiment analysis tool that classifies text as positive, negative, or neutral, detects specific emotions (joy, anger, sadness, etc.), and can process CSV batches of reviews, tweets, or feedback at scale.

---

## Learning Goals

- Use JSON mode for structured classification outputs
- Process batches of data with rate limiting
- Map emotions to structured categories
- Build a simple CSV pipeline for real-world data

---

## Architecture

```
Input (text / CSV file)
        ↓
LLM classification (JSON output)
        ↓
Structured sentiment + emotion + confidence
        ↓
Results table / CSV export
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai pandas
```

### Step 2: Sentiment Classifier

```python
# analyzer.py
import json
from openai import OpenAI

client = OpenAI()

SENTIMENT_PROMPT = """Analyze the sentiment of the following text.

Return JSON with this exact structure:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <0.0-1.0>,
  "emotions": ["joy", "anger", "sadness", "fear", "surprise", "disgust"],
  "intensity": "low" | "medium" | "high",
  "key_phrases": ["phrase1", "phrase2"],
  "summary": "One sentence explanation of the sentiment"
}}

Only include emotions that are present. Return an empty list if no strong emotion.

Text: {text}"""


def analyze_sentiment(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SENTIMENT_PROMPT.format(text=text[:1000])}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def analyze_batch(texts: list[str], delay: float = 0.5) -> list[dict]:
    """Analyze a list of texts with basic rate limiting."""
    import time
    results = []
    for i, text in enumerate(texts):
        print(f"  Analyzing {i+1}/{len(texts)}...", end="\r")
        result = analyze_sentiment(text)
        result["text"] = text[:100]  # Store truncated text for reference
        results.append(result)
        if i < len(texts) - 1:
            time.sleep(delay)
    print()
    return results
```

### Step 3: CSV Pipeline

```python
# pipeline.py
import pandas as pd
from analyzer import analyze_sentiment, analyze_batch


def process_csv(input_path: str, text_column: str, output_path: str = None):
    """Process a CSV file and add sentiment columns."""
    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

    texts = df[text_column].fillna("").tolist()
    print(f"Analyzing {len(texts)} texts...")
    results = analyze_batch(texts)

    df["sentiment"] = [r.get("sentiment", "") for r in results]
    df["confidence"] = [r.get("confidence", 0) for r in results]
    df["emotions"] = [", ".join(r.get("emotions", [])) for r in results]
    df["intensity"] = [r.get("intensity", "") for r in results]

    output = output_path or input_path.replace(".csv", "_sentiment.csv")
    df.to_csv(output, index=False)
    print(f"Saved to {output}")

    # Print summary
    print("\n--- Summary ---")
    counts = df["sentiment"].value_counts()
    for sentiment, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")

    return df
```

### Step 4: CLI

```python
# main.py
import argparse
from analyzer import analyze_sentiment
from pipeline import process_csv


def main():
    parser = argparse.ArgumentParser(description="AI Sentiment Analyzer")
    subparsers = parser.add_subparsers(dest="command")

    # Single text analysis
    text_parser = subparsers.add_parser("text", help="Analyze a single text")
    text_parser.add_argument("text", help="Text to analyze")

    # CSV batch processing
    csv_parser = subparsers.add_parser("csv", help="Process a CSV file")
    csv_parser.add_argument("file", help="Path to CSV file")
    csv_parser.add_argument("--column", "-c", default="text", help="Column name containing text")
    csv_parser.add_argument("--output", "-o", help="Output CSV path")

    args = parser.parse_args()

    if args.command == "text":
        result = analyze_sentiment(args.text)
        sentiment = result["sentiment"].upper()
        confidence = result["confidence"]
        emotions = ", ".join(result.get("emotions", [])) or "none"

        emoji = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(sentiment, "")
        print(f"\n{emoji} Sentiment: {sentiment} (confidence: {confidence:.0%})")
        print(f"Intensity: {result.get('intensity', 'N/A')}")
        print(f"Emotions: {emotions}")
        print(f"Summary: {result.get('summary', '')}")
        if result.get("key_phrases"):
            print(f"Key phrases: {', '.join(result['key_phrases'])}")

    elif args.command == "csv":
        process_csv(args.file, args.column, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

### Step 5: Run

```bash
# Single text
python main.py text "This product completely exceeded my expectations! Amazing quality."

# CSV batch
python main.py csv reviews.csv --column review_text --output results.csv
```

### Sample Output

```
😊 Sentiment: POSITIVE (confidence: 96%)
Intensity: high
Emotions: joy, surprise
Summary: The reviewer expresses strong satisfaction and delight with the product quality.
Key phrases: exceeded expectations, amazing quality
```

---

## Extension Ideas

1. **Aspect-based sentiment** — detect sentiment per aspect (price, quality, service)
2. **Real-time Twitter/Reddit analysis** — connect to social media APIs
3. **Trend dashboard** — track sentiment over time with Plotly charts
4. **Comparison mode** — compare sentiment across two CSV datasets
5. **Multilingual support** — add language detection and translate before analysis

---

## What to Learn Next

- **AI data analyst** → [AI Data Analyst](/projects/ai-data-analyst/)
- **Structured outputs** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
