---
title: "Build an AI Data Analyst: Chat with Your CSV Data"
description: "Build a natural language data analysis tool that lets you ask questions about CSV datasets. Generates and executes Python pandas code, produces charts, and explains insights automatically."
date: "2026-03-10"
slug: "ai-data-analyst"
level: "Intermediate"
time: "4–5 hours"
stack: "Python, OpenAI API, pandas, matplotlib, Streamlit"
keywords: ["AI data analyst Python", "chat with CSV AI", "LLM code execution data analysis"]
---

## Project Overview

A conversational data analyst that loads a CSV file, understands its schema, and answers natural language questions by generating and executing pandas/matplotlib code. Ask "What's the average revenue by region?" and get both the answer and the chart.

---

## Learning Goals

- Generate and safely execute Python code from LLM output
- Build schema-aware prompts for accurate code generation
- Handle code execution errors and retry automatically
- Display dynamic charts in Streamlit

---

## Architecture

```
CSV file upload
        ↓
Schema extraction (column names, types, sample rows)
        ↓
User question + schema → LLM → Python code
        ↓
Execute code (sandboxed) → result + optional chart
        ↓
LLM explains the result in plain English
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai pandas matplotlib streamlit
```

### Step 2: Schema Extractor

```python
# schema.py
import pandas as pd


def extract_schema(df: pd.DataFrame) -> str:
    """Build a schema description for the prompt."""
    lines = [
        f"Dataset: {len(df)} rows × {len(df.columns)} columns",
        f"Columns:",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_null = df[col].isna().sum()
        if df[col].dtype == "object":
            sample = df[col].dropna().head(3).tolist()
            unique = df[col].nunique()
            lines.append(f"  - {col} (text, {unique} unique values, sample: {sample}, nulls: {n_null})")
        elif "int" in dtype or "float" in dtype:
            stats = df[col].describe()
            lines.append(f"  - {col} (numeric, min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, nulls={n_null})")
        elif "datetime" in dtype:
            lines.append(f"  - {col} (datetime, range: {df[col].min()} to {df[col].max()}, nulls={n_null})")
        else:
            lines.append(f"  - {col} ({dtype}, nulls={n_null})")

    lines.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    return "\n".join(lines)
```

### Step 3: Code Generator + Executor

```python
# analyst.py
import io
import sys
import traceback
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI
from schema import extract_schema

client = OpenAI()

CODE_PROMPT = """You are a data analyst. Generate Python code to answer the user's question.

Available variables:
- `df`: pandas DataFrame (already loaded)
- `plt`: matplotlib.pyplot (already imported)

Dataset schema:
{schema}

Rules:
- Use `df` directly, do not reload the file
- If producing a chart: call plt.savefig('chart.png', bbox_inches='tight', dpi=150) at the end
- If computing a value: store the final result in a variable named `result` and print it
- Do not use plt.show()
- Keep code concise

Question: {question}

Return ONLY executable Python code, no markdown fences, no explanation."""


def generate_code(question: str, schema: str, history: list = None) -> str:
    messages = [{"role": "user", "content": CODE_PROMPT.format(schema=schema, question=question)}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def execute_code(code: str, df: pd.DataFrame) -> dict:
    """Execute generated code and capture output."""
    namespace = {"df": df, "pd": pd, "plt": plt}
    stdout_capture = io.StringIO()
    chart_path = None

    try:
        exec(code, namespace)
        chart_path = "chart.png" if "savefig" in code else None
        output = stdout_capture.getvalue() or str(namespace.get("result", ""))
        return {"success": True, "output": output, "chart": chart_path, "namespace": namespace}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def explain_result(question: str, result: str, code: str) -> str:
    """Generate a plain English explanation of the result."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""The user asked: "{question}"
The analysis produced: {result[:500]}
Explain the result in 2-3 clear sentences for a business audience."""}],
        max_tokens=200,
        temperature=0.3,
    )
    return response.choices[0].message.content


def analyze(question: str, df: pd.DataFrame, max_retries: int = 2) -> dict:
    schema = extract_schema(df)
    code = generate_code(question, schema)

    for attempt in range(max_retries + 1):
        result = execute_code(code, df)
        if result["success"]:
            explanation = explain_result(question, result["output"], code)
            return {
                "code": code,
                "output": result["output"],
                "chart": result["chart"],
                "explanation": explanation,
            }
        elif attempt < max_retries:
            # Ask LLM to fix the error
            fix_prompt = f"The following Python code produced an error:\n\n{code}\n\nError: {result['error']}\n\nFix the code. Return ONLY the corrected Python code."
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.1,
            )
            code = response.choices[0].message.content.strip()

    return {"error": f"Failed after {max_retries + 1} attempts: {result['error']}", "code": code}
```

### Step 4: Streamlit App

```python
# app.py
import streamlit as st
import pandas as pd
from analyst import analyze

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Upload a CSV and ask questions in plain English")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded: {len(df)} rows × {len(df.columns)} columns")

    with st.expander("Preview data"):
        st.dataframe(df.head(10))

    question = st.text_input("Ask a question about your data",
        placeholder="What is the total revenue by region? Show a bar chart.")

    if st.button("Analyze", type="primary") and question:
        with st.spinner("Analyzing..."):
            result = analyze(question, df)

        if "error" in result:
            st.error(f"Analysis failed: {result['error']}")
            with st.expander("Generated code"):
                st.code(result.get("code", ""), language="python")
        else:
            if result.get("explanation"):
                st.info(result["explanation"])

            if result.get("chart"):
                st.image(result["chart"])

            if result.get("output"):
                st.subheader("Result")
                st.text(result["output"])

            with st.expander("View generated code"):
                st.code(result["code"], language="python")
```

### Step 5: Run

```bash
streamlit run app.py
```

---

## Extension Ideas

1. **Multi-turn conversation** — maintain context across follow-up questions
2. **SQL mode** — generate SQL queries against DuckDB instead of pandas
3. **Auto EDA** — run exploratory data analysis automatically on upload
4. **Export dashboard** — save a collection of charts as a PDF report
5. **Database connector** — connect to PostgreSQL or BigQuery instead of CSV

---

## What to Learn Next

- **AI agents with tools** → [Tool Use and Function Calling](/blog/tool-use-and-function-calling/)
- **Multi-agent systems** → [Multi-Agent Research System](/projects/multi-agent-research-system/)
