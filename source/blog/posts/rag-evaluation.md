---
title: "RAG Evaluation Metrics Explained"
description: "Learn how to evaluate RAG systems using faithfulness, answer relevancy, context precision, and recall metrics. Includes RAGAS implementation and LLM-as-judge patterns."
date: "2026-03-15"
slug: "rag-evaluation"
keywords: ["rag evaluation metrics", "ragas evaluation", "rag faithfulness", "rag answer relevancy", "llm evaluation rag", "context precision recall"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-15"
---

# RAG Evaluation Metrics Explained

The most common RAG evaluation strategy is: ask a few questions, read the answers, decide they look good, and ship. This works until something breaks in production — wrong answers, answers that ignore retrieved context, retrieved context that misses the relevant documents — and you have no measurement infrastructure to diagnose which component failed.

A RAG system has two failure surfaces: retrieval and generation. Retrieval failures mean the right chunks aren't in the context. Generation failures mean the LLM ignores or misrepresents the chunks it was given. Treating the system as a black box and only measuring end-to-end quality makes it nearly impossible to identify which layer is the problem.

Good RAG evaluation is not optional — it's the feedback loop that makes iterative improvement possible. This guide covers the standard evaluation framework (RAGAS), how to implement it, and how to interpret the results to drive concrete improvements.

For the full pipeline context, see the [RAG Architecture Guide](/blog/rag-architecture-guide).

---

## Concept Overview

RAG evaluation has four core metrics, each measuring a different aspect of system quality:

1. **Faithfulness** — does the answer contain only information supported by the retrieved context? (measures hallucination)
2. **Answer Relevancy** — does the answer actually address the question asked? (measures response quality)
3. **Context Precision** — are the retrieved chunks relevant to the question? (measures retrieval quality)
4. **Context Recall** — does the retrieved context contain all the information needed to answer the question? (measures retrieval completeness)

RAGAS (Retrieval Augmented Generation Assessment) is the most widely adopted framework for computing these metrics. It uses LLM-as-a-judge internally, which means it requires an LLM API but does not require human-labeled ground truth for faithfulness and relevancy.

---

## How It Works

![Architecture diagram](/assets/diagrams/rag-evaluation-diagram-1.png)

Context recall is the only metric that requires ground truth answers. The others use LLM-as-a-judge patterns that can evaluate without labeled data.

---

## Implementation Example

### Install and Setup RAGAS

```bash
pip install ragas langchain langchain-openai langchain-community chromadb datasets
export OPENAI_API_KEY="sk-..."
```

### Build a Test Dataset

```python
# evaluation_dataset.py
# Build an evaluation dataset — ideally from real user queries

evaluation_samples = [
    {
        "question": "What is the return policy for digital products?",
        "ground_truth": "Digital products are non-refundable once the license key has been activated. Physical products can be returned within 30 days of purchase.",
        # ground_truth is needed for context_recall only
    },
    {
        "question": "How long does standard shipping take?",
        "ground_truth": "Standard shipping takes 5-7 business days for domestic orders and 10-14 business days for international orders.",
    },
    {
        "question": "Can I upgrade my plan mid-cycle?",
        "ground_truth": "Plan upgrades take effect immediately and are prorated for the remaining billing period. Downgrades take effect at the next billing cycle.",
    },
    {
        "question": "What payment methods are accepted?",
        "ground_truth": "We accept Visa, Mastercard, American Express, PayPal, and bank transfers for orders over $500.",
    },
    {
        "question": "Is there a free trial available?",
        "ground_truth": "Yes, a 14-day free trial is available for all plans. No credit card is required to start the trial.",
    },
]

print(f"Evaluation dataset: {len(evaluation_samples)} samples")
```

### Run Your RAG Pipeline and Collect Results

```python
# run_pipeline.py
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load vector store
vs = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="documents"
)

PROMPT = PromptTemplate(
    template="""Answer ONLY from the provided context. If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    retriever=vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)


def run_pipeline_for_evaluation(samples: list) -> list:
    """Run RAG pipeline on all samples and collect answers + contexts."""
    results = []

    for sample in samples:
        result = qa_chain.invoke({"query": sample["question"]})

        results.append({
            "question": sample["question"],
            "answer": result["result"],
            "contexts": [doc.page_content for doc in result["source_documents"]],
            "ground_truth": sample.get("ground_truth", ""),
        })

    return results


pipeline_results = run_pipeline_for_evaluation(evaluation_samples)
print(f"Collected {len(pipeline_results)} pipeline results")
```

### RAGAS Evaluation

```python
# evaluate.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_correctness
)
from datasets import Dataset

# Convert pipeline results to RAGAS dataset format
eval_data = {
    "question": [r["question"] for r in pipeline_results],
    "answer": [r["answer"] for r in pipeline_results],
    "contexts": [r["contexts"] for r in pipeline_results],
    "ground_truth": [r["ground_truth"] for r in pipeline_results],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
# Note: RAGAS uses the OPENAI_API_KEY to power its LLM-as-judge calls
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,           # answer grounded in context?
        answer_relevancy,       # answer relevant to question?
        context_precision,      # retrieved chunks relevant?
        context_recall,         # retrieved chunks complete? (needs ground_truth)
    ]
)

print(result)
# Output: {'faithfulness': 0.91, 'answer_relevancy': 0.87, 'context_precision': 0.83, 'context_recall': 0.79}

# Convert to pandas for analysis
df = result.to_pandas()
print("\nPer-question breakdown:")
print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string())
```

### Diagnosing with Per-Question Scores

```python
def diagnose_failures(df) -> dict:
    """
    Classify each question by failure type based on metric scores.
    This tells you which component to fix.
    """
    LOW = 0.7   # threshold for "low" score

    diagnoses = []
    for _, row in df.iterrows():
        issues = []

        if row["context_precision"] < LOW and row["context_recall"] < LOW:
            issues.append("RETRIEVAL_FAILURE: both precision and recall are low — wrong chunks retrieved")

        elif row["context_precision"] < LOW:
            issues.append("RETRIEVAL_NOISE: retrieved irrelevant chunks — tune retriever or add filtering")

        elif row["context_recall"] < LOW:
            issues.append("RETRIEVAL_INCOMPLETE: missing relevant chunks — check chunking or increase k")

        if row["faithfulness"] < LOW:
            issues.append("HALLUCINATION: answer not supported by context — strengthen grounding prompt")

        if row["answer_relevancy"] < LOW:
            issues.append("OFF_TOPIC_ANSWER: answer doesn't address the question — check prompt or LLM behavior")

        diagnoses.append({
            "question": row["question"][:60],
            "issues": issues if issues else ["PASS"],
            "scores": {
                "faithfulness": round(row["faithfulness"], 3),
                "relevancy": round(row["answer_relevancy"], 3),
                "precision": round(row["context_precision"], 3),
                "recall": round(row["context_recall"], 3),
            }
        })

    return diagnoses


diagnoses = diagnose_failures(df)
for d in diagnoses:
    if d["issues"] != ["PASS"]:
        print(f"\nQ: {d['question']}")
        print(f"Scores: {d['scores']}")
        print(f"Issues: {d['issues']}")
```

### LLM-as-Judge for Custom Metrics

When RAGAS metrics don't cover your specific quality requirements, implement custom LLM judges:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

CITATION_JUDGE_PROMPT = PromptTemplate(
    template="""Evaluate whether the answer correctly cites sources.

Answer: {answer}
Context (numbered sources): {context}

Score the answer from 0.0 to 1.0 on citation accuracy:
- 1.0: All claims are supported and correctly attributed to specific sources
- 0.5: Some claims supported but citations are missing or imprecise
- 0.0: No source attribution or citations contradict the context

Respond with ONLY a number between 0.0 and 1.0.""",
    input_variables=["answer", "context"]
)

COMPLETENESS_JUDGE_PROMPT = PromptTemplate(
    template="""Evaluate whether the answer is complete.

Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Does the answer cover all key points from the ground truth?
Score from 0.0 to 1.0:
- 1.0: Answer is complete — covers all key points
- 0.5: Answer is partially complete — covers main point but misses details
- 0.0: Answer is incomplete or misses the main point

Respond with ONLY a number between 0.0 and 1.0.""",
    input_variables=["question", "answer", "ground_truth"]
)


class CustomRAGEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)   # use stronger model for judging

    def score_citation(self, answer: str, context: list[str]) -> float:
        formatted_context = "\n".join([f"[{i+1}] {c[:300]}" for i, c in enumerate(context)])
        response = (CITATION_JUDGE_PROMPT | self.llm).invoke({
            "answer": answer,
            "context": formatted_context
        })
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.0

    def score_completeness(self, question: str, answer: str, ground_truth: str) -> float:
        response = (COMPLETENESS_JUDGE_PROMPT | self.llm).invoke({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth
        })
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.0

    def evaluate(self, samples: list) -> list:
        results = []
        for sample in samples:
            citation_score = self.score_citation(sample["answer"], sample["contexts"])
            completeness_score = self.score_completeness(
                sample["question"], sample["answer"], sample["ground_truth"]
            )
            results.append({
                **sample,
                "citation_score": citation_score,
                "completeness_score": completeness_score
            })
        return results


evaluator = CustomRAGEvaluator()
custom_results = evaluator.evaluate(pipeline_results)

for r in custom_results:
    print(f"Q: {r['question'][:50]}")
    print(f"  Citation: {r['citation_score']:.2f} | Completeness: {r['completeness_score']:.2f}")
```

### Automated Evaluation Pipeline

```python
# Run evaluation on every code change or nightly
import json
from datetime import datetime
from pathlib import Path

def run_full_evaluation(pipeline_results: list, output_dir: str = "./eval_results") -> dict:
    """Run full evaluation suite and save results."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # RAGAS evaluation
    eval_data = {
        "question": [r["question"] for r in pipeline_results],
        "answer": [r["answer"] for r in pipeline_results],
        "contexts": [r["contexts"] for r in pipeline_results],
        "ground_truth": [r["ground_truth"] for r in pipeline_results],
    }
    dataset = Dataset.from_dict(eval_data)
    ragas_result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

    summary = {
        "timestamp": timestamp,
        "sample_count": len(pipeline_results),
        "faithfulness": round(float(ragas_result["faithfulness"]), 4),
        "answer_relevancy": round(float(ragas_result["answer_relevancy"]), 4),
        "context_precision": round(float(ragas_result["context_precision"]), 4),
        "context_recall": round(float(ragas_result["context_recall"]), 4),
    }

    # Save results
    output_path = Path(output_dir) / f"eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation saved to {output_path}")
    print(f"Summary: {summary}")

    # Alert on regressions
    THRESHOLDS = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.75,
    }

    for metric, threshold in THRESHOLDS.items():
        if summary[metric] < threshold:
            print(f"ALERT: {metric} = {summary[metric]} below threshold {threshold}")

    return summary

summary = run_full_evaluation(pipeline_results)
```

---

## Interpreting Evaluation Results

| Metric | Low Score Diagnosis | Fix |
|---|---|---|
| Faithfulness < 0.8 | LLM generating beyond context | Strengthen grounding prompt |
| Answer Relevancy < 0.8 | Answer off-topic or incomplete | Improve prompt or increase k |
| Context Precision < 0.75 | Irrelevant chunks retrieved | Tune retriever, add filtering, use reranker |
| Context Recall < 0.75 | Missing relevant chunks | Increase k, improve chunking, check index |

---

## Best Practices

**Evaluate retrieval and generation separately.** A low faithfulness score with high context precision means the LLM is the problem. A low context precision score means retrieval is the problem. Don't try to fix both at once.

**Build the evaluation dataset from production logs.** The most valuable test questions are real user queries, not questions you wrote yourself. Extract the 100 most common queries from production logs and use those as your evaluation set.

**Run evaluation before and after every significant change.** Changing chunk size, embedding model, k value, or the prompt can all affect metrics. Run the evaluation suite as part of your deployment pipeline.

**Use a stronger model for judging than for generation.** RAGAS LLM-as-judge calls use the model you configure. Use GPT-4o for judging even if you use GPT-4o-mini for generation. The judge needs to accurately assess the quality of the generation.

**Track metrics over time.** A single evaluation snapshot tells you where you are. Tracking metrics over time tells you whether you're improving and whether changes caused regressions.

---

## Common Mistakes

**Evaluating only with questions the system can answer.** A system that refuses to answer difficult questions will score perfectly on faithfulness and well on context precision. Include out-of-scope questions in your evaluation set to test the full behavior spectrum.

**Treating RAGAS scores as absolute truth.** RAGAS uses LLM-as-judge, which is imperfect. It's an excellent proxy for human judgment but not a substitute. Periodically have a human review a sample of responses alongside RAGAS scores to calibrate.

**Not version-controlling the evaluation dataset.** As your system improves, update your evaluation questions. Track which version of the eval set produced which scores so results are comparable.

**Ignoring context recall in favor of easier-to-compute metrics.** Context recall requires ground truth and is more expensive to compute, so teams skip it. This means they're not measuring retrieval completeness — which is often where RAG systems fail.

---

## Summary

RAG evaluation requires measuring both retrieval and generation quality independently. RAGAS provides four complementary metrics: faithfulness (hallucination detection), answer relevancy (response quality), context precision (retrieval quality), and context recall (retrieval completeness). Together they give you a complete picture of where the system is succeeding and where it is failing.

The investment in evaluation infrastructure pays off quickly. Teams with good evaluation can iterate in hours rather than days because they can immediately measure whether a change improved or regressed the system.

---

## Related Articles

- [RAG Architecture Guide](/blog/rag-architecture-guide) — the full pipeline that evaluation measures
- [Context Window Optimization in RAG Systems](/blog/context-window-rag) — improving context precision
- [Chunking Strategies for RAG Pipelines](/blog/rag-chunking-strategies) — improving context recall

---

## FAQ

**What is RAGAS?**
RAGAS (Retrieval Augmented Generation Assessment) is an open-source evaluation framework for RAG systems. It implements LLM-as-a-judge evaluation for faithfulness, answer relevancy, context precision, and context recall without requiring human-labeled data for most metrics.

**Do I need human labels to use RAGAS?**
For faithfulness, answer relevancy, and context precision — no. RAGAS uses LLM-as-judge. For context recall, you need ground truth answers for each question. The ground truth can be written by a subject-matter expert or extracted from document content.

**How large should my evaluation dataset be?**
Fifty questions is the minimum for meaningful aggregate statistics. One hundred to two hundred is the right target for production monitoring. Include a mix of query types: simple factual, complex multi-document, out-of-scope, and adversarial.

**What faithfulness score should I target?**
A faithfulness score above 0.85 is a reasonable production target. Scores above 0.90 indicate the grounding prompt is working well. Scores below 0.80 suggest the model is regularly generating beyond what the context supports.

**Can I use RAGAS with open-source models?**
Yes. RAGAS supports any LangChain-compatible LLM. Replace the default OpenAI evaluator with Ollama, HuggingFace, or any other model. Note that smaller models produce less reliable evaluation scores — for judging quality, larger models are more accurate judges.
