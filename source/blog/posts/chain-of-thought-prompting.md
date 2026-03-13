---
title: "Chain-of-Thought Prompting Explained for Developers"
description: "Learn how chain-of-thought prompting improves reasoning in large language models."
date: "2026-03-13"
slug: "chain-of-thought-prompting"
keywords: ["chain of thought prompting", "CoT prompting", "LLM reasoning", "few-shot CoT", "chain of thought explained"]
---

## Learning Objectives

- Understand why chain-of-thought improves LLM accuracy
- Apply zero-shot and few-shot CoT techniques
- Use self-consistency to improve reliability
- Implement Tree-of-Thought for complex problems
- Know when to use CoT and when it's overkill

---

## What Is Chain-of-Thought?

Chain-of-thought (CoT) prompting guides an LLM to show its reasoning step by step before reaching a conclusion. This significantly improves accuracy on math, logic, and multi-step reasoning tasks.

**Without CoT:**
> Q: If a store sells 3 shirts for $45, how much do 7 shirts cost?
> A: $95 ❌

**With CoT:**
> Q: If a store sells 3 shirts for $45, how much do 7 shirts cost? Let's think step by step.
> A: First, find the price per shirt: $45 ÷ 3 = $15. Then multiply by 7: $15 × 7 = $105. The answer is $105. ✅

---

## Technique 1: Zero-Shot CoT

Simply add "Let's think step by step." to any prompt. Surprisingly effective — no examples needed.

```python
from openai import OpenAI

client = OpenAI()

def zero_shot_cot(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful reasoning assistant."},
            {"role": "user",   "content": f"{question}\n\nLet's think step by step."},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# Math
print(zero_shot_cot("A train travels 60 mph. It departs at 2 PM and arrives at 5:30 PM. How far did it travel?"))

# Logic
print(zero_shot_cot("All programmers drink coffee. Some coffee drinkers are introverts. Are all programmers introverts?"))

# Code
print(zero_shot_cot("A Python list contains [3, 1, 4, 1, 5, 9, 2, 6]. What is the output of sorted(set(lst))?"))
```

**Other effective triggers:**
- "Let's work through this carefully."
- "First, let me identify what we know..."
- "I'll break this down into steps."
- "Step 1:"

---

## Technique 2: Few-Shot CoT

Provide worked examples that demonstrate the reasoning pattern:

```python
FEW_SHOT_COT_EXAMPLES = """
Q: Tom has 3 times as many marbles as Jerry. Together they have 48 marbles. How many does Tom have?
A: Let Jerry have x marbles. Then Tom has 3x. Together: x + 3x = 4x = 48. So x = 12. Tom has 3 × 12 = 36 marbles.

Q: A recipe needs 2.5 cups of flour for 12 cookies. How much flour for 30 cookies?
A: Flour per cookie = 2.5 ÷ 12 = 0.2083 cups. For 30 cookies: 30 × 0.2083 = 6.25 cups of flour.

Q: {question}
A:"""


def few_shot_cot(question: str) -> str:
    prompt = FEW_SHOT_COT_EXAMPLES.format(question=question)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


print(few_shot_cot("A car dealership sold 40% of its inventory in March and 25% of what remained in April. If it started with 200 cars, how many are left?"))
```

**Few-shot CoT works best when:**
- The problem type is consistent (all math, all logic, etc.)
- Examples are high quality and representative
- The reasoning pattern generalizes to new problems

---

## Technique 3: Self-Consistency

Generate multiple reasoning paths and take the majority answer. Dramatically improves accuracy on hard problems.

```python
from collections import Counter
import re

def self_consistency(question: str, n_samples: int = 5) -> str:
    answers = []

    for i in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Solve step by step. End with 'Final answer: [number/answer]'"},
                {"role": "user",   "content": question},
            ],
            temperature=0.7,  # non-zero temperature for diverse paths
        )
        text = response.choices[0].message.content

        # Extract final answer
        match = re.search(r'[Ff]inal answer[:\s]+(.+?)(?:\n|$)', text)
        if match:
            answers.append(match.group(1).strip())
        else:
            # Fall back to last number mentioned
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            if numbers:
                answers.append(numbers[-1])

    # Return most common answer
    counter = Counter(answers)
    majority_answer, count = counter.most_common(1)[0]

    print(f"Answers: {dict(counter)}")
    print(f"Majority ({count}/{n_samples}): {majority_answer}")
    return majority_answer


# Works especially well for math problems where LLMs make arithmetic errors
result = self_consistency(
    "A box has 15 red and 10 blue balls. You pick 3 balls without replacement. "
    "What's the probability all 3 are red?",
    n_samples=5,
)
```

**When to use:** High-stakes reasoning where accuracy matters more than cost. 5 samples is usually enough.

---

## Technique 4: Step-Back Prompting

For complex questions, first ask a more general question to retrieve background knowledge:

```python
def step_back_prompting(specific_question: str) -> str:
    # Step 1: Generate a more general "step-back" question
    step_back_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate a more general question that provides background knowledge helpful for answering the specific question."},
            {"role": "user",   "content": f"Specific question: {specific_question}\n\nMore general question:"},
        ],
        temperature=0,
    )
    general_q = step_back_response.choices[0].message.content

    # Step 2: Answer the general question
    general_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": general_q}],
        temperature=0,
    )
    background = general_response.choices[0].message.content

    # Step 3: Answer original with background context
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Use this background knowledge:\n\n{background}"},
            {"role": "user",   "content": specific_question},
        ],
        temperature=0,
    )
    return final_response.choices[0].message.content


print(step_back_prompting("Why does PyTorch use dynamic computation graphs instead of static ones?"))
```

---

## Technique 5: Tree of Thoughts (ToT)

Explore multiple reasoning paths simultaneously and prune dead ends. Best for creative/exploratory problems.

```python
def tree_of_thoughts(problem: str, n_thoughts: int = 3, depth: int = 2) -> str:
    """Simplified Tree of Thoughts implementation."""

    def generate_thoughts(context: str, n: int) -> list[str]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Generate {n} different approaches or next steps. Number them 1-{n}."},
                {"role": "user",   "content": context},
            ],
            temperature=0.8,
        )
        text = response.choices[0].message.content
        thoughts = re.split(r'\n\d+\.', text)
        return [t.strip() for t in thoughts if t.strip()][:n]

    def evaluate_thought(thought: str, problem: str) -> float:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rate the promise of this approach for solving the problem (1-10). Reply with just a number."},
                {"role": "user",   "content": f"Problem: {problem}\n\nApproach: {thought}"},
            ],
            temperature=0,
        )
        try:
            return float(re.search(r'\d+', response.choices[0].message.content).group())
        except:
            return 5.0

    # Generate and evaluate initial thoughts
    initial_thoughts = generate_thoughts(f"Problem: {problem}\n\nList {n_thoughts} approaches:", n_thoughts)
    scored_thoughts = [(t, evaluate_thought(t, problem)) for t in initial_thoughts]
    best_thought = max(scored_thoughts, key=lambda x: x[1])

    # Expand best thought
    expanded = generate_thoughts(
        f"Problem: {problem}\n\nBest approach so far: {best_thought[0]}\n\nExpand this approach:",
        n=2,
    )

    # Final synthesis
    context = f"Problem: {problem}\n\nBest approach: {best_thought[0]}\n\nExpanded ideas: {'; '.join(expanded)}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Synthesize the best ideas into a complete solution."},
            {"role": "user",   "content": context},
        ],
    )
    return response.choices[0].message.content
```

---

## When to Use Which Technique

| Technique | Best For | Cost | Complexity |
|-----------|---------|------|-----------|
| Zero-shot CoT | Quick improvement on any reasoning task | Low | None |
| Few-shot CoT | Consistent problem types (math, code) | Low | Medium (writing examples) |
| Self-consistency | High-accuracy math/logic | Medium (5× API calls) | Low |
| Step-back | Complex multi-concept questions | Low | Low |
| Tree of Thoughts | Creative/exploratory problems | High (many calls) | High |

---

## Measuring CoT Effectiveness

```python
def evaluate_cot(test_cases: list[tuple], n_runs: int = 3) -> dict:
    """Compare plain prompting vs CoT."""
    plain_correct = 0
    cot_correct   = 0

    for question, expected in test_cases:
        # Plain
        for _ in range(n_runs):
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
                temperature=0,
            )
            if expected.lower() in r.choices[0].message.content.lower():
                plain_correct += 1

        # CoT
        for _ in range(n_runs):
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"{question}\nLet's think step by step."}],
                temperature=0,
            )
            if expected.lower() in r.choices[0].message.content.lower():
                cot_correct += 1

    total = len(test_cases) * n_runs
    return {
        "plain_accuracy": plain_correct / total,
        "cot_accuracy":   cot_correct / total,
        "improvement":    cot_correct / total - plain_correct / total,
    }
```

---

## Troubleshooting

**CoT gives wrong final answer despite correct reasoning**
→ Add "Your final answer must be X format." Add explicit verification step: "Now double-check your answer."

**Model skips steps**
→ Use few-shot examples that demonstrate granular step-by-step reasoning. Add "Show every step, do not skip."

**CoT is too verbose**
→ Add length guidance: "Reason in at most 150 words, then give your answer."

---

## FAQ

**Does CoT work with all models?**
Most reliably with GPT-4, Claude, and larger models. Smaller models (< 7B) benefit less. The reasoning capability must exist in the model for CoT to help — it can't compensate for missing knowledge.

**Is CoT always worth the extra tokens?**
No. For simple factual questions or classification, CoT adds cost without benefit. Reserve it for multi-step reasoning, math, logic, and code analysis.

---

## What to Learn Next

- **Prompt engineering foundation** → [Prompt Engineering Guide](/blog/prompt-engineering-guide/)
- **All prompting techniques** → [Prompt Engineering Techniques](/blog/prompt-engineering-techniques/)
- **In-context learning with examples** → [Few-Shot Prompting Explained](/blog/few-shot-prompting-explained/)
