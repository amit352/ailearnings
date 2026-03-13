---
title: "Few-Shot Prompting Explained – Improve LLM Output"
description: "Understand how few-shot prompting works and how developers guide LLM responses."
date: "2026-03-13"
slug: "few-shot-prompting-explained"
keywords: ["few-shot prompting", "few-shot learning", "LLM examples", "in-context learning", "prompt examples"]
---

# Few-Shot Prompting Explained – Improve LLM Output

One of the most reliable ways to improve LLM output quality is to show the model examples of what you want. This is called few-shot prompting, and it consistently outperforms vague instructions alone. For developers building AI features, few-shot prompting is a go-to technique for getting predictable, well-formatted responses.

---

## What is Few-Shot Prompting

Few-shot prompting is the practice of providing a small number of input/output examples in the prompt before presenting your actual task. The model learns the pattern from the examples and applies it to new inputs.

The term comes from machine learning — where "few-shot learning" means training a model on very few examples. In the context of LLMs, it refers to examples provided directly in the prompt at inference time, without any model retraining. This is also called **in-context learning**.

- **Zero-shot**: No examples — just the instruction
- **One-shot**: One example provided
- **Few-shot**: Two to five examples provided
- **Many-shot**: Five or more examples (becoming common with larger context windows)

---

## Why Few-Shot Prompting Matters for Developers

When you specify output format in prose — "return a JSON object with company and role fields" — the model might comply, or it might add extra explanation, use slightly different field names, or wrap the JSON in markdown code fences.

When you show an example of exactly what you want, the model calibrates to your format precisely. This matters most in production systems where code parses the model's output.

Few-shot prompting is also critical when:
- The task has domain-specific conventions the model might not guess
- The output requires a nuanced tone or style that is hard to describe
- You need consistent formatting across hundreds of different inputs
- Zero-shot prompting produces inconsistent or low-quality results

For more on where few-shot fits among other techniques, see [prompt engineering techniques](/blog/prompt-engineering-techniques/).

---

## How Few-Shot Prompting Works

The model uses the examples to infer:
1. The expected input format
2. The expected output format
3. The scope of the task (what to include/exclude)
4. The style and tone of the response

### Basic Template

```
[Task description]

Input: [example 1 input]
Output: [example 1 output]

Input: [example 2 input]
Output: [example 2 output]

Input: [your actual input]
Output:
```

### Key Principles

**Quality over quantity** — Two excellent examples outperform five mediocre ones. Pick examples that are clear, representative, and cover edge cases.

**Coverage** — Examples should represent the range of inputs you expect. If your task has different input types, include examples of each.

**Consistency** — All examples must follow the same format. Inconsistent formatting confuses the model about what the expected output actually is.

**End on the right pattern** — The last line before the model generates should follow the same pattern as the examples. Do not add extra instructions after the examples — they break the pattern.

---

## Practical Examples

### Structured Extraction

```
Extract the company name and job title from job posting text.

Input: "We are hiring a Senior ML Engineer at Stripe to build fraud detection models."
Output: {"company": "Stripe", "role": "Senior ML Engineer"}

Input: "Anthropic is looking for a Research Scientist focused on AI safety."
Output: {"company": "Anthropic", "role": "Research Scientist"}

Input: "Join the infrastructure team at Cloudflare as a Staff Engineer."
Output:
```

### Sentiment Classification with Nuance

```
Classify the sentiment as Positive, Negative, Mixed, or Neutral.

Input: "The product is well-designed but shipping took three weeks."
Output: Mixed

Input: "Absolutely love it. Best purchase I've made this year."
Output: Positive

Input: "It arrived on time but stopped working after two days."
Output: Negative

Input: "It's fine. Does what it says."
Output:
```

### Code Transformation

```
Convert Python 2 print statements to Python 3 syntax.

Input: print "Hello, world!"
Output: print("Hello, world!")

Input: print "Value:", x
Output: print("Value:", x)

Input: print "Error:", msg, "in line", line_num
Output:
```

### Python Implementation

```python
from openai import OpenAI

client = OpenAI()

def few_shot_extract(job_posting: str) -> str:
    examples = [
        ("We are hiring a Senior ML Engineer at Stripe.",
         '{"company": "Stripe", "role": "Senior ML Engineer"}'),
        ("Anthropic is looking for a Research Scientist.",
         '{"company": "Anthropic", "role": "Research Scientist"}'),
    ]

    messages = [
        {"role": "system", "content":
         "Extract company name and job title. Return only valid JSON."}
    ]

    for user_text, assistant_text in examples:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": job_posting})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content
```

Using alternating user/assistant messages to encode examples is the modern pattern for chat-based APIs. It is more reliable than embedding examples in a single text block.

---

## Tools and Frameworks

**LangChain FewShotPromptTemplate** — A template class that formats few-shot examples automatically. Supports dynamic example selection based on similarity to the input.

**LangChain ExampleSelector** — Selects the most relevant examples from a larger pool using semantic similarity. Useful when you have many examples and want to inject the most relevant ones dynamically.

**DSPy** — Optimizes few-shot example selection automatically from a dataset. Reduces the manual work of picking good examples.

**OpenAI fine-tuning** — When you have hundreds of examples, fine-tuning converts them into model weights. More expensive upfront but produces faster, cheaper inference than injecting examples every request.

---

## Common Mistakes

**Too few examples for complex tasks** — For simple classification, one or two examples may suffice. For complex extraction with many edge cases, you need five or more. Match example count to task complexity.

**Non-representative examples** — If all your examples show easy, clean inputs, the model will struggle on messy real-world data. Include examples with typos, abbreviations, and edge cases.

**Examples that contradict each other** — Inconsistent examples are worse than no examples. The model cannot infer a consistent rule from contradictory data.

**Putting instructions after the examples** — The model expects the task input to follow the example pattern. Adding post-example instructions breaks the format and reduces accuracy.

**Not testing on held-out examples** — Always validate few-shot performance on examples not used in the prompt. The model should generalize, not just repeat.

---

## Best Practices

- **Start with three examples** — This is usually the sweet spot for cost vs. quality. Add more only if performance is insufficient.
- **Use real data for examples** — Examples from your actual production data perform better than manually crafted ones.
- **Order examples thoughtfully** — Put the most prototypical example last, closest to the model's generation. The last example has the strongest influence on output format.
- **Test the prompt, not just the examples** — The task description and format still matter. Few-shot examples do not compensate for a badly worded instruction.
- **Version your example sets** — When you update examples, track what changed and why. Example quality drift is a common silent failure.

---

## Summary

Few-shot prompting is one of the most reliable tools in a developer's AI toolkit. By providing concrete examples of the expected input/output pattern, you give the model precise calibration that prose instructions alone cannot achieve.

Use it whenever format consistency matters, whenever zero-shot underperforms, and whenever your task has domain-specific conventions. Keep examples representative, consistent, and high-quality.

For foundational context, see [prompt engineering guide](/blog/prompt-engineering-guide/). For reasoning-focused techniques, see [chain-of-thought prompting](/blog/chain-of-thought-prompting/). For the complete list of techniques, see [prompt engineering techniques](/blog/prompt-engineering-techniques/).
