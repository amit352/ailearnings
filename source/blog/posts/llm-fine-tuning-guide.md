---
title: "LLM Fine-Tuning Guide – Customize AI Models"
description: "Learn how developers fine-tune large language models."
date: "2026-03-13"
slug: "llm-fine-tuning-guide"
keywords: ["LLM fine-tuning", "fine-tune LLM", "fine-tuning guide", "how to fine-tune LLM", "custom AI model"]
---

# LLM Fine-Tuning Guide – Customize AI Models

Pre-trained LLMs are general-purpose. They handle a wide range of tasks but may not speak your domain's language, follow your company's style, or excel at your specific task format. Fine-tuning adapts a pre-trained model to a specific use case using a curated dataset, without training from scratch. This guide explains when and how to fine-tune effectively.

---

## What is LLM Fine-Tuning

Fine-tuning is the process of continuing a pre-trained model's training on a smaller, task-specific dataset. The model's parameters — already initialized from pre-training on billions of documents — are updated to improve performance on your specific task.

The result is a model that has the general capabilities of the base model plus specialization on your data. Common fine-tuning goals:
- Match a specific output format or style consistently
- Perform well on a narrow domain (medical, legal, code in a specific language)
- Reduce hallucination on domain-specific facts
- Follow complex instructions reliably

Fine-tuning is one option among several. It competes with prompting and RAG for most use cases.

---

## Why Fine-Tuning Matters for Developers

Prompting is usually the right first step. It is faster, cheaper, and requires no training infrastructure. But prompting has limits:

- You cannot reliably teach a model new facts through prompting (the model may ignore injected context)
- Few-shot examples take tokens — at scale, this is expensive
- Some tasks require consistency across thousands of calls that prompting alone cannot guarantee
- If your use case requires a specific writing style, tone, or structured output format at low latency, a fine-tuned smaller model often outperforms a larger prompted model

For most applications, try prompting first, then RAG, then fine-tuning. Fine-tuning is the most complex and expensive option but can be the most powerful for the right use cases.

---

## How Fine-Tuning Works

### Full Fine-Tuning

All model parameters are updated. Most effective but requires significant GPU memory. A 7B model needs ~28GB VRAM for full fine-tuning. Impractical for most developers without specialized infrastructure.

### Parameter-Efficient Fine-Tuning (PEFT)

Only a small subset of parameters are updated. The original model weights are frozen. Methods include:

**LoRA (Low-Rank Adaptation)** — Adds small trainable matrices to attention layers. Reduces trainable parameters by 10,000x compared to full fine-tuning. See [LoRA fine-tuning explained](/blog/lora-fine-tuning-explained/) for details.

**QLoRA** — LoRA with the base model quantized to 4-bit. Enables fine-tuning large models on a single consumer GPU. See [QLoRA explained](/blog/qlora-explained/).

**Prefix tuning** — Adds trainable prefix tokens to the input. Only the prefix parameters are updated.

### Supervised Fine-Tuning (SFT)

The most common approach. Train on labeled examples of the format: input → desired output.

```python
# Training data format (OpenAI style)
training_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are a Python code reviewer."},
            {"role": "user", "content": "Review this function:\ndef add(a, b):\n    return a + b"},
            {"role": "assistant", "content": "The function looks correct for basic addition. Consider adding type hints: def add(a: int | float, b: int | float) -> int | float. Also add a docstring describing parameters and return type."}
        ]
    },
    # ... hundreds more examples
]
```

---

## Practical Examples

### Fine-Tuning with Hugging Face + LoRA

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,       # QLoRA: load in 4-bit
    device_map="auto"
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show ~0.1% of total params

# Load your dataset
train_data = Dataset.from_list([
    {"text": "<|system|>You are a code reviewer.<|user|>Review this code...<|assistant|>..."},
    # ... more examples
])

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    tokenizer=tokenizer,
)
trainer.train()
```

### OpenAI Fine-Tuning API (Managed)

```python
from openai import OpenAI
import json

client = OpenAI()

# Upload training file
with open("training_data.jsonl", "rb") as f:
    response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3}
)
print(f"Job ID: {job.id}")

# Check status
status = client.fine_tuning.jobs.retrieve(job.id)
print(f"Status: {status.status}")
print(f"Fine-tuned model: {status.fine_tuned_model}")
```

---

## Tools and Frameworks

**Hugging Face TRL** — The standard library for fine-tuning LLMs. SFTTrainer, DPOTrainer, and PPOTrainer cover the main training paradigms.

**Unsloth** — Optimized LoRA fine-tuning that runs 2–5x faster than standard implementations with lower memory usage.

**LLaMA Factory** — Web UI for fine-tuning open-source models. Good for non-code workflows.

**OpenAI fine-tuning API** — Managed fine-tuning for GPT models. No infrastructure management. Best for teams using OpenAI models.

**Axolotl** — Flexible fine-tuning framework supporting many model architectures and training methods.

---

## Common Mistakes

**Too little data** — Fine-tuning on fewer than 50–100 examples rarely produces consistent improvement. For meaningful specialization, plan for hundreds to thousands of examples.

**Low-quality training data** — The model will learn from every example, including bad ones. Curate training data carefully. Quality matters more than quantity.

**Fine-tuning when prompting would suffice** — Fine-tuning is expensive and slow to iterate. Exhaust prompting and RAG options before committing to a fine-tuning workflow.

**No evaluation set** — Hold out 10–20% of examples for evaluation. Without it, you cannot measure whether fine-tuning improved performance.

**Over-fitting on small datasets** — Too many epochs on too little data causes the model to memorize training examples and fail to generalize. Monitor validation loss.

---

## Best Practices

- **Curate, don't just collect** — Training data quality is the most important factor in fine-tuning success. Review every example before including it.
- **Start with LoRA** — Unless you have a strong reason for full fine-tuning, start with LoRA or QLoRA. Lower memory, faster iteration, and often comparable performance.
- **Use a consistent data format** — All training examples must follow the same prompt format. Inconsistency confuses the model.
- **Evaluate on task-specific benchmarks** — Define success metrics before fine-tuning. Measure against them before and after training.
- **Keep the base model accessible** — Fine-tuned models may regress on general tasks. Keep the base model available as a fallback.

---

## Summary

Fine-tuning adapts a pre-trained LLM to your specific use case using a curated dataset. PEFT methods like LoRA and QLoRA make this accessible without massive GPU infrastructure.

Use fine-tuning when prompting and RAG have been exhausted and you need consistent behavior, domain-specific language, or a specific output format at scale. Invest in high-quality training data — it is the primary determinant of fine-tuning success.

For efficient fine-tuning with minimal compute, see [LoRA fine-tuning explained](/blog/lora-fine-tuning-explained/). For the most memory-efficient approach, see [QLoRA explained](/blog/qlora-explained/).
