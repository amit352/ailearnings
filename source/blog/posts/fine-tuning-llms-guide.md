---
title: "Fine-Tuning LLMs: Complete Guide to Instruction Tuning and LoRA"
description: "Learn how to fine-tune large language models — full fine-tuning vs LoRA vs QLoRA, dataset preparation, training with Hugging Face, and avoiding common pitfalls."
date: "2026-03-10"
slug: "fine-tuning-llms-guide"
keywords: ["fine-tuning LLMs", "LoRA fine-tuning", "LLM fine-tuning guide", "instruction tuning"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-13"
---

# Fine-Tuning LLMs: Complete Guide to Instruction Tuning and LoRA

Fine-tuning is not the first thing you should try when an LLM underperforms on your task — it is the last thing. Improve your prompt first, add few-shot examples, try RAG if you need domain knowledge. Fine-tuning is powerful but expensive, slow to iterate, and genuinely necessary for a narrower set of problems than developers typically assume. When it is the right tool, though, it produces results that no amount of prompt engineering can match: consistent output format, deep domain adaptation, dramatic latency reductions by distilling large model behavior into smaller models. This guide covers when to fine-tune, how to prepare data correctly, and how to train efficiently with LoRA and QLoRA.

---

## When Fine-Tuning is the Right Choice

Before starting a fine-tuning project, work through this decision table honestly:

| Situation | Better Approach |
|-----------|----------------|
| Model ignores format instructions | Fine-tune |
| Model lacks domain-specific terminology or reasoning | Fine-tune |
| Latency is too high — need a smaller, faster model | Fine-tune a smaller model |
| You need private data kept out of third-party APIs | Fine-tune a local model |
| You need to update knowledge frequently | RAG — fine-tuning requires retraining |
| You need to answer questions about specific documents | RAG — fine-tuning doesn't inject passages |
| The model just needs better instructions | Improve your prompt first |
| You have fewer than 50 examples | Not enough data — use few-shot prompting |

The most common waste in fine-tuning projects: spending two weeks preparing a dataset and training runs to achieve what a better system prompt would have done in an afternoon.

---

## Types of Fine-Tuning

### Full Fine-Tuning

Update all model parameters on your training data. This has the highest performance ceiling but requires enormous compute — proportionally similar to the original pre-training. Impractical for models above 1B parameters without a dedicated GPU cluster.

### Supervised Fine-Tuning (SFT / Instruction Tuning)

Fine-tune on instruction-response pairs. This is how GPT-3 became InstructGPT — showing the model examples of good instructions and responses makes it dramatically better at following instructions. Most fine-tuning projects use SFT.

### RLHF (Reinforcement Learning from Human Feedback)

Fine-tune using human preference comparisons. Humans rate two model responses, the model learns to prefer what humans prefer. This is how ChatGPT, Claude, and other aligned models are created. Very complex and expensive — not applicable to most developer projects.

### LoRA (Low-Rank Adaptation)

Adds small trainable adapter matrices to frozen model layers. Reduces trainable parameters by 100–1000x compared to full fine-tuning. Near-equivalent quality at a tiny fraction of the compute cost. This is the standard approach for most fine-tuning projects today.

### QLoRA (Quantized LoRA)

Quantizes the base model to 4-bit format, then applies LoRA adapters. Enables fine-tuning a 70B model on a single 48GB GPU. The practical default for developer-accessible hardware. See [QLoRA explained](/blog/qlora-explained/) for the technique in depth.

---

## Dataset Preparation

Dataset quality is the most important factor in fine-tuning quality. 500 carefully curated examples consistently outperforms 10,000 scraped examples with noisy labels.

### The Instruction-Following Format

Modern LLMs use a chat template format with system, user, and assistant turns:

```python
# Standard example format — adjust to match your base model's template
{
    "messages": [
        {"role": "system",    "content": "You are a SQL expert who writes clean, commented SQL."},
        {"role": "user",      "content": "Get all customers who made a purchase in the last 30 days"},
        {"role": "assistant", "content": "SELECT DISTINCT c.customer_id, c.name, c.email\nFROM customers c\nINNER JOIN orders o ON c.customer_id = o.customer_id\nWHERE o.created_at >= CURRENT_DATE - INTERVAL '30 days'\nORDER BY c.name;"}
    ]
}
```

### Build and Save a Training Dataset

```python
import json

dataset = [
    {
        "messages": [
            {"role": "system",    "content": "You are a SQL expert."},
            {"role": "user",      "content": "Find all users where age > 30"},
            {"role": "assistant", "content": "SELECT * FROM users WHERE age > 30;"}
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": "You are a SQL expert."},
            {"role": "user",      "content": "Count orders by status"},
            {"role": "assistant", "content": "SELECT status, COUNT(*) AS order_count\nFROM orders\nGROUP BY status\nORDER BY order_count DESC;"}
        ]
    },
    # ... more examples
]

# Save as JSONL (one JSON object per line)
with open("train.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

# Create a small validation split (10–20% of data)
split = int(len(dataset) * 0.8)
with open("val.jsonl", "w") as f:
    for item in dataset[split:]:
        f.write(json.dumps(item) + "\n")

print(f"Training: {split} examples, Validation: {len(dataset) - split} examples")
```

**Data quality checklist:**
- Every response is correct (have a domain expert review a sample)
- Responses have consistent format and tone
- Coverage is diverse — include edge cases and difficult examples
- Length is appropriate — don't include artificially short or long responses
- System prompt matches what you will use at inference

---

## Fine-Tuning with LoRA and QLoRA

### Install Dependencies

```bash
pip install transformers trl peft bitsandbytes datasets accelerate
```

### Load Base Model with 4-bit Quantization (QLoRA)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "meta-llama/Llama-3.2-3B-Instruct"

# 4-bit quantization configuration for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # important for causal LM training
```

### Configure LoRA

```python
from peft import LoraConfig, prepare_model_for_kbit_training

# Prepare quantized model for training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,              # rank — higher = more capacity, more trainable params
    lora_alpha=32,     # scaling factor (rule of thumb: 2×r)
    target_modules=[   # apply LoRA to these weight matrices
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Rank selection guide:**
- `r=8` — simple task adaptation, minimal memory overhead
- `r=16` — good default for most instruction-following tasks
- `r=64` — complex tasks, large vocabulary domains, 70B+ models

### Training with SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load datasets
train_dataset = load_dataset("json", data_files={"train": "train.jsonl"})["train"]
eval_dataset  = load_dataset("json", data_files={"train": "val.jsonl"})["train"]

training_args = SFTConfig(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # effective batch size = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,                    # evaluate on val set every 50 steps
    bf16=True,
    max_seq_length=2048,
    optim="paged_adamw_32bit",        # paged optimizer for QLoRA
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()
trainer.save_model("./fine-tuned-model")
print("Training complete. Adapter saved.")
```

---

## Merging LoRA Weights for Deployment

The LoRA adapter is a small file (50–200MB) that modifies the base model's behavior when loaded on top of it. For production inference, merge the adapter into the base model for slightly faster inference and simpler deployment.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

# Load base model in full precision for merging
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load and merge LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, "./fine-tuned-model")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged-model")

tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./merged-model")
print("Merged model saved.")
```

Keep the adapter files. Merging is not reversible unless you have both the base model and adapter saved.

---

## Running Inference

```python
from transformers import pipeline

# Load merged model
pipe = pipeline(
    "text-generation",
    model="./merged-model",
    tokenizer=tokenizer,
    device_map="auto",
)

# Test a few examples from your use case
test_messages = [
    {"role": "system", "content": "You are a SQL expert."},
    {"role": "user",   "content": "Get the top 10 customers by total order value"},
]

output = pipe(
    test_messages,
    max_new_tokens=256,
    do_sample=False,  # greedy decoding for deterministic SQL
    temperature=None,
    top_p=None,
)
print(output[0]["generated_text"][-1]["content"])
```

---

## Evaluation

### Watch Validation Loss During Training

If validation loss starts increasing while training loss continues decreasing, you are overfitting. Stop training and use the checkpoint from before validation loss started rising.

```
Step  50: train_loss=1.82 eval_loss=1.85  ← good
Step 100: train_loss=1.54 eval_loss=1.58  ← good
Step 150: train_loss=1.31 eval_loss=1.55  ← overfitting starting
Step 200: train_loss=1.10 eval_loss=1.72  ← overfitting confirmed, use step 100 checkpoint
```

### Manual Evaluation Set

Create 50–100 test examples that were never seen during training. For each, manually evaluate:
- Is the answer correct?
- Is the format what you specified?
- Are there any hallucinations or errors?

Track these as a percentage — your fine-tuned model should score significantly higher than the base model on your specific task.

---

## Common Mistakes

**Not applying the chat template correctly** — Each base model has a specific chat template that adds special tokens around messages. Applying the wrong template (or none at all) causes the model to see malformed training data and produce garbage output at inference.

**Training the model to reproduce the prompt** — The loss should be computed only on the assistant's response tokens, not on the system and user turns. `SFTTrainer` handles this automatically with the messages format; verify that `packing=False` and the dataset format is correct.

**Catastrophic forgetting** — Fine-tuning on a narrow task can cause the model to forget general capabilities. Mitigation: use fewer epochs (1–2), use a lower rank (r=8 or r=16), and optionally mix in 5–10% general instruction data.

**Too high a learning rate** — QLoRA is more sensitive to learning rate than standard LoRA. If training loss spikes or oscillates, reduce from `2e-4` to `1e-4` or `5e-5`.

**Skipping gradient checkpointing** — `gradient_checkpointing=True` reduces peak GPU memory usage by recomputing activations during the backward pass. Enable it when training large models with limited memory, at the cost of ~20% slower training.

**Merging before validating** — Run your evaluation on the LoRA adapter before merging. Once merged, reverting requires keeping both checkpoints.

---

## What to Learn Next

Fine-tuning sits at the intersection of data engineering, ML training, and deployment. Each adjacent topic adds leverage:

- **QLoRA in depth — the memory-efficient technique behind this guide** → [QLoRA Explained](/blog/qlora-explained/)
- **When to use RAG instead of fine-tuning** → [How to Build a RAG Application](/blog/build-rag-app/)
- **Deploy your fine-tuned model** → [Deploying AI Applications](/blog/deploying-ai-applications/)
