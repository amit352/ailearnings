---
title: "Fine-Tuning LLMs: Complete Guide to Instruction Tuning and LoRA"
description: "Learn how to fine-tune large language models — full fine-tuning vs LoRA vs QLoRA, dataset preparation, training with Hugging Face, and avoiding common pitfalls."
date: "2026-03-10"
slug: "fine-tuning-llms-guide"
keywords: ["fine-tuning LLMs", "LoRA fine-tuning", "LLM fine-tuning guide", "instruction tuning"]
---

## Learning Objectives

- Understand when fine-tuning is the right choice vs prompting
- Prepare instruction-following datasets
- Apply LoRA and QLoRA for efficient fine-tuning
- Train a model with Hugging Face's `transformers` and `trl` libraries
- Evaluate fine-tuned models and avoid common mistakes

---

## When to Fine-Tune (and When Not To)

Fine-tuning is not always the right answer. Before deciding, ask:

| Situation | Recommendation |
|-----------|---------------|
| You need better instruction following | Fine-tune |
| You need domain-specific knowledge | Fine-tune or RAG |
| You need consistent output format | Fine-tune |
| You need to update knowledge frequently | RAG is better |
| You need to inject private documents | RAG is better |
| You just need better prompts | Improve prompts first |

**Always try prompt engineering before fine-tuning.** Fine-tuning is expensive, slow to iterate, and overkill for many tasks.

---

## Types of Fine-Tuning

### Full Fine-Tuning
Update all model parameters. Highest performance ceiling but requires enormous compute (the same as pre-training scale, proportionally).

### Instruction Fine-Tuning (SFT)
Supervised fine-tuning on instruction-response pairs. Teaches the model to follow instructions. The foundation of most chat models.

### RLHF (Reinforcement Learning from Human Feedback)
Fine-tunes using human preference data. Used to create alignment and safety. Very complex — most developers don't need this.

### LoRA (Low-Rank Adaptation)
Adds small trainable matrices to frozen model layers. Reduces trainable parameters by 10–1000×. Near-full-fine-tuning quality at a fraction of the cost.

### QLoRA
Quantizes the base model to 4-bit, adds LoRA adapters. Enables fine-tuning a 70B model on a single 48GB GPU.

---

## Dataset Preparation

### Instruction-Following Format

```python
# Standard chat template (most modern LLMs)
{
    "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user",   "content": "Write a Python function to reverse a string."},
        {"role": "assistant", "content": "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"}
    ]
}
```

### Prepare Dataset File

```python
import json

# Build your dataset as a list of conversation dicts
dataset = [
    {
        "messages": [
            {"role": "system",    "content": "You are a SQL expert."},
            {"role": "user",      "content": "Convert this to SQL: get all users over 30"},
            {"role": "assistant", "content": "SELECT * FROM users WHERE age > 30;"}
        ]
    },
    # ... more examples
]

with open('train.jsonl', 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')
```

**Data quality tips:**
- 100–1000 high-quality examples often beats 10,000 poor ones
- Ensure diverse coverage of your use cases
- Match the format your base model was trained on

---

## Fine-Tuning with LoRA (Using Hugging Face + TRL)

### Install Dependencies

```bash
pip install transformers trl peft bitsandbytes datasets accelerate
```

### Load Base Model with 4-bit Quantization (QLoRA)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "meta-llama/Llama-3.2-3B-Instruct"

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
```

### Configure LoRA

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,              # rank — higher = more capacity, more params
    lora_alpha=32,     # scaling factor (typically 2×r)
    target_modules=[   # which layers to add LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Rule of thumb for rank `r`:**
- `r=8`: minimum, works for simple tasks
- `r=16`: good default
- `r=64`: high capacity, more memory

### Training with SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": "train.jsonl"})["train"]

training_args = SFTConfig(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch size = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    fp16=False,
    bf16=True,
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()
trainer.save_model("./fine-tuned-model")
```

---

## Merging LoRA Weights

After training, merge the LoRA adapter into the base model for deployment:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(base_model, "./fine-tuned-model")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

---

## Running Inference

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./merged-model",
    tokenizer=tokenizer,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a SQL expert."},
    {"role": "user",   "content": "Get all orders from 2025"},
]

output = pipe(messages, max_new_tokens=256, do_sample=True, temperature=0.7)
print(output[0]["generated_text"][-1]["content"])
```

---

## Evaluation

### Held-Out Validation Loss

Monitor loss on a held-out validation set during training. If validation loss increases while training loss decreases — you're overfitting.

```python
training_args = SFTConfig(
    ...
    eval_strategy="steps",
    eval_steps=50,
)
trainer = SFTTrainer(
    ...
    eval_dataset=eval_dataset,
)
```

### Manual Evaluation

Create a test set of 50–100 examples and manually rate model responses on:
- Correctness
- Format adherence
- Tone/style

### Automated Benchmarks

```python
# Use lm-evaluation-harness for standardized benchmarks
# pip install lm-eval
# lm_eval --model hf --model_args pretrained=./merged-model --tasks mmlu --device cuda
```

---

## Troubleshooting

**Loss doesn't decrease**
- Check that the chat template is applied correctly
- Verify labels are set properly (only train on assistant responses, not user inputs)
- Reduce learning rate

**Model generates gibberish**
- Learning rate too high — try `1e-4` or lower
- Sequence length too short — increase `max_seq_length`
- Base model and chat template mismatch

**Out-of-memory (OOM)**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use `gradient_checkpointing=True`

**Fine-tuned model forgets previous capabilities (catastrophic forgetting)**
- Train on fewer epochs (1–2)
- Mix in general instruction data (5–10% of your dataset)
- Use a lower rank LoRA

---

## FAQ

**How much data do I need?**
For instruction fine-tuning: 100 high-quality examples can meaningfully change behavior. 1000–5000 is solid. Tens of thousands for more general capabilities.

**Should I fine-tune or use RAG?**
Use RAG when you need to inject documents, knowledge that changes frequently, or cited sources. Use fine-tuning when you need to change behavior, tone, output format, or domain expertise.

**What GPU do I need?**
For QLoRA fine-tuning: 8B model → 1× RTX 4090 (24GB) or A100 (40GB). 70B model → 1× A100 80GB or 2× A100 40GB. For production training runs use cloud GPUs (Lambda Labs, RunPod, Vast.ai).

---

## What to Learn Next

- **Transformer architecture** → [Transformer Architecture Explained](/blog/transformer-architecture-explained/)
- **RAG systems** → [RAG Tutorial Step by Step](/blog/rag-tutorial-step-by-step/)
- **Deploying AI apps** → deploying-ai-applications
- **LLM roadmap** → [AI Roadmap for Developers](/blog/ai-roadmap-for-developers/)
