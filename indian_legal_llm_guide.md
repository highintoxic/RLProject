# 🏛️ Building an Indian Legal Reasoning LLM (1.5B Parameters)
### A Complete Pipeline: From Dataset → Fine-tuning → Evaluation

> **Stack:** Qwen2.5-1.5B-Instruct · DeepSeek-R1 (Teacher) · ILDC + NyayaAnumana · LoRA + GRPO · Kaggle/Colab Pro

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Environment Setup](#2-environment-setup)
3. [Load the Base Model](#3-load-the-base-model)
4. [Load Indian Legal Datasets](#4-load-indian-legal-datasets)
5. [Generate Chain-of-Thought Data via DeepSeek-R1](#5-generate-chain-of-thought-data-via-deepseek-r1)
6. [SFT Fine-tuning with LoRA](#6-sft-fine-tuning-with-lora)
7. [RL Fine-tuning with GRPO](#7-rl-fine-tuning-with-grpo)
8. [Testing & Evaluation](#8-testing--evaluation)
9. [Memory Budget & Execution Order](#9-memory-budget--execution-order)
10. [Expected Results](#10-expected-results)

---

## 1. Architecture Overview

```
ILDC + NyayaAnumana Dataset
        │
        ▼
DeepSeek-R1 (Teacher) ──► Chain-of-Thought Data
        │
        ▼
Qwen2.5-1.5B (Student Base Model)
        │
   ┌────┴────┐
   ▼         ▼
 SFT       GRPO RL
(LoRA)   (Rewards)
   └────┬────┘
        ▼
Indian Legal Reasoning Model
        │
        ▼
  Evaluation Suite
(ROUGE + BERTScore + LLM Judge)
```

**Key Design Decisions:**
- **Base Model:** Qwen2.5-1.5B-Instruct — native Hindi+English, Apache 2.0 license
- **Teacher Model:** DeepSeek-R1 via API (~$1–2 total cost for 500 cases)
- **Training Method:** LoRA (trains only ~0.8% of parameters) → fits in 16GB GPU
- **RL Method:** GRPO (much lighter than PPO, same method DeepSeek-R1 used)
- **Legal Grounding:** IPC, BNS (2023), CrPC, BNSS, Evidence Act, SC precedents

---

## 2. Environment Setup

```python
# Cell 1 — Run once at the start of your Kaggle/Colab notebook
!pip install -q transformers datasets peft trl accelerate bitsandbytes
!pip install -q sentencepiece protobuf huggingface_hub
!pip install -q rouge-score bert-score nltk evaluate
!pip install -q openai

from huggingface_hub import login
login("YOUR_HF_TOKEN")  # from huggingface.co/settings/tokens
```

---

## 3. Load the Base Model

```python
# Cell 2 — Load Qwen2.5-1.5B in 4-bit quantization
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 4-bit quantization — cuts memory from ~3GB to ~1.2GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"Model loaded. Memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
# Expected: ~1.5 GB
```

---

## 4. Load Indian Legal Datasets

```python
# Cell 3 — Load ILDC + NyayaAnumana
from datasets import load_dataset

# ILDC — Indian Legal Documents Corpus
ildc = load_dataset("rceborg/ildc", split="train")

# NyayaAnumana — structured Indian legal reasoning
nyaya = load_dataset("NyayaAnumana/NyayaAnumana", split="train")

print(f"ILDC samples: {len(ildc)}")
print(f"NyayaAnumana samples: {len(nyaya)}")
```

```python
# Cell 4 — Format into bilingual instruction tuning format
def format_sample(example):
    system = """You are an expert Indian legal assistant.
आप भारतीय कानून के विशेषज्ञ हैं।
You know IPC, BNS (2023), CrPC, BNSS, Evidence Act, and Supreme Court precedents."""

    facts    = example.get("facts") or example.get("text", "")[:1500]
    judgment = example.get("judgment") or example.get("label", "")

    return {
        "text": f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
निम्नलिखित मामले पर विचार करें / Consider the following case:

{facts}

What is the legal reasoning and applicable sections?<|im_end|>
<|im_start|>assistant
{judgment}<|im_end|>"""
    }

dataset = ildc.map(format_sample, remove_columns=ildc.column_names)
dataset = dataset.train_test_split(test_size=0.05)

print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")
```

---

## 5. Generate Chain-of-Thought Data via DeepSeek-R1

> **Cost estimate:** ~$1–2 for 500 cases using `deepseek-reasoner`
> Get your API key at: [platform.deepseek.com](https://platform.deepseek.com)

```python
# Cell 5 — Generate CoT from teacher model
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_DEEPSEEK_KEY",
    base_url="https://api.deepseek.com"
)

COT_PROMPT = """You are a Senior Indian Supreme Court judge.

Analyze this case step by step:
1. Identify applicable sections (IPC/BNS/CrPC)
2. Recall relevant precedents
3. Apply legal reasoning
4. State your judgment

IMPORTANT: Where relevant, respond in both Hindi and English.

Case Facts:
{facts}"""

def get_cot_from_teacher(facts: str) -> dict:
    resp = client.chat.completions.create(
        model="deepseek-reasoner",    # DeepSeek-R1
        messages=[{"role": "user", "content": COT_PROMPT.format(facts=facts)}],
        max_tokens=2048
    )
    return {
        "facts": facts,
        "reasoning": resp.choices[0].message.reasoning_content,  # R1's <think> trace
        "answer":    resp.choices[0].message.content
    }
```

```python
# Cell 6 — Run over 500 cases and save (don't regenerate!)
import json

cot_data = []

for i, case in enumerate(ildc.select(range(500))):
    facts = case.get("facts", case.get("text", ""))[:1000]
    try:
        result = get_cot_from_teacher(facts)
        cot_data.append(result)
        if i % 50 == 0:
            print(f"Done {i}/500")
    except Exception as e:
        print(f"Skipped {i}: {e}")

# Save immediately — protects against notebook crashes
with open("cot_data.jsonl", "w") as f:
    for row in cot_data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(cot_data)} CoT samples")
```

```python
# Cell 7 — Format CoT data for training
from datasets import Dataset

def format_cot_sample(row):
    return {
        "text": f"""<|im_start|>system
You are an expert Indian legal reasoner. Think step by step.<|im_end|>
<|im_start|>user
{row['facts']}<|im_end|>
<|im_start|>assistant
<think>
{row['reasoning']}
</think>

{row['answer']}<|im_end|>"""
    }

cot_dataset = Dataset.from_list([format_cot_sample(r) for r in cot_data])
print(f"CoT dataset ready: {len(cot_dataset)} samples")
```

---

## 6. SFT Fine-tuning with LoRA

```python
# Cell 8 — Configure LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                  # rank — higher = more capacity
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params ~12M / 1.5B total (~0.8%) ✓
```

```python
# Cell 9 — Train with SFTTrainer
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./indian-legal-1.5b",
    num_train_epochs=3,
    per_device_train_batch_size=2,     # small batch for 16GB GPU
    gradient_accumulation_steps=8,     # effective batch size = 16
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=False, fp16=True,             # T4/P100 don't support bf16
    logging_steps=10,
    save_steps=100,
    max_seq_length=1024,               # keep short to save memory
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=cot_dataset,
    args=training_args,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./indian-legal-1.5b-lora")
print("SFT training complete. Model saved.")
```

---

## 7. RL Fine-tuning with GRPO

> GRPO is much lighter than PPO — it's also the exact method DeepSeek-R1 used internally.

```python
# Cell 10 — Define reward functions (rule-based, no reward model needed)
import re

def reward_has_legal_citation(completions, **kwargs):
    """Reward for citing actual IPC/BNS/CrPC sections"""
    rewards = []
    for c in completions:
        citations = re.findall(
            r'(Section\s+\d+[A-Z]?\s+(?:IPC|BNS|CrPC|BNSS|IEA)|'
            r'(?:IPC|BNS)\s+[Ss]ection\s+\d+)',
            c
        )
        rewards.append(min(len(citations) * 0.3, 1.0))  # cap at 1.0
    return rewards

def reward_has_reasoning(completions, **kwargs):
    """Reward for structured step-by-step reasoning"""
    rewards = []
    for c in completions:
        score = 0.0
        if "<think>" in c and "</think>" in c:
            score += 0.4
        keywords = ["therefore", "अतः", "hence", "held", "observed", "ratio"]
        score += min(sum(0.1 for k in keywords if k.lower() in c.lower()), 0.6)
        rewards.append(score)
    return rewards

def reward_bilingual(completions, **kwargs):
    """Reward for Hindi + English mixed output"""
    rewards = []
    for c in completions:
        has_hindi   = bool(re.search(r'[\u0900-\u097F]', c))
        has_english = bool(re.search(r'[a-zA-Z]', c))
        rewards.append(0.5 if (has_hindi and has_english) else 0.0)
    return rewards
```

```python
# Cell 11 — GRPO training
from trl import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    output_dir="./indian-legal-1.5b-grpo",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_generations=4,                 # generate 4 responses, reward best
    max_new_tokens=512,
    num_train_epochs=1,
    fp16=True,
    report_to="none",
)

grpo_trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[
        reward_has_legal_citation,
        reward_has_reasoning,
        reward_bilingual,
    ],
    args=grpo_config,
    train_dataset=cot_dataset,
)

grpo_trainer.train()
grpo_trainer.save_model("./indian-legal-1.5b-final")
print("GRPO training complete.")
```

---

## 8. Testing & Evaluation

### 8.1 Load Both Models for Comparison

```python
# Cell 12 — Load base and fine-tuned models side by side
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH  = "./indian-legal-1.5b-lora"

# Base model (before fine-tuning)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# Fine-tuned model (LoRA adapter on top)
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

# Load ILDC test split
ildc_test = load_dataset("rceborg/ildc", split="test")
print(f"Test cases: {len(ildc_test)}")
```

### 8.2 Inference Helper

```python
# Cell 13 — Generate a judgment from any model
def generate_judgment(model, facts: str, max_new_tokens=512) -> str:
    prompt = f"""<|im_start|>system
You are an expert Indian legal reasoner. Think step by step using IPC/BNS sections.<|im_end|>
<|im_start|>user
निम्नलिखित मामले पर विचार करें:

{facts[:1200]}

Provide legal reasoning and judgment.<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Quick sanity check
sample = ildc_test[0]
print("=== FACTS ===")
print(sample["text"][:500])
print("\n=== MODEL OUTPUT ===")
print(generate_judgment(finetuned_model, sample["text"]))
```

### 8.3 ROUGE + BERTScore (100 samples)

```python
# Cell 14 — Automated metric evaluation
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

def evaluate_rouge_bert(model, dataset, n_samples=100):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    predictions, references = [], []

    for i, case in enumerate(dataset.select(range(n_samples))):
        facts     = case.get("text", "")
        reference = case.get("judgment") or case.get("label") or ""
        if not reference:
            continue

        prediction = generate_judgment(model, facts)

        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
        predictions.append(prediction)
        references.append(reference)

        if i % 10 == 0:
            print(f"Progress: {i}/{n_samples}")

    P, R, F1 = bert_score(
        predictions, references,
        lang="en",
        model_type="distilbert-base-uncased",  # lightweight for Kaggle
        verbose=False
    )

    return {
        "rouge1":     np.mean(rouge1_scores),
        "rouge2":     np.mean(rouge2_scores),
        "rougeL":     np.mean(rougeL_scores),
        "bert_score": F1.mean().item(),
        "n_samples":  len(predictions),
    }

print("Evaluating BASE model...")
base_scores = evaluate_rouge_bert(base_model, ildc_test, n_samples=100)

print("\nEvaluating FINE-TUNED model...")
ft_scores = evaluate_rouge_bert(finetuned_model, ildc_test, n_samples=100)

# Print before/after comparison table
print("\n" + "="*55)
print(f"{'Metric':<20} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
print("="*55)
for metric in ["rouge1", "rouge2", "rougeL", "bert_score"]:
    base_val = base_scores[metric]
    ft_val   = ft_scores[metric]
    delta    = ft_val - base_val
    arrow    = "▲" if delta > 0 else "▼"
    print(f"{metric:<20} {base_val:>10.4f} {ft_val:>12.4f} {arrow}{abs(delta):>8.4f}")
print("="*55)
```

### 8.4 LLM-as-Judge (Most Important Test)

> **Cost:** ~$0.05 for 30 cases using `deepseek-chat`

```python
# Cell 15 — DeepSeek judges reasoning quality
import json

JUDGE_PROMPT = """You are a strict Indian legal examiner evaluating AI-generated judgments.

Score the following AI judgment on these 4 criteria (0-10 each):

1. LEGAL_ACCURACY     — Are IPC/BNS/CrPC sections correctly identified?
2. REASONING_CHAIN    — Is the step-by-step logic coherent and sound?
3. REFERENCE_FIDELITY — How well does it align with the ground truth judgment?
4. COMPLETENESS       — Are all key legal issues addressed?

Case Facts:
{facts}

Ground Truth Judgment:
{reference}

AI Generated Judgment:
{prediction}

Respond ONLY in this JSON format, nothing else:
{{
  "legal_accuracy": <0-10>,
  "reasoning_chain": <0-10>,
  "reference_fidelity": <0-10>,
  "completeness": <0-10>,
  "overall": <0-10>,
  "feedback": "<one sentence explanation>"
}}"""

def llm_judge(facts: str, reference: str, prediction: str) -> dict:
    resp = client.chat.completions.create(
        model="deepseek-chat",       # cheaper than R1 for scoring
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                facts=facts[:800],
                reference=reference[:600],
                prediction=prediction[:600]
            )
        }],
        max_tokens=300,
        temperature=0.0,             # deterministic scoring
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "parse failed", "raw": resp.choices[0].message.content}


def run_llm_judge_eval(model, dataset, n_samples=30):
    results = []
    for i, case in enumerate(dataset.select(range(n_samples))):
        facts     = case.get("text", "")
        reference = case.get("judgment") or case.get("label") or ""
        if not reference:
            continue

        prediction = generate_judgment(model, facts)
        scores     = llm_judge(facts, reference, prediction)

        results.append({"case_id": i, "scores": scores, "prediction": prediction[:300]})

        if i % 5 == 0:
            print(f"Judged {i}/{n_samples} | Last overall: {scores.get('overall', 'N/A')}")

    return results


print("Running LLM Judge on fine-tuned model (30 cases)...")
judge_results = run_llm_judge_eval(finetuned_model, ildc_test, n_samples=30)

criteria = ["legal_accuracy", "reasoning_chain", "reference_fidelity", "completeness", "overall"]
print("\n=== LLM JUDGE SCORES (out of 10) ===")
for c in criteria:
    vals = [r["scores"].get(c, 0) for r in judge_results if "error" not in r["scores"]]
    print(f"  {c:<25} {np.mean(vals):.2f} ± {np.std(vals):.2f}")
```

### 8.5 Side-by-Side Manual Analysis

```python
# Cell 16 — Print base vs fine-tuned comparison for any case
def print_comparison(case_idx: int):
    case      = ildc_test[case_idx]
    facts     = case.get("text", "")
    reference = case.get("judgment") or case.get("label") or ""

    base_out = generate_judgment(base_model, facts)
    ft_out   = generate_judgment(finetuned_model, facts)

    print("=" * 70)
    print("FACTS:")
    print(facts[:600])
    print("\n--- GROUND TRUTH ---")
    print(reference[:400])
    print("\n--- BASE MODEL ---")
    print(base_out[:400])
    print("\n--- FINE-TUNED MODEL ---")
    print(ft_out[:400])
    print("=" * 70)

# Inspect a few cases manually
for idx in [0, 5, 12, 20]:
    print_comparison(idx)
```

### 8.6 Export Full Evaluation Report

```python
# Cell 17 — Save everything to CSV
import pandas as pd

rows = []
for r in judge_results:
    row = {"case_id": r["case_id"], "preview": r["prediction"][:150]}
    row.update(r["scores"])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("legal_eval_report.csv", index=False)

print("\n=== FINAL EVALUATION SUMMARY ===")
print(f"Samples evaluated : {len(df)}")
print(f"Avg Overall Score : {df['overall'].mean():.2f} / 10")
print(f"Avg Legal Accuracy: {df['legal_accuracy'].mean():.2f} / 10")
print(f"Avg Reasoning     : {df['reasoning_chain'].mean():.2f} / 10")
print(f"\nTop 5 best cases:\n{df.nlargest(5, 'overall')[['case_id','overall','feedback']]}")
print(f"\nBottom 5 cases:\n{df.nsmallest(5, 'overall')[['case_id','overall','feedback']]}")
```

---

## 9. Memory Budget & Execution Order

### GPU Memory Budget (Kaggle T4/P100 — 16GB)

| Component | Memory |
|---|---|
| Model (4-bit quantized) | ~1.5 GB |
| LoRA adapters | ~0.3 GB |
| Optimizer states | ~4.0 GB |
| Batch (seq length 1024) | ~6.0 GB |
| Activations + overhead | ~2.0 GB |
| **Total** | **~13.8 GB ✓** |

### Recommended Execution Order

```
Day 1 — Cells 1–4   : Setup + load datasets + format for SFT
Day 2 — Cells 5–7   : Generate 500 CoT samples via DeepSeek-R1 (~$1–2)
Day 3 — Cells 8–9   : SFT training with LoRA (~3 hrs on Kaggle T4)
Day 4 — Cells 10–11 : GRPO RL fine-tuning (~2 hrs)
Day 5 — Cells 12–17 : Full evaluation suite + export report
```

### Total Cost Estimate

| Item | Cost |
|---|---|
| DeepSeek-R1 (500 CoT samples) | ~$1.50 |
| DeepSeek Chat (30 judge evals) | ~$0.05 |
| Kaggle GPU (free tier) | $0.00 |
| **Total** | **~$1.55** |

---

## 10. Expected Results

For a **1.5B model** fine-tuned on Indian legal data, these are realistic targets:

| Metric | Weak | Acceptable | Good (Target) |
|---|---|---|---|
| ROUGE-L | < 0.10 | 0.15–0.25 | **> 0.30** |
| BERTScore F1 | < 0.75 | 0.78–0.83 | **> 0.85** |
| LLM Judge Overall | < 4/10 | 5–6/10 | **> 7/10** |
| LLM Judge Legal Accuracy | < 3/10 | 5–6/10 | **> 7/10** |

> Hitting **BERTScore > 0.80** and **LLM Judge > 6/10** on a 1.5B model is a strong result — competitive with much larger general-purpose models on this narrow domain.

---

## Key Pitfalls to Avoid

**1. IPC vs BNS confusion** — The BNS replaced IPC in 2023. Train your model on both and teach it to map between them (e.g., IPC 302 → BNS 101 for murder charges).

**2. Citation hallucination** — Small models invent section numbers. The `reward_has_legal_citation` GRPO reward directly penalises this.

**3. Notebook crashes during CoT generation** — Always save `cot_data.jsonl` incrementally (every 50 samples) so you don't lose API spend.

**4. Hindi tokenization** — Qwen2.5 handles Devanagari natively. Do not use LLaMA-based models for this task without verifying their Hindi vocabulary first.

**5. Long judgment truncation** — ILDC judgments can exceed 50,000 words. Truncate facts to 1,200 tokens max during training to avoid OOM errors on Kaggle.

---

*Built with: Qwen2.5 · DeepSeek-R1 · HuggingFace TRL · ILDC · NyayaAnumana*
