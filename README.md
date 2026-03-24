# 🏛️ Indian Legal Reasoning LLM (1.5B Parameters)

> **Building a domain-specific Indian legal reasoning model using knowledge distillation and reinforcement learning.**

## Architecture

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

## 🔑 Required API Keys

| Key | Source | Purpose | Cost |
|-----|--------|---------|------|
| **HuggingFace Token** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Model & dataset access | Free |
| **DeepSeek API Key** | [platform.deepseek.com](https://platform.deepseek.com) | CoT generation + LLM Judge | ~$1.55 |

### Setting API Keys

**Kaggle:** Add secrets via *Add-ons → Secrets*:
- `HF_TOKEN`
- `DEEPSEEK_API_KEY`

**Local/Colab:** Set environment variables:
```bash
export HF_TOKEN="hf_..."
export DEEPSEEK_API_KEY="sk-..."
```

## 📁 Project Structure

```
RLProject/
├── README.md
├── requirements.txt
├── .gitignore
├── indian_legal_llm_guide.md     # Original reference guide
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                 # All hyperparameters & constants
│   ├── model_utils.py            # Model loading, LoRA, inference
│   ├── data_utils.py             # Dataset loading & formatting
│   ├── cot_generator.py          # DeepSeek-R1 CoT generation
│   ├── reward_functions.py       # GRPO reward functions
│   └── evaluation.py             # ROUGE, BERTScore, LLM Judge
└── notebooks/                    # Ordered execution notebooks
    ├── 01_setup_and_data.ipynb
    ├── 02_cot_generation.ipynb
    ├── 03_sft_training.ipynb
    ├── 04_grpo_training.ipynb
    └── 05_evaluation.ipynb
```

## 🚀 Execution Order

| Day | Notebook | Duration | GPU Required |
|-----|----------|----------|-------------|
| 1 | `01_setup_and_data.ipynb` | ~15 min | ✅ Yes |
| 2 | `02_cot_generation.ipynb` | ~1 hr | ❌ No (API calls) |
| 3 | `03_sft_training.ipynb` | ~3 hrs | ✅ Yes (T4/P100) |
| 4 | `04_grpo_training.ipynb` | ~2 hrs | ✅ Yes (T4/P100) |
| 5 | `05_evaluation.ipynb` | ~1 hr | ✅ Yes |

## 💾 GPU Memory Budget (Kaggle T4/P100 — 16GB)

| Component | Memory |
|-----------|--------|
| Model (4-bit quantized) | ~1.5 GB |
| LoRA adapters | ~0.3 GB |
| Optimizer states | ~4.0 GB |
| Batch (seq length 1024) | ~6.0 GB |
| Activations + overhead | ~2.0 GB |
| **Total** | **~13.8 GB ✓** |

## 📊 Expected Results

| Metric | Weak | Acceptable | Good (Target) |
|--------|------|------------|---------------|
| ROUGE-L | < 0.10 | 0.15–0.25 | **> 0.30** |
| BERTScore F1 | < 0.75 | 0.78–0.83 | **> 0.85** |
| LLM Judge Overall | < 4/10 | 5–6/10 | **> 7/10** |
| LLM Judge Legal Accuracy | < 3/10 | 5–6/10 | **> 7/10** |

## ⚠️ Key Pitfalls

1. **IPC vs BNS confusion** — BNS replaced IPC in 2023; model must learn both.
2. **Citation hallucination** — GRPO reward penalises invented section numbers.
3. **Notebook crashes** — CoT data saved incrementally every 50 samples.
4. **Hindi tokenization** — Qwen2.5 handles Devanagari natively (don't use LLaMA).
5. **Long judgment truncation** — Facts truncated to 1,200 tokens to avoid OOM.

---

*Stack: Qwen2.5 · DeepSeek-R1 · HuggingFace TRL · ILDC · NyayaAnumana*
