"""
config.py — Central configuration for the Indian Legal LLM pipeline.

All hyperparameters, model IDs, paths, and prompts live here so notebooks
stay clean and changes propagate everywhere at once.
"""

import os
import torch

# ─────────────────────────────────────────────
# API Keys (loaded from env / Kaggle secrets)
# ─────────────────────────────────────────────

def _get_secret(name: str) -> str:
    """Try Kaggle secrets first, then fall back to env var."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        val = os.environ.get(name, "")
        if not val:
            print(f"⚠️  {name} not found. Set it via env var or Kaggle Secrets.")
        return val


HF_TOKEN          = _get_secret("HF_TOKEN")
DEEPSEEK_API_KEY  = _get_secret("DEEPSEEK_API_KEY")

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Auto-detect: T4/P100 don't support bfloat16, use float16 instead
_COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

BNB_CONFIG = {
    "load_in_4bit":              True,
    "bnb_4bit_quant_type":       "nf4",
    "bnb_4bit_compute_dtype":    _COMPUTE_DTYPE,
    "bnb_4bit_use_double_quant": True,
}

# ─────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────

LORA_R             = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ─────────────────────────────────────────────
# SFT Training
# ─────────────────────────────────────────────

SFT_OUTPUT_DIR           = "./indian-legal-1.5b"
SFT_LORA_SAVE_DIR        = "./indian-legal-1.5b-lora"
SFT_EPOCHS               = 3
SFT_BATCH_SIZE            = 2
SFT_GRAD_ACCUM_STEPS      = 8
SFT_LEARNING_RATE          = 2e-4
SFT_WARMUP_RATIO           = 0.05
SFT_MAX_SEQ_LENGTH         = 1024

# ─────────────────────────────────────────────
# GRPO Training
# ─────────────────────────────────────────────

GRPO_OUTPUT_DIR          = "./indian-legal-1.5b-grpo"
GRPO_FINAL_SAVE_DIR      = "./indian-legal-1.5b-final"
GRPO_LEARNING_RATE       = 5e-6
GRPO_BATCH_SIZE          = 1
GRPO_GRAD_ACCUM_STEPS    = 16
GRPO_NUM_GENERATIONS     = 4
GRPO_MAX_NEW_TOKENS      = 512
GRPO_EPOCHS              = 1

# ─────────────────────────────────────────────
# DeepSeek API
# ─────────────────────────────────────────────

DEEPSEEK_BASE_URL    = "https://api.deepseek.com"
DEEPSEEK_R1_MODEL    = "deepseek-reasoner"
DEEPSEEK_CHAT_MODEL  = "deepseek-chat"

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

ILDC_DATASET       = "labofsahil/Indian-Supreme-Court-Judgments"
NYAYA_DATASET      = "anuragiiser/ILDC_expert"
COT_DATA_PATH      = "cot_data.jsonl"
COT_NUM_SAMPLES    = 500
TEST_SPLIT_RATIO   = 0.05
MAX_FACTS_LENGTH   = 1500
MAX_FACTS_COT      = 1000

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

EVAL_N_SAMPLES       = 100
JUDGE_N_SAMPLES      = 30
EVAL_REPORT_PATH     = "legal_eval_report.csv"

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Indian legal assistant.
आप भारतीय कानून के विशेषज्ञ हैं।
You know IPC, BNS (2023), CrPC, BNSS, Evidence Act, and Supreme Court precedents."""

COT_PROMPT = """You are a Senior Indian Supreme Court judge.

Analyze this case step by step:
1. Identify applicable sections (IPC/BNS/CrPC)
2. Recall relevant precedents
3. Apply legal reasoning
4. State your judgment

IMPORTANT: Where relevant, respond in both Hindi and English.

Case Facts:
{facts}"""

INFERENCE_SYSTEM_PROMPT = (
    "You are an expert Indian legal reasoner. "
    "Think step by step using IPC/BNS sections."
)

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
