"""
model_utils.py — Model loading, LoRA configuration, and inference helpers.
"""

import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.config import (
    MODEL_ID, BNB_CONFIG, _COMPUTE_DTYPE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    INFERENCE_SYSTEM_PROMPT,
)


def debug_resources(label: str = ""):
    """Print current CPU RAM and GPU VRAM usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  [{label}] RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB "
              f"({mem.percent}% used, {mem.available/1e9:.1f} GB free)")
    except ImportError:
        print(f"  [{label}] psutil not installed — skipping RAM check")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            name = torch.cuda.get_device_name(i)
            print(f"  [{label}] GPU {i} ({name}): {alloc:.2f}/{total:.1f} GB")


def load_base_model(model_id: str = MODEL_ID):
    """Load Qwen2.5-1.5B with 4-bit quantization.

    Optimised for Kaggle T4 (16GB VRAM, ~13GB RAM):
    - torch_dtype=float16 to halve CPU RAM during load
    - low_cpu_mem_usage=True for lazy weight init
    - attn_implementation="eager" to avoid flash-attn memory spikes
    - device_map={"": 0} to skip auto device map computation
    """
    print("🔄 Loading model...")
    debug_resources("BEFORE")

    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Force garbage collection before the big allocation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=_COMPUTE_DTYPE,
        device_map={"": 0},            # direct to GPU, skip auto mapping overhead
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",   # avoids flash-attn extra buffers
    )

    debug_resources("AFTER")
    print(f"✅ Model loaded: {model_id}")

    return model, tokenizer


def apply_lora(model):
    """Wrap model with LoRA adapters for parameter-efficient training."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config


def load_finetuned_model(lora_path: str, model_id: str = MODEL_ID):
    """Reload base model + LoRA adapter for inference/evaluation."""
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=_COMPUTE_DTYPE,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    finetuned_model = PeftModel.from_pretrained(base_model, lora_path)
    finetuned_model.eval()

    print(f"✅ Fine-tuned model loaded from: {lora_path}")
    return finetuned_model, tokenizer


def generate_judgment(
    model,
    tokenizer,
    facts: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """Generate a legal judgment from the model given case facts."""
    prompt = f"""<|im_start|>system
{INFERENCE_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
निम्नलिखित मामले पर विचार करें:

{facts[:1200]}

Provide legal reasoning and judgment.<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
