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
    """Print current CPU RAM, GPU VRAM, and disk usage."""
    import psutil
    
    mem = psutil.virtual_memory()
    print(f"  [{label}]")
    print(f"    CPU RAM: {mem.used / 1e9:.2f} / {mem.total / 1e9:.2f} GB "
          f"({mem.percent}% used, {mem.available / 1e9:.2f} GB free)")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            name = torch.cuda.get_device_name(i)
            print(f"    GPU {i} ({name}): {alloc:.2f} GB alloc / "
                  f"{reserved:.2f} GB reserved / {total:.1f} GB total")
    else:
        print("    GPU: ❌ CUDA not available")
    
    disk = psutil.disk_usage('/')
    print(f"    Disk: {disk.used / 1e9:.1f} / {disk.total / 1e9:.1f} GB "
          f"({disk.percent}% used)")


def load_base_model(model_id: str = MODEL_ID):
    """Load Qwen2.5-1.5B with 4-bit quantization.
    
    Returns:
        tuple: (model, tokenizer)
    """
    print("=" * 50)
    print("🔍 DEBUG: Model Loading Diagnostics")
    print("=" * 50)
    
    # Step 0: Check environment
    debug_resources("BEFORE ANYTHING")
    
    print(f"\n  Model ID:      {model_id}")
    print(f"  Compute dtype: {_COMPUTE_DTYPE}")
    print(f"  BNB config:    {BNB_CONFIG}")
    print(f"  Python torch:  {torch.__version__}")
    
    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes:  {bnb.__version__}")
    except Exception as e:
        print(f"  bitsandbytes:  ❌ ERROR: {e}")
    
    # Step 1: Create quantization config
    print("\n--- Step 1: Creating BitsAndBytesConfig ---")
    try:
        bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
        print("  ✅ BNB config created")
    except Exception as e:
        print(f"  ❌ BNB config FAILED: {e}")
        raise
    
    # Step 2: Load tokenizer
    print("\n--- Step 2: Loading tokenizer ---")
    debug_resources("BEFORE TOKENIZER")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"  ✅ Tokenizer loaded (vocab: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"  ❌ Tokenizer FAILED: {e}")
        raise
    debug_resources("AFTER TOKENIZER")
    
    # Step 3: Free memory before model load
    print("\n--- Step 3: Clearing memory before model load ---")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    debug_resources("AFTER GC")
    
    # Step 4: Load model
    print("\n--- Step 4: Loading model (this is where it usually crashes) ---")
    print(f"  torch_dtype={_COMPUTE_DTYPE}")
    print(f"  device_map='auto'")
    print(f"  low_cpu_mem_usage=True")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=_COMPUTE_DTYPE,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print("  ✅ Model loaded successfully!")
    except Exception as e:
        print(f"\n  ❌ MODEL LOAD FAILED: {type(e).__name__}: {e}")
        debug_resources("AT FAILURE")
        raise
    
    debug_resources("AFTER MODEL LOAD")
    
    print("\n" + "=" * 50)
    print("✅ Model loading complete!")
    print("=" * 50)
    
    return model, tokenizer


def apply_lora(model):
    """Wrap model with LoRA adapters for parameter-efficient training.
    
    Returns:
        PeftModel with LoRA adapters applied.
    """
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
    """Reload base model + LoRA adapter for inference/evaluation.
    
    Returns:
        tuple: (finetuned_model, tokenizer)
    """
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=_COMPUTE_DTYPE,   # load in fp16 to halve CPU RAM
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
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
    """Generate a legal judgment from the model given case facts.
    
    Args:
        model: The language model (base or fine-tuned).
        tokenizer: The tokenizer.
        facts: Case facts text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
    
    Returns:
        str: Generated judgment text.
    """
    prompt = f"""<|im_start|>system
{INFERENCE_SYSTEM_PROMPT}<|im_end|>
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
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
