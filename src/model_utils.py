"""
model_utils.py — Model loading, LoRA configuration, and inference helpers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.config import (
    MODEL_ID, BNB_CONFIG,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    INFERENCE_SYSTEM_PROMPT,
)


def load_base_model(model_id: str = MODEL_ID):
    """Load Qwen2.5-1.5B with 4-bit quantization.
    
    Returns:
        tuple: (model, tokenizer)
    """
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"✅ Model loaded: {model_id}")
    print(f"   Memory used: {mem_gb:.2f} GB")
    
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
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
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
