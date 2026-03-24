"""
cot_generator.py — Chain-of-Thought generation via DeepSeek-R1 teacher model.
"""

import json
from openai import OpenAI

from src.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    DEEPSEEK_R1_MODEL, COT_PROMPT,
    COT_DATA_PATH, COT_NUM_SAMPLES, MAX_FACTS_COT,
)


def create_deepseek_client(api_key: str = None) -> OpenAI:
    """Create an OpenAI-compatible client for DeepSeek API."""
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        raise ValueError(
            "DeepSeek API key not found. "
            "Set DEEPSEEK_API_KEY env var or pass it directly."
        )
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def get_cot_from_teacher(client: OpenAI, facts: str) -> dict:
    """Get chain-of-thought reasoning from DeepSeek-R1 for a single case.
    
    Returns:
        dict with keys: 'facts', 'reasoning', 'answer'
    """
    resp = client.chat.completions.create(
        model=DEEPSEEK_R1_MODEL,
        messages=[{
            "role": "user",
            "content": COT_PROMPT.format(facts=facts),
        }],
        max_tokens=2048,
    )
    return {
        "facts": facts,
        "reasoning": resp.choices[0].message.reasoning_content,
        "answer": resp.choices[0].message.content,
    }


def generate_cot_batch(
    ildc_dataset,
    client: OpenAI = None,
    n_samples: int = COT_NUM_SAMPLES,
    save_path: str = COT_DATA_PATH,
    save_every: int = 50,
) -> list:
    """Generate CoT data for N cases from ILDC, saving incrementally.
    
    Args:
        ildc_dataset: The ILDC dataset (HuggingFace).
        client: OpenAI client for DeepSeek. Created if None.
        n_samples: Number of cases to process.
        save_path: Path to save JSONL output.
        save_every: Save checkpoint every N samples.
    
    Returns:
        list of dicts with CoT data.
    """
    if client is None:
        client = create_deepseek_client()
    
    cot_data = []
    
    for i, case in enumerate(ildc_dataset.select(range(n_samples))):
        facts = case.get("facts", case.get("text", ""))[:MAX_FACTS_COT]
        try:
            result = get_cot_from_teacher(client, facts)
            cot_data.append(result)
            
            if (i + 1) % save_every == 0:
                _save_jsonl(cot_data, save_path)
                print(f"  💾 Checkpoint saved at {i + 1}/{n_samples}")
                
            if i % 50 == 0:
                print(f"  ✅ Done {i}/{n_samples}")
                
        except Exception as e:
            print(f"  ⚠️  Skipped {i}: {e}")
    
    # Final save
    _save_jsonl(cot_data, save_path)
    print(f"\n✅ Saved {len(cot_data)} CoT samples to {save_path}")
    
    return cot_data


def _save_jsonl(data: list, path: str):
    """Save list of dicts to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
