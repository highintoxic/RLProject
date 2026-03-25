"""
cot_generator.py — Chain-of-Thought generation via teacher model (OpenRouter).
"""

import json
from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    REASONER_MODEL, COT_PROMPT,
    COT_DATA_PATH, COT_NUM_SAMPLES, MAX_FACTS_COT,
)


def create_openrouter_client(api_key: str = None) -> OpenAI:
    """Create an OpenAI-compatible client for OpenRouter."""
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. "
            "Set it via env var or Kaggle Secrets."
        )
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


def get_cot_from_teacher(
    client: OpenAI,
    facts: str,
    model: str = None,
) -> dict:
    """Get chain-of-thought reasoning from teacher model for a single case.

    Args:
        client: OpenRouter client.
        facts: Case facts text.
        model: Override the default reasoner model.

    Returns:
        dict with keys: 'facts', 'reasoning', 'answer'
    """
    model_id = model or REASONER_MODEL
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{
            "role": "user",
            "content": COT_PROMPT.format(facts=facts),
        }],
        max_tokens=2048,
    )

    # OpenRouter wraps R1's reasoning in `reasoning_content` if available,
    # otherwise it may be in the main `content`. Handle None safely.
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    answer = msg.content or ""

    # If R1 put everything in reasoning_content and nothing in content
    if not answer and reasoning:
        answer = reasoning

    return {
        "facts": facts,
        "reasoning": reasoning,
        "answer": answer,
    }


def generate_cot_batch(
    ildc_dataset,
    client: OpenAI = None,
    n_samples: int = COT_NUM_SAMPLES,
    save_path: str = COT_DATA_PATH,
    save_every: int = 50,
    model: str = None,
) -> list:
    """Generate CoT data for N cases, saving incrementally.

    Args:
        ildc_dataset: The dataset (HuggingFace).
        client: OpenRouter client. Created if None.
        n_samples: Number of cases to process.
        save_path: Path to save JSONL output.
        save_every: Save checkpoint every N samples.
        model: Override the default reasoner model.

    Returns:
        list of dicts with CoT data.
    """
    if client is None:
        client = create_openrouter_client()

    cot_data = []

    for i, case in enumerate(ildc_dataset.select(range(n_samples))):
        facts = case.get("facts", case.get("text", ""))[:MAX_FACTS_COT]
        try:
            result = get_cot_from_teacher(client, facts, model=model)
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
