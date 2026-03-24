"""
data_utils.py — Dataset loading, formatting, and preparation.
"""

from datasets import load_dataset, Dataset

from src.config import (
    ILDC_DATASET, NYAYA_DATASET,
    SYSTEM_PROMPT, TEST_SPLIT_RATIO,
    MAX_FACTS_LENGTH, COT_DATA_PATH,
)


def load_legal_datasets():
    """Load ILDC and NyayaAnumana datasets.
    
    Returns:
        tuple: (ildc_dataset, nyaya_dataset)
    """
    ildc = load_dataset(ILDC_DATASET, split="train")
    nyaya = load_dataset(NYAYA_DATASET, split="train")
    
    print(f"✅ ILDC samples:         {len(ildc)}")
    print(f"✅ NyayaAnumana samples: {len(nyaya)}")
    
    return ildc, nyaya


def format_sample(example: dict) -> dict:
    """Format a single dataset example into bilingual instruction-tuning format.
    
    Uses Qwen's ChatML template: <|im_start|>role ... <|im_end|>
    """
    facts    = example.get("facts") or example.get("text", "")[:MAX_FACTS_LENGTH]
    judgment = example.get("judgment") or example.get("label", "")
    
    return {
        "text": f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
निम्नलिखित मामले पर विचार करें / Consider the following case:

{facts}

What is the legal reasoning and applicable sections?<|im_end|>
<|im_start|>assistant
{judgment}<|im_end|>"""
    }


def format_cot_sample(row: dict) -> dict:
    """Format a CoT example (with reasoning trace) for training.
    
    Args:
        row: Dict with keys 'facts', 'reasoning', 'answer'.
    """
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


def prepare_sft_dataset(ildc):
    """Format ILDC into instruction-tuning dataset and split.
    
    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    dataset = ildc.map(format_sample, remove_columns=ildc.column_names)
    dataset = dataset.train_test_split(test_size=TEST_SPLIT_RATIO)
    
    print(f"✅ Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")
    return dataset


def load_cot_dataset(path: str = COT_DATA_PATH):
    """Load saved CoT JSONL file and format for training.
    
    Returns:
        Dataset ready for SFT training.
    """
    import json
    
    with open(path, "r", encoding="utf-8") as f:
        cot_data = [json.loads(line) for line in f]
    
    cot_dataset = Dataset.from_list([format_cot_sample(r) for r in cot_data])
    print(f"✅ CoT dataset loaded: {len(cot_dataset)} samples")
    return cot_dataset
