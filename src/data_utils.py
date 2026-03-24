"""
data_utils.py — Dataset loading, formatting, and preparation.
"""

from datasets import load_dataset, Dataset

from src.config import (
    ILDC_DATASET, NYAYA_DATASET,
    SYSTEM_PROMPT, TEST_SPLIT_RATIO,
    MAX_FACTS_LENGTH, COT_DATA_PATH,
)


def _get_text(example: dict) -> str:
    """Extract the main text field from a dataset example.
    
    Handles different column names across datasets:
    - 'facts', 'text', 'case_text', 'document', 'Text'
    """
    for key in ("facts", "text", "case_text", "document", "Text"):
        if key in example and example[key]:
            return str(example[key])
    # Fall back to first string column
    for key, val in example.items():
        if isinstance(val, str) and len(val) > 50:
            return val
    return ""


def _get_label(example: dict) -> str:
    """Extract the label/judgment field."""
    for key in ("judgment", "label", "decision", "Label", "Decision"):
        if key in example and example[key] is not None:
            return str(example[key])
    return ""


def load_legal_datasets():
    """Load primary and secondary Indian legal datasets.
    
    Returns:
        tuple: (primary_dataset, secondary_dataset_or_None)
    """
    print(f"Loading primary dataset: {ILDC_DATASET}")
    primary = load_dataset(ILDC_DATASET, split="train")
    print(f"✅ Primary: {len(primary)} samples | Columns: {primary.column_names}")

    secondary = None
    try:
        print(f"Loading secondary dataset: {NYAYA_DATASET}")
        secondary = load_dataset(NYAYA_DATASET, split="train")
        print(f"✅ Secondary: {len(secondary)} samples | Columns: {secondary.column_names}")
    except Exception as e:
        print(f"⚠️  Secondary dataset unavailable: {e}")

    return primary, secondary


def format_sample(example: dict) -> dict:
    """Format a single dataset example into bilingual instruction-tuning format.
    
    Uses Qwen's ChatML template: <|im_start|>role ... <|im_end|>
    """
    facts    = _get_text(example)[:MAX_FACTS_LENGTH]
    judgment = _get_label(example)

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
    """Format a CoT example (with reasoning trace) for training."""
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


def prepare_sft_dataset(dataset):
    """Format dataset into instruction-tuning format and split.
    
    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    formatted = dataset.map(format_sample, remove_columns=dataset.column_names)
    split_data = formatted.train_test_split(test_size=TEST_SPLIT_RATIO)

    print(f"✅ Train: {len(split_data['train'])} | Test: {len(split_data['test'])}")
    return split_data


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
