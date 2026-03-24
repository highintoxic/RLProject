"""
evaluation.py — Evaluation suite: ROUGE, BERTScore, LLM-as-Judge, reporting.
"""

import json
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

from src.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL,
    JUDGE_PROMPT, EVAL_REPORT_PATH,
)
from src.model_utils import generate_judgment


# ─────────────────────────────────────────────
# ROUGE + BERTScore
# ─────────────────────────────────────────────

def evaluate_rouge_bert(model, tokenizer, dataset, n_samples: int = 100) -> dict:
    """Run ROUGE and BERTScore evaluation on n samples.
    
    Args:
        model: Language model (base or fine-tuned).
        tokenizer: Tokenizer instance.
        dataset: Test dataset with 'text' and 'judgment'/'label' fields.
        n_samples: Number of samples to evaluate.
    
    Returns:
        dict with rouge1, rouge2, rougeL, bert_score, n_samples.
    """
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
        
        prediction = generate_judgment(model, tokenizer, facts)
        
        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
        predictions.append(prediction)
        references.append(reference)
        
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_samples}")
    
    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    
    return {
        "rouge1":     np.mean(rouge1_scores),
        "rouge2":     np.mean(rouge2_scores),
        "rougeL":     np.mean(rougeL_scores),
        "bert_score": F1.mean().item(),
        "n_samples":  len(predictions),
    }


def print_comparison_table(base_scores: dict, ft_scores: dict):
    """Print a before/after comparison table of evaluation metrics."""
    print("\n" + "=" * 55)
    print(f"{'Metric':<20} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("=" * 55)
    for metric in ["rouge1", "rouge2", "rougeL", "bert_score"]:
        base_val = base_scores[metric]
        ft_val   = ft_scores[metric]
        delta    = ft_val - base_val
        arrow    = "▲" if delta > 0 else "▼"
        print(f"{metric:<20} {base_val:>10.4f} {ft_val:>12.4f} {arrow}{abs(delta):>8.4f}")
    print("=" * 55)


# ─────────────────────────────────────────────
# LLM-as-Judge
# ─────────────────────────────────────────────

def _get_judge_client():
    """Create OpenAI client for DeepSeek judge."""
    from openai import OpenAI
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def llm_judge(client, facts: str, reference: str, prediction: str) -> dict:
    """Score a single prediction using DeepSeek-Chat as judge.
    
    Returns:
        dict with scores for legal_accuracy, reasoning_chain, etc.
    """
    resp = client.chat.completions.create(
        model=DEEPSEEK_CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                facts=facts[:800],
                reference=reference[:600],
                prediction=prediction[:600],
            ),
        }],
        max_tokens=300,
        temperature=0.0,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "parse failed", "raw": resp.choices[0].message.content}


def run_llm_judge_eval(model, tokenizer, dataset, n_samples: int = 30) -> list:
    """Run LLM-as-Judge evaluation over n_samples.
    
    Returns:
        list of dicts with case_id, scores, prediction preview.
    """
    client = _get_judge_client()
    results = []
    
    for i, case in enumerate(dataset.select(range(n_samples))):
        facts     = case.get("text", "")
        reference = case.get("judgment") or case.get("label") or ""
        if not reference:
            continue
        
        prediction = generate_judgment(model, tokenizer, facts)
        scores     = llm_judge(client, facts, reference, prediction)
        
        results.append({
            "case_id": i,
            "scores": scores,
            "prediction": prediction[:300],
        })
        
        if i % 5 == 0:
            print(f"  Judged {i}/{n_samples} | "
                  f"Last overall: {scores.get('overall', 'N/A')}")
    
    return results


def print_judge_scores(judge_results: list):
    """Print summary of LLM judge scores."""
    criteria = [
        "legal_accuracy", "reasoning_chain",
        "reference_fidelity", "completeness", "overall",
    ]
    print("\n=== LLM JUDGE SCORES (out of 10) ===")
    for c in criteria:
        vals = [
            r["scores"].get(c, 0)
            for r in judge_results
            if "error" not in r["scores"]
        ]
        if vals:
            print(f"  {c:<25} {np.mean(vals):.2f} ± {np.std(vals):.2f}")


# ─────────────────────────────────────────────
# Side-by-Side Comparison
# ─────────────────────────────────────────────

def print_side_by_side(base_model, finetuned_model, tokenizer, dataset, case_idx: int):
    """Print base vs fine-tuned model outputs for a single case."""
    case      = dataset[case_idx]
    facts     = case.get("text", "")
    reference = case.get("judgment") or case.get("label") or ""
    
    base_out = generate_judgment(base_model, tokenizer, facts)
    ft_out   = generate_judgment(finetuned_model, tokenizer, facts)
    
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


# ─────────────────────────────────────────────
# Report Export
# ─────────────────────────────────────────────

def export_report(judge_results: list, save_path: str = EVAL_REPORT_PATH) -> pd.DataFrame:
    """Export evaluation results to CSV.
    
    Returns:
        DataFrame with all results.
    """
    rows = []
    for r in judge_results:
        row = {"case_id": r["case_id"], "preview": r["prediction"][:150]}
        row.update(r["scores"])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    
    print(f"\n=== FINAL EVALUATION SUMMARY ===")
    print(f"Samples evaluated : {len(df)}")
    if "overall" in df.columns:
        print(f"Avg Overall Score : {df['overall'].mean():.2f} / 10")
    if "legal_accuracy" in df.columns:
        print(f"Avg Legal Accuracy: {df['legal_accuracy'].mean():.2f} / 10")
    if "reasoning_chain" in df.columns:
        print(f"Avg Reasoning     : {df['reasoning_chain'].mean():.2f} / 10")
    
    print(f"\n📄 Report saved to: {save_path}")
    return df
