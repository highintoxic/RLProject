"""
legaldelta.py — LegalDelta framework: Information Gain reward + dual-mode dataset.

Based on: "LegalDelta: Enhancing Legal Reasoning via RL with CoT-guided
Information Gain" (ICASSP 2026, arXiv 2508.12281).

Core idea: Measure how much the CoT *actually helps* the answer.
If reasoning changes the answer distribution → reward.
If reasoning is filler → penalise.
"""

import re
import json
import torch
import torch.nn.functional as F
from openai import OpenAI

from src.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    DEEPSEEK_R1_MODEL, DEEPSEEK_CHAT_MODEL,
)


# ═══════════════════════════════════════════════
# Dual-Mode Prompts
# ═══════════════════════════════════════════════

DIRECT_PROMPT = """You are an Indian legal expert.
Given the case facts below, directly state the judgment outcome and applicable sections.
Be concise — no step-by-step reasoning.

Case Facts: {facts}

Judgment:"""

COT_PROMPT = """You are a Senior Indian Supreme Court judge.
Reason step by step through this case:
1. Identify applicable IPC/BNS/CrPC sections
2. Recall relevant Supreme Court precedents
3. Apply legal syllogism (Major premise → Minor premise → Conclusion)
4. State final judgment

IMPORTANT: Where relevant, respond in both Hindi and English.

Case Facts: {facts}"""


# ═══════════════════════════════════════════════
# Dual-Mode Data Generation
# ═══════════════════════════════════════════════

def create_deepseek_client(api_key: str = None) -> OpenAI:
    """Create OpenAI-compatible client for DeepSeek API."""
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        raise ValueError("DEEPSEEK_API_KEY not found.")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def generate_dual_mode_pair(client: OpenAI, facts: str) -> dict:
    """Generate a (direct_answer, CoT_answer) pair for one case.

    This is LegalDelta Stage 1: distill from LRM (Large Reasoning Model).

    Returns:
        dict with keys: facts, direct_answer, cot_reasoning, cot_answer
    """
    # Mode 1: Direct answer (fast, cheap — deepseek-chat)
    direct_resp = client.chat.completions.create(
        model=DEEPSEEK_CHAT_MODEL,
        messages=[{"role": "user", "content": DIRECT_PROMPT.format(facts=facts)}],
        max_tokens=256,
        temperature=0.1,
    )

    # Mode 2: CoT answer (deep reasoning — deepseek-reasoner / R1)
    cot_resp = client.chat.completions.create(
        model=DEEPSEEK_R1_MODEL,
        messages=[{"role": "user", "content": COT_PROMPT.format(facts=facts)}],
        max_tokens=1024,
    )

    return {
        "facts": facts,
        "direct_answer": direct_resp.choices[0].message.content,
        "cot_reasoning": cot_resp.choices[0].message.reasoning_content,
        "cot_answer": cot_resp.choices[0].message.content,
    }


def generate_dual_mode_batch(
    ildc_dataset,
    client: OpenAI = None,
    n_samples: int = 300,
    save_path: str = "dual_mode_data.jsonl",
    save_every: int = 50,
) -> list:
    """Generate dual-mode pairs for N cases, saving incrementally.

    Args:
        ildc_dataset: ILDC HuggingFace dataset.
        client: DeepSeek OpenAI client.
        n_samples: Number of cases.
        save_path: JSONL output path.
        save_every: Checkpoint interval.

    Returns:
        list of dual-mode dicts.
    """
    if client is None:
        client = create_deepseek_client()

    data = []

    for i, case in enumerate(ildc_dataset.select(range(n_samples))):
        facts = case.get("facts", case.get("text", ""))[:1000]
        try:
            pair = generate_dual_mode_pair(client, facts)
            pair["reference"] = str(case.get("label", ""))
            data.append(pair)

            if (i + 1) % save_every == 0:
                _save_jsonl(data, save_path)
                print(f"  💾 Checkpoint: {i+1}/{n_samples} saved")

            if i % 50 == 0:
                print(f"  ✅ Done {i}/{n_samples}")

        except Exception as e:
            print(f"  ⚠️  Skipped {i}: {e}")

    _save_jsonl(data, save_path)
    print(f"\n✅ Saved {len(data)} dual-mode pairs to {save_path}")
    return data


def _save_jsonl(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dual_mode_data(path: str = "dual_mode_data.jsonl") -> list:
    """Load saved dual-mode JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(data)} dual-mode pairs from {path}")
    return data


# ═══════════════════════════════════════════════
# Information Gain Reward (LegalDelta's core)
# ═══════════════════════════════════════════════

def compute_information_gain(
    model,
    tokenizer,
    facts: str,
    direct_answer: str,
    cot_response: str,
    max_length: int = 512,
) -> float:
    """
    Core LegalDelta reward:
    IG = KL( P(answer | facts + CoT) || P(answer | facts only) )

    High IG → CoT meaningfully changed the answer distribution → reward.
    Low IG  → CoT added nothing → penalise.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        facts: Case facts.
        direct_answer: The answer text to measure probability of.
        cot_response: The CoT reasoning text.
        max_length: Max token length for inputs.

    Returns:
        float: Information gain score (clamped to [0, 2]).
    """
    model.eval()
    device = next(model.parameters()).device

    def get_answer_log_probs(prompt: str, answer: str) -> torch.Tensor:
        """Get per-token log probs of `answer` conditioned on `prompt`."""
        full_text = prompt + answer
        inputs = tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=max_length,
        ).to(device)

        prompt_ids = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=max_length,
        )
        prompt_len = prompt_ids["input_ids"].shape[1]

        # Need at least 1 answer token
        if inputs["input_ids"].shape[1] <= prompt_len:
            return torch.tensor([0.0], device=device)

        with torch.no_grad():
            outputs = model(**inputs)
            # logits for positions prompt_len-1 to end-1 predict answer tokens
            logits = outputs.logits[0, prompt_len - 1:-1]
            answer_ids = inputs["input_ids"][0, prompt_len:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                1, answer_ids.unsqueeze(1)
            ).squeeze(1)

        return token_log_probs

    try:
        # P(answer | facts only)
        direct_prompt = f"Case Facts: {facts[:400]}\n\nJudgment: "
        log_p_direct = get_answer_log_probs(direct_prompt, direct_answer[:200])

        # P(answer | facts + CoT)
        cot_prompt = (
            f"Case Facts: {facts[:400]}\n\n"
            f"<think>{cot_response[:300]}</think>\n\nJudgment: "
        )
        log_p_cot = get_answer_log_probs(cot_prompt, direct_answer[:200])

        # Align lengths (take minimum)
        min_len = min(len(log_p_direct), len(log_p_cot))
        if min_len == 0:
            return 0.0

        log_p_direct = log_p_direct[:min_len]
        log_p_cot = log_p_cot[:min_len]

        # KL(P_cot || P_direct)
        kl_div = (
            log_p_cot.exp() * (log_p_cot - log_p_direct)
        ).mean().item()

        return max(0.0, min(kl_div, 2.0))

    except Exception as e:
        print(f"  ⚠️ IG computation error: {e}")
        return 0.0


# ═══════════════════════════════════════════════
# Multidimensional Reward
# ═══════════════════════════════════════════════

def compute_legaldelta_reward(
    facts: str,
    direct_answer: str,
    cot_full_response: str,
    reference: str,
    ig_score: float,
    w_ig: float = 0.5,
    w_structure: float = 0.3,
    w_domain: float = 0.2,
) -> dict:
    """
    LegalDelta's multidimensional reward:
    1. Information Gain (core novelty)
    2. Structural coherence (legal syllogism structure)
    3. Indian legal domain specificity (IPC/BNS citations)

    Returns:
        dict with total, ig, structure, domain scores.
    """
    # Component 1: Information Gain (already computed)
    r_ig = ig_score

    # Component 2: Structural coherence
    has_think = 1.0 if ("<think>" in cot_full_response and
                        "</think>" in cot_full_response) else 0.0
    legal_keywords = [
        "therefore", "hence", "held", "observed",
        "ratio decidendi", "prima facie", "mens rea",
        "अतः", "निर्णय", "धारा",
    ]
    keyword_score = sum(
        0.1 for kw in legal_keywords
        if kw.lower() in cot_full_response.lower()
    )
    r_structure = min(has_think + keyword_score, 1.0)

    # Component 3: Indian legal domain specificity
    ipc_citations = re.findall(
        r'(?:Section|Sec\.?)\s*\d+[A-Z]?\s*(?:IPC|BNS|CrPC|BNSS|IEA)',
        cot_full_response,
    )
    sc_refs = re.findall(
        r'(?:AIR|SCC|SCR)\s*\d{4}',
        cot_full_response,
    )
    r_domain = min(len(ipc_citations) * 0.3 + len(sc_refs) * 0.2, 1.0)

    # Weighted total
    total = w_ig * r_ig + w_structure * r_structure + w_domain * r_domain

    return {
        "total": total,
        "ig": r_ig,
        "structure": r_structure,
        "domain": r_domain,
    }


def legaldelta_reward_for_grpo(
    model, tokenizer, dual_mode_data: list
):
    """Create a GRPO-compatible reward function using LegalDelta rewards.

    Returns a callable that takes (completions, **kwargs) -> list[float].
    """
    # Pre-index by facts for lookup
    data_by_facts = {}
    for row in dual_mode_data:
        key = row["facts"][:100]
        data_by_facts[key] = row

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for completion in completions:
            # Try to compute IG if we have matching data
            ig = 0.0
            direct = ""

            # Match completion to original data
            for key, row in data_by_facts.items():
                if key[:50] in (prompts[0] if prompts else ""):
                    direct = row.get("direct_answer", "")
                    try:
                        ig = compute_information_gain(
                            model, tokenizer,
                            row["facts"], direct, completion,
                        )
                    except Exception:
                        ig = 0.0
                    break

            result = compute_legaldelta_reward(
                facts="",
                direct_answer=direct,
                cot_full_response=completion,
                reference="",
                ig_score=ig,
            )
            rewards.append(result["total"])

        return rewards

    return reward_fn
