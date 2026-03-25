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
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    REASONER_MODEL, CHAT_MODEL,
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

def create_openrouter_client(api_key: str = None) -> OpenAI:
    """Create OpenAI-compatible client for OpenRouter."""
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found.")
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


def generate_dual_mode_pair(
    client: OpenAI,
    facts: str,
    reasoner_model: str = None,
    chat_model: str = None,
) -> dict:
    """Generate a (direct_answer, CoT_answer) pair for one case.

    This is LegalDelta Stage 1: distill from LRM (Large Reasoning Model).

    Args:
        client: OpenRouter client.
        facts: Case facts text.
        reasoner_model: Model for CoT generation (default: config REASONER_MODEL).
        chat_model: Model for direct answers (default: config CHAT_MODEL).

    Returns:
        dict with keys: facts, direct_answer, cot_reasoning, cot_answer
    """
    r_model = reasoner_model or REASONER_MODEL
    c_model = chat_model or CHAT_MODEL

    # Mode 1: Direct answer (fast, cheap)
    direct_resp = client.chat.completions.create(
        model=c_model,
        messages=[{"role": "user", "content": DIRECT_PROMPT.format(facts=facts)}],
        max_tokens=256,
        temperature=0.1,
    )

    # Mode 2: CoT answer (deep reasoning via OpenRouter)
    cot_resp = client.chat.completions.create(
        model=r_model,
        messages=[{"role": "user", "content": COT_PROMPT.format(facts=facts)}],
        max_tokens=1024,
        extra_body={"reasoning": {"enabled": True}},
    )

    # Safely extract content — OpenRouter returns reasoning in reasoning_details
    direct_msg = direct_resp.choices[0].message
    direct_answer = direct_msg.content or ""

    cot_msg = cot_resp.choices[0].message
    # reasoning_details is a list of dicts, extract text from each
    raw_details = getattr(cot_msg, "reasoning_details", None) or []
    if isinstance(raw_details, list):
        cot_reasoning = "\n".join(
            d.get("content", "") if isinstance(d, dict) else str(d)
            for d in raw_details
        )
    else:
        cot_reasoning = str(raw_details)
    cot_answer = cot_msg.content or ""

    # Fallback: if content is empty, try reasoning_content (DeepSeek native)
    if not direct_answer:
        direct_answer = getattr(direct_msg, "reasoning_content", None) or ""
    if not cot_answer and cot_reasoning:
        cot_answer = cot_reasoning

    if not direct_answer and not cot_answer:
        print(f"  ⚠️ Both responses empty! Direct: {direct_msg}, CoT: {cot_msg}")

    return {
        "facts": facts,
        "direct_answer": direct_answer,
        "cot_reasoning": cot_reasoning,
        "cot_answer": cot_answer,
    }


def generate_dual_mode_batch(
    ildc_dataset,
    client: OpenAI = None,
    n_samples: int = 300,
    save_path: str = "dual_mode_data.jsonl",
    save_every: int = 50,
    reasoner_model: str = None,
    chat_model: str = None,
) -> list:
    """Generate dual-mode pairs for N cases, saving incrementally.

    Args:
        ildc_dataset: ILDC HuggingFace dataset.
        client: OpenRouter client.
        n_samples: Number of cases.
        save_path: JSONL output path.
        save_every: Checkpoint interval.
        reasoner_model: Override reasoner model.
        chat_model: Override chat model.

    Returns:
        list of dual-mode dicts.
    """
    if client is None:
        client = create_openrouter_client()

    data = []

    for i, case in enumerate(ildc_dataset.select(range(n_samples))):
        facts = case.get("facts", case.get("text", ""))[:1000]
        try:
            pair = generate_dual_mode_pair(
                client, facts,
                reasoner_model=reasoner_model,
                chat_model=chat_model,
            )
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
    """
    model.eval()
    device = next(model.parameters()).device

    def get_answer_log_probs(prompt: str, answer: str) -> torch.Tensor:
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

        if inputs["input_ids"].shape[1] <= prompt_len:
            return torch.tensor([0.0], device=device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, prompt_len - 1:-1]
            answer_ids = inputs["input_ids"][0, prompt_len:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                1, answer_ids.unsqueeze(1)
            ).squeeze(1)

        return token_log_probs

    try:
        direct_prompt = f"Case Facts: {facts[:400]}\n\nJudgment: "
        log_p_direct = get_answer_log_probs(direct_prompt, direct_answer[:200])

        cot_prompt = (
            f"Case Facts: {facts[:400]}\n\n"
            f"<think>{cot_response[:300]}</think>\n\nJudgment: "
        )
        log_p_cot = get_answer_log_probs(cot_prompt, direct_answer[:200])

        min_len = min(len(log_p_direct), len(log_p_cot))
        if min_len == 0:
            return 0.0

        log_p_direct = log_p_direct[:min_len]
        log_p_cot = log_p_cot[:min_len]

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
    """
    r_ig = ig_score

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

    ipc_citations = re.findall(
        r'(?:Section|Sec\.?)\s*\d+[A-Z]?\s*(?:IPC|BNS|CrPC|BNSS|IEA)',
        cot_full_response,
    )
    sc_refs = re.findall(
        r'(?:AIR|SCC|SCR)\s*\d{4}',
        cot_full_response,
    )
    r_domain = min(len(ipc_citations) * 0.3 + len(sc_refs) * 0.2, 1.0)

    total = w_ig * r_ig + w_structure * r_structure + w_domain * r_domain

    return {
        "total": total,
        "ig": r_ig,
        "structure": r_structure,
        "domain": r_domain,
    }
