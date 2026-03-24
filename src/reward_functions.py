"""
reward_functions.py — Rule-based reward functions for GRPO training.

These are used instead of a reward model — much lighter and interpretable.
"""

import re


def reward_has_legal_citation(completions: list, **kwargs) -> list:
    """Reward for citing actual IPC/BNS/CrPC sections.
    
    Looks for patterns like 'Section 302 IPC', 'BNS Section 101', etc.
    Score: 0.3 per citation, capped at 1.0.
    """
    rewards = []
    for c in completions:
        citations = re.findall(
            r'(Section\s+\d+[A-Z]?\s+(?:IPC|BNS|CrPC|BNSS|IEA)|'
            r'(?:IPC|BNS)\s+[Ss]ection\s+\d+)',
            c,
        )
        rewards.append(min(len(citations) * 0.3, 1.0))
    return rewards


def reward_has_reasoning(completions: list, **kwargs) -> list:
    """Reward for structured step-by-step reasoning.
    
    - 0.4 for using <think>...</think> tags
    - Up to 0.6 for legal reasoning keywords (therefore, hence, held, etc.)
    """
    rewards = []
    for c in completions:
        score = 0.0
        if "<think>" in c and "</think>" in c:
            score += 0.4
        keywords = ["therefore", "अतः", "hence", "held", "observed", "ratio"]
        score += min(sum(0.1 for k in keywords if k.lower() in c.lower()), 0.6)
        rewards.append(score)
    return rewards


def reward_bilingual(completions: list, **kwargs) -> list:
    """Reward for Hindi + English mixed output.
    
    Score: 0.5 if both Hindi (Devanagari) and English characters present,
    else 0.0.
    """
    rewards = []
    for c in completions:
        has_hindi   = bool(re.search(r'[\u0900-\u097F]', c))
        has_english = bool(re.search(r'[a-zA-Z]', c))
        rewards.append(0.5 if (has_hindi and has_english) else 0.0)
    return rewards


# Convenience list of all reward functions for GRPO trainer
ALL_REWARD_FUNCS = [
    reward_has_legal_citation,
    reward_has_reasoning,
    reward_bilingual,
]
