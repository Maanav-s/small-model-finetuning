"""The GRPO reward — score one rollout's menu answer to a scalar in [0, 1].

Phase 3 (GRPO RL) reward, built by *composing* the pure per-episode terms in
eval_metrics.py (the same scoring contract the eval harness uses), so the reward
and the eval can't drift. Nothing here does I/O; it takes a candidate answer
(raw model text OR an already-parsed final_json dict) and returns a reward, so it
is importable from a TRL reward callback, a notebook, or a test with no GPU/network.

Design — shaped, gated, bounded to [0, 1]:
  reward = w_valid·schema_valid                      # emit a valid menu AT ALL
         + schema_valid·( w_found·found_correct      # right found/abstain call
                        + w_content·content          # completeness (+ correctness)
                        + w_price·price )             # price agreement / coverage
An unparseable or schema-invalid answer scores **0** (the floor). A valid, found-
correct, fully-matching menu scores **1.0**. The intermediate shaping is the point:
within a GRPO group (G rollouts for one restaurant) a *more complete* valid menu
outranks a thin one, which outranks an **empty** answer. That gradient targets the
v1 failure mode head-on — the dominant SFT failure was an *empty* final answer
(non-termination), not a wrong menu (see notes/experiments.md), so the reward's
first job is to make "produce a valid, complete menu" strictly beat "give up".

Two modes (pick per-restaurant by whether a teacher reference is available):
  - REFERENCE-BASED (recommended): score against the teacher's final_json via
    score_episode. `content` = item-level F1 (recall AND precision), so padding
    the menu with invented items is *penalised* (precision drops) — the reward is
    not hackable by hallucinating length. `found_correct` grades abstention, and
    dietary-conditioned episodes are handled for free when the reference is the
    teacher's menu *under the same restriction* (matching a filtered menu rewards
    compliance without a separate per-item dietary classifier).
  - REFERENCE-FREE (fallback): no teacher menu, so `content` is a *saturating*
    function of raw item count and `price` is price coverage. This CANNOT detect
    hallucination (no precision signal) and rewards claiming a menu even when the
    restaurant is unfindable, so prefer reference-based; use this only on
    restaurants known to be findable. See REWARD-HACKING notes on the two modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from eval_metrics import score_episode, self_report
from schema import extract_json


@dataclass(frozen=True)
class RewardWeights:
    """Component weights; sum to 1.0 so a perfect answer scores exactly 1.0.

    valid   — floor credit for emitting ANY schema-valid menu (targets the
              empty-output failure: a valid-but-thin menu must beat "").
    found   — correct found/abstain call (score_episode.found_correct).
    content — completeness: F1 vs the reference (ref mode) or a saturating item
              count (ref-free). The dominant term — this is what "a good menu" means.
    price   — price agreement on matched items (ref mode) or price coverage (ref-free).
    """

    valid: float = 0.10
    found: float = 0.20
    content: float = 0.60
    price: float = 0.10

    def total(self) -> float:
        return self.valid + self.found + self.content + self.price


DEFAULT_WEIGHTS = RewardWeights()

# Reference-free content saturation: content = 1 - exp(-n_items / TAU). Diminishing
# returns so the model isn't rewarded without bound for padding item count (the only
# lever ref-free mode has against hallucination). At TAU=15: ~10 items -> 0.49,
# ~20 -> 0.74, ~30 -> 0.86, ~45 -> 0.95. Teacher menus average ~39 items.
ITEM_SATURATION_TAU = 15.0


def _as_menu(candidate) -> dict | None:
    """Coerce a candidate answer to a parsed menu dict, or None if unusable.

    Accepts a raw model string (parsed via schema.extract_json, the SAME
    tolerant parser the eval harness and corpus builder use) or an
    already-parsed final_json dict. A None / empty / unparseable answer -> None,
    which every mode scores as the 0 floor (this is the empty-output case).
    """
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        return candidate
    if isinstance(candidate, str):
        obj, _err = extract_json(candidate)
        return obj
    return None


def _saturating_item_content(n_items: int) -> float:
    """Bounded [0, 1) completeness proxy for reference-free mode (see TAU note)."""
    return 1.0 - math.exp(-max(0, n_items) / ITEM_SATURATION_TAU)


def menu_reward(
    candidate,
    reference=None,
    *,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    breakdown: bool = False,
):
    """Scalar reward in [0, 1] for one candidate menu answer.

    candidate : raw model text OR a parsed final_json dict (None/"" -> 0 floor).
    reference : the teacher's final_json dict to score against (REFERENCE-BASED
                mode). Pass None for REFERENCE-FREE mode.
    weights   : component weights (RewardWeights).
    breakdown : if True, return (reward, components_dict) for logging/debugging.

    A schema-invalid or unparseable answer scores 0 regardless of mode.
    """
    menu = _as_menu(candidate)
    comp = {"schema_valid": 0.0, "found": 0.0, "content": 0.0, "price": 0.0}

    if reference is not None:
        s = score_episode(menu if menu is not None else {}, reference)
        if s["schema_valid"]:
            comp["schema_valid"] = 1.0
            comp["found"] = 1.0 if s["found_correct"] else 0.0
            correct_abstention = (
                s["found_reference"] is False and s["found_candidate"] is False
            )
            if correct_abstention:
                # Nothing to extract and the model correctly abstained: full
                # content+price credit so a right abstention ranks near the top.
                comp["content"], comp["price"] = 1.0, 1.0
            else:
                comp["content"] = s["f1"] if s["f1"] is not None else 0.0
                comp["price"] = s["price_agreement"] if s["price_agreement"] is not None else 0.0
    else:
        # Reference-free: self-report only (no precision signal -- see docstring).
        sr = self_report(menu)
        if sr["schema_valid"]:
            comp["schema_valid"] = 1.0
            comp["found"] = 1.0 if sr["found"] else 0.0
            comp["content"] = _saturating_item_content(sr["n_items"])
            comp["price"] = sr["price_coverage"] if sr["price_coverage"] is not None else 0.0

    reward = (
        weights.valid * comp["schema_valid"]
        + comp["schema_valid"] * (
            weights.found * comp["found"]
            + weights.content * comp["content"]
            + weights.price * comp["price"]
        )
    )
    reward = max(0.0, min(1.0, reward))
    if breakdown:
        return reward, {**comp, "reward": reward}
    return reward


def _completion_text(completion) -> str:
    """Pull the answer text from a TRL completion (a str, or a conversational
    message list whose last message's `content` is the final assistant text)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
    return str(completion or "")


def make_grpo_reward(*, weights: RewardWeights = DEFAULT_WEIGHTS):
    """Build a TRL-GRPO reward callback over menu_reward.

    Returns `reward_func(completions, reference=None, **kwargs) -> list[float]`,
    the shape TRL's GRPOTrainer calls: `completions` is the batch of rollouts and
    any dataset columns (here `reference`, a per-example teacher final_json or
    None) arrive as parallel-list kwargs. Reference-free falls out when the column
    is absent or an entry is None.

    NOTE: this scores the completion TEXT. When the agentic rollout captures the
    final_json directly (the tool loop's parsed answer), prefer passing that in as
    the completion so scoring doesn't re-parse -- the rollout wiring (Phase 3 #3)
    will decide which. menu_reward accepts either, so this stays stable.
    """
    def reward_func(completions, reference=None, **_kwargs) -> list[float]:
        n = len(completions)
        refs = reference if reference is not None else [None] * n
        return [
            menu_reward(_completion_text(c), r, weights=weights)
            for c, r in zip(completions, refs)
        ]

    return reward_func
