"""The GRPO reward — PURE RL, no teacher reference (Phase 3).

Scores a rollout from signals intrinsic to the episode only: the schema, the
model's own found/abstain call, and whether the menu is GROUNDED in the tool
outputs the agent actually scraped. There is deliberately NO comparison to a
teacher menu -- the teacher generated the SFT warm-start, but GRPO optimizes
against the environment, not against Claude, so the policy can in principle
exceed the teacher instead of being capped at imitating it.

Split into SEPARATE reward functions (TRL's GRPOTrainer takes a `reward_funcs`
list, sums them with `reward_weights`, and logs each term's mean independently --
so `structure`, `found`, and `grounding` each get their own training curve to
watch). The three live terms:

  structure  schema_valid (jsonschema vs MENU_SCHEMA)            -> {0, 1}
  found      committed to a menu: valid AND found AND items>0    -> {0, 1}
  grounding  are the items REAL -- each item name present in the -> [-penalty, ~1]
             tool-response text the agent scraped this episode;
             rewards grounded volume, PENALISES ungrounded items

  dietary    (STUB, weight 0) compliance of a conditioned menu with its dietary
             restriction -- a semantic judgment (is this dish vegetarian?) that
             needs world knowledge, so it is deferred to a small LOCALLY-inferenced
             judge model. See dietary_reward + notes below.

Why grounding replaces the old F1-vs-teacher term: without a reference we lose the
precision signal that stopped the model padding the menu with invented items. The
replacement is FAITHFULNESS TO THE RETRIEVED EVIDENCE -- an item counts only if its
name actually appears in what the agent scraped. Hallucinated items are not in the
evidence, so they are penalised; and since an unfindable restaurant yields no menu
in the scraped pages, confidently inventing one also scores negative. This makes
"honestly abstain" (found=false -> grounding 0) strictly better than "confidently
hallucinate" (grounding < 0), while a grounded, complete menu scores highest.

REWARD ORDERING (the gradient GRPO follows), with default weights + penalty:
  schema-invalid            ~0     (hard floor: structure/found/grounding all gate on valid)
  confident hallucination   < 0    (grounding penalty dominates)
  honest abstention          +     (structure only; grounding neutral)
  grounded, complete menu    ++    (all three positive)

EVIDENCE PLUMBING (important): grounding needs the concatenated tool-response text
for the rollout. The clean path is the GRPO rollout (a TRL `rollout_func` running
our agent loop) attaching per-trajectory `final_json` and `evidence` kwargs -- we
run the tool loop ourselves anyway, so both are in hand. make_grpo_rewards() reads
those kwargs first and only falls back to best-effort parsing of the completion.
This is the one part that must be validated against TRL's actual reward-callback
inputs on GPU (see the GRPO wiring / notes).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from eval_metrics import flatten_items, is_schema_valid, normalize_name
from schema import extract_json


@dataclass(frozen=True)
class RewardWeights:
    """Per-term weights, passed to GRPOConfig.reward_weights in list order
    (structure, found, grounding, dietary). Not required to sum to 1 -- GRPO
    normalises advantages within each generation group, so only the RELATIVE
    magnitudes matter. grounding dominates because "are the items real" is the
    hardest-won and most game-prone signal.
    """

    structure: float = 0.2
    found: float = 0.2
    grounding: float = 0.6
    dietary: float = 0.0  # stub until the local dietary judge exists

    def as_list(self) -> list[float]:
        return [self.structure, self.found, self.grounding, self.dietary]


DEFAULT_WEIGHTS = RewardWeights()

# grounded-item completeness saturates: 1 - exp(-n_grounded / TAU). Diminishing
# returns so volume alone can't be farmed. TAU=20 keeps the gradient meaningful
# across the realistic menu size (teacher menus average ~39 items): completeness is
# ≈0.39 at 10 grounded items, 0.63 at 20, 0.86 at 40 -- so extracting MORE of a
# large menu keeps paying, instead of flattening by ~30 items (TAU=15 did).
GROUND_SATURATION_TAU = 20.0

# How hard hallucinated items are punished. The grounding term is
#   completeness·grounded_frac - HALLUCINATION_PENALTY·max(0, ungrounded_frac - SLACK)
# so a 0%-grounded menu scores about -HALLUCINATION_PENALTY·(1 - SLACK). This is the
# key knob for where a partly-hallucinated menu crosses BELOW honest abstention.
HALLUCINATION_PENALTY = 1.0

# Ungrounded-fraction tolerance before the penalty bites. is_grounded is a strict
# substring test (high precision, but it FALSE-NEGATIVES on legitimately reworded
# items -- "Margherita" on the page vs "Margherita Pizza" in the answer). Without
# slack, that matcher noise would penalise real menus. 0.15 lets ~1-in-7 items go
# unmatched as assumed-real before any penalty applies, so the penalty targets
# genuine hallucination, not matcher imperfection. Revisit against real rollout data.
GROUNDING_SLACK = 0.15

# Gemma renders a tool result as <|tool_response>...<tool_response|>. When evidence
# isn't supplied as a kwarg we recover it from those spans in the completion text
# (NOT the whole completion -- the final answer's own item names must not count as
# their own grounding).
_TOOL_RESPONSE_RE = re.compile(r"<\|tool_response>(.*?)<tool_response\|>", re.DOTALL)


# ---------------------------------------------------------------------------
# Candidate + evidence coercion
# ---------------------------------------------------------------------------
def _as_menu(candidate) -> dict | None:
    """A candidate answer -> a parsed menu dict, or None (unusable -> the floor).
    Accepts a dict (parsed final_json) or a raw string (schema.extract_json)."""
    if isinstance(candidate, dict):
        return candidate
    if isinstance(candidate, str):
        obj, _err = extract_json(candidate)
        return obj
    return None


def is_grounded(item_name, normalized_evidence: str) -> bool:
    """True if the item name appears (normalized substring) in the scraped evidence.

    Both sides go through eval_metrics.normalize_name (casefold, punctuation ->
    space, whitespace collapsed), then a contiguous substring test. Contiguous (not
    token-subset) on purpose: it is HIGH-PRECISION -- it won't falsely ground a
    hallucinated dish just because its common food-words ("grilled", "salad") each
    occur somewhere on the page. The cost is under-grounding a legitimately reworded
    item; for an anti-hallucination signal, precision is the right side to favour (a
    real item wrongly flagged loses some reward; a fake item wrongly grounded
    defeats the whole term).
    """
    nn = normalize_name(item_name)
    return bool(nn) and nn in normalized_evidence


def count_grounded(items: list[dict], evidence: str) -> tuple[int, int]:
    """(n_grounded, n_total) item names found in `evidence`."""
    norm = normalize_name(evidence or "")
    n_grounded = sum(is_grounded(it.get("name"), norm) for it in items) if norm else 0
    return n_grounded, len(items)


# ---------------------------------------------------------------------------
# The live component scores (pure; each in its own range)
# ---------------------------------------------------------------------------
def structure_score(menu) -> float:
    """1.0 if the answer is a schema-valid menu, else 0.0."""
    return 1.0 if is_schema_valid(menu) else 0.0


def found_score(menu) -> float:
    """1.0 if the model committed to a menu (valid AND found AND at least one item).

    Rewards *finding* a menu over abstaining. Deliberately independent of grounding
    so its curve reads cleanly as "is the policy attempting menus?" -- the grounding
    term separately decides whether those attempts are real, and its penalty makes a
    hallucinated attempt net-negative overall.
    """
    if not is_schema_valid(menu):
        return 0.0
    return 1.0 if (menu.get("found") and flatten_items(menu)) else 0.0


def grounding_score(menu, evidence: str) -> float:
    """Faithfulness of the menu to the scraped evidence, in [-HALLUCINATION_PENALTY, ~1].

    0.0 when there is nothing to grade (invalid, abstained, or no items -- no claim
    to be faithful about). Otherwise rewards the VOLUME of grounded items
    (saturating) scaled by the grounded FRACTION, minus a penalty on the ungrounded
    fraction -- so a fully grounded menu is positive and a fully hallucinated one is
    -HALLUCINATION_PENALTY.
    """
    if not is_schema_valid(menu):
        return 0.0
    items = flatten_items(menu)
    if not items:
        return 0.0
    n_grounded, n_total = count_grounded(items, evidence)
    grounded_frac = n_grounded / n_total
    completeness = 1.0 - math.exp(-n_grounded / GROUND_SATURATION_TAU)
    # reward real-item volume (scaled by cleanliness), penalise the hallucinated
    # fraction beyond the SLACK tolerance for matcher false-negatives.
    penalty = HALLUCINATION_PENALTY * max(0.0, (1.0 - grounded_frac) - GROUNDING_SLACK)
    return completeness * grounded_frac - penalty


def dietary_score(menu, dietary_restrictions, judge=None) -> float:
    """STUB (returns 0.0 without a judge). Compliance of a conditioned menu with its
    restriction.

    Grading this needs a per-item semantic judgment ("is Chicken Alfredo
    vegetarian?", "is this gluten-free?") -- world knowledge a substring/keyword rule
    can't supply reliably. The plan is a small LOCALLY-inferenced judge model (a
    compact instruct model or a fine-tuned classifier) exposed here as
    `judge(item, restrictions) -> compliant?`, scoring the net compliant fraction for
    conditioned episodes only. Until that exists this returns 0.0 and carries weight
    0, so it never perturbs training.
    """
    if judge is None or not dietary_restrictions:
        return 0.0
    items = flatten_items(menu)
    if not items:
        return 0.0
    ok = sum(bool(judge(it, dietary_restrictions)) for it in items)
    violations = len(items) - ok
    return (ok - violations) / len(items)


# ---------------------------------------------------------------------------
# Composite (for eval/logging/tests -- TRL uses the split functions below)
# ---------------------------------------------------------------------------
def menu_reward(candidate, evidence: str = "", *, weights: RewardWeights = DEFAULT_WEIGHTS,
                dietary_restrictions=None, dietary_judge=None, breakdown: bool = False):
    """Weighted sum of the component scores for one rollout (teacher-free).

    candidate : raw model text or a parsed final_json dict.
    evidence  : concatenated tool-response text the agent scraped this episode
                (grounding is scored against this -- empty => everything ungrounded).
    Returns the scalar reward (can be NEGATIVE for a hallucinated menu -- that is
    the point), or (reward, components) if breakdown=True.
    """
    menu = _as_menu(candidate)
    comp = {
        "structure": structure_score(menu),
        "found": found_score(menu),
        "grounding": grounding_score(menu, evidence),
        "dietary": dietary_score(menu, dietary_restrictions, dietary_judge),
    }
    reward = (
        weights.structure * comp["structure"]
        + weights.found * comp["found"]
        + weights.grounding * comp["grounding"]
        + weights.dietary * comp["dietary"]
    )
    if breakdown:
        return reward, {**comp, "reward": reward}
    return reward


# ---------------------------------------------------------------------------
# Completion / evidence extraction for the TRL callbacks
# ---------------------------------------------------------------------------
def _completion_text(completion) -> str:
    """The final answer text from a TRL completion (a str, or a message list whose
    last message content is the final assistant text)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
    return str(completion or "")


def _evidence_from_completion(completion) -> str:
    """Best-effort tool-response text from a completion when `evidence` isn't passed.

    Preferred path is the rollout attaching `evidence` explicitly (see module
    docstring). Fallback: pull tool-response content from a conversational message
    list (role 'tool'/'tool_response', or bundled `tool_responses`), or from the
    Gemma <|tool_response>...<tool_response|> spans in a rendered string. Crucially
    this excludes the final answer, so an item can't ground itself.
    """
    if isinstance(completion, str):
        return "\n".join(_TOOL_RESPONSE_RE.findall(completion))
    if isinstance(completion, list):
        parts: list[str] = []
        for m in completion:
            if not isinstance(m, dict):
                continue
            if m.get("role") in ("tool", "tool_response"):
                parts.append(str(m.get("content") or ""))
            for tr in m.get("tool_responses") or []:
                if isinstance(tr, dict):
                    parts.append(str(tr.get("response") or ""))
        return "\n".join(parts)
    return ""


def make_grpo_rewards(*, weights: RewardWeights = DEFAULT_WEIGHTS, include_dietary: bool = False):
    """Build the TRL-GRPO `reward_funcs` list + matching `reward_weights`.

    Returns (funcs, reward_weights). Pass both to GRPOTrainer/GRPOConfig; TRL calls
    each `func(completions, **columns) -> list[float]`, sums them weighted, and logs
    each term's mean under its __name__ (structure_reward / found_reward /
    grounding_reward). Each example's menu and evidence come from the rollout kwargs
    `final_json` and `evidence` when present (the clean path), else are parsed from
    the completion. `dietary_reward` is included only when include_dietary=True; it
    carries weight 0 and returns 0.0 until a local judge is wired in.
    """
    def _resolve(completions, kwargs):
        n = len(completions)
        fj = kwargs.get("final_json") or [None] * n
        ev = kwargs.get("evidence") or [None] * n
        menus, evids = [], []
        for i, c in enumerate(completions):
            menus.append(_as_menu(fj[i]) if fj[i] is not None else _as_menu(_completion_text(c)))
            evids.append(ev[i] if ev[i] is not None else _evidence_from_completion(c))
        return menus, evids

    def structure_reward(completions, **kwargs) -> list[float]:
        menus, _ = _resolve(completions, kwargs)
        return [structure_score(m) for m in menus]

    def found_reward(completions, **kwargs) -> list[float]:
        menus, _ = _resolve(completions, kwargs)
        return [found_score(m) for m in menus]

    def grounding_reward(completions, **kwargs) -> list[float]:
        menus, evids = _resolve(completions, kwargs)
        return [grounding_score(m, e or "") for m, e in zip(menus, evids)]

    funcs = [structure_reward, found_reward, grounding_reward]
    reward_weights = [weights.structure, weights.found, weights.grounding]

    if include_dietary:
        def dietary_reward(completions, dietary_restrictions=None, **kwargs) -> list[float]:
            menus, _ = _resolve(completions, kwargs)
            dr = dietary_restrictions or [None] * len(menus)
            # judge stays None: returns 0.0 until a local judge is wired in (see
            # dietary_score). Present so the curve/slot exists ahead of time.
            return [dietary_score(m, dr[i]) for i, m in enumerate(menus)]
        funcs.append(dietary_reward)
        reward_weights.append(weights.dietary)

    return funcs, reward_weights
