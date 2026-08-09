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
for the rollout. make_grpo_rewards() resolves each example's menu + evidence in
priority order: explicit `final_json`/`evidence` kwargs (a custom rollout_func), the
RAW `completion_ids` decoded with the supplied tokenizer (the path train_grpo.py
uses -- see make_grpo_rewards for why TRL's own parsed messages LOSE Gemma final
answers mid-episode), then best-effort parsing of the TRL completion object.
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
# The other Gemma wire spans _final_answer_from_wire must strip before extract_json:
# a dangling tool call's arguments and a thought span can both contain '{', and
# extract_json decodes from the FIRST brace it sees.
_TOOL_CALL_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
_THOUGHT_RE = re.compile(r"<\|channel>[^\n]*\n.*?<channel\|>", re.DOTALL)
_THOUGHT_OPEN_RE = re.compile(r"<\|channel>.*\Z", re.DOTALL)  # clipped: opener, no close
# Path-A signature: the rollout's last content is a COMPLETE tool call (optionally
# followed by turn/eos markers and whitespace). TRL rolled back the tool result that
# would have overflowed the budget and dropped the sample, so the episode ends here
# with no final answer -- while still ending on a stop token, which is why TRL's own
# truncation test cannot see it. See _aborted_flags.
_DANGLING_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>.*?<tool_call\|>(?:\s|<turn\|>|<eos>|<pad>)*\Z", re.DOTALL
)


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


def _final_answer_from_wire(raw: str) -> str:
    """The final assistant answer text from a RAW Gemma wire completion (decoded with
    skip_special_tokens=False).

    Take the tail after the LAST <|tool_response> span (so scraped pages can never
    leak into the answer), then strip tool-call spans and thought spans -- both can
    carry braces that would fool extract_json's first-'{' scan. A completion clipped
    inside an unterminated thought span strips to nothing, which is correct: it never
    answered. Whatever remains (typically `<turn|>`/turn-header debris around the
    model's final text) is handed to schema.extract_json, which tolerates
    surrounding junk.
    """
    idx = 0
    for m in _TOOL_RESPONSE_RE.finditer(raw):
        idx = m.end()
    tail = raw[idx:]
    tail = _TOOL_CALL_RE.sub(" ", tail)
    tail = _THOUGHT_RE.sub(" ", tail)
    tail = _THOUGHT_OPEN_RE.sub(" ", tail)
    return tail.strip()


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


def make_grpo_rewards(*, weights: RewardWeights = DEFAULT_WEIGHTS, include_dietary: bool = False,
                      tokenizer=None, num_generations: int | None = None,
                      neutralize_truncated: bool = True):
    """Build the TRL-GRPO `reward_funcs` list + matching `reward_weights`.

    Returns (funcs, reward_weights). Pass both to GRPOTrainer/GRPOConfig; TRL calls
    each `func(completions, **columns) -> list[float]`, sums them weighted, and logs
    each term's mean under its __name__ (structure_reward / found_reward /
    grounding_reward). `dietary_reward` is included only when include_dietary=True;
    it carries weight 0 and returns 0.0 until a local judge is wired in.

    Each example's menu and evidence resolve in priority order:
      1. rollout kwargs `final_json` / `evidence` (a custom rollout_func attaching
         them explicitly);
      2. `tokenizer` + the `completion_ids` kwarg TRL always passes: decode the RAW
         wire text ourselves and extract answer/evidence from Gemma's markers;
      3. the TRL-parsed completion messages (string or message-list).

    TRUNCATION NEUTRALIZATION (`neutralize_truncated`, needs `tokenizer` +
    `num_generations`): a rollout the tool loop CUT OFF at max_completion_length did
    not fail -- it was stopped by the budget. TRL rolls back the tool result that
    would overflow and drops the sample from the loop (grpo_trainer.py ~1584), so the
    completion ends on a dangling tool call with no final answer and every term here
    scores exactly 0.0. That zero is not a random draw: it lands on the rollouts that
    scraped the MOST pages, so within a group it drags the mean down and hands
    POSITIVE advantage to the sibling that answered after one scrape -- training the
    policy away from the persistence SFT distilled. `mask_truncated_completions=True`
    removes such a sample's own tokens from the loss but NOT its reward from the group
    mean/std (advantages are computed over the raw group, grpo_trainer.py ~2145), so
    the sibling inflation survives masking. Here we close that half: a truncated
    sample's score is replaced by the mean of its TERMINATED group siblings. Adding a
    point equal to the mean leaves the mean unchanged and gives that sample ~zero
    advantage, so it becomes inert rather than actively mis-teaching. (It does shrink
    the group std slightly, mildly inflating the surviving advantages.) Groups that
    are entirely truncated are left alone -- their scores are already equal, so the
    advantage is zero and the group simply contributes nothing.

    Truncation is detected the same way TRL detects it (last token not EOS/pad), which
    means the tokenizer's `eos_token_id` MUST be Gemma's turn terminator `<turn|>`
    (106), NOT `<eos>` (1) -- see train_grpo.py, which sets it. With the stock
    tokenizer every Gemma rollout looks truncated and this would neutralize the whole
    batch.

    Path 2 exists because path 3 is BROKEN for Gemma mid-episode turns (measured
    2026-08-08 on transformers 5.14.1 / TRL 1.9.2, live rollouts: 16/16 final
    answers parsed to content='' -> every reward 0, zero gradient). Gemma bundles
    tool calls + responses INSIDE one assistant turn, so the final answer is a turn
    CONTINUATION after a token-concatenated prefix ending in <tool_response|>, with
    a SECOND thought span -- a shape the streaming response parser mis-handles (a
    fresh single-turn parse of the same text is fine). The raw ids are authoritative
    and immune to that parser: ALWAYS pass `tokenizer` when training Gemma.
    """
    def _resolve(completions, kwargs):
        n = len(completions)
        fj = kwargs.get("final_json") or [None] * n
        ev = kwargs.get("evidence") or [None] * n
        ids_list = kwargs.get("completion_ids") if tokenizer is not None else None
        menus, evids = [], []
        for i, c in enumerate(completions):
            raw = None
            if ids_list is not None and i < len(ids_list) and ids_list[i] is not None:
                raw = tokenizer.decode(ids_list[i], skip_special_tokens=False)
            if fj[i] is not None:
                menus.append(_as_menu(fj[i]))
            elif raw is not None:
                menus.append(_as_menu(_final_answer_from_wire(raw)))
            else:
                menus.append(_as_menu(_completion_text(c)))
            if ev[i] is not None:
                evids.append(ev[i])
            elif raw is not None:
                evids.append("\n".join(_TOOL_RESPONSE_RE.findall(raw)))
            else:
                evids.append(_evidence_from_completion(c))
        return menus, evids

    def _aborted_flags(menus, kwargs, n) -> list[bool] | None:
        """Which rollouts died on the completion budget rather than on their own merits.

        TWO shapes, and only the first is visible to TRL:
          B. cut mid-stream -- the post-tool turn overshot and was sliced
             (grpo_trainer.py ~1642), so the last token is not EOS/pad. This is
             exactly TRL's own `completions/clipped_ratio` test.
          A. dangling tool call -- the tool loop refused a result that would
             overflow, rolled it back and dropped the sample (grpo_trainer.py
             ~1584). The rollout then ENDS CLEANLY on its own tool-call tokens, so
             its last token IS a stop token and TRL counts it as *terminated*. It
             is invisible to clipped_ratio and to mask_truncated_completions alike
             -- measured on the 2026-08-08 run: 22.9% of 1680 logged rollouts, mean
             advantage -0.239, i.e. GRPO steadily punishing "called another tool".
        Path A is detected here instead: no parseable final answer AND the raw wire
        ends on a tool-call span. Returns None when we cannot tell (no tokenizer, no
        completion_ids) -- callers then leave scores untouched.
        """
        if tokenizer is None or not neutralize_truncated:
            return None
        ids_list = kwargs.get("completion_ids")
        if not ids_list or len(ids_list) < n:
            return None
        stop = {getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)}
        stop.discard(None)
        flags = []
        for i in range(n):
            ids = ids_list[i]
            cut_mid_stream = bool(stop) and bool(ids) and ids[-1] not in stop
            dangling = False
            if menus[i] is None and ids:
                raw = tokenizer.decode(ids, skip_special_tokens=False)
                dangling = bool(_DANGLING_TOOL_CALL_RE.search(raw))
            flags.append(cut_mid_stream or dangling)
        return flags

    def _neutralize(scores: list[float], menus, kwargs) -> list[float]:
        """Replace each budget-aborted rollout's score with its group's clean mean.

        The replacement equals the mean of the clean siblings, so the group mean is
        unchanged and the aborted sample's advantage is EXACTLY 0 -- it contributes
        no gradient of its own and inflates no sibling's. That makes TRL's
        `mask_truncated_completions` redundant for our purposes (and it could not
        see Path A anyway).
        """
        flags = _aborted_flags(menus, kwargs, len(scores))
        if not flags or not any(flags):
            return scores
        g = num_generations
        if not g or g < 2 or len(scores) % g:
            return scores  # can't identify group boundaries -> leave it alone
        out = list(scores)
        for start in range(0, len(scores), g):
            group = range(start, start + g)
            kept = [scores[i] for i in group if not flags[i]]
            if not kept or len(kept) == g:
                continue  # all truncated (already equal) or none truncated
            mean_kept = sum(kept) / len(kept)
            for i in group:
                if flags[i]:
                    out[i] = mean_kept
        return out

    def structure_reward(completions, **kwargs) -> list[float]:
        menus, _ = _resolve(completions, kwargs)
        return _neutralize([structure_score(m) for m in menus], menus, kwargs)

    def found_reward(completions, **kwargs) -> list[float]:
        menus, _ = _resolve(completions, kwargs)
        return _neutralize([found_score(m) for m in menus], menus, kwargs)

    def grounding_reward(completions, **kwargs) -> list[float]:
        menus, evids = _resolve(completions, kwargs)
        return _neutralize([grounding_score(m, e or "") for m, e in zip(menus, evids)],
                           menus, kwargs)

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
