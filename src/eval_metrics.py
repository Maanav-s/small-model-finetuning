"""WS-G shared metrics: score a candidate menu JSON against a reference menu JSON.

Pure, deterministic, zero-network functions over already-parsed menu dicts (the
`final_json` of a trace, contract 1.5 in notes/phase2_plan.md). This module is
the scoring contract the same way schema.py is the shape contract: the eval CLI
(scripts/eval_menu.py) and the Phase 3 GRPO reward both compose these per-episode
terms, so nothing here does I/O and everything stays importable (plan WS-G:
"keep the scoring functions importable, not buried in `__main__`").

Per-episode terms (score_episode) and how they map onto the future GRPO reward:
  schema_valid        jsonschema vs MENU_SCHEMA          -> gating term (0 gates the rest)
  found_correct       candidate `found` == reference's   -> abstention/hallucination term
  precision/recall/f1 item-level, fuzzy name matching    -> the completeness/content term
  price_agreement     on matched items                   -> correctness term
  section/item count deltas                              -> diagnostics only (report, not reward)

Metrics that don't apply to an episode are None -- e.g. content scores when both
sides are found=false (a correct abstention has no items to grade), or precision
when the candidate made no item claims at all. aggregate() means over the defined
values and reports the per-metric n, so N/A episodes never silently drag a mean.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import jsonschema

from schema import MENU_SCHEMA

# An item-name pair at or above this similarity counts as the same item. Chosen
# against measured values: typos ("Ceasar"/"Caesar" 0.92 seq), token reorders
# ("Pizza Margherita" 1.0 jaccard) and spell-outs ("BBQ"/"Barbecue ..." 0.82 seq)
# land above it; genuinely different dishes ("Galbi Set"/"Bulgogi Set" 0.60,
# "Pad Thai"/"Pad See Ew" 0.50) land below.
ITEM_MATCH_THRESHOLD = 0.75

# Two numeric prices within this are "the same" (float round-trips, ".99" vs ".990").
PRICE_TOLERANCE = 0.005

_PUNCT_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------
def is_schema_valid(obj) -> bool:
    """True iff `obj` is a dict that validates against MENU_SCHEMA."""
    if not isinstance(obj, dict):
        return False
    try:
        jsonschema.validate(obj, MENU_SCHEMA)
        return True
    except jsonschema.ValidationError:
        return False


# ---------------------------------------------------------------------------
# Fuzzy item-name matching
# ---------------------------------------------------------------------------
def normalize_name(name) -> str:
    """Item-name normal form: casefold, punctuation -> space, collapse whitespace."""
    return " ".join(_PUNCT_RE.sub(" ", str(name or "").casefold()).split())


def name_similarity(a, b) -> float:
    """Similarity in [0, 1] between two item names.

    max of three signals over the normalized names: character SequenceMatcher
    ratio (catches typos/spell-outs), token-set Jaccard (catches reorders), and
    token-set containment (catches suffix decorations like "Spring Rolls (2)" --
    only when the smaller name has >=2 tokens, so a single generic token like
    "Pizza" can't swallow every pizza on the menu).
    """
    na, nb = normalize_name(a), normalize_name(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb)
    containment = len(ta & tb) / min(len(ta), len(tb)) if min(len(ta), len(tb)) >= 2 else 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    return max(seq, jaccard, containment)


def prices_equal(a, b) -> bool:
    """Both unknown (None) counts as agreement; numbers agree within tolerance;
    known-vs-unknown is a disagreement (the model is told never to guess)."""
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= PRICE_TOLERANCE
    return False


def flatten_items(menu_json) -> list[dict]:
    """All item dicts across sections, in menu order (defensive: skips non-dicts
    and nameless items so a near-valid menu can still be inspected)."""
    items = []
    for section in (menu_json or {}).get("menu") or []:
        if not isinstance(section, dict):
            continue
        for item in section.get("items") or []:
            if isinstance(item, dict) and item.get("name"):
                items.append(item)
    return items


def match_items(candidate_items, reference_items, *, threshold=ITEM_MATCH_THRESHOLD):
    """Greedy one-to-one assignment of candidate items to reference items.

    All cross pairs at/above `threshold` are considered, best similarity first
    (ties: price agreement, then menu order -- fully deterministic). One-to-one,
    so duplicated candidate names can't double-claim a single reference item.
    Returns [(cand_idx, ref_idx, similarity), ...].
    """
    pairs = []
    for ci, c in enumerate(candidate_items):
        for ri, r in enumerate(reference_items):
            sim = name_similarity(c.get("name"), r.get("name"))
            if sim >= threshold:
                pairs.append((sim, prices_equal(c.get("price"), r.get("price")), ci, ri))
    pairs.sort(key=lambda p: (-p[0], not p[1], p[2], p[3]))

    matches, used_c, used_r = [], set(), set()
    for sim, _agree, ci, ri in pairs:
        if ci in used_c or ri in used_r:
            continue
        used_c.add(ci)
        used_r.add(ri)
        matches.append((ci, ri, sim))
    return matches


# ---------------------------------------------------------------------------
# Per-episode scoring -- the GRPO-reward-shaped unit
# ---------------------------------------------------------------------------
def score_episode(candidate_json, reference_json, *, threshold=ITEM_MATCH_THRESHOLD) -> dict:
    """Score one candidate menu dict against one reference menu dict.

    Both args are parsed `final_json`-shaped dicts (candidate may be None or
    malformed -- that's a schema_valid=False episode, not an exception). The
    reference is trusted as-is; callers should pre-filter unusable references.

    found combinations (found_correct grades abstention; content metrics only
    grade what can be graded):
      ref T / cand T -> precision, recall, f1, prices, deltas all scored
      ref T / cand F -> missed menu: recall=0, f1=0; precision None (no claims)
      ref F / cand T -> hallucinated menu: precision=0, f1=0; recall None
      ref F / cand F -> correct abstention: content metrics all None
    A schema-invalid candidate is treated as no answer: found_correct=False,
    and if the reference had a menu, recall=0 / f1=0.
    """
    ref_found = bool(reference_json.get("found"))
    ref_items = flatten_items(reference_json) if ref_found else []

    score = {
        "schema_valid": is_schema_valid(candidate_json),
        "found_reference": ref_found,
        "found_candidate": None,
        "found_correct": False,
        "precision": None,
        "recall": None,
        "f1": None,
        "price_agreement": None,
        "n_matched": 0,
        "n_candidate_items": 0,
        "n_reference_items": len(ref_items),
        "section_count_delta": None,
        "item_count_delta": None,
        "matches": [],
    }

    if not score["schema_valid"]:
        # No usable answer. A reference menu the candidate failed to produce is
        # a completeness miss (recall 0), same as an abstention on a findable menu.
        if ref_found:
            score["recall"], score["f1"] = 0.0, 0.0
        return score

    cand_found = bool(candidate_json.get("found"))
    score["found_candidate"] = cand_found
    score["found_correct"] = cand_found == ref_found

    if not ref_found and not cand_found:
        return score  # correct abstention -- nothing to grade
    if ref_found and not cand_found:
        score["recall"], score["f1"] = 0.0, 0.0
        return score

    cand_items = flatten_items(candidate_json)
    score["n_candidate_items"] = len(cand_items)
    if not ref_found:  # cand_found is True here
        score["precision"], score["f1"] = 0.0, 0.0
        return score

    # Both found=true: the full content comparison.
    matches = match_items(cand_items, ref_items, threshold=threshold)
    score["n_matched"] = len(matches)
    score["matches"] = [
        (cand_items[ci]["name"], ref_items[ri]["name"], round(sim, 4)) for ci, ri, sim in matches
    ]
    n_c, n_r, n_m = len(cand_items), len(ref_items), len(matches)
    if n_c == 0 and n_r == 0:
        score["precision"] = score["recall"] = score["f1"] = 1.0  # trivially perfect
    else:
        p = n_m / n_c if n_c else None  # no claims -> precision N/A, not vacuous 1.0
        r = n_m / n_r if n_r else None  # nothing to recall -> N/A
        score["precision"], score["recall"] = p, r
        pr = (p or 0.0) + (r or 0.0)
        score["f1"] = (2 * (p or 0.0) * (r or 0.0) / pr) if pr else 0.0
    if matches:
        agree = sum(
            prices_equal(cand_items[ci].get("price"), ref_items[ri].get("price"))
            for ci, ri, _ in matches
        )
        score["price_agreement"] = agree / len(matches)
    score["section_count_delta"] = (
        len(candidate_json.get("menu") or []) - len(reference_json.get("menu") or [])
    )
    score["item_count_delta"] = n_c - n_r
    return score


# ---------------------------------------------------------------------------
# Reference-free self-report (for a split with no reference set yet)
# ---------------------------------------------------------------------------
def self_report(candidate_json) -> dict:
    """Reference-free episode stats: validity, found, size, price coverage."""
    items = flatten_items(candidate_json)
    n_priced = sum(isinstance(i.get("price"), (int, float)) for i in items)
    return {
        "schema_valid": is_schema_valid(candidate_json),
        "found": bool(candidate_json.get("found")) if isinstance(candidate_json, dict) else None,
        "n_sections": len((candidate_json or {}).get("menu") or []) if isinstance(candidate_json, dict) else 0,
        "n_items": len(items),
        "n_priced_items": n_priced,
        "price_coverage": (n_priced / len(items)) if items else None,
    }


# ---------------------------------------------------------------------------
# Abstention vs the WS-F findability label
# ---------------------------------------------------------------------------
def abstention_outcome(candidate_json, findable: bool) -> str:
    """Grade the candidate's found/not-found call against the labels.jsonl
    `findable` label: one of correct_find | correct_abstain | false_abstain
    (gave up on a findable menu) | false_find (claimed an unfindable one --
    the hallucination-risk bucket). Invalid candidates count as abstentions."""
    cand_found = bool(isinstance(candidate_json, dict) and candidate_json.get("found"))
    if findable:
        return "correct_find" if cand_found else "false_abstain"
    return "false_find" if cand_found else "correct_abstain"


# ---------------------------------------------------------------------------
# Aggregation across a split
# ---------------------------------------------------------------------------
def _mean_n(scores, key):
    vals = [s[key] for s in scores if s.get(key) is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def aggregate(scores: list[dict]) -> dict:
    """Split-level aggregate of score_episode dicts: rates over all episodes,
    means over the episodes where each metric is defined (with that n)."""
    out = {"n_episodes": len(scores)}
    if not scores:
        return out
    n = len(scores)
    out["schema_valid_rate"] = sum(s["schema_valid"] for s in scores) / n
    out["found_accuracy"] = sum(s["found_correct"] for s in scores) / n
    for key in ("precision", "recall", "f1", "price_agreement"):
        out[f"{key}_mean"], out[f"{key}_n"] = _mean_n(scores, key)
    for key in ("section_count_delta", "item_count_delta"):
        vals = [s[key] for s in scores if s.get(key) is not None]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        out[f"{key}_mean_abs"] = sum(abs(v) for v in vals) / len(vals) if vals else None
        out[f"{key}_n"] = len(vals)
    return out


def aggregate_self_reports(reports: list[dict]) -> dict:
    """Split-level aggregate of self_report dicts."""
    out = {"n_episodes": len(reports)}
    if not reports:
        return out
    n = len(reports)
    out["schema_valid_rate"] = sum(r["schema_valid"] for r in reports) / n
    out["found_rate"] = sum(bool(r["found"]) for r in reports) / n
    out["mean_sections"] = sum(r["n_sections"] for r in reports) / n
    out["mean_items"] = sum(r["n_items"] for r in reports) / n
    out["price_coverage_mean"], out["price_coverage_n"] = _mean_n(reports, "price_coverage")
    return out
