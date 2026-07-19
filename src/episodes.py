"""Shared episode planning for the corpus/eval/GRPO builders.

The seeded free + dietary-conditioned episode plan is used in THREE places -- the
teacher corpus build (scripts/corpus/build_corpus.py), the eval harness
(scripts/eval/eval.py), and the trace-free GRPO dataset (scripts/datasets/build_grpo.py).
It lives here once so those three can't drift (v1 kept copies in lockstep by hand).

An "episode" is `{"row": <restaurant dict>, "restrictions": <normalized list | []>}`.
`[]` restrictions == a free (full-menu) episode; a non-empty list == a
dietary-conditioned episode whose target is the filtered menu. The trace/eval key
for an episode is `trace_id_for(row["restaurant_id"], restrictions)`.
"""

from __future__ import annotations

import random

from corpus import trace_id_for  # noqa: E402  (src/ is on sys.path via the entry script)
from prompts import normalize_dietary_restrictions  # noqa: E402

# Abort an episode RUN (corpus build or eval) after this many consecutive failures
# -- a broken key/tool/server should not burn the whole selection. Shared by
# scripts/corpus/build_corpus.py and scripts/eval/eval.py.
MAX_CONSECUTIVE_FAILURES = 5

# Dietary restrictions sampled (in this fixed, rotated order) across the
# conditioned slice -- spread across the axes (diet type, single allergens,
# religious, combinations) so the student generalizes to unseen phrasings rather
# than memorizing one label. Each entry is fed to build_system_prompt as-is;
# comma-separated entries become multiple ANDed restrictions via
# normalize_dietary_restrictions.
DIETARY_POOL = [
    "vegetarian",
    "vegan",
    "gluten-free",
    "dairy-free",
    "no peanuts",
    "no nuts (peanuts or tree nuts)",
    "no shellfish",
    "halal",
    "kosher",
    "pescatarian",
    "keto (low-carb)",
    "vegetarian, no peanuts",
    "gluten-free, dairy-free",
]


def seeded_order(rows: list[dict], seed: int) -> list[dict]:
    """The prefix-stable restaurant order: one seeded shuffle of `rows`.

    `rows` should already be in a deterministic base order (corpus.iter_restaurants
    yields rid-sorted), so a single seeded shuffle gives a reproducible order whose
    front is reused by the conditioned slice. Returns a new list; `rows` is
    unmodified.
    """
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def plan_episodes(rows: list[dict], total: int | None, conditioned_frac: float) -> list[dict]:
    """Plan the mixed corpus: a list of {"row", "restrictions"} episodes.

    `total` is the whole episode budget (None -> one free episode per row);
    `conditioned_frac` of it is dietary-conditioned. Free episodes take the first
    n_free restaurants of the (already seeded) order. Conditioned episodes REUSE
    the front of that same order (episode i uses rows[i % len(rows)]) so they hit
    the warm cache and pair CONTRASTIVELY with the free episodes, rotating through
    DIETARY_POOL for variety. Deduped by trace_id (rid / rid__slug) so a
    wrap-around can't plan the same (restaurant, restriction) twice.
    """
    if not 0.0 <= conditioned_frac <= 1.0:
        raise ValueError(f"conditioned_frac must be in [0, 1], got {conditioned_frac}")
    if not rows:
        return []
    n = total if total is not None else len(rows)
    n_cond = round(n * conditioned_frac)
    n_free = n - n_cond

    episodes: list[dict] = []
    seen: set[str] = set()

    def add(row, restrictions):
        tid = trace_id_for(row["restaurant_id"], restrictions or None)
        if tid not in seen:
            seen.add(tid)
            episodes.append({"row": row, "restrictions": restrictions})

    for row in rows[:n_free]:
        add(row, [])
    for i in range(n_cond):
        row = rows[i % len(rows)]
        restrictions = normalize_dietary_restrictions(DIETARY_POOL[i % len(DIETARY_POOL)])
        add(row, restrictions)
    return episodes
