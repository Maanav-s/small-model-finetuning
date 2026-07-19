"""v2 GRPO export: corpus.sqlite grpo-split restaurants -> student-prompt rollout seeds.

This is the v2 rebuild of scripts/build_grpo.py, now **TRACE-FREE**. GRPO generates
the trajectory ON-POLICY at train time, so a row needs only the PROMPT to roll out
from -- and the v2 reward is teacher-free (src/reward.py: structure + found +
GROUNDING in the scraped evidence), so there is no teacher `reference` to store and
no teacher trace to read. Rows are built directly from the `grpo`-split restaurants
(`iter_restaurants(split="grpo")`) plus seeded dietary sampling.

Because ONLY grpo-split restaurants are read, the v1 eval-leak guard is gone: the DB
split is disjoint by restaurant_id, so an eval restaurant can never surface here.

Prompt = the SHIPPED student view (same as eval): system = build_system_prompt(
dietary, variant="student") (teacher guidance absent; the dietary restriction --
target-defining -- kept), then the user episode input "{name}, {city}". The tool
DECLARATIONS are NOT in the prompt: GRPOTrainer renders them from the tool callables
at train time, exactly as the agent loop and eval do (tools=TOOLS passed to
apply_chat_template). The row is therefore STUDENT-AGNOSTIC -- the chat template is
applied at train time, not baked in here.

Episode planning mirrors the corpus builder's mixed free+conditioned plan (seeded,
prefix-stable) so the GRPO restaurant set explores the same free/conditioned mix:

  * restriction-FREE episodes (restrictions=[]) take the front of the seeded order.
  * restriction-CONDITIONED episodes REUSE the front of that same order (row i uses
    rows[i % len(rows)]), rotating through DIETARY_POOL, so a restaurant can get both
    a free and a filtered rollout seed. --conditioned-frac sets the conditioned share
    of the --limit episode budget (default 0.0 = pure free). Deduped by trace_id_for
    (rid / rid__slug) so a wrap-around can't plan the same (restaurant, restriction)
    twice.

Output (one JSON object per line in data/grpo/train.jsonl -- train_grpo.py reads only
`prompt`):
  {
    "restaurant_id": "<rid>",
    "dietary_restrictions": null | ["vegetarian", ...],
    "found": null,                       # trace-free: unknown (no teacher outcome)
    "prompt": [ {"role":"system","content": <student prompt>},
                {"role":"user","content": <episode input>} ]
  }

A sidecar `<out>.meta.json` records provenance (git sha, corpus md5, seed,
conditioned-frac, row counts).

  uv run python scripts/datasets/build_grpo.py                       # grpo split -> data/grpo/train.jsonl
  uv run python scripts/datasets/build_grpo.py --conditioned-frac 0.4  # 60% free + 40% conditioned
  uv run python scripts/datasets/build_grpo.py --limit 8             # quick smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/.
sys.path.insert(0, str(REPO_ROOT / "src"))

from corpus import open_corpus  # noqa: E402
from episodes import plan_episodes, seeded_order  # noqa: E402
from prompts import build_system_prompt  # noqa: E402
from run_meta import git_sha, md5_file  # noqa: E402


def episode_to_row(episode: dict) -> dict:
    """One planned episode -> one GRPO dataset row (student-agnostic prompt).

    `found` is null: trace-free GRPO has no teacher outcome, so whether a menu is
    findable is unknown until the rollout. train_grpo.py reads only `prompt`.
    """
    row = episode["row"]
    restrictions = episode["restrictions"]  # normalized list; [] == free
    episode_input = f"{row['name']}, {row['city']}"
    system_prompt = build_system_prompt(restrictions or None, variant="student")
    return {
        "restaurant_id": row["restaurant_id"],
        "dietary_restrictions": restrictions or None,
        "found": None,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": episode_input},
        ],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                   help="corpus.sqlite (the v2 single source of truth)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "grpo" / "train.jsonl")
    p.add_argument("--conditioned-frac", type=float, default=0.0,
                   help="fraction of the episode budget that is dietary-restriction "
                        "conditioned (default 0.0 = pure free; 0.4 gives a 3:2 free:"
                        "conditioned split). Conditioned episodes reuse the front of the "
                        "seeded order (contrastive free/filtered pairs).")
    p.add_argument("--limit", type=int, default=None,
                   help="TOTAL episode budget (free + conditioned); default: one free "
                        "episode per grpo-split restaurant")
    p.add_argument("--seed", type=int, default=42, help="selection-order seed (default 42; keep fixed)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # grpo-split restaurants only -> no eval-leak possible (splits are disjoint by
    # restaurant_id). iter_restaurants already yields in restaurant_id order; one
    # seeded shuffle gives the prefix-stable order build_corpus uses.
    with open_corpus(args.corpus, create=False) as cx:
        rows = seeded_order(cx.iter_restaurants(split="grpo"), args.seed)

    episodes = plan_episodes(rows, args.limit, args.conditioned_frac)
    n_free = sum(not e["restrictions"] for e in episodes)
    n_cond = len(episodes) - n_free

    out_rows = [episode_to_row(e) for e in episodes]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- provenance sidecar ----------------------------------------------
    meta_path = args.out.with_name(args.out.name + ".meta.json")
    meta = {
        "git_sha": git_sha(REPO_ROOT),
        "corpus_path": str(args.corpus),
        "corpus_md5": md5_file(args.corpus),
        "prompt_variant": "student",
        "seed": args.seed,
        "conditioned_frac": args.conditioned_frac,
        "limit": args.limit,
        "n_rows": len(out_rows),
        "n_free": n_free,
        "n_conditioned": n_cond,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== build_grpo summary =====")
    print(f"grpo restaurants: {len(rows)}  (seed {args.seed}, conditioned-frac "
          f"{args.conditioned_frac}, limit {args.limit})")
    print(f"written        : {len(out_rows)} -> {args.out}  "
          f"(free {n_free}, conditioned {n_cond})")
    print(f"meta           : {meta_path}")


if __name__ == "__main__":
    main()
