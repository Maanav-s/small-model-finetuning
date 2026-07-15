"""Build the GRPO training dataset from teacher traces (Phase 3).

Unlike build_sft.py -- which bakes the full teacher trajectory into a trainable
sequence -- GRPO generates the trajectory ON-POLICY at train time, so a row needs
essentially just the PROMPT to roll out from. `scripts/train_grpo.py` hands each
prompt to TRL's GRPOTrainer (with the web_search/scrape_url tools + vLLM), which
produces G tool-use rollouts, and `src/reward.py` scores each rollout from the
episode itself (schema + found + GROUNDING in the scraped evidence).

Prompt = the SHIPPED student view (same as eval): system = build_system_prompt(
dietary, variant="student") (teacher guidance absent; the dietary restriction --
target-defining -- kept), then the user episode input. The tool DECLARATIONS are
NOT in the prompt: GRPOTrainer renders them from the tool callables, exactly as
the agent loop and eval do (tools=TOOLS passed to apply_chat_template).

Reference = the teacher's cleaned final_json, stored as a compact JSON STRING.
IMPORTANT: this is **analysis-only metadata** -- the reward is now teacher-free
(pure RL: it does NOT score against this reference; see src/reward.py). It is kept
only so offline eval/analysis can compare rollouts to the teacher if wanted. The
`dietary_restrictions` column IS used at train time (a future local dietary judge
grades conditioned menus against it).

Output (one JSON object per line in data/grpo/train.jsonl):
  {
    "restaurant_id": "<rid>",
    "dietary_restrictions": null | ["vegetarian", ...],
    "found": true | false,
    "prompt": [ {"role":"system","content": <student prompt>},
                {"role":"user","content": <episode input>} ],
    "reference": "<compact JSON string of the teacher final_json>",
    "meta": {"model": "<teacher>", "source_trace": "<file>"}
  }

  uv run python scripts/build_grpo.py                    # data/traces/* -> data/grpo/train.jsonl
  uv run python scripts/build_grpo.py --limit 10         # quick smoke
  uv run python scripts/build_grpo.py --splits data/splits.json   # DROP eval-split rids (leakage guard)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/, the
# SFT builder in scripts/ (we reuse its trace-reading helpers so the two builders
# read a trace identically).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_sft import (  # noqa: E402  -- reuse, don't reimplement, the trace readers
    _text_of,
    clean_final_answer,
    is_rejected,
    load_reject_set,
)
from prompts import build_system_prompt  # noqa: E402


def trace_to_grpo_row(trace: dict) -> dict:
    """One teacher trace -> one GRPO dataset row. Raises ValueError (a skip reason).

    Pure + tokenizer-free: builds the student prompt messages and the reference
    string. Does NOT render the chat template (GRPOTrainer does that on-policy).
    """
    tmsgs = trace.get("messages") or []
    if not tmsgs:
        raise ValueError("trace has no messages")
    if tmsgs[0].get("role") != "user":
        raise ValueError("first message is not a user episode-input turn")
    episode_input = _text_of(tmsgs[0]["content"])
    if not episode_input:
        raise ValueError("empty episode input")

    last = tmsgs[-1]
    if last.get("role") != "assistant":
        raise ValueError("last message is not an assistant turn")
    cleaned = clean_final_answer(_text_of(last["content"]))
    if cleaned is None:
        raise ValueError("final answer did not parse as JSON")
    reference_str, final_obj = cleaned

    dietary = trace.get("dietary_restrictions")
    system_prompt = build_system_prompt(dietary, variant="student")
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": episode_input},
    ]
    return {
        "restaurant_id": trace.get("restaurant_id"),
        "dietary_restrictions": dietary,
        "found": bool(final_obj.get("found")),
        "prompt": prompt,
        "reference": reference_str,
        "meta": {"model": trace.get("model"), "source_trace": None},
    }


def load_eval_rids(splits_path: Path | None) -> set[str]:
    """restaurant_ids in the EVAL split, to DROP (never train GRPO on eval data).

    splits.json is `{"train": [...], "eval": [...]}` of restaurant_ids. Missing
    file / key -> empty set (no filtering), so this is an opt-in leakage guard.
    """
    if splits_path is None or not splits_path.exists():
        return set()
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    return set(data.get("eval") or [])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "data" / "traces")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "grpo" / "train.jsonl")
    p.add_argument("--splits", type=Path, default=None,
                   help="splits.json; when given, DROP eval-split restaurant_ids (leakage guard)")
    p.add_argument("--reject-list", type=Path, default=None,
                   help="optional file of restaurant_ids / trace filenames (one per line) to DROP")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N trace files (sorted) -- for quick tests")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    reject = load_reject_set(args.reject_list)
    eval_rids = load_eval_rids(args.splits)
    trace_files = sorted(args.traces_dir.glob("*.json"))
    if args.limit is not None:
        trace_files = trace_files[: args.limit]
    if not trace_files:
        sys.exit(f"no traces found in {args.traces_dir}")

    rows: list[dict] = []
    skipped: dict[str, int] = {}
    rejected = leaked = 0

    def skip(reason: str, name: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        print(f"  [skip] {name}: {reason}")

    for path in trace_files:
        name = path.name
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            skip(f"unreadable trace ({e})", name)
            continue
        rid = trace.get("restaurant_id", name[:-5])
        if is_rejected(name, rid, reject):
            rejected += 1
            print(f"  [reject] {name}: on reject list")
            continue
        if rid in eval_rids:
            leaked += 1
            print(f"  [drop] {name}: eval-split restaurant (leakage guard)")
            continue
        try:
            row = trace_to_grpo_row(trace)
        except ValueError as e:
            skip(str(e), name)
            continue
        row["meta"]["source_trace"] = name
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_true = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_true += bool(row["found"])

    print("\n===== build_grpo summary =====")
    print(f"traces scanned : {len(trace_files)}")
    print(f"rejected (list): {rejected}")
    print(f"dropped (eval) : {leaked}")
    print(f"skipped        : {sum(skipped.values())}")
    for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {reason}")
    n_false = len(rows) - n_true
    frac = (n_false / len(rows)) if rows else 0.0
    print(f"written        : {len(rows)} -> {args.out}  "
          f"(found=true {n_true}, found=false {n_false}, frac {frac:.3f})")


if __name__ == "__main__":
    main()
