"""Export a split's DB traces as candidate files -- so the TEACHER gets scored by
the same harness as the students, for free (no second inference pass).

WHY THIS EXISTS: `eval.py` scores a *directory of candidate traces* against the
teacher's DB traces. The teacher itself can't be scored that way -- it IS the
reference, so pairing it against itself yields P=R=F1=1.0 by construction and says
nothing. What we actually want for the expert row of the results table is its
SELF-REPORT (schema-valid %, found %, mean sections/items, price coverage): the
exact metric shape the students report, so the three rows are comparable.

`eval.py --self-report` already computes that from a candidate dir, and it skips
production entirely when every planned candidate file already exists ("all
candidates already exist -- skipping the run"). So: dump the teacher's eval traces
into a candidate dir in the candidate-file shape, then point eval.py at it. No
model, no GPU, no API, no cache.

    # 1. the teacher populated corpus.sqlite (scripts/corpus/build_corpus.py --split eval)
    # 2. dump those traces as candidates
    uv run python scripts/eval/dump_reference.py results/20260807/teacher/candidates
    # 3. score them with the same scorer the students use
    uv run python scripts/eval/eval.py results/20260807/teacher/candidates \
        --model vllm --self-report --limit 500 --conditioned-frac 0.4 \
        --json results/20260807/teacher/report.json

The dumped file is a superset of a candidate trace: it carries the reference-only
fields (`grounding`, `queries`, `urls`, `n_messages`) too, which the scorer ignores
but which make the dump a self-contained artifact worth archiving next to the
report. `messages` (the full trajectory, by far the biggest column) is NOT dumped --
the DB remains the place to read trajectories from.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from corpus import VALID_SPLITS, open_corpus  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path,
                   help="directory to write <trace_id>.json candidate files into")
    p.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                   help="corpus.sqlite (default data/corpus.sqlite)")
    p.add_argument("--split", default="eval", choices=sorted(VALID_SPLITS),
                   help="which split's traces to dump (default eval)")
    p.add_argument("--trace-source", default="teacher",
                   help="trace_source filter (default 'teacher'; pass '' for all sources)")
    p.add_argument("--include-rejected", action="store_true",
                   help="also dump traces flagged rejected by the cleaning pass "
                        "(default: skip them, matching what eval.py uses as reference)")
    return p.parse_args(argv)


def candidate_from_trace(trace: dict) -> dict:
    """A DB trace -> the candidate-file shape eval.py's loader/scorer reads.

    Keys through `captured_at` mirror eval.run_one's candidate exactly (so the file
    is indistinguishable from a produced candidate); the rest are reference-only
    extras the scorer ignores.
    """
    return {
        "trace_id": trace["trace_id"],
        "restaurant_id": trace["restaurant_id"],
        "restaurant_name": trace.get("restaurant_name"),
        "episode_input": trace.get("episode_input"),
        "model": trace["model"],
        "prompt_variant": trace["prompt_variant"],
        "dietary_restrictions": trace.get("dietary_restrictions"),
        "final_json": trace["final_json"],
        "schema_valid": bool(trace["schema_valid"]),
        "captured_at": trace["captured_at"],
        # reference-only extras (ignored by the scorer, kept for the archive)
        "trace_source": trace["trace_source"],
        "grounding": trace.get("grounding"),
        "queries": trace.get("queries"),
        "urls": trace.get("urls"),
        "n_messages": len(trace.get("messages") or []),
        "dumped_from": "corpus.sqlite",
    }


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open_corpus(args.corpus, create=False) as cx:
        traces = list(cx.iter_traces(split=args.split,
                                     include_rejected=args.include_rejected,
                                     trace_source=args.trace_source or None))
    if not traces:
        sys.exit(f"no {args.split!r} traces in {args.corpus} "
                 f"(source={args.trace_source!r}) -- run the teacher over that split first "
                 f"(scripts/corpus/build_corpus.py --split {args.split})")

    n_found = 0
    for trace in traces:
        cand = candidate_from_trace(trace)
        n_found += bool((cand["final_json"] or {}).get("found"))
        (args.out_dir / f"{trace['trace_id']}.json").write_text(
            json.dumps(cand, ensure_ascii=False), encoding="utf-8")

    n_cond = sum(bool(t.get("dietary_restrictions")) for t in traces)
    print(f"wrote {len(traces)} candidate files to {args.out_dir} "
          f"({len(traces) - n_cond} free + {n_cond} conditioned; "
          f"found=true on {n_found}/{len(traces)})")
    print(f"model(s): {sorted({t['model'] for t in traces})}")


if __name__ == "__main__":
    main()
