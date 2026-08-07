"""Render a directory of eval report.json files as one comparison table (Markdown).

The eval harness writes a `report.json` per model (`eval.py --json`). Those files
are the permanent, committed record (`results/<run-set>/<model>.json`); this script
turns a whole run-set into the human-readable table that goes at the top of
`results/README.md` -- so "which model won" is answerable from the repo without
opening W&B or a pod.

Handles BOTH report modes in one table:
  * `paired`      -- scored against the teacher's DB reference traces (P/R/F1,
                     found-accuracy, abstention buckets).
  * `self-report` -- reference-free (schema-valid, found=true, sizes). The teacher's
                     own row is necessarily this mode: it IS the reference, so
                     pairing it against itself would print 1.000 and mean nothing.
Columns that don't apply to a mode render as `--`.

    uv run python scripts/eval/summarize.py results/20260807-eval500
    uv run python scripts/eval/summarize.py results/20260807-eval500 -o results/README.md
    uv run python scripts/eval/summarize.py results/20260807-eval500 --slice conditioned

Ordering: reports are listed teacher-first (self-report rows before paired ones),
then by descending found-rate, so the table reads as "the ceiling, then the students".
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report_dir", type=Path, nargs="?", default=DEFAULT_RESULTS,
                   help=f"directory of report.json files, searched recursively "
                        f"(default {DEFAULT_RESULTS})")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write the Markdown table here (default: print to stdout)")
    p.add_argument("--slice", default="all", choices=["all", "free", "conditioned"],
                   help="which aggregate slice to tabulate (default all)")
    return p.parse_args(argv)


def pct(value):
    return "--" if value is None else f"{100 * value:.1f}%"


def num(value, places=2):
    return "--" if value is None else f"{value:.{places}f}"


def label_for(report: dict, path: Path) -> str:
    """Prefer the checkpoint's run_id (the adapter lineage), else the model id, else
    the filename -- whichever most specifically names what was evaluated."""
    ckpt = report.get("checkpoint") or {}
    if ckpt.get("run_id"):
        return str(ckpt["run_id"])
    model_id = report.get("model_id")
    if model_id and model_id not in {"vllm", "gemma", "claude"}:
        return Path(str(model_id)).name
    return path.stem


def row_for(report: dict, path: Path, slice_name: str) -> dict:
    agg = (report.get("aggregate") or {}).get(slice_name) or {}
    cache = report.get("cache") or {}
    run = report.get("run") or {}
    absten = report.get("abstention") or {}
    n_findable = absten.get("correct_find", 0) + absten.get("false_abstain", 0)
    return {
        "model": label_for(report, path),
        "mode": report.get("mode", "?"),
        "n": agg.get("n_episodes", 0),
        "schema_valid": pct(agg.get("schema_valid_rate")),
        # self-report calls it found_rate (did it answer); paired calls it
        # found_accuracy (did its found flag MATCH the reference). Different
        # questions -- the mode column is what tells them apart.
        "found": pct(agg.get("found_rate", agg.get("found_accuracy"))),
        "f1": num(agg.get("f1_mean"), 3),
        "precision": num(agg.get("precision_mean"), 3),
        "recall": num(agg.get("recall_mean"), 3),
        "items": num(agg.get("mean_items", agg.get("item_count_delta_mean")), 1),
        "price_cov": num(agg.get("price_coverage_mean"), 3),
        "false_find": str(absten["false_find"]) if "false_find" in absten else "--",
        "false_abstain": (f"{absten['false_abstain']}/{n_findable}"
                          if "false_abstain" in absten else "--"),
        "cache_hit": pct(cache.get("hit_rate")),
        "cache_policy": report.get("cache_policy", "--"),
        "eps_min": num(run.get("eps_per_min"), 1),
        "failed": str(run.get("n_failed", "--")),
        "_sort": (0 if report.get("mode") == "self-report" else 1,
                  -(agg.get("found_rate") or agg.get("found_accuracy") or 0)),
    }


COLUMNS = [
    ("model", "model"), ("mode", "mode"), ("n", "n"),
    ("schema_valid", "schema-valid"), ("found", "found"),
    ("f1", "item F1"), ("precision", "P"), ("recall", "R"),
    ("items", "mean items"), ("price_cov", "price cov"),
    ("false_find", "false-find"), ("false_abstain", "false give-up"),
    ("cache_hit", "cache hit-rate"), ("cache_policy", "policy"),
    ("eps_min", "eps/min"), ("failed", "failed"),
]


def render(rows: list[dict], slice_name: str, report_dir: Path) -> str:
    head = "| " + " | ".join(h for _, h in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(str(r[k]) for k, _ in COLUMNS) + " |" for r in rows]
    note = (
        f"\n_Slice: **{slice_name}**. `found` means found=true rate for self-report rows "
        f"and found-flag ACCURACY vs the reference for paired rows. Item F1/P/R and the "
        f"abstention columns exist only for paired rows -- the teacher is the reference, "
        f"so it is self-reported by construction. `cache hit-rate` is lookups served from "
        f"`cache.sqlite`; a low value means the model explored off the warmed distribution._\n"
    )
    return "\n".join([f"### Eval comparison - `{report_dir.name or report_dir}`",
                      "", head, sep, *body, note])


def main(argv=None):
    args = parse_args(argv)
    paths = sorted(p for p in args.report_dir.rglob("*.json") if p.name != "meta.json")
    if not paths:
        sys.exit(f"no report JSON files under {args.report_dir}")

    rows = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        if "aggregate" not in report:  # not an eval report (e.g. a dataset meta file)
            continue
        rows.append(row_for(report, path, args.slice))
    if not rows:
        sys.exit(f"no eval reports (files with an 'aggregate' key) under {args.report_dir}")

    rows.sort(key=lambda r: r.pop("_sort"))
    table = render(rows, args.slice, args.report_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
