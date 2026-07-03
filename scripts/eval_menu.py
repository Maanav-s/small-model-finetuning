"""WS-G: eval/validation harness -- score menu-JSON outputs on a split.

Pure-local scorer over already-recorded outputs: it only READS trace files,
never runs episodes and never touches the network, so it is deterministic and
free to re-run. (Producing a candidate set is a separate step -- run the eval
split through a runner with `--cache-policy canned` against the WS-C2-warmed
cache and record traces; this script then grades those files. See
notes/phase2_plan.md WS-G.) All per-episode scoring lives in
src/eval_metrics.py -- the shared, importable functions the Phase 3 GRPO
reward will compose -- this CLI only loads files, joins, and formats.

Inputs are directories of per-restaurant JSON files: full trace files
(contract 1.5 -- the menu is read from `final_json`) or bare menu-JSON files.
Join key is the trace's `restaurant_id`, falling back to the filename stem.

  # candidate set vs reference set (joined on restaurant_id)
  uv run python scripts/eval_menu.py data/traces_gemma --reference data/traces

  # no reference set yet: validity / found-rate / size stats only
  uv run python scripts/eval_menu.py data/traces --self-report

  # optional extras (both modes)
  ... --labels data/labels.jsonl   # correct-abstention vs WS-F findability labels
  ... --json report.json           # machine-readable dump of everything printed
  ... --per-episode                # one scored line per restaurant
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Shared modules live in src/ (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval_metrics import (  # noqa: E402
    ITEM_MATCH_THRESHOLD,
    abstention_outcome,
    aggregate,
    aggregate_self_reports,
    score_episode,
    self_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate_dir", type=Path,
                        help="directory of candidate trace/menu JSON files (one per restaurant)")
    parser.add_argument("--reference", type=Path, default=None,
                        help="directory of reference trace/menu JSON files to score against")
    parser.add_argument("--self-report", action="store_true",
                        help="no reference set: report validity/found/size stats only")
    parser.add_argument("--labels", type=Path, default=None,
                        help="labels.jsonl (WS-F) for correct-abstention scoring")
    parser.add_argument("--threshold", type=float, default=ITEM_MATCH_THRESHOLD,
                        help=f"fuzzy item-name match threshold (default {ITEM_MATCH_THRESHOLD})")
    parser.add_argument("--per-episode", action="store_true", help="print one scored line per restaurant")
    parser.add_argument("--json", type=Path, default=None, help="also write the full report as JSON")
    args = parser.parse_args()
    if bool(args.reference) == bool(args.self_report):
        parser.error("exactly one of --reference DIR or --self-report is required")
    return args


def load_menus(dir_path: Path) -> tuple[dict, list[str]]:
    """{restaurant_id: menu dict | None} from a directory of JSON files.

    A file with a `final_json` key is a trace (contract 1.5); anything else is
    taken as a bare menu JSON. Unreadable files score as None (an invalid
    candidate) rather than killing the run; their ids are returned for the report.
    """
    menus, unreadable = {}, []
    for path in sorted(dir_path.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            menus[path.stem] = None
            unreadable.append(path.stem)
            continue
        if isinstance(obj, dict) and "final_json" in obj:
            menus[obj.get("restaurant_id") or path.stem] = obj["final_json"]
        else:
            menus[path.stem] = obj
    return menus, unreadable


def load_labels(path: Path) -> dict:
    """{restaurant_id: findable bool} from labels.jsonl (WS-F)."""
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["restaurant_id"]] = bool(row.get("findable"))
    return labels


def score_abstention(menus: dict, labels: dict) -> dict:
    """Bucket every labeled candidate by abstention_outcome; return counts."""
    counts = {"correct_find": 0, "correct_abstain": 0, "false_abstain": 0, "false_find": 0}
    for rid, menu in menus.items():
        if rid in labels:
            counts[abstention_outcome(menu, labels[rid])] += 1
    counts["n_labeled"] = sum(counts.values())
    return counts


def fmt(value, pct=False) -> str:
    """One number for the report; N/A for metrics with no defined episodes."""
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%" if pct else f"{value:.3f}"


def print_abstention(counts: dict):
    n_unfindable = counts["correct_abstain"] + counts["false_find"]
    n_findable = counts["correct_find"] + counts["false_abstain"]
    print(f"abstention vs labels ({counts['n_labeled']} labeled):")
    print(f"  correct-abstention (findable=false): {counts['correct_abstain']}/{n_unfindable}"
          f"  (false-find, hallucination risk: {counts['false_find']})")
    print(f"  found when findable:                 {counts['correct_find']}/{n_findable}"
          f"  (false give-ups: {counts['false_abstain']})")


def run_paired(args) -> dict:
    candidates, cand_unreadable = load_menus(args.candidate_dir)
    references, ref_unreadable = load_menus(args.reference)

    joined = sorted(set(candidates) & set(references))
    # A reference must at least carry a usable found flag; skip (and count) the rest.
    usable = [rid for rid in joined
              if isinstance(references[rid], dict) and "found" in references[rid]]
    episodes = {rid: score_episode(candidates[rid], references[rid], threshold=args.threshold)
                for rid in usable}
    agg = aggregate(list(episodes.values()))

    print("===== WS-G eval report (candidate vs reference) =====")
    print(f"candidate: {args.candidate_dir} ({len(candidates)} files, {len(cand_unreadable)} unreadable)")
    print(f"reference: {args.reference} ({len(references)} files, {len(ref_unreadable)} unreadable)")
    print(f"joined on restaurant_id: {len(joined)}  scored: {len(usable)} "
          f"(skipped {len(joined) - len(usable)} unusable references; "
          f"candidate-only: {len(candidates) - len(joined)}, reference-only: {len(references) - len(joined)})")
    if args.per_episode:
        for rid in usable:
            s = episodes[rid]
            print(f"  {rid}  valid={int(s['schema_valid'])} found_ok={int(s['found_correct'])} "
                  f"P={fmt(s['precision'])} R={fmt(s['recall'])} F1={fmt(s['f1'])} "
                  f"price={fmt(s['price_agreement'])} "
                  f"items={s['n_candidate_items']}/{s['n_reference_items']} (matched {s['n_matched']})")
    print(f"schema-valid:     {fmt(agg.get('schema_valid_rate'), pct=True)}")
    print(f"found accuracy:   {fmt(agg.get('found_accuracy'), pct=True)}")
    print(f"item precision:   {fmt(agg.get('precision_mean'))}  (n={agg.get('precision_n', 0)})")
    print(f"item recall:      {fmt(agg.get('recall_mean'))}  (n={agg.get('recall_n', 0)})")
    print(f"item F1:          {fmt(agg.get('f1_mean'))}  (n={agg.get('f1_n', 0)})")
    print(f"price agreement:  {fmt(agg.get('price_agreement_mean'))}  (n={agg.get('price_agreement_n', 0)})")
    print(f"section-count delta (cand-ref): mean {fmt(agg.get('section_count_delta_mean'))}  "
          f"mean|delta| {fmt(agg.get('section_count_delta_mean_abs'))}  (n={agg.get('section_count_delta_n', 0)})")
    print(f"item-count delta    (cand-ref): mean {fmt(agg.get('item_count_delta_mean'))}  "
          f"mean|delta| {fmt(agg.get('item_count_delta_mean_abs'))}  (n={agg.get('item_count_delta_n', 0)})")

    report = {"mode": "paired", "candidate_dir": str(args.candidate_dir),
              "reference_dir": str(args.reference), "threshold": args.threshold,
              "aggregate": agg, "episodes": episodes,
              "unreadable": {"candidate": cand_unreadable, "reference": ref_unreadable}}
    if args.labels and args.labels.exists():
        report["abstention"] = score_abstention(candidates, load_labels(args.labels))
        print_abstention(report["abstention"])
    return report


def run_self_report(args) -> dict:
    candidates, unreadable = load_menus(args.candidate_dir)
    episodes = {rid: self_report(menu) for rid, menu in sorted(candidates.items())}
    agg = aggregate_self_reports(list(episodes.values()))

    print("===== WS-G self-report (no reference) =====")
    print(f"candidate: {args.candidate_dir} ({len(candidates)} files, {len(unreadable)} unreadable)")
    if args.per_episode:
        for rid, r in episodes.items():
            print(f"  {rid}  valid={int(r['schema_valid'])} found={r['found']} "
                  f"sections={r['n_sections']} items={r['n_items']} "
                  f"price_coverage={fmt(r['price_coverage'])}")
    print(f"schema-valid:   {fmt(agg.get('schema_valid_rate'), pct=True)}")
    print(f"found=true:     {fmt(agg.get('found_rate'), pct=True)}")
    print(f"mean sections:  {fmt(agg.get('mean_sections'))}")
    print(f"mean items:     {fmt(agg.get('mean_items'))}")
    print(f"price coverage: {fmt(agg.get('price_coverage_mean'))}  (n={agg.get('price_coverage_n', 0)})")

    report = {"mode": "self-report", "candidate_dir": str(args.candidate_dir),
              "aggregate": agg, "episodes": episodes, "unreadable": {"candidate": unreadable}}
    if args.labels and args.labels.exists():
        report["abstention"] = score_abstention(candidates, load_labels(args.labels))
        print_abstention(report["abstention"])
    return report


def main():
    args = parse_args()
    report = run_paired(args) if args.reference else run_self_report(args)
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote JSON report: {args.json}")


if __name__ == "__main__":
    main()
