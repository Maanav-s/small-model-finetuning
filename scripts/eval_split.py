"""WS-G: PRODUCE + score an eval-split candidate set (the run half of the harness).

scripts/eval_menu.py only SCORES two directories of already-recorded traces; it
cannot generate a candidate set. This script fills that gap: it runs a chosen
model (Claude or the fine-tuned Gemma student) over the **eval split** under a
frozen (`canned`) cache to produce candidate traces, then scores them against a
teacher reference set -- printing a WS-G report with a **free vs. dietary-
conditioned** breakdown.

The eval plan is produced by the SAME planner the training corpus uses
(scripts/build_corpus.py: `load_seeded_rows` -> `plan_episodes`), so the eval set
is a seed-reproducible mix of restriction-FREE episodes plus a conditioned slice
whose restrictions are drawn from the SAME `DIETARY_POOL`, but over **eval-split**
restaurants. Episodes are rendered with the **student** system prompt
(`variant="student"`) -- that is what we ship and therefore what we must measure;
conditioned episodes keep their restriction visible (a dietary restriction is a
target-defining input, never distilled away -- see CLAUDE.md).

RUN ORDERING (important -- `canned` needs a warmed eval cache):
  A frozen (`canned`) run cannot fetch: any tool call whose key is absent from the
  cache returns the canned constant, and any that *would* have needed the network
  scores nothing but abstention. So run these in order:

    1. Build the REFERENCE set AND warm the eval-split cache in one live pass:
         uv run python scripts/build_corpus.py --split eval --conditioned-frac 0.4 \
             --cache-policy live --limit <N>
       This writes teacher traces to data/traces/ (point --reference at that dir,
       or a copy) and populates data/cache.sqlite with every query/URL -- including
       the restriction-specific queries the teacher issues -- for the eval split.
    2. Score a candidate model against that reference over the frozen cache:
         uv run python scripts/eval_split.py data/candidates_gemma \
             --model gemma --model-path <ckpt> \
             --cache-policy canned --reference data/traces_eval \
             --conditioned-frac 0.4

  Keep --split/--seed/--conditioned-frac/--limit identical between the two so the
  candidate and reference plans (hence the per-episode trace filenames) line up.

A candidate episode that hits a genuine cache miss (CacheMiss, e.g. the student
explores a URL/mode the teacher never did under `--cache-policy error`) is NOT
fatal: it is recorded as a failed/empty candidate (final_json=None,
schema_valid=False) and counted, exactly as build_corpus treats a per-episode
exception. Unexpected exceptions still trip build_corpus's MAX_CONSECUTIVE_FAILURES
abort so a broken key/checkpoint can't silently burn the whole run.

  # plan only, no API/GPU (like build_corpus --list)
  uv run python scripts/eval_split.py cand/ --model claude --list --limit 100 --conditioned-frac 0.4

  # candidates only (no reference yet): schema-valid / found-rate / mean-items self-stats
  uv run python scripts/eval_split.py cand/ --model claude

  # candidates + paired scoring with a free/conditioned breakdown
  uv run python scripts/eval_split.py cand/ --model claude --reference data/traces_eval

Requires BRAVE_API_KEY (search backend is built even when frozen) and, for
--model claude, ANTHROPIC_API_KEY (repo-root .env).
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/, the
# Claude loop in src/claude/, the episode planner in scripts/ (build_corpus).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "claude"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import jsonschema  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# Episode planner + trace-name contract, imported (not reimplemented) so the eval
# plan is IDENTICAL to how the train corpus is planned.
from build_corpus import (  # noqa: E402
    DIETARY_POOL,  # noqa: F401 -- re-exported for callers/tests
    MAX_CONSECUTIVE_FAILURES,
    episode_trace_name,
    load_seeded_rows,
    plan_episodes,
)
from cache import Cache, CacheMiss  # noqa: E402
from claude_agent import MODEL_ID as CLAUDE_MODEL_ID  # noqa: E402
from claude_agent import run_episode as claude_run_episode  # noqa: E402
from eval_metrics import aggregate, aggregate_self_reports, score_episode, self_report  # noqa: E402
from prompts import build_system_prompt  # noqa: E402
from schema import MENU_SCHEMA, extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

GEMMA_MODEL_ID = "google/gemma-4-E4B-it"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("candidate_dir", type=Path,
                        help="directory to write candidate trace JSON files into (one per episode)")
    parser.add_argument("--model", choices=["claude", "gemma"], required=True,
                        help="which runner produces the candidates")
    parser.add_argument("--model-path", default=None,
                        help="gemma: a local merged HF checkpoint dir (a fine-tuned student) to "
                             "load instead of the base model; claude: an optional model-id override")
    parser.add_argument("--adapter-path", default=None,
                        help="gemma: load a LoRA adapter dir on top of the (4-bit) base model, "
                             "instead of a fully-merged --model-path checkpoint. Evaluates the "
                             "adapter without materializing/pulling the ~15GB merged model.")
    parser.add_argument("--reference", type=Path, default=None,
                        help="dir of teacher eval traces to score against (build_corpus.py "
                             "--split eval). Omit to run candidates only + print self-stats.")
    parser.add_argument("--limit", type=int, default=None,
                        help="TOTAL episode budget (free + conditioned); default: one free "
                             "episode per eval-split restaurant")
    parser.add_argument("--conditioned-frac", type=float, default=0.4,
                        help="conditioned share of the episode budget (default 0.4; mirror the "
                             "value used to build the reference set)")
    parser.add_argument("--split", default="eval",
                        help="which split to run (default eval)")
    parser.add_argument("--seed", type=int, default=42,
                        help="selection-order seed (default 42; keep it == the reference build)")
    parser.add_argument("--cache-policy", choices=["live", "canned", "error"], default="canned",
                        help="cache miss policy (default canned = frozen/reproducible eval)")
    parser.add_argument("--cache-path", default=str(REPO_ROOT / "data" / "cache.sqlite"),
                        help="sqlite cache path (default data/cache.sqlite)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="dir holding restaurants.jsonl + splits.json (default data/)")
    parser.add_argument("--workers", type=int, default=None,
                        help="thread-pool size (default 3 for claude; FORCED to 1 for gemma -- a "
                             "single-GPU model is not thread-safe)")
    parser.add_argument("--per-episode", action="store_true",
                        help="print one scored line per episode")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the full report as JSON")
    parser.add_argument("--list", action="store_true",
                        help="print the planned episodes and exit (no API/GPU calls)")
    args = parser.parse_args(argv)

    # Single GPU model isn't thread-safe: gemma is forced to one worker.
    if args.workers is None:
        args.workers = 1 if args.model == "gemma" else 3
    elif args.model == "gemma" and args.workers != 1:
        print("[warn] gemma runner is single-GPU / not thread-safe -- forcing --workers 1")
        args.workers = 1
    return args


def model_label(args) -> str:
    """The `model` field stamped into each candidate trace (checkpoint or base id)."""
    if args.model == "gemma":
        return args.model_path or GEMMA_MODEL_ID
    return args.model_path or CLAUDE_MODEL_ID


# ---------------------------------------------------------------------------
# Runner construction (Claude loads no torch; Gemma imported lazily)
# ---------------------------------------------------------------------------
def build_runner(args, label):
    """Return `runner(episode_input, tools, registry, system_prompt) -> final_text`.

    Claude drives the Anthropic API; Gemma loads a local (possibly fine-tuned)
    checkpoint. Torch/transformers are imported only inside the gemma branch so
    the Claude path and the tests never require a GPU.
    """
    if args.model == "claude":
        import anthropic

        client = anthropic.Anthropic()

        def runner(episode_input, tools, registry, system_prompt):
            # claude_run_episode is a module global on purpose -- tests monkeypatch it.
            final_text, _messages = claude_run_episode(
                client, episode_input, tools, registry, system_prompt, model=label
            )
            return final_text

        return runner

    # gemma: load once, reuse. --model-path points load_model's from_pretrained
    # source at a merged student checkpoint. load_model (src/gemma/model.py) reads
    # its module-global MODEL_ID, so we retarget that instead of editing model.py --
    # this keeps its SDPA GQA patch + 4-bit/device_map setup intact.
    sys.path.insert(0, str(REPO_ROOT / "src" / "gemma"))
    import model as gemma_model  # noqa: E402
    from agent import run_episode as gemma_run_episode  # noqa: E402
    from model import load_model  # noqa: E402

    if args.model_path:
        gemma_model.MODEL_ID = args.model_path
    gmodel, tokenizer = load_model(quantize=True)
    if args.adapter_path:
        # QLoRA-style inference: LoRA adapter applied on top of the 4-bit base
        # (avoids needing the merged bf16 checkpoint on disk). The base is MODEL_ID
        # (hub id / GEMMA_MODEL_PATH); do not also pass --model-path here.
        from peft import PeftModel  # noqa: E402
        gmodel = PeftModel.from_pretrained(gmodel, args.adapter_path)
        gmodel.eval()

    def runner(episode_input, tools, registry, system_prompt):
        return gemma_run_episode(
            gmodel, tokenizer, episode_input, tools, registry, system_prompt
        )

    return runner


# ---------------------------------------------------------------------------
# One candidate episode -> one candidate trace file
# ---------------------------------------------------------------------------
def run_one(runner, episode, tools, registry, system_prompt, args, label, candidate_dir):
    """Run one episode through `runner`, write a candidate trace, return a summary.

    A CacheMiss (a tool call absent from the frozen cache under --cache-policy
    error) is caught and recorded as a failed/empty candidate rather than raised,
    so one frozen-miss can't kill the run.
    """
    row = episode["row"]
    restrictions = episode["restrictions"]  # normalized list; [] == free
    rid = row["restaurant_id"]
    episode_input = f"{row['name']}, {row['city']}"

    cache_miss = False
    parse_err = None
    try:
        final_text = runner(episode_input, tools, registry, system_prompt)
        final_json, parse_err = extract_json(final_text)
    except CacheMiss as exc:
        cache_miss = True
        parse_err = f"cache miss: {exc}"
        final_json = None

    schema_valid = False
    if final_json is not None:
        try:
            jsonschema.validate(final_json, MENU_SCHEMA)
            schema_valid = True
        except jsonschema.ValidationError:
            pass

    trace = {
        "restaurant_id": rid,
        "restaurant_name": row["name"],
        "episode_input": episode_input,
        "model": label,
        "prompt_variant": "student",  # eval always uses the shipped student prompt
        "dietary_restrictions": restrictions or None,
        "cache_policy": args.cache_policy,
        "final_json": final_json,     # contract-1.5 compatible: eval_metrics reads this
        "schema_valid": schema_valid,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    if parse_err:
        trace["parse_error"] = parse_err
    if cache_miss:
        trace["cache_miss"] = True

    # Atomic write so an interrupted run can't leave a torn file the idempotent
    # skip would then treat as done.
    path = candidate_dir / episode_trace_name(rid, restrictions)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

    found = bool(final_json and final_json.get("found"))
    n_items = sum(len(s.get("items", [])) for s in (final_json or {}).get("menu", []) or [])
    return {"rid": rid, "name": row["name"], "schema_valid": schema_valid, "found": found,
            "items": n_items, "conditioned": bool(restrictions), "cache_miss": cache_miss}


# ---------------------------------------------------------------------------
# Candidate production
# ---------------------------------------------------------------------------
def produce_candidates(args, episodes, todo, candidate_dir):
    """Run every `todo` episode through the chosen runner; write candidate traces."""
    if args.model == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is required for --model claude (repo-root .env)")

    label = model_label(args)
    cache = Cache(args.cache_path, miss_policy=args.cache_policy)
    # Tools/registry are restriction-independent (only the system prompt embeds the
    # restriction), so build tools ONCE and memoize the student prompt per unique
    # restriction -- the same pattern build_corpus.main uses.
    tools, registry, _ = setup_tools(dietary_restrictions=None, variant="student", cache=cache)
    prompt_cache: dict[tuple, str] = {}

    def prompt_for(restrictions):
        key = tuple(restrictions)
        if key not in prompt_cache:
            prompt_cache[key] = build_system_prompt(list(key) or None, variant="student")
        return prompt_cache[key]

    runner = build_runner(args, label)

    results, failures = [], []
    consecutive_failures = 0
    fail_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, runner, e, tools, registry, prompt_for(e["restrictions"]),
                        args, label, candidate_dir): e
            for e in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            e = futures[fut]
            row = e["row"]
            try:
                summary = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one bad episode must not kill the run
                with fail_lock:
                    failures.append((row["restaurant_id"], row["name"], repr(exc)))
                    consecutive_failures += 1
                    print(f"[{i}/{len(todo)}] FAILED {row['name']!r}: {exc!r}")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"aborting: {consecutive_failures} consecutive failures")
                        for f in futures:
                            f.cancel()
                        break
                continue
            with fail_lock:
                consecutive_failures = 0
            results.append(summary)
            diet = "conditioned" if summary["conditioned"] else "free"
            flag = " [CACHE MISS]" if summary["cache_miss"] else ""
            print(f"[{i}/{len(todo)}] {summary['name']!r} ({diet}): "
                  f"schema_valid={summary['schema_valid']} found={summary['found']} "
                  f"items={summary['items']}{flag}")

    n_miss = sum(r["cache_miss"] for r in results)
    print("\n===== candidate run summary =====")
    print(f"episodes completed: {len(results)}  failed: {len(failures)}  "
          f"cache-misses (recorded empty): {n_miss}  "
          f"skipped (pre-existing): {len(episodes) - len(todo)}")
    for rid, name, err in failures:
        print(f"  FAILED {rid} {name!r}: {err}")
    print(f"cache stats: {cache.stats()}")
    cache.close()


# ---------------------------------------------------------------------------
# Scoring: load candidate/reference dirs, join on the episode FILENAME
# ---------------------------------------------------------------------------
def load_traces_by_filename(dir_path):
    """{filename: trace-or-menu obj (None if unreadable)} for a dir of JSON files.

    Keyed on the FULL filename (episode_trace_name), NOT restaurant_id: a
    restaurant's free and conditioned episodes share a restaurant_id but have
    distinct filenames (<rid>.json vs <rid>__<slug>.json), so joining on the id
    would collide the two.
    """
    out, unreadable = {}, []
    for path in sorted(dir_path.glob("*.json")):
        try:
            out[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            out[path.name] = None
            unreadable.append(path.name)
    return out, unreadable


def menu_of(obj):
    """The scored menu dict from a loaded object: a trace's `final_json` (contract
    1.5) or a bare menu JSON; None passes through as an invalid candidate."""
    if isinstance(obj, dict) and "final_json" in obj:
        return obj["final_json"]
    return obj


def is_conditioned(filename, obj):
    """Free vs. dietary-conditioned. Prefer the trace's recorded restriction;
    fall back to the filename's `__<slug>` marker (episode_trace_name)."""
    if isinstance(obj, dict) and "dietary_restrictions" in obj:
        return bool(obj.get("dietary_restrictions"))
    return "__" in Path(filename).stem


def _fmt(value, pct=False):
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%" if pct else f"{value:.3f}"


def _print_block(label, agg):
    print(f"--- {label} (n={agg.get('n_episodes', 0)}) ---")
    print(f"  schema-valid:    {_fmt(agg.get('schema_valid_rate'), pct=True)}")
    print(f"  found accuracy:  {_fmt(agg.get('found_accuracy'), pct=True)}")
    print(f"  item precision:  {_fmt(agg.get('precision_mean'))}  (n={agg.get('precision_n', 0)})")
    print(f"  item recall:     {_fmt(agg.get('recall_mean'))}  (n={agg.get('recall_n', 0)})")
    print(f"  item F1:         {_fmt(agg.get('f1_mean'))}  (n={agg.get('f1_n', 0)})")
    print(f"  price agreement: {_fmt(agg.get('price_agreement_mean'))}  (n={agg.get('price_agreement_n', 0)})")
    print(f"  section-count delta (cand-ref): mean {_fmt(agg.get('section_count_delta_mean'))}  "
          f"mean|delta| {_fmt(agg.get('section_count_delta_mean_abs'))}  (n={agg.get('section_count_delta_n', 0)})")
    print(f"  item-count delta    (cand-ref): mean {_fmt(agg.get('item_count_delta_mean'))}  "
          f"mean|delta| {_fmt(agg.get('item_count_delta_mean_abs'))}  (n={agg.get('item_count_delta_n', 0)})")


def run_paired(args, candidate_dir):
    cand_objs, cand_unreadable = load_traces_by_filename(candidate_dir)
    ref_objs, ref_unreadable = load_traces_by_filename(args.reference)

    joined = sorted(set(cand_objs) & set(ref_objs))
    # A reference must at least carry a usable found flag; skip (and count) the rest.
    usable = [fn for fn in joined
              if isinstance(menu_of(ref_objs[fn]), dict) and "found" in menu_of(ref_objs[fn])]
    episodes = {fn: score_episode(menu_of(cand_objs[fn]), menu_of(ref_objs[fn])) for fn in usable}
    conditioned = {fn: is_conditioned(fn, cand_objs[fn]) for fn in usable}

    all_scores = list(episodes.values())
    free_scores = [episodes[fn] for fn in usable if not conditioned[fn]]
    cond_scores = [episodes[fn] for fn in usable if conditioned[fn]]
    agg_all, agg_free, agg_cond = (aggregate(all_scores), aggregate(free_scores), aggregate(cond_scores))

    print("===== WS-G eval report (candidate vs reference) =====")
    print(f"model: {args.model} ({model_label(args)})   split: {args.split}   "
          f"seed: {args.seed}   conditioned-frac: {args.conditioned_frac}")
    print(f"candidate: {candidate_dir} ({len(cand_objs)} files, {len(cand_unreadable)} unreadable)")
    print(f"reference: {args.reference} ({len(ref_objs)} files, {len(ref_unreadable)} unreadable)")
    print(f"joined on episode filename: {len(joined)}  scored: {len(usable)} "
          f"(skipped {len(joined) - len(usable)} unusable references; "
          f"candidate-only: {len(cand_objs) - len(joined)}, reference-only: {len(ref_objs) - len(joined)})")
    if args.per_episode:
        for fn in usable:
            s = episodes[fn]
            tag = "cond" if conditioned[fn] else "free"
            print(f"  [{tag}] {fn}  valid={int(s['schema_valid'])} found_ok={int(s['found_correct'])} "
                  f"P={_fmt(s['precision'])} R={_fmt(s['recall'])} F1={_fmt(s['f1'])} "
                  f"price={_fmt(s['price_agreement'])} "
                  f"items={s['n_candidate_items']}/{s['n_reference_items']} (matched {s['n_matched']})")
    _print_block("all", agg_all)
    _print_block("free", agg_free)
    _print_block("conditioned", agg_cond)

    return {
        "mode": "paired", "model": args.model, "model_id": model_label(args),
        "split": args.split, "seed": args.seed, "conditioned_frac": args.conditioned_frac,
        "candidate_dir": str(candidate_dir), "reference_dir": str(args.reference),
        "aggregate": {"all": agg_all, "free": agg_free, "conditioned": agg_cond},
        "episodes": {fn: {**episodes[fn], "conditioned": conditioned[fn]} for fn in usable},
        "unreadable": {"candidate": cand_unreadable, "reference": ref_unreadable},
    }


def run_self_report(args, candidate_dir):
    cand_objs, unreadable = load_traces_by_filename(candidate_dir)
    reports = {fn: self_report(menu_of(obj)) for fn, obj in sorted(cand_objs.items())}
    conditioned = {fn: is_conditioned(fn, cand_objs[fn]) for fn in reports}

    agg_all = aggregate_self_reports(list(reports.values()))
    agg_free = aggregate_self_reports([reports[fn] for fn in reports if not conditioned[fn]])
    agg_cond = aggregate_self_reports([reports[fn] for fn in reports if conditioned[fn]])

    print("===== WS-G self-report (no reference) =====")
    print(f"model: {args.model} ({model_label(args)})   split: {args.split}")
    print(f"candidate: {candidate_dir} ({len(cand_objs)} files, {len(unreadable)} unreadable)")
    if args.per_episode:
        for fn, r in reports.items():
            tag = "cond" if conditioned[fn] else "free"
            print(f"  [{tag}] {fn}  valid={int(r['schema_valid'])} found={r['found']} "
                  f"sections={r['n_sections']} items={r['n_items']} "
                  f"price_coverage={_fmt(r['price_coverage'])}")
    for label, agg in (("all", agg_all), ("free", agg_free), ("conditioned", agg_cond)):
        print(f"--- {label} (n={agg.get('n_episodes', 0)}) ---")
        print(f"  schema-valid:   {_fmt(agg.get('schema_valid_rate'), pct=True)}")
        print(f"  found=true:     {_fmt(agg.get('found_rate'), pct=True)}")
        print(f"  mean sections:  {_fmt(agg.get('mean_sections'))}")
        print(f"  mean items:     {_fmt(agg.get('mean_items'))}")
        print(f"  price coverage: {_fmt(agg.get('price_coverage_mean'))}  (n={agg.get('price_coverage_n', 0)})")

    return {
        "mode": "self-report", "model": args.model, "model_id": model_label(args),
        "split": args.split, "candidate_dir": str(candidate_dir),
        "aggregate": {"all": agg_all, "free": agg_free, "conditioned": agg_cond},
        "episodes": {fn: {**reports[fn], "conditioned": conditioned[fn]} for fn in reports},
        "unreadable": {"candidate": unreadable},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    rows = load_seeded_rows(args.data_dir, args.split, args.seed)
    episodes = plan_episodes(rows, args.limit, args.conditioned_frac)
    candidate_dir = args.candidate_dir
    candidate_dir.mkdir(parents=True, exist_ok=True)

    n_free = sum(not e["restrictions"] for e in episodes)
    n_cond = len(episodes) - n_free
    todo = [e for e in episodes
            if not (candidate_dir / episode_trace_name(e["row"]["restaurant_id"], e["restrictions"])).exists()]
    print(f"plan: {len(episodes)} episodes ({n_free} free + {n_cond} conditioned; "
          f"{args.split} split, seed {args.seed}, limit {args.limit}, "
          f"conditioned-frac {args.conditioned_frac}); "
          f"{len(episodes) - len(todo)} candidates exist, {len(todo)} to run")
    if args.list:
        for e in episodes:
            r = e["row"]
            name = episode_trace_name(r["restaurant_id"], e["restrictions"])
            done = "done" if (candidate_dir / name).exists() else "todo"
            diet = ", ".join(e["restrictions"]) if e["restrictions"] else "-"
            print(f"  [{done}] {name}  {r['name']}, {r['city']} ({r.get('country', '?')})  diet=[{diet}]")
        return

    if todo:
        produce_candidates(args, episodes, todo, candidate_dir)
    else:
        print("all candidates already exist -- skipping the run, scoring what's on disk")

    report = run_paired(args, candidate_dir) if args.reference else run_self_report(args, candidate_dir)
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote JSON report: {args.json}")


if __name__ == "__main__":
    main()
