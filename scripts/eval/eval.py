"""WS-G: PRODUCE + SCORE an eval-split candidate set (the whole harness, one script).

This is the v2 merge of scripts/eval_split.py (produce candidates + score) and
scripts/eval_menu.py (the pure scorer). Everything it needs now comes from
`corpus.sqlite` via src/corpus.py -- no loose files:

  * The eval PLAN is the SAME seed-reproducible free+conditioned mix the training
    corpus uses: iter_restaurants(split="eval") -> seeded shuffle -> plan_episodes
    (reused from build_corpus, with its DIETARY_POOL + --conditioned-frac). So the
    eval set is restriction-FREE episodes plus a conditioned slice drawn from the
    same DIETARY_POOL, over the EVAL-split restaurants, rendered with the STUDENT
    system prompt (what we ship, hence what we must measure).
  * The REFERENCE set is the teacher's eval traces read straight from the DB
    (cx.iter_traces(split="eval")), NOT a --reference directory of files.
  * "FINDABLE" is DERIVED, not a labels.jsonl file (retired): a restaurant is
    findable iff its teacher eval trace found a menu (found=true). Abstention is
    graded against that DB-derived signal.

Candidate and reference join on the TRACE ID (`<rid>` for a free episode,
`<rid>__<diet-slug>` for a conditioned one) -- the candidate trace files are named
`<trace_id>.json`, and the reference traces carry `trace_id`, so a restaurant's
free and conditioned episodes never collide.

RUN ORDERING (important -- `canned` needs a warmed eval cache):
  A frozen (`canned`) run cannot fetch: any tool call whose key is absent from the
  cache returns the canned constant, and any that *would* have needed the network
  scores nothing but abstention. So run these in order:

    1. Build the REFERENCE set AND warm the eval-split cache in one live pass
       (writes teacher eval traces into corpus.sqlite + populates data/cache.sqlite
       with every query/URL -- including the restriction-specific queries):
         uv run python scripts/corpus/build_corpus.py --split eval \
             --conditioned-frac 0.4 --cache-policy live --limit <N>
    2. Score a candidate model against that reference over the frozen cache:
         uv run python scripts/eval/eval.py data/candidates_gemma \
             --model gemma --model-path <ckpt> \
             --cache-policy canned --conditioned-frac 0.4

  Keep --split/--seed/--conditioned-frac/--limit identical between the two so the
  candidate and reference plans (hence the per-episode trace ids) line up.

A candidate episode that hits a genuine cache miss (CacheMiss, e.g. the student
explores a URL/mode the teacher never did under `--cache-policy error`) is NOT
fatal: it is recorded as a failed/empty candidate (final_json=None,
schema_valid=False) and counted, exactly as build_corpus treats a per-episode
exception. Unexpected exceptions still trip MAX_CONSECUTIVE_FAILURES so a broken
key/checkpoint can't silently burn the whole run.

  # plan only, no API/GPU (like build_corpus --list)
  uv run python scripts/eval/eval.py cand/ --model claude --list --limit 100 --conditioned-frac 0.4

  # candidates only (no scoring), self-report self-stats
  uv run python scripts/eval/eval.py cand/ --model claude --self-report

  # candidates + paired scoring (reference from the DB) with a free/conditioned breakdown
  uv run python scripts/eval/eval.py cand/ --model claude

REPORTING. --json writes the permanent record (results/<run-set>/<model>.json; see
results/README.md). Alongside the scores it stamps `run` (throughput, workers, the
full failure list) and `cache` (hits/misses/writes + the derived `hit_rate`) --
without the hit rate a score can't be read honestly, since a model that explored off
the warmed distribution is partly being scored on the cache rather than on itself.
The same numbers go to Weights & Biases iff WANDB_API_KEY is set (project from
WANDB_PROJECT, default 'menu-eval'; --wandb-name labels the model, --no-wandb opts
out). W&B is NEVER fatal: a missing package or a failed init degrades to console-only.

Requires BRAVE_API_KEY (search backend is built even when frozen) and, for
--model claude, ANTHROPIC_API_KEY (repo-root .env).
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Nested-script path convention (see CLAUDE.md / notes/v2_rebuild_plan.md): shared
# modules in src/, the Claude loop in src/claude/. The vLLM/Gemma runner paths are
# added lazily inside build_runner.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "claude"))

import jsonschema  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# The seeded episode planner + DIETARY_POOL live in src/episodes.py -- reused (not
# reimplemented) so the eval plan is IDENTICAL to how the train corpus is planned
# (same seeded free+conditioned mix, same DIETARY_POOL).
from episodes import (  # noqa: E402
    DIETARY_POOL,  # noqa: F401 -- re-exported for callers/tests
    MAX_CONSECUTIVE_FAILURES,
    plan_episodes,
    seeded_order,
)
from backends import preflight_browser  # noqa: E402
from cache import Cache, CacheMiss  # noqa: E402
from claude_agent import MODEL_ID as CLAUDE_MODEL_ID  # noqa: E402
from claude_agent import run_episode as claude_run_episode  # noqa: E402
from corpus import VALID_SPLITS, open_corpus, trace_id_for  # noqa: E402
from eval_metrics import (  # noqa: E402
    abstention_outcome,
    aggregate,
    aggregate_self_reports,
    score_episode,
    self_report,
)
from prompts import build_system_prompt  # noqa: E402
from schema import MENU_SCHEMA, extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

GEMMA_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_WANDB_PROJECT = "menu-eval"
PROGRESS_EVERY = 25  # aggregate progress line every N completed episodes


# ---------------------------------------------------------------------------
# Trace-id / filename contract (the candidate<->reference join key)
# ---------------------------------------------------------------------------
def episode_trace_id(rid: str, restrictions: list[str]) -> str:
    """Canonical trace id for an episode: free -> '<rid>', conditioned ->
    '<rid>__<diet-slug>' (corpus.trace_id_for). == the reference trace_id, so
    candidates and references join on it."""
    return trace_id_for(rid, restrictions or None)


def candidate_filename(rid: str, restrictions: list[str]) -> str:
    """Candidate trace filename: '<trace_id>.json'."""
    return f"{episode_trace_id(rid, restrictions)}.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("candidate_dir", type=Path,
                        help="directory to write candidate trace JSON files into (one per episode)")
    parser.add_argument("--model", choices=["claude", "gemma", "vllm"], required=True,
                        help="which runner produces the candidates. 'vllm' drives an "
                             "OpenAI-compatible vLLM server (teacher / tool-parser models); "
                             "'gemma' + --gemma-vllm-base-url serves the student via vLLM.")
    parser.add_argument("--model-path", default=None,
                        help="gemma: a local merged HF checkpoint dir (a fine-tuned student) to "
                             "load instead of the base model; claude: an optional model-id override")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vllm: OpenAI-compatible base URL of the vLLM server")
    parser.add_argument("--served-model-name", default=None,
                        help="vllm: served model name on the vLLM server (--served-model-name at "
                             "serve time). Also the Gemma completions model when --gemma-vllm-base-url is set.")
    parser.add_argument("--gemma-vllm-base-url", default=None,
                        help="gemma: serve the student via a vLLM /v1/completions server at this URL "
                             "(fast + concurrent) instead of loading HF weights locally. Renders with "
                             "our own template/parser (agent.generate_turn's vLLM path).")
    parser.add_argument("--adapter-path", default=None,
                        help="gemma: load a LoRA adapter dir on top of the (4-bit) base model, "
                             "instead of a fully-merged --model-path checkpoint. Evaluates the "
                             "adapter without materializing/pulling the ~15GB merged model.")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                        help="corpus.sqlite (eval plan + reference traces; default data/corpus.sqlite)")
    parser.add_argument("--self-report", action="store_true",
                        help="score candidates ALONE (validity / found-rate / size self-stats); "
                             "skip paired scoring even if the corpus has eval reference traces")
    parser.add_argument("--limit", type=int, default=None,
                        help="TOTAL episode budget (free + conditioned); default: one free "
                             "episode per eval-split restaurant")
    parser.add_argument("--conditioned-frac", type=float, default=0.4,
                        help="conditioned share of the episode budget (default 0.4; mirror the "
                             "value used to build the reference set)")
    parser.add_argument("--split", default="eval", choices=list(VALID_SPLITS),
                        help="which split to run (default eval)")
    parser.add_argument("--seed", type=int, default=42,
                        help="selection-order seed (default 42; keep it == the reference build)")
    parser.add_argument("--cache-policy", choices=["live", "canned", "error"], default="canned",
                        help="cache miss policy (default canned = frozen/reproducible eval)")
    parser.add_argument("--cache-path", default=str(REPO_ROOT / "data" / "cache.sqlite"),
                        help="sqlite cache path (default data/cache.sqlite)")
    parser.add_argument("--workers", type=int, default=None,
                        help="thread-pool size (default 3 for claude; FORCED to 1 for gemma -- a "
                             "single-GPU model is not thread-safe; 16 for vLLM paths)")
    parser.add_argument("--per-episode", action="store_true",
                        help="print one scored line per episode")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the full report as JSON (the eval report.json)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="disable Weights & Biases logging even when WANDB_API_KEY is set. "
                             "By default the eval logs to W&B iff WANDB_API_KEY is in the "
                             "environment (project from WANDB_PROJECT, default "
                             f"{DEFAULT_WANDB_PROJECT!r}); without the key it is a no-op.")
    parser.add_argument("--wandb-name", default=None,
                        help="W&B run name (default: $WANDB_NAME, else wandb's own). Use it to "
                             "label the model under test, e.g. 'teacher-qwen235b' / 'gemma-base'.")
    parser.add_argument("--list", action="store_true",
                        help="print the planned episodes and exit (no API/GPU calls)")
    args = parser.parse_args(argv)

    # Concurrency: a local single-GPU HF gemma runner isn't thread-safe (forced to
    # 1 worker). vLLM (server) runners -- '--model vllm' or gemma via
    # --gemma-vllm-base-url -- ARE concurrent, so default them high.
    local_gemma = args.model == "gemma" and not args.gemma_vllm_base_url
    is_vllm = args.model == "vllm" or (args.model == "gemma" and args.gemma_vllm_base_url)
    if args.workers is None:
        args.workers = 1 if local_gemma else (16 if is_vllm else 3)
    elif local_gemma and args.workers != 1:
        print("[warn] local gemma runner is single-GPU / not thread-safe -- forcing --workers 1")
        args.workers = 1
    return args


def model_label(args) -> str:
    """The `model` field stamped into each candidate trace (checkpoint or base id)."""
    if args.model == "gemma":
        return args.model_path or GEMMA_MODEL_ID
    if args.model == "vllm":
        return args.served_model_name or "vllm"
    return args.model_path or CLAUDE_MODEL_ID


# ---------------------------------------------------------------------------
# eval -> weights lineage: the scored checkpoint's meta.json (§6 of the v2 plan)
# ---------------------------------------------------------------------------
def read_checkpoint_meta(model_path) -> dict | None:
    """Best-effort read of a checkpoint's meta.json for eval->weights traceability.

    Looks next to the checkpoint dir and one level up (the v2 layout stores
    meta.json beside the adapter/ dir). Returns the parsed meta dict, or None if
    absent/unreadable -- callers tolerate absence.
    """
    if not model_path:
        return None
    p = Path(model_path)
    for cand in (p / "meta.json", p.parent / "meta.json"):
        try:
            if cand.is_file():
                return json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def checkpoint_lineage(args) -> dict:
    """{model, label, path, run_id, md5, meta} for the scored checkpoint.

    `run_id` + `md5` are pulled from the checkpoint's meta.json when present (md5 is
    the checkpoint's own md5 if the meta records one -- tolerate absence -> null);
    the full meta is stashed so every recorded hash (base_ref/dataset) travels with
    the report. Read whenever --model-path is a local checkpoint dir (harmless for a
    claude model-id override: read_checkpoint_meta just finds no file).
    """
    meta = read_checkpoint_meta(args.model_path)
    return {
        "model": args.model,
        "label": model_label(args),
        "path": args.model_path,
        "run_id": meta.get("run_id") if meta else None,
        "md5": meta.get("md5") if meta else None,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# W&B telemetry (same contract as build_corpus: key-gated and NEVER fatal)
# ---------------------------------------------------------------------------
def init_wandb(args, n_todo: int):
    """Start a W&B run iff WANDB_API_KEY is set and --no-wandb was not passed.

    Returns the run (or None). NEVER fatal: a missing wandb package or a failed init
    degrades to console-only logging -- a metered eval on a $12/hr pod must not die
    because telemetry is misconfigured. Reads WANDB_PROJECT / WANDB_ENTITY from the
    environment (the caller exports them), defaulting the project only.
    """
    if args.no_wandb or not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] WANDB_API_KEY is set but the wandb package is not installed "
              "(pip install wandb); logging to console only")
        return None
    lineage = checkpoint_lineage(args)
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            name=args.wandb_name or os.environ.get("WANDB_NAME") or None,
            job_type="eval",
            config={
                "model": args.model, "model_id": model_label(args),
                "model_path": args.model_path, "adapter_path": args.adapter_path,
                "base_url": args.base_url, "gemma_vllm_base_url": args.gemma_vllm_base_url,
                "served_model_name": args.served_model_name,
                "split": args.split, "limit": args.limit, "seed": args.seed,
                "conditioned_frac": args.conditioned_frac, "workers": args.workers,
                "cache_policy": args.cache_policy, "cache_path": args.cache_path,
                "prompt_variant": "student", "n_todo": n_todo,
                "checkpoint_run_id": lineage["run_id"], "checkpoint_md5": lineage["md5"],
            },
        )
    except Exception as exc:  # noqa: BLE001 -- telemetry must never break the eval
        print(f"[wandb] init failed ({exc!r}); logging to console only")
        return None
    print(f"[wandb] logging to {run.url}")
    return run


def wandb_log(run, metrics: dict, step: int | None = None) -> None:
    """Log to W&B, swallowing telemetry errors (a hiccup must not kill the eval)."""
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed at step {step} ({exc!r}); continuing")


def wandb_summarize(run, report: dict) -> None:
    """Flatten the finished report's headline numbers into W&B summary + a final log.

    Everything the report holds that is a scalar we care about comparing across the
    three models: per-slice aggregates, abstention buckets, and the cache hit rate
    (the whole point of tracking it -- a low hit rate means the model explored off
    the warmed distribution and its score is partly a cache artifact).
    """
    if run is None:
        return
    flat: dict[str, float] = {}
    for slice_name, agg in (report.get("aggregate") or {}).items():
        for k, v in (agg or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[f"{slice_name}/{k}"] = v
    for k, v in (report.get("abstention") or {}).items():
        flat[f"abstention/{k}"] = v
    for k, v in (report.get("cache") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[f"cache/{k}"] = v
    for k, v in (report.get("run") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[f"run/{k}"] = v
    wandb_log(run, flat)
    try:
        run.summary.update(flat)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] summary update failed ({exc!r}); continuing")


def cache_report(cache) -> dict:
    """cache.stats() + the derived hit rate (None when no lookups happened).

    `hit_rate` is over LOOKUPS (hits / (hits + misses)) -- writes are a subset of
    misses under a live policy, so counting them would double-count.
    """
    stats = dict(cache.stats())
    lookups = stats["hits"] + stats["misses"]
    stats["lookups"] = lookups
    stats["hit_rate"] = (stats["hits"] / lookups) if lookups else None
    return stats


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

    if args.model == "vllm":
        # OpenAI-compatible vLLM server (teacher / tool-parser models). No torch.
        sys.path.insert(0, str(REPO_ROOT / "src" / "serving"))
        from openai_agent import build_client  # noqa: E402
        from openai_agent import run_episode as vllm_run_episode  # noqa: E402

        client = build_client(args.base_url)
        served = args.served_model_name or "teacher"

        def runner(episode_input, tools, registry, system_prompt):
            final_text, _messages = vllm_run_episode(
                client, served, episode_input, tools, registry, system_prompt
            )
            return final_text

        return runner

    # gemma via a vLLM completions server: render with OUR template, decode on vLLM.
    # Tokenizer-only (no HF weights loaded locally), so it's concurrent + fast.
    if args.gemma_vllm_base_url:
        from transformers import AutoTokenizer  # noqa: E402

        sys.path.insert(0, str(REPO_ROOT / "src" / "gemma"))
        sys.path.insert(0, str(REPO_ROOT / "src" / "serving"))
        from agent import run_episode as gemma_run_episode  # noqa: E402
        from openai_agent import build_client, build_gemma_completions  # noqa: E402

        tokenizer = AutoTokenizer.from_pretrained(args.model_path or GEMMA_MODEL_ID)
        # Pass the tokenizer so max_tokens gets clamped to the server's context window.
        # Without it, a long (tool-heavy) episode 400s and eval scores it FAILED --
        # silently penalising exactly the episodes that gathered the most. See
        # openai_agent.build_gemma_completions.
        vllm_generate = build_gemma_completions(
            build_client(args.gemma_vllm_base_url), args.served_model_name or "gemma-menu",
            tokenizer=tokenizer,
        )

        def runner(episode_input, tools, registry, system_prompt):
            return gemma_run_episode(None, tokenizer, episode_input, tools, registry,
                                     system_prompt, vllm_generate=vllm_generate)

        return runner

    # gemma (local HF): load once, reuse. --model-path points load_model's from_pretrained
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
    tid = episode_trace_id(rid, restrictions)

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
        "trace_id": tid,             # the candidate<->reference join key
        "restaurant_id": rid,
        "restaurant_name": row["name"],
        "episode_input": episode_input,
        "model": label,
        "prompt_variant": "student",  # eval always uses the shipped student prompt
        "dietary_restrictions": restrictions or None,
        "cache_policy": args.cache_policy,
        "final_json": final_json,     # eval_metrics reads the menu from here
        "schema_valid": schema_valid,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    if parse_err:
        trace["parse_error"] = parse_err
    if cache_miss:
        trace["cache_miss"] = True

    # Atomic write so an interrupted run can't leave a torn file the idempotent
    # skip would then treat as done.
    path = candidate_dir / candidate_filename(rid, restrictions)
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
def produce_candidates(args, episodes, todo, candidate_dir, run=None):
    """Run every `todo` episode through the chosen runner; write candidate traces.

    Returns `(run_stats, cache_stats)`: the production-side counters (completed /
    failed / recorded cache-misses / throughput) and the cache hit-rate report, both
    of which get folded into the report JSON and W&B by main().
    """
    if args.model == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is required for --model claude (repo-root .env)")

    label = model_label(args)

    # Fail before episode 1, not on scrape 1 of a long eval: under live/error the
    # tools scrape through the local browser (canned replays only and never launches
    # one). Same discipline as build_corpus/warm_cache/train_grpo.
    if args.cache_policy != "canned":
        browser_error = preflight_browser()
        if browser_error:
            sys.exit(f"browser preflight failed, refusing to start:\n  {browser_error}")

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
    t_start = time.monotonic()

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
                wandb_log(run, {"processed": i, "failed": len(failures),
                                "episode/failed": 1}, step=i)
                continue
            with fail_lock:
                consecutive_failures = 0
            results.append(summary)
            diet = "conditioned" if summary["conditioned"] else "free"
            flag = " [CACHE MISS]" if summary["cache_miss"] else ""
            print(f"[{i}/{len(todo)}] {summary['name']!r} ({diet}): "
                  f"schema_valid={summary['schema_valid']} found={summary['found']} "
                  f"items={summary['items']}{flag}")

            # Live telemetry: running rates + the cache hit rate as it evolves, so a
            # run that is quietly missing the warm shows up early instead of at the end.
            elapsed = time.monotonic() - t_start
            cs = cache_report(cache)
            wandb_log(run, {
                "processed": i, "completed": len(results), "failed": len(failures),
                "pct_done": i / len(todo),
                "eps_per_min": i / elapsed * 60 if elapsed > 0 else 0.0,
                "found_rate": sum(r["found"] for r in results) / len(results),
                "schema_valid_rate": sum(r["schema_valid"] for r in results) / len(results),
                "cache_miss_rate": sum(r["cache_miss"] for r in results) / len(results),
                "cache/hits": cs["hits"], "cache/misses": cs["misses"],
                "cache/writes": cs["writes"],
                "cache/hit_rate": cs["hit_rate"] if cs["hit_rate"] is not None else 0.0,
                "episode/found": int(summary["found"]),
                "episode/schema_valid": int(summary["schema_valid"]),
                "episode/items": summary["items"],
                "episode/conditioned": int(summary["conditioned"]),
            }, step=i)
            if i % PROGRESS_EVERY == 0:
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(todo) - i) / rate if rate > 0 else 0.0
                print(f"[progress] {i}/{len(todo)} ({100 * i / len(todo):.0f}%)  "
                      f"{rate * 60:.1f} eps/min  ETA {eta / 60:.0f}m  |  "
                      f"found={100 * sum(r['found'] for r in results) / len(results):.0f}%  "
                      f"cache hit-rate="
                      f"{100 * (cs['hit_rate'] or 0):.1f}% ({cs['hits']}/{cs['lookups']})")

    elapsed = time.monotonic() - t_start
    n_miss = sum(r["cache_miss"] for r in results)
    cache_stats = cache_report(cache)
    print("\n===== candidate run summary =====")
    print(f"episodes completed: {len(results)}  failed: {len(failures)}  "
          f"cache-misses (recorded empty): {n_miss}  "
          f"skipped (pre-existing): {len(episodes) - len(todo)}")
    for rid, name, err in failures:
        print(f"  FAILED {rid} {name!r}: {err}")
    hr = cache_stats["hit_rate"]
    print(f"cache: hit-rate {'n/a' if hr is None else f'{100 * hr:.1f}%'} "
          f"({cache_stats['hits']} hits / {cache_stats['misses']} misses / "
          f"{cache_stats['writes']} writes, policy={cache_stats['miss_policy']})")
    print(f"wall clock: {elapsed / 60:.1f} min "
          f"({len(todo) / elapsed * 60 if elapsed > 0 else 0:.1f} eps/min)")
    cache.close()

    run_stats = {
        "n_planned": len(episodes),
        "n_todo": len(todo),
        "n_skipped_existing": len(episodes) - len(todo),
        "n_completed": len(results),
        "n_failed": len(failures),
        "n_cache_miss_recorded": n_miss,
        "elapsed_s": elapsed,
        "eps_per_min": (len(todo) / elapsed * 60) if elapsed > 0 else 0.0,
        "workers": args.workers,
        "failures": [{"restaurant_id": rid, "name": name, "error": err}
                     for rid, name, err in failures],
    }
    return run_stats, cache_stats


# ---------------------------------------------------------------------------
# Scoring: load candidate files + DB reference traces, join on trace_id
# ---------------------------------------------------------------------------
def load_candidates(candidate_dir):
    """{trace_id: trace-or-menu obj (None if unreadable)} for a dir of JSON files.

    Keyed on the candidate's stored `trace_id` (falling back to the filename stem):
    a restaurant's free and conditioned episodes have distinct trace ids
    (<rid> vs <rid>__<slug>), so keying on the id never collides the two.
    """
    out, unreadable = {}, []
    for path in sorted(candidate_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            out[path.stem] = None
            unreadable.append(path.stem)
            continue
        tid = obj.get("trace_id") if isinstance(obj, dict) else None
        out[tid or path.stem] = obj
    return out, unreadable


def load_reference(cx, split):
    """{trace_id: reference trace dict} -- the teacher eval traces straight from the
    DB (the reference set; {} if none exist yet)."""
    return {t["trace_id"]: t for t in cx.iter_traces(split=split)}


def menu_of(obj):
    """The scored menu dict from a loaded object: a trace's `final_json` (candidate
    file or DB reference trace) or a bare menu JSON; None passes through as an
    invalid candidate."""
    if isinstance(obj, dict) and "final_json" in obj:
        return obj["final_json"]
    return obj


def is_conditioned(trace_id, obj):
    """Free vs. dietary-conditioned. Prefer the trace's recorded restriction; fall
    back to the trace id's `__<slug>` marker."""
    if isinstance(obj, dict) and "dietary_restrictions" in obj:
        return bool(obj.get("dietary_restrictions"))
    return "__" in trace_id


def score_abstention(cand_menus, findable):
    """Bucket every candidate by abstention_outcome against the DB-derived findable
    signal (findable = the reference trace found a menu). Returns counts."""
    counts = {"correct_find": 0, "correct_abstain": 0, "false_abstain": 0, "false_find": 0}
    for tid, menu in cand_menus.items():
        counts[abstention_outcome(menu, findable[tid])] += 1
    counts["n_labeled"] = sum(v for k, v in counts.items() if k != "n_labeled")
    return counts


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


def print_abstention(counts):
    n_unfindable = counts["correct_abstain"] + counts["false_find"]
    n_findable = counts["correct_find"] + counts["false_abstain"]
    print(f"--- abstention vs DB-derived findable ({counts['n_labeled']} episodes) ---")
    print(f"  correct-abstention (findable=false): {counts['correct_abstain']}/{n_unfindable}"
          f"  (false-find, hallucination risk: {counts['false_find']})")
    print(f"  found when findable:                 {counts['correct_find']}/{n_findable}"
          f"  (false give-ups: {counts['false_abstain']})")


def run_paired(args, reference, candidate_dir):
    """Score candidate files against the DB reference traces, joined on trace_id."""
    cand_objs, cand_unreadable = load_candidates(candidate_dir)

    joined = sorted(set(cand_objs) & set(reference))
    # A reference must at least carry a usable found flag; skip (and count) the rest.
    usable = [tid for tid in joined
              if isinstance(menu_of(reference[tid]), dict) and "found" in menu_of(reference[tid])]
    episodes = {tid: score_episode(menu_of(cand_objs[tid]), menu_of(reference[tid])) for tid in usable}
    conditioned = {tid: is_conditioned(tid, cand_objs[tid]) for tid in usable}
    # "findable" is DERIVED, not a labels.jsonl file (retired): the teacher eval
    # trace's own found flag is the ground truth for whether a menu exists.
    findable = {tid: bool(reference[tid].get("found")) for tid in usable}
    abstention = score_abstention({tid: menu_of(cand_objs[tid]) for tid in usable}, findable)

    all_scores = list(episodes.values())
    free_scores = [episodes[tid] for tid in usable if not conditioned[tid]]
    cond_scores = [episodes[tid] for tid in usable if conditioned[tid]]
    agg_all, agg_free, agg_cond = (aggregate(all_scores), aggregate(free_scores), aggregate(cond_scores))

    print("===== WS-G eval report (candidate vs reference) =====")
    print(f"model: {args.model} ({model_label(args)})   split: {args.split}   "
          f"seed: {args.seed}   conditioned-frac: {args.conditioned_frac}")
    print(f"candidate: {candidate_dir} ({len(cand_objs)} files, {len(cand_unreadable)} unreadable)")
    print(f"reference: {args.corpus} ({len(reference)} eval traces, from the DB)")
    print(f"joined on trace_id: {len(joined)}  scored: {len(usable)} "
          f"(skipped {len(joined) - len(usable)} unusable references; "
          f"candidate-only: {len(cand_objs) - len(joined)}, reference-only: {len(reference) - len(joined)})")
    if args.per_episode:
        for tid in usable:
            s = episodes[tid]
            tag = "cond" if conditioned[tid] else "free"
            print(f"  [{tag}] {tid}  valid={int(s['schema_valid'])} found_ok={int(s['found_correct'])} "
                  f"P={_fmt(s['precision'])} R={_fmt(s['recall'])} F1={_fmt(s['f1'])} "
                  f"price={_fmt(s['price_agreement'])} "
                  f"items={s['n_candidate_items']}/{s['n_reference_items']} (matched {s['n_matched']})")
    _print_block("all", agg_all)
    _print_block("free", agg_free)
    _print_block("conditioned", agg_cond)
    print_abstention(abstention)

    return {
        "mode": "paired", "model": args.model, "model_id": model_label(args),
        "checkpoint": checkpoint_lineage(args),
        "split": args.split, "seed": args.seed, "conditioned_frac": args.conditioned_frac,
        "corpus": str(args.corpus), "cache_policy": args.cache_policy,
        "candidate_dir": str(candidate_dir), "n_reference_traces": len(reference),
        "aggregate": {"all": agg_all, "free": agg_free, "conditioned": agg_cond},
        "abstention": abstention,
        "episodes": {tid: {**episodes[tid], "conditioned": conditioned[tid],
                           "findable": findable[tid]} for tid in usable},
        "unreadable": {"candidate": cand_unreadable},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_self_report(args, candidate_dir):
    cand_objs, unreadable = load_candidates(candidate_dir)
    reports = {tid: self_report(menu_of(obj)) for tid, obj in sorted(cand_objs.items())}
    conditioned = {tid: is_conditioned(tid, cand_objs[tid]) for tid in reports}

    agg_all = aggregate_self_reports(list(reports.values()))
    agg_free = aggregate_self_reports([reports[tid] for tid in reports if not conditioned[tid]])
    agg_cond = aggregate_self_reports([reports[tid] for tid in reports if conditioned[tid]])

    print("===== WS-G self-report (no reference) =====")
    print(f"model: {args.model} ({model_label(args)})   split: {args.split}")
    print(f"candidate: {candidate_dir} ({len(cand_objs)} files, {len(unreadable)} unreadable)")
    if args.per_episode:
        for tid, r in reports.items():
            tag = "cond" if conditioned[tid] else "free"
            print(f"  [{tag}] {tid}  valid={int(r['schema_valid'])} found={r['found']} "
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
        "checkpoint": checkpoint_lineage(args),
        "split": args.split, "candidate_dir": str(candidate_dir),
        "corpus": str(args.corpus), "cache_policy": args.cache_policy,
        "aggregate": {"all": agg_all, "free": agg_free, "conditioned": agg_cond},
        "episodes": {tid: {**reports[tid], "conditioned": conditioned[tid]} for tid in reports},
        "unreadable": {"candidate": unreadable},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_seeded_rows(cx, split, seed):
    """The seeded, prefix-stable restaurant order for a split: iter (sorted by id),
    one seeded shuffle -- the same order v1 produced, now from the DB instead of
    restaurants.jsonl + splits.json."""
    return seeded_order(cx.iter_restaurants(split=split), seed)


def main(argv=None):
    args = parse_args(argv)
    candidate_dir = args.candidate_dir
    candidate_dir.mkdir(parents=True, exist_ok=True)

    with open_corpus(args.corpus, create=False) as cx:
        rows = load_seeded_rows(cx, args.split, args.seed)
        episodes = plan_episodes(rows, args.limit, args.conditioned_frac)

        n_free = sum(not e["restrictions"] for e in episodes)
        n_cond = len(episodes) - n_free
        todo = [e for e in episodes
                if not (candidate_dir / candidate_filename(e["row"]["restaurant_id"], e["restrictions"])).exists()]
        print(f"plan: {len(episodes)} episodes ({n_free} free + {n_cond} conditioned; "
              f"{args.split} split, seed {args.seed}, limit {args.limit}, "
              f"conditioned-frac {args.conditioned_frac}); "
              f"{len(episodes) - len(todo)} candidates exist, {len(todo)} to run")
        if args.list:
            for e in episodes:
                r = e["row"]
                tid = episode_trace_id(r["restaurant_id"], e["restrictions"])
                done = "done" if (candidate_dir / f"{tid}.json").exists() else "todo"
                diet = ", ".join(e["restrictions"]) if e["restrictions"] else "-"
                print(f"  [{done}] {tid}  {r['name']}, {r['city']}  diet=[{diet}]")
            return

        run = init_wandb(args, len(todo))
        run_stats = cache_stats = None
        if todo:
            run_stats, cache_stats = produce_candidates(args, episodes, todo,
                                                        candidate_dir, run=run)
        else:
            print("all candidates already exist -- skipping the run, scoring what's on disk")

        # Reference = teacher eval traces from the DB. Fall back to self-report when
        # there are none (or --self-report is set).
        reference = {} if args.self_report else load_reference(cx, args.split)
        if reference:
            report = run_paired(args, reference, candidate_dir)
        else:
            if not args.self_report:
                print(f"[note] no {args.split}-split reference traces in {args.corpus} "
                      "-- self-report only (build them with scripts/corpus/build_corpus.py)")
            report = run_self_report(args, candidate_dir)

    # Production-side facts (throughput, failures, cache hit rate) travel WITH the
    # scores -- a report whose cache hit rate is unknown can't be read honestly.
    report["run"] = run_stats
    report["cache"] = cache_stats
    wandb_summarize(run, report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote JSON report: {args.json}")
        if run is not None:
            try:
                run.save(str(args.json), policy="now")
            except Exception as exc:  # noqa: BLE001 -- artifact upload is best-effort
                print(f"[wandb] report upload failed ({exc!r}); the local file is authoritative")
    if run is not None:
        try:
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] finish failed ({exc!r})")


if __name__ == "__main__":
    main()
