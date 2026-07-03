"""WS-C1/C3: run the Claude teacher over train restaurants -> SFT traces + cache.

Each episode drives claude_agent.run_episode over the shared live tools wrapped
in the content-addressed cache (miss_policy="live"), so the run populates
data/cache.sqlite as a side effect and records one trace per restaurant to
data/traces/<restaurant_id>.json (contract 1.5 in phase2_plan.md).

Staged per the restructured WS-C: the same script serves the ~100-episode pilot
(WS-C1) and the later sized bulk run (WS-C3) — selection order is seeded and
prefix-stable (sort by restaurant_id, one seeded shuffle), so a --limit 100 run
is always a strict subset of a --limit 1000 run, and existing traces are skipped
(idempotent / resumable).

  uv run python scripts/build_corpus.py --limit 100            # WS-C1 pilot
  uv run python scripts/build_corpus.py --limit 100 --list     # show selection, no API
  uv run python scripts/build_corpus.py                        # all train rows (WS-C3)

Episodes run RESTRICTION-FREE (no dietary restrictions) with the teacher prompt
variant; the episode input is "{name}, {city}" (TEST_RESTAURANT's shape).
Requires ANTHROPIC_API_KEY + BRAVE_API_KEY (repo-root .env).
"""

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Shared modules live in src/, the Claude loop in src/claude/ (flat-import,
# script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "claude"))

import anthropic  # noqa: E402
import jsonschema  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from cache import Cache  # noqa: E402
from claude_agent import MODEL_ID, run_episode  # noqa: E402
from schema import MENU_SCHEMA, extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

PROMPT_VARIANT = "teacher"  # what generates SFT data; the student re-render comes later
# Abort the run if this many episodes fail in a row -- a broken key/tool should
# not burn API budget across the whole selection.
MAX_CONSECUTIVE_FAILURES = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="episode count (prefix of the seeded order; default: all in split)")
    parser.add_argument("--split", choices=["train", "eval"], default="train",
                        help="which split to run (default train; eval only for harness debugging)")
    parser.add_argument("--workers", type=int, default=3,
                        help="thread-pool size (one pooled Chromium per worker; keep <=4 on the 15GB box)")
    parser.add_argument("--model", default=MODEL_ID, help=f"teacher model id (default {MODEL_ID})")
    parser.add_argument("--cache-policy", choices=["live", "canned", "error"], default="live",
                        help="cache miss policy (default live: fetch+store, the populate pass)")
    parser.add_argument("--cache-path", default=str(REPO_ROOT / "data" / "cache.sqlite"))
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--seed", type=int, default=42, help="selection-order seed (default 42; keep fixed)")
    parser.add_argument("--list", action="store_true",
                        help="print the selected restaurants and exit (no API calls)")
    return parser.parse_args()


def load_selection(data_dir: Path, split: str, seed: int, limit: int | None) -> list[dict]:
    """The seeded, prefix-stable episode order: sort by id, one shuffle, slice."""
    rows = [json.loads(line) for line in open(data_dir / "restaurants.jsonl", encoding="utf-8")]
    splits = json.load(open(data_dir / "splits.json", encoding="utf-8"))
    rows = [r for r in rows if splits.get(r["restaurant_id"]) == split]
    rows.sort(key=lambda r: r["restaurant_id"])
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def serialize_content(content):
    """Message content -> JSON-safe: SDK content blocks become dicts."""
    if isinstance(content, str):
        return content
    return [b if isinstance(b, dict) else b.model_dump() for b in content]


def extract_tool_calls(messages) -> tuple[list[str], list[str], int]:
    """(queries, urls, total tool calls) from the raw message list, in order."""
    queries, urls, n_calls = [], [], 0
    for m in messages:
        if m["role"] != "assistant" or isinstance(m["content"], str):
            continue
        for block in m["content"]:
            is_dict = isinstance(block, dict)
            btype = block.get("type") if is_dict else getattr(block, "type", None)
            if btype != "tool_use":
                continue
            n_calls += 1
            name = block["name"] if is_dict else block.name
            args = (block["input"] if is_dict else block.input) or {}
            if name == "web_search":
                queries.append(args.get("query", ""))
            elif name == "scrape_url":
                urls.append(args.get("url", ""))
    return queries, urls, n_calls


def run_one(client, row, tools, registry, system_prompt, args, cache, traces_dir: Path) -> dict:
    """One episode -> one trace file. Returns a small summary dict."""
    rid = row["restaurant_id"]
    episode_input = f"{row['name']}, {row['city']}"
    final_text, messages = run_episode(
        client, episode_input, tools, registry, system_prompt, model=args.model
    )
    final_json, parse_err = extract_json(final_text)
    schema_valid = False
    if final_json is not None:
        try:
            jsonschema.validate(final_json, MENU_SCHEMA)
            schema_valid = True
        except jsonschema.ValidationError:
            pass
    queries, urls, n_calls = extract_tool_calls(messages)

    trace = {
        "restaurant_id": rid,
        "restaurant_name": row["name"],
        "episode_input": episode_input,
        "model": args.model,
        "prompt_variant": PROMPT_VARIANT,
        "dietary_restrictions": None,
        "cache_version": cache.cache_version,
        "messages": [
            {"role": m["role"], "content": serialize_content(m["content"])} for m in messages
        ],
        "queries": queries,
        "urls": urls,
        "final_json": final_json,
        "schema_valid": schema_valid,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    if parse_err:
        trace["parse_error"] = parse_err

    # Atomic write: a killed run must not leave a torn trace that the idempotent
    # skip would then treat as done.
    path = traces_dir / f"{rid}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

    found = bool(final_json and final_json.get("found"))
    n_items = sum(len(s.get("items", [])) for s in (final_json or {}).get("menu", []) or [])
    return {"rid": rid, "name": row["name"], "schema_valid": schema_valid,
            "found": found, "tool_calls": n_calls, "items": n_items}


def main():
    args = parse_args()
    selection = load_selection(args.data_dir, args.split, args.seed, args.limit)
    traces_dir = args.data_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    todo = [r for r in selection if not (traces_dir / f"{r['restaurant_id']}.json").exists()]
    print(f"selection: {len(selection)} {args.split} restaurants "
          f"(seed {args.seed}, limit {args.limit}); {len(selection) - len(todo)} traces exist, "
          f"{len(todo)} to run")
    if args.list:
        for r in selection:
            done = "done" if (traces_dir / f"{r['restaurant_id']}.json").exists() else "todo"
            print(f"  [{done}] {r['restaurant_id']}  {r['name']}, {r['city']} ({r['country']})")
        return
    if not todo:
        print("nothing to do")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is required (repo-root .env)")

    cache = Cache(args.cache_path, miss_policy=args.cache_policy)
    tools, registry, system_prompt = setup_tools(
        offline=False, dietary_restrictions=None, variant=PROMPT_VARIANT, cache=cache
    )
    client = anthropic.Anthropic()

    results, failures = [], []
    consecutive_failures = 0
    fail_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, client, row, tools, registry, system_prompt, args, cache, traces_dir): row
            for row in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                summary = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
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
            print(f"[{i}/{len(todo)}] {summary['name']!r}: schema_valid={summary['schema_valid']} "
                  f"found={summary['found']} tool_calls={summary['tool_calls']} items={summary['items']}")

    print("\n===== corpus build summary =====")
    print(f"episodes completed: {len(results)}  failed: {len(failures)}  "
          f"skipped (pre-existing): {len(selection) - len(todo)}")
    if results:
        n = len(results)
        print(f"schema-valid: {sum(r['schema_valid'] for r in results)}/{n} "
              f"({100 * sum(r['schema_valid'] for r in results) / n:.1f}%)")
        print(f"found=true:   {sum(r['found'] for r in results)}/{n} "
              f"({100 * sum(r['found'] for r in results) / n:.1f}%)")
        print(f"mean tool calls/episode: {sum(r['tool_calls'] for r in results) / n:.2f}")
        print(f"mean items (found only): "
              f"{(sum(r['items'] for r in results if r['found']) / max(1, sum(r['found'] for r in results))):.1f}")
    for rid, name, err in failures:
        print(f"  FAILED {rid} {name!r}: {err}")
    print(f"cache stats: {cache.stats()}")
    cache.close()


if __name__ == "__main__":
    main()
