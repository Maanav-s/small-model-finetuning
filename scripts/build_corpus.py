"""WS-C1/C3: run the Claude teacher over train restaurants -> SFT traces + cache.

Each episode drives claude_agent.run_episode over the shared live tools wrapped
in the content-addressed cache (miss_policy="live"), so the run populates
data/cache.sqlite as a side effect and records one trace per EPISODE to
data/traces/ (contract 1.5 in notes/phase2_plan.md).

Two kinds of episode make up one mixed corpus (see notes/phase2_plan.md, WS-C3):

  * restriction-FREE episodes -- no dietary restrictions, full menu target. The
    base agentic skill. Trace file: data/traces/<restaurant_id>.json.
  * restriction-CONDITIONED episodes -- a dietary restriction (sampled from
    DIETARY_POOL) is slotted into the system prompt, so the target is the
    diet-filtered menu. Teaches the filtering skill. Because a dietary
    restriction is a per-episode INPUT that changes the target, it is visible to
    BOTH teacher and student (it is NOT distilled away -- see CLAUDE.md /
    notes/phase2_plan.md). Trace file: data/traces/<restaurant_id>__<slug>.json.

--conditioned-frac sets the conditioned share of the --limit episode budget
(default 0.0 = pure free; the sized WS-C3 run uses 0.4 for a 3:2 free:conditioned
split). Conditioned episodes REUSE the front of the seeded restaurant order, so
(a) they are guaranteed cache hits against the warmed corpus and (b) the same
restaurant gets both a free trace and a filtered trace -- the contrastive pair
the student needs to learn "condition the filter on the restriction". Selection
order is seeded (sort by restaurant_id, one seeded shuffle); existing traces are
skipped (idempotent / resumable).

  uv run python scripts/build_corpus.py --limit 100                       # free-only pilot
  uv run python scripts/build_corpus.py --limit 1000 --conditioned-frac 0.4  # 600 free + 400 conditioned
  uv run python scripts/build_corpus.py --limit 1000 --conditioned-frac 0.4 --list  # plan only, no API

The episode input is "{name}, {city}" (TEST_RESTAURANT's shape); the restriction,
when present, lives in the system prompt (never the input).
Requires ANTHROPIC_API_KEY + BRAVE_API_KEY (repo-root .env).
"""

import argparse
import json
import os
import random
import re
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
from prompts import build_system_prompt, normalize_dietary_restrictions  # noqa: E402
from schema import MENU_SCHEMA, extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

PROMPT_VARIANT = "teacher"  # default: what generates SFT data (--prompt-variant to override)
# Abort the run if this many episodes fail in a row -- a broken key/tool should
# not burn API budget across the whole selection.
MAX_CONSECUTIVE_FAILURES = 5

# Dietary restrictions sampled (in this fixed, rotated order) across the
# conditioned slice. Spread across the axes -- diet type, single allergens,
# religious, and a couple of combinations -- so the student learns to generalize
# to unseen phrasings rather than memorizing one label. Each entry is fed to
# build_system_prompt as-is (a string; comma-separated entries become multiple
# ANDed restrictions via normalize_dietary_restrictions).
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


def restriction_slug(restrictions: list[str]) -> str:
    """Filesystem-safe tag for a normalized restriction list (trace filename)."""
    slug = re.sub(r"[^a-z0-9]+", "-", "-".join(restrictions).lower()).strip("-")
    return slug[:60] or "diet"


def episode_trace_name(rid: str, restrictions: list[str]) -> str:
    """Trace filename for an episode. Free -> <rid>.json (back-compat with the
    pre-conditioned run); conditioned -> <rid>__<slug>.json (distinct key so a
    restaurant's free and filtered traces never collide)."""
    return f"{rid}.json" if not restrictions else f"{rid}__{restriction_slug(restrictions)}.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="TOTAL episode budget (free + conditioned); default: one free "
                             "episode per restaurant in the split")
    parser.add_argument("--conditioned-frac", type=float, default=0.0,
                        help="fraction of the episode budget that is dietary-restriction "
                             "conditioned (default 0.0 = pure free; 0.4 gives a 3:2 free:"
                             "conditioned split). Conditioned episodes reuse the front of the "
                             "seeded order (warm-cache hits + contrastive free/filtered pairs).")
    parser.add_argument("--split", choices=["train", "eval"], default="train",
                        help="which split to run (default train; eval only for harness debugging)")
    parser.add_argument("--workers", type=int, default=3,
                        help="thread-pool size (one pooled Chromium per worker; keep <=4 on the 15GB box)")
    parser.add_argument("--model", default=MODEL_ID, help=f"teacher model id (default {MODEL_ID})")
    parser.add_argument("--prompt-variant", choices=["teacher", "student"], default=PROMPT_VARIANT,
                        help=f"system-prompt variant (default {PROMPT_VARIANT}; 'student' is for "
                             "prompt-sufficiency smoke tests, not SFT-corpus generation)")
    parser.add_argument("--cache-policy", choices=["live", "canned", "error"], default="live",
                        help="cache miss policy (default live: fetch+store, the populate pass)")
    parser.add_argument("--cache-path", default=str(REPO_ROOT / "data" / "cache.sqlite"))
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--seed", type=int, default=42, help="selection-order seed (default 42; keep fixed)")
    parser.add_argument("--list", action="store_true",
                        help="print the selected restaurants and exit (no API calls)")
    return parser.parse_args()


def load_seeded_rows(data_dir: Path, split: str, seed: int) -> list[dict]:
    """The seeded, prefix-stable restaurant order: sort by id, one shuffle."""
    rows = [json.loads(line) for line in open(data_dir / "restaurants.jsonl", encoding="utf-8")]
    splits = json.load(open(data_dir / "splits.json", encoding="utf-8"))
    rows = [r for r in rows if splits.get(r["restaurant_id"]) == split]
    rows.sort(key=lambda r: r["restaurant_id"])
    random.Random(seed).shuffle(rows)
    return rows


def plan_episodes(rows: list[dict], total: int | None, conditioned_frac: float) -> list[dict]:
    """Plan the mixed corpus: a list of {"row", "restrictions"} episodes.

    total (--limit) is the whole episode budget; conditioned_frac of it is
    dietary-conditioned. Free episodes take the first n_free restaurants of the
    seeded order (restrictions=[]). Conditioned episodes REUSE the front of that
    same order (row i uses rows[i % len(rows)]) so they hit the warm cache and
    pair contrastively with the free traces, rotating through DIETARY_POOL for
    variety. Deduped by trace filename so a wrap-around can't plan the same
    (restaurant, restriction) twice.
    """
    if not 0.0 <= conditioned_frac <= 1.0:
        raise ValueError(f"--conditioned-frac must be in [0, 1], got {conditioned_frac}")
    n = total if total is not None else len(rows)
    n_cond = round(n * conditioned_frac)
    n_free = n - n_cond

    episodes, seen = [], set()

    def add(row, restrictions):
        name = episode_trace_name(row["restaurant_id"], restrictions)
        if name not in seen:
            seen.add(name)
            episodes.append({"row": row, "restrictions": restrictions})

    for row in rows[:n_free]:
        add(row, [])
    for i in range(n_cond):
        row = rows[i % len(rows)]
        restrictions = normalize_dietary_restrictions(DIETARY_POOL[i % len(DIETARY_POOL)])
        add(row, restrictions)
    return episodes


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


def run_one(client, episode, tools, registry, system_prompt, args, cache, traces_dir: Path) -> dict:
    """One episode -> one trace file. Returns a small summary dict."""
    row = episode["row"]
    restrictions = episode["restrictions"]  # normalized list; [] == free
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
        "prompt_variant": args.prompt_variant,
        # None for a free episode, the restriction phrase list for a conditioned
        # one -- the target-defining input the student must see (see CLAUDE.md).
        "dietary_restrictions": restrictions or None,
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
    path = traces_dir / episode_trace_name(rid, restrictions)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

    found = bool(final_json and final_json.get("found"))
    n_items = sum(len(s.get("items", [])) for s in (final_json or {}).get("menu", []) or [])
    return {"rid": rid, "name": row["name"], "schema_valid": schema_valid,
            "found": found, "tool_calls": n_calls, "items": n_items,
            "conditioned": bool(restrictions)}


def main():
    args = parse_args()
    rows = load_seeded_rows(args.data_dir, args.split, args.seed)
    episodes = plan_episodes(rows, args.limit, args.conditioned_frac)
    traces_dir = args.data_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    n_free = sum(not e["restrictions"] for e in episodes)
    n_cond = len(episodes) - n_free
    todo = [e for e in episodes
            if not (traces_dir / episode_trace_name(e["row"]["restaurant_id"], e["restrictions"])).exists()]
    print(f"plan: {len(episodes)} episodes ({n_free} free + {n_cond} conditioned; "
          f"{args.split} split, seed {args.seed}, limit {args.limit}, "
          f"conditioned-frac {args.conditioned_frac}); "
          f"{len(episodes) - len(todo)} traces exist, {len(todo)} to run")
    if args.list:
        for e in episodes:
            r = e["row"]
            name = episode_trace_name(r["restaurant_id"], e["restrictions"])
            done = "done" if (traces_dir / name).exists() else "todo"
            diet = ", ".join(e["restrictions"]) if e["restrictions"] else "-"
            print(f"  [{done}] {name}  {r['name']}, {r['city']} ({r['country']})  diet=[{diet}]")
        return
    if not todo:
        print("nothing to do")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is required (repo-root .env)")

    cache = Cache(args.cache_path, miss_policy=args.cache_policy)
    # Tools/registry are restriction-independent; only the system prompt embeds
    # the restriction, so build the prompts per unique restriction and memoize.
    tools, registry, _ = setup_tools(
        dietary_restrictions=None, variant=args.prompt_variant, cache=cache
    )
    prompt_cache: dict[tuple, str] = {}

    def prompt_for(restrictions: list[str]) -> str:
        key = tuple(restrictions)
        if key not in prompt_cache:
            prompt_cache[key] = build_system_prompt(list(key) or None, variant=args.prompt_variant)
        return prompt_cache[key]

    client = anthropic.Anthropic()

    results, failures = [], []
    consecutive_failures = 0
    fail_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, client, e, tools, registry, prompt_for(e["restrictions"]),
                        args, cache, traces_dir): e
            for e in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            e = futures[fut]
            row = e["row"]
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
            diet = "conditioned" if summary["conditioned"] else "free"
            print(f"[{i}/{len(todo)}] {summary['name']!r} ({diet}): "
                  f"schema_valid={summary['schema_valid']} found={summary['found']} "
                  f"tool_calls={summary['tool_calls']} items={summary['items']}")

    print("\n===== corpus build summary =====")
    print(f"episodes completed: {len(results)}  failed: {len(failures)}  "
          f"skipped (pre-existing): {len(episodes) - len(todo)}")
    if results:
        _report(results, "all")
        cond = [r for r in results if r["conditioned"]]
        free = [r for r in results if not r["conditioned"]]
        if cond and free:
            _report(free, "free")
            _report(cond, "conditioned")
    for rid, name, err in failures:
        print(f"  FAILED {rid} {name!r}: {err}")
    print(f"cache stats: {cache.stats()}")
    cache.close()


def _report(results: list[dict], label: str) -> None:
    """Print schema-valid / found / tool-call / item stats for a result subset."""
    n = len(results)
    n_found = sum(r["found"] for r in results)
    print(f"[{label}] n={n}  "
          f"schema-valid: {sum(r['schema_valid'] for r in results)}/{n} "
          f"({100 * sum(r['schema_valid'] for r in results) / n:.1f}%)  "
          f"found=true: {n_found}/{n} ({100 * n_found / n:.1f}%)  "
          f"mean tool calls: {sum(r['tool_calls'] for r in results) / n:.2f}  "
          f"mean items (found): {sum(r['items'] for r in results if r['found']) / max(1, n_found):.1f}")


if __name__ == "__main__":
    main()
