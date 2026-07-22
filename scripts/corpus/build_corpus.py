"""Run the teacher over sft/eval restaurants -> traces in corpus.sqlite + cache.

Adapts v1 scripts/build_corpus.py to the v2 store. The plan (which restaurants,
which dietary conditioning) is read from corpus.sqlite and the resulting traces
are written back to it via corpus.Corpus.write_trace -- no more per-episode
data/traces/*.json files. Grounding is computed AT WRITE (grounding.grounding_for_trace,
the old audit_grounding.py post-scan folded in) and stored on the trace, and
trace_source is set to 'teacher'.

Two teachers, one trace shape (--teacher):
  * claude (default) -- the working v1 path: src/claude/claude_agent.run_episode
    over the Anthropic API. Its messages are already Anthropic content blocks
    (SDK objects, serialized here).
  * vllm -- src/serving/openai_agent.run_episode over an OpenAI-compatible vLLM
    server (--teacher-base-url / --teacher-model). Its OpenAI-shaped messages are
    NORMALIZED to the SAME canonical Anthropic content-block shape before write, so
    every trace field is identical regardless of which teacher produced it (the
    canonical shape is what build_sft parses and grounding scores).

Two kinds of episode make up one mixed corpus (see notes/phase2_plan.md, WS-C3):
  * restriction-FREE -- no dietary restrictions, full-menu target. trace_id = <rid>.
  * restriction-CONDITIONED -- a restriction (sampled from DIETARY_POOL) is slotted
    into the system prompt, so the target is the diet-filtered menu. Because the
    restriction is a per-episode INPUT that changes the target, it is visible to
    BOTH teacher and student (NOT distilled away). trace_id = <rid>__<slug>.

--conditioned-frac sets the conditioned share of the --limit episode budget
(default 0.0). Conditioned episodes REUSE the front of the seeded restaurant order,
so they hit the warm cache AND pair contrastively with the free traces. Selection
order is seeded (iter_restaurants is rid-sorted, then one seeded shuffle); existing
traces are skipped via cx.has_trace (idempotent / resumable).

GRPO is trace-free (plan §2): --split only accepts sft/eval, never grpo.

  uv run python scripts/corpus/build_corpus.py --limit 100                         # free-only pilot (claude)
  uv run python scripts/corpus/build_corpus.py --limit 1000 --conditioned-frac 0.4 # 600 free + 400 conditioned
  uv run python scripts/corpus/build_corpus.py --limit 1000 --conditioned-frac 0.4 --list  # plan only, no API
  uv run python scripts/corpus/build_corpus.py --teacher vllm --teacher-model gemma-teacher --limit 100

Requires BRAVE_API_KEY (repo-root .env; scrape runs locally), plus ANTHROPIC_API_KEY
(--teacher claude) or a reachable vLLM server (--teacher vllm).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules live in src/; the teacher loops in src/claude and src/serving
# (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "claude"))
sys.path.insert(0, str(REPO_ROOT / "src" / "serving"))

import anthropic  # noqa: E402
import jsonschema  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from backends import preflight_browser  # noqa: E402
from cache import Cache  # noqa: E402
from claude_agent import MODEL_ID, run_episode as claude_run_episode  # noqa: E402
from corpus import open_corpus, trace_id_for  # noqa: E402
from episodes import MAX_CONSECUTIVE_FAILURES, plan_episodes, seeded_order  # noqa: E402
from grounding import grounding_for_trace  # noqa: E402
from openai_agent import build_client as openai_build_client  # noqa: E402
from openai_agent import run_episode as vllm_run_episode  # noqa: E402
from prompts import build_system_prompt  # noqa: E402
from schema import MENU_SCHEMA, extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

PROMPT_VARIANT = "teacher"  # default: what generates SFT data (--prompt-variant to override)
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"

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
    parser.add_argument("--split", choices=["sft", "eval"], default="sft",
                        help="which split to run (default sft; eval also allowed). GRPO is "
                             "trace-free -- 'grpo' is intentionally NOT a choice (plan §2).")
    parser.add_argument("--teacher", choices=["claude", "vllm"], default="claude",
                        help="teacher backend (default claude = Anthropic API; vllm = an "
                             "OpenAI-compatible vLLM server, see --teacher-base-url/-model)")
    parser.add_argument("--teacher-base-url", default=DEFAULT_VLLM_BASE_URL,
                        help=f"vLLM OpenAI-compatible base URL (--teacher vllm; default {DEFAULT_VLLM_BASE_URL})")
    parser.add_argument("--teacher-model", default=None,
                        help="served model name for --teacher vllm (required in that mode)")
    parser.add_argument("--teacher-timeout", type=float, default=900.0,
                        help="per-request wall-clock budget in seconds for --teacher vllm "
                             "(default 900). Sized from DECODE SPEED, not from patience: at "
                             "the 9.5 tok/s measured for 235B-FP8 under --enforce-eager, the "
                             "p90 episode (~4k generated tokens) needs ~425 s and the longest "
                             "seen ~740 s. openai_agent.build_client defaults to 300, which "
                             "would fail a tenth of the corpus as 'timeouts' that are really "
                             "just a slow teacher. Lower it for a fast one.")
    parser.add_argument("--model", default=MODEL_ID,
                        help=f"Claude teacher model id for --teacher claude (default {MODEL_ID})")
    parser.add_argument("--workers", type=int, default=3,
                        help="thread-pool size for network episodes (one pooled Chromium per "
                             "worker; keep <=4 on the 15GB box). Traces are written single-"
                             "threaded on the main thread as episodes complete (SQLite writer).")
    parser.add_argument("--prompt-variant", choices=["teacher", "student"], default=PROMPT_VARIANT,
                        help=f"system-prompt variant (default {PROMPT_VARIANT}; 'student' is for "
                             "prompt-sufficiency smoke tests, not SFT-corpus generation)")
    parser.add_argument("--cache-policy", choices=["live", "canned", "error"], default="live",
                        help="cache miss policy (default live: fetch+store, the populate pass)")
    parser.add_argument("--cache-path", default=str(REPO_ROOT / "data" / "cache.sqlite"))
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                        help="corpus.sqlite path (default data/corpus.sqlite)")
    parser.add_argument("--seed", type=int, default=42, help="selection-order seed (default 42; keep fixed)")
    parser.add_argument("--list", action="store_true",
                        help="print the planned episodes and exit (no API calls)")
    return parser.parse_args()


def load_seeded_rows(cx, split: str, seed: int) -> list[dict]:
    """The seeded, prefix-stable restaurant order: rid-sorted (iter_restaurants),
    then one seeded shuffle -- identical to v1's order given the same rid set."""
    return seeded_order(cx.iter_restaurants(split=split), seed)


# ---------------------------------------------------------------------------
# Message normalization -> the CANONICAL Anthropic content-block trace shape
# ---------------------------------------------------------------------------
def serialize_content(content):
    """Claude message content -> JSON-safe: SDK content blocks become dicts."""
    if isinstance(content, str):
        return content
    return [b if isinstance(b, dict) else b.model_dump() for b in content]


def _as_text(content) -> str:
    """Flatten an OpenAI message `content` (str | content-parts list | None) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def openai_messages_to_anthropic(messages: list[dict]) -> list[dict]:
    """Normalize an OpenAI-shaped trajectory (openai_agent.run_episode) into the
    CANONICAL Anthropic content-block shape build_sft parses + grounding scores.

    - the leading system turn is DROPPED (canonical traces carry no system turn;
      build_sft rebuilds it under the student prompt);
    - a user turn's string content becomes one {type:text} block;
    - an assistant turn's tool_calls become {type:tool_use, id, name, input} blocks
      (arguments JSON-decoded), preceded by its own text if any;
    - each role="tool" result becomes a {type:tool_result, tool_use_id, content}
      block BUNDLED into a single user turn following the assistant that called it
      (mirrors the Claude loop, so a run's shape is teacher-independent).
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue  # canonical trace has no system turn
        if role == "user":
            out.append({"role": "user",
                        "content": [{"type": "text", "text": _as_text(m.get("content"))}]})
        elif role == "assistant":
            blocks: list[dict] = []
            text = m.get("content")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id"),
                               "name": fn.get("name"), "input": args})
            if not blocks:
                blocks.append({"type": "text", "text": text if isinstance(text, str) else ""})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
                     "content": _as_text(m.get("content"))}
            prev = out[-1] if out else None
            if (prev and prev["role"] == "user" and prev["content"]
                    and prev["content"][0].get("type") == "tool_result"):
                prev["content"].append(block)  # bundle consecutive tool results
            else:
                out.append({"role": "user", "content": [block]})
    return out


def extract_tool_calls(messages) -> tuple[list[str], list[str], int]:
    """(queries, urls, total tool calls) from the canonical message list, in order."""
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


def run_teacher(client, teacher: str, episode_input: str, tools, registry,
                system_prompt: str, teacher_model: str) -> tuple[str, list[dict]]:
    """Dispatch one episode to the chosen teacher; return (final_text, canonical
    messages) -- messages already normalized to the Anthropic content-block shape."""
    if teacher == "vllm":
        final_text, raw = vllm_run_episode(
            client, teacher_model, episode_input, tools, registry, system_prompt
        )
        return final_text, openai_messages_to_anthropic(raw)
    final_text, raw = claude_run_episode(
        client, episode_input, tools, registry, system_prompt, model=teacher_model
    )
    messages = [{"role": m["role"], "content": serialize_content(m["content"])} for m in raw]
    return final_text, messages


# ---------------------------------------------------------------------------
# One episode -> one trace dict (NO corpus writes here -- see main; SQLite is a
# single writer, so run_one runs on a worker thread and returns the trace, and the
# MAIN thread does cx.write_trace as futures complete).
# ---------------------------------------------------------------------------
def run_one(client, episode, tools, registry, system_prompt, args, cache, teacher_model) -> dict:
    """Run one episode and BUILD its trace dict (grounding computed here). Returns
    the trace; the caller writes it to the corpus on the main thread."""
    row = episode["row"]
    restrictions = episode["restrictions"]  # normalized list; [] == free
    rid = row["restaurant_id"]
    episode_input = f"{row['name']}, {row['city']}"

    final_text, messages = run_teacher(
        client, args.teacher, episode_input, tools, registry, system_prompt, teacher_model
    )
    final_json, parse_err = extract_json(final_text)
    schema_valid = False
    if final_json is not None:
        try:
            jsonschema.validate(final_json, MENU_SCHEMA)
            schema_valid = True
        except jsonschema.ValidationError:
            pass
    queries, urls, _n_calls = extract_tool_calls(messages)

    trace = {
        "restaurant_id": rid,
        "model": teacher_model,
        "prompt_variant": args.prompt_variant,
        # None for a free episode, the restriction phrase list for a conditioned
        # one -- the target-defining input the student must see (see CLAUDE.md).
        "dietary_restrictions": restrictions or None,
        "trace_source": "teacher",
        "cache_version": cache.cache_version,
        "messages": messages,
        "queries": queries,
        "urls": urls,
        "final_json": final_json,
        "found": bool(final_json and final_json.get("found")),
        "schema_valid": schema_valid,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    if parse_err:
        trace["parse_error"] = parse_err
    # Grounding AT WRITE (folds in the old audit_grounding.py post-scan): fraction
    # of extracted item names that appear in this trace's own captured tool output.
    grounding, unmatched = grounding_for_trace(trace)
    trace["grounding"] = grounding
    trace["unmatched_items"] = unmatched
    return trace


def trace_summary(trace: dict, row: dict) -> dict:
    """Small per-episode summary for the run report (no DB access)."""
    fj = trace.get("final_json") or {}
    return {
        "rid": trace["restaurant_id"], "name": row["name"],
        "schema_valid": trace["schema_valid"], "found": trace["found"],
        "tool_calls": len(trace["queries"]) + len(trace["urls"]),
        "items": sum(len(s.get("items", [])) for s in fj.get("menu", []) or []),
        "conditioned": bool(trace["dietary_restrictions"]),
    }


def main():
    args = parse_args()

    with open_corpus(args.db, create=False) as cx:
        rows = load_seeded_rows(cx, args.split, args.seed)
        if not rows:
            sys.exit(f"no restaurants in split {args.split!r} (harvest + assign splits first?)")
        episodes = plan_episodes(rows, args.limit, args.conditioned_frac)

        n_free = sum(not e["restrictions"] for e in episodes)
        n_cond = len(episodes) - n_free
        todo = [e for e in episodes
                if not cx.has_trace(trace_id_for(e["row"]["restaurant_id"], e["restrictions"] or None))]
        print(f"plan: {len(episodes)} episodes ({n_free} free + {n_cond} conditioned; "
              f"{args.split} split, seed {args.seed}, limit {args.limit}, "
              f"conditioned-frac {args.conditioned_frac}); "
              f"{len(episodes) - len(todo)} traces exist, {len(todo)} to run")

        if args.list:
            for e in episodes:
                r = e["row"]
                tid = trace_id_for(r["restaurant_id"], e["restrictions"] or None)
                done = "done" if cx.has_trace(tid) else "todo"
                diet = ", ".join(e["restrictions"]) if e["restrictions"] else "-"
                print(f"  [{done}] {tid}  {r['name']}, {r['city']}  diet=[{diet}]")
            return
        if not todo:
            print("nothing to do")
            return

        # Resolve the teacher + its client (after --list so plan-only needs no key).
        if args.teacher == "vllm":
            teacher_model = args.teacher_model
            if not teacher_model:
                sys.exit("--teacher vllm requires --teacher-model (the vLLM served model name)")
            if not os.environ.get("BRAVE_API_KEY"):
                sys.exit("BRAVE_API_KEY is required (repo-root .env)")
            client = openai_build_client(args.teacher_base_url, timeout=args.teacher_timeout)
        else:  # claude
            teacher_model = args.model
            if not os.environ.get("ANTHROPIC_API_KEY"):
                sys.exit("ANTHROPIC_API_KEY is required (repo-root .env)")
            if not os.environ.get("BRAVE_API_KEY"):
                sys.exit("BRAVE_API_KEY is required (repo-root .env)")
            client = anthropic.Anthropic()
        cx.set_meta("teacher_model", teacher_model)

        # Fail before the first episode, not after the last. A browser that cannot
        # launch makes EVERY scrape an infra failure, and nothing downstream notices:
        # scrape returns a sentinel rather than raising, so the episode "succeeds"
        # with found=false and the MAX_CONSECUTIVE_FAILURES guard below -- which only
        # counts episodes that RAISE -- can never fire. That is the hole that let a
        # warm run grind six hours at 100% infra (2026-07-20); here the same hole
        # bills a metered teacher pod the whole way. One launch turns it into an exit.
        if args.cache_policy != "canned":  # canned replays only; it never launches a browser
            browser_error = preflight_browser()
            if browser_error:
                sys.exit(f"browser preflight failed, refusing to start:\n  {browser_error}")

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

        results, failures = [], []
        consecutive_failures = 0

        # Network episodes run in the pool; cx.write_trace happens ONLY here on the
        # main thread as each future resolves (corpus.Corpus is a single-writer
        # SQLite connection -- workers must not touch it).
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, client, e, tools, registry, prompt_for(e["restrictions"]),
                            args, cache, teacher_model): e
                for e in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                e = futures[fut]
                row = e["row"]
                try:
                    trace = fut.result()
                    # A trace with no extractable JSON has no training target and
                    # cannot be stored -- traces.final_json is NOT NULL by design, so
                    # a menu-less episode is a FAILED episode, not a row. Enforce that
                    # here (with a legible reason) rather than letting the DB reject it
                    # with a cryptic IntegrityError. The write is INSIDE this try for
                    # the same reason: on 2026-07-21 an unguarded main-thread write of
                    # exactly this None aborted a 2252-episode build after ~50 good
                    # ones. Both paths are idempotent -- no row is written, so a re-run
                    # retries the episode.
                    if trace.get("final_json") is None:
                        reason = trace.get("parse_error") or "no JSON in final turn"
                        raise ValueError(f"teacher emitted no parseable menu JSON ({reason})")
                    cx.write_trace(trace)  # MAIN-THREAD-ONLY write
                except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
                    failures.append((row["restaurant_id"], row["name"], repr(exc)))
                    consecutive_failures += 1
                    print(f"[{i}/{len(todo)}] FAILED {row['name']!r}: {exc!r}")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"aborting: {consecutive_failures} consecutive failures")
                        for f in futures:
                            f.cancel()
                        break
                    continue
                consecutive_failures = 0
                summary = trace_summary(trace, row)
                results.append(summary)
                diet = "conditioned" if summary["conditioned"] else "free"
                print(f"[{i}/{len(todo)}] {summary['name']!r} ({diet}): "
                      f"schema_valid={summary['schema_valid']} found={summary['found']} "
                      f"tool_calls={summary['tool_calls']} items={summary['items']} "
                      f"grounding={trace['grounding']}")

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
        print(f"corpus traces: {cx.trace_count()} total")
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
