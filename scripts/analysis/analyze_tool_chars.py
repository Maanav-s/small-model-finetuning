"""READ-ONLY analysis: can the MAX_TOOL_CHARS=75000 scrape cap be safely LOWERED?

The v2 rebuild of the v1 script -- traces now come from `corpus.sqlite` (via
src/corpus.py) instead of `data/traces/*.json`, and the student re-render goes
through the v2 build_sft at scripts/datasets/build_sft.py. Everything else is
verbatim from v1.

Motivation (see CLAUDE.md "Web tools"/"Scrape-result slimming" + notes/phase2_plan.md
WS-I / Part 5): SFT episode token lengths have a long tail (p50~14k, p95~95k,
max~209k) driven by big scrapes x multiple calls. Lowering the scrape cap is a
train==inference-consistent lever (the student sees the same cap at inference), but
it MUST NOT clip real menus. This script measures, over the corpus traces:

  1. scrape-length distribution (per call + per episode total), and how many scrape
     results sit at/near the 75k cap (>= 74900 chars -> raw was even larger).
  2. MENU REACH -- for each FOUND episode, concatenate its scrape results in call
     order and find the FIRST case-insensitive offset of every extracted item name
     and section name. The "deepest matched position" is the offset by which all
     MATCHABLE strings have appeared; the match rate says how much to trust it.
  3. CAP SIMULATION for C in {15k,20k,30k,40k,50k,75k}: (a) %found-episodes whose
     deepest-matched position <= C (menu fully within first C chars); (b) re-cap
     every scrape to C, re-render each episode under the STUDENT prompt via the same
     path scripts/datasets/build_sft.py uses (Gemma tokenizer), and estimate the
     per-episode token-length distribution + how many episodes fall under max_length=32768.
  4. a recommended MAX_TOOL_CHARS.

Nothing is mutated. Numeric findings are written to data/review/tool_chars_report.json.

  uv run python scripts/analysis/analyze_tool_chars.py
  uv run python scripts/analysis/analyze_tool_chars.py --limit 50   # quick smoke
  uv run python scripts/analysis/analyze_tool_chars.py --no-tokens  # skip the tokenizer pass
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules in src/, the Gemma loader in src/gemma/, the v2 SFT render path in
# scripts/datasets/ (nested-script path convention -- see CLAUDE.md / §7).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "gemma"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "datasets"))

from corpus import VALID_SPLITS, open_corpus  # noqa: E402

CANDIDATE_CAPS = [15000, 20000, 30000, 40000, 50000, 75000]
NEAR_CAP_CHARS = 74900   # >= this after the 75k cap => the raw page was clipped
MAX_LENGTH_DEFAULT = 32768  # train_sft.py default max_length


# ---------------------------------------------------------------------------
# Trace parsing (pure, no tokenizer)
# ---------------------------------------------------------------------------
def _result_text(content) -> str:
    """String body of a tool_result block (a str, or a list of text blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def parse_trace(trace: dict) -> dict:
    """Extract the per-episode facts this analysis needs.

    Builds an id->name map from tool_use blocks (so scrape results are told apart
    from web_search results), then walks the conversation in order pairing each
    tool_use with the tool_result that answers it.

    Returns a dict with:
      found            : bool (final_json.found)
      scrapes          : list[str]  scrape_url result texts, IN CALL ORDER
      searches         : list[str]  web_search result texts, in call order
      menu_strings     : list[str]  section names + item names (found episodes)
    """
    msgs = trace.get("messages", [])
    id_to_name: dict[str, str] = {}
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    id_to_name[b.get("id")] = b.get("name")

    scrapes: list[str] = []
    searches: list[str] = []
    for m in msgs:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                continue
            name = id_to_name.get(b.get("tool_use_id"))
            text = _result_text(b.get("content"))
            if name == "scrape_url":
                scrapes.append(text)
            elif name == "web_search":
                searches.append(text)

    fj = trace.get("final_json") or {}
    found = bool(fj.get("found"))
    menu_strings: list[str] = []
    for section in fj.get("menu") or []:
        if not isinstance(section, dict):
            continue
        sname = section.get("section")
        if isinstance(sname, str) and sname.strip():
            menu_strings.append(sname.strip())
        for item in section.get("items") or []:
            if isinstance(item, dict):
                iname = item.get("name")
                if isinstance(iname, str) and iname.strip():
                    menu_strings.append(iname.strip())

    return {
        "found": found,
        "scrapes": scrapes,
        "searches": searches,
        "menu_strings": menu_strings,
    }


# ---------------------------------------------------------------------------
# Menu reach (pure)
# ---------------------------------------------------------------------------
def menu_reach(scrapes: list[str], menu_strings: list[str]) -> dict | None:
    """Deepest-matched-position + match rate for one FOUND episode.

    Two offset notions are computed per matched menu string (section + item names):

      * concat offset  -- first occurrence (end = pos+len) in the scrapes JOINED in
        call order. This is the descriptive "reach" stat item 2 asks for.
      * per-call offset -- because MAX_TOOL_CHARS caps EACH scrape call
        independently, a string survives a per-call cap C iff it appears within the
        first C chars of AT LEAST ONE scrape. So we take the MIN over every scrape
        it appears in of (local pos + len): the smallest per-call cap that RETAINS
        that string. This is the accurate guardrail for lowering the cap.

    deepest_pos      = max concat-end over matched strings.
    deepest_percall  = max per-call-min-end over matched strings = the smallest
                       per-call cap that retains the WHOLE (matchable) menu.
    Returns None if there are no menu strings to match.
    """
    if not menu_strings:
        return None
    lowers = [s.lower() for s in scrapes]
    concat_lower = "\n".join(lowers)
    # offset where each scrape STARTS inside the "\n"-joined concat.
    starts, off = [], 0
    for s in lowers:
        starts.append(off)
        off += len(s) + 1  # +1 for the join "\n"
    total_chars = len("\n".join(scrapes))

    concat_ends: list[int] = []
    percall_ends: list[int] = []
    n_matched = 0
    for raw in menu_strings:
        s = raw.lower()
        slen = len(raw)
        cpos = concat_lower.find(s)
        if cpos < 0:
            continue
        n_matched += 1
        concat_ends.append(cpos + slen)
        # smallest per-call retention offset across every scrape it appears in
        best = None
        for hay, st in zip(lowers, starts):
            lpos = hay.find(s)
            if lpos >= 0:
                end = lpos + slen
                best = end if best is None else min(best, end)
        percall_ends.append(best)
    n = len(menu_strings)
    return {
        "n_strings": n,
        "n_matched": n_matched,
        "match_rate": n_matched / n if n else 0.0,
        "deepest_pos": max(concat_ends) if concat_ends else 0,
        "deepest_percall": max(percall_ends) if percall_ends else 0,
        "total_scrape_chars": total_chars,
    }


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------
def pct(sorted_vals: list, q: float):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def dist(vals: list) -> dict:
    s = sorted(vals)
    if not s:
        return {"n": 0}
    return {
        "n": len(s),
        "min": s[0],
        "p50": pct(s, 0.50),
        "p90": pct(s, 0.90),
        "p95": pct(s, 0.95),
        "p99": pct(s, 0.99),
        "max": s[-1],
        "mean": round(sum(s) / len(s), 1),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_traces(corpus_path: Path, split: str | None, limit: int | None) -> list[dict]:
    """Non-rejected corpus traces (optionally one split), first `limit` in trace_id
    order. Read-only (create=False)."""
    with open_corpus(corpus_path, create=False) as cx:
        traces = cx.iter_traces(split=split, include_rejected=False)
        if limit is not None:
            traces = itertools.islice(traces, limit)
        return list(traces)


def run_token_simulation(traces, caps, max_length):
    """Per-cap per-episode STUDENT-rendered token lengths via the build_sft path.

    Monkeypatches tools.MAX_TOOL_CHARS to each candidate cap so build_sft's
    transform_tool_result (-> tools._cap, which reads the module global) re-caps
    every tool result exactly as inference would, then renders under the student
    prompt with the Gemma tokenizer. Returns {cap: [token_len, ...]} over the
    episodes that re-render cleanly, plus the count that failed to render.

    build_sft is imported defensively from scripts/datasets/ (the v2 location). If
    it (or the gated Gemma tokenizer) is unavailable, the token sim is skipped and
    the rest of the report still prints.
    """
    import tools as tools_mod
    try:
        import build_sft  # scripts/datasets/build_sft.py (v2 SFT render path)
        from build_sft import build_gemma_messages, token_length
    except ImportError as e:
        print(f"  [warn] scripts/datasets/build_sft unavailable ({e}) -- skipping token simulation")
        return None, 0

    try:
        tok = build_sft.load_tokenizer()
    except Exception as e:  # noqa: BLE001 -- gated model / no HF cache: degrade, don't crash
        print(f"  [warn] could not load the Gemma tokenizer ({e}) -- skipping token simulation")
        return None, 0
    if tok is None:
        print("  [warn] Gemma tokenizer unavailable -- skipping token simulation")
        return None, 0

    original_cap = tools_mod.MAX_TOOL_CHARS
    per_cap: dict[int, list[int]] = {c: [] for c in caps}
    render_fail = 0
    sink = io.StringIO()  # swallow tools._cap truncation warnings during renders
    # Pre-filter to traces that build_gemma_messages accepts (once, at the max cap).
    usable = []
    tools_mod.MAX_TOOL_CHARS = max(caps)
    with contextlib.redirect_stdout(sink):
        for trace in traces:
            try:
                build_gemma_messages(trace)
                usable.append(trace)
            except ValueError:
                render_fail += 1
    print(f"  token sim: {len(usable)} episodes render cleanly, {render_fail} skipped")

    for cap in caps:
        tools_mod.MAX_TOOL_CHARS = cap
        with contextlib.redirect_stdout(sink):
            for trace in usable:
                try:
                    messages, _, _ = build_gemma_messages(trace)
                    per_cap[cap].append(token_length(tok, messages))
                except ValueError:
                    pass
        print(f"    cap {cap:>6}: rendered {len(per_cap[cap])} episodes")
    tools_mod.MAX_TOOL_CHARS = original_cap
    return per_cap, render_fail


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                    help="corpus.sqlite (read-only; default data/corpus.sqlite)")
    ap.add_argument("--split", default=None, choices=list(VALID_SPLITS),
                    help="only analyze traces whose restaurant is in this split (default: all)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "review" / "tool_chars_report.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N traces (trace_id order) -- for smoke tests")
    ap.add_argument("--no-tokens", action="store_true",
                    help="skip the (slower) Gemma-tokenizer cap simulation")
    args = ap.parse_args()

    if not args.corpus.is_file():
        sys.exit(f"no corpus at {args.corpus}")
    traces = load_traces(args.corpus, args.split, args.limit)
    if not traces:
        sys.exit(f"no traces in {args.corpus}")

    # --- parse everything once -------------------------------------------
    parsed = [parse_trace(trace) for trace in traces]

    per_scrape_lens: list[int] = []
    near_cap = 0
    per_episode_total: list[int] = []
    scrapes_per_ep: list[int] = []
    n_found = 0
    reach_rows: list[dict] = []

    for p in parsed:
        ep_scrape_lens = [len(s) for s in p["scrapes"]]
        per_scrape_lens.extend(ep_scrape_lens)
        near_cap += sum(1 for L in ep_scrape_lens if L >= NEAR_CAP_CHARS)
        per_episode_total.append(sum(ep_scrape_lens))
        scrapes_per_ep.append(len(ep_scrape_lens))
        if p["found"]:
            n_found += 1
            r = menu_reach(p["scrapes"], p["menu_strings"])
            if r is not None:
                reach_rows.append(r)

    scrape_dist = dist(per_scrape_lens)
    ep_total_dist = dist(per_episode_total)
    deepest_positions = [r["deepest_pos"] for r in reach_rows]          # concat
    deepest_percall = [r["deepest_percall"] for r in reach_rows]        # per-call cap
    deepest_dist = dist(deepest_positions)
    deepest_percall_dist = dist(deepest_percall)
    match_rates = [r["match_rate"] for r in reach_rows]
    overall_match_rate = (sum(r["n_matched"] for r in reach_rows) /
                          sum(r["n_strings"] for r in reach_rows)) if reach_rows else 0.0
    per_ep_match_rate = (sum(match_rates) / len(match_rates)) if match_rates else 0.0

    # --- cap simulation: menu coverage -----------------------------------
    # PRIMARY guardrail = per-call coverage (the cap is per scrape call). The
    # concat-based number is also kept -- it is pessimistic (penalizes a late item
    # only because an EARLIER scrape in the same episode was long) but is the
    # literal item-3a metric.
    n_reach = len(reach_rows)
    coverage = {}
    coverage_concat = {}
    for c in CANDIDATE_CAPS:
        coverage[c] = (sum(1 for d in deepest_percall if d <= c) / n_reach) if n_reach else 0.0
        coverage_concat[c] = (sum(1 for d in deepest_positions if d <= c) / n_reach) if n_reach else 0.0

    # --- cap simulation: token lengths -----------------------------------
    token_sim = None
    if not args.no_tokens:
        print("\nRunning Gemma-tokenizer cap simulation (student re-render)...")
        per_cap, render_fail = run_token_simulation(traces, CANDIDATE_CAPS, MAX_LENGTH_DEFAULT)
        if per_cap is not None:
            token_sim = {}
            for c in CANDIDATE_CAPS:
                lens = per_cap[c]
                d = dist(lens)
                under = sum(1 for L in lens if L <= MAX_LENGTH_DEFAULT)
                d["frac_under_maxlen"] = round(under / len(lens), 4) if lens else 0.0
                token_sim[c] = d

    # --- build report ----------------------------------------------------
    report = {
        "corpus": str(args.corpus),
        "split": args.split,
        "n_traces": len(parsed),
        "n_found_episodes": n_found,
        "near_cap_chars_threshold": NEAR_CAP_CHARS,
        "current_cap": 75000,
        "max_length_default": MAX_LENGTH_DEFAULT,
        "scrape_calls_total": len(per_scrape_lens),
        "scrape_length_chars": scrape_dist,
        "scrapes_at_or_near_cap": near_cap,
        "scrapes_per_episode": dist(scrapes_per_ep),
        "per_episode_total_scrape_chars": ep_total_dist,
        "menu_reach": {
            "n_found_with_menu": n_reach,
            "overall_item_match_rate": round(overall_match_rate, 4),
            "mean_per_episode_match_rate": round(per_ep_match_rate, 4),
            "deepest_matched_position_chars_concat": deepest_dist,
            "deepest_matched_offset_chars_percall": deepest_percall_dist,
        },
        "cap_simulation": {
            str(c): {
                "menu_fully_covered_frac_percall": round(coverage[c], 4),
                "menu_fully_covered_frac_concat": round(coverage_concat[c], 4),
                **({"tokens": token_sim[c]} if token_sim else {}),
            }
            for c in CANDIDATE_CAPS
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- print clean report ----------------------------------------------
    _print_report(report)
    print(f"\nnumeric report -> {args.out}")


def _fmt(d: dict) -> str:
    if d.get("n", 0) == 0:
        return "(no data)"
    return (f"n={d['n']} min={d['min']} p50={d['p50']} p90={d['p90']} "
            f"p95={d['p95']} p99={d['p99']} max={d['max']} mean={d['mean']}")


def _print_report(r: dict) -> None:
    print("\n" + "=" * 72)
    print("MAX_TOOL_CHARS cap analysis")
    print("=" * 72)
    print(f"traces: {r['n_traces']}   found episodes: {r['n_found_episodes']}   "
          f"scrape calls: {r['scrape_calls_total']}")

    print("\n-- 1. Scrape-length distribution (chars) --")
    print(f"  per scrape call : {_fmt(r['scrape_length_chars'])}")
    print(f"  at/near {r['near_cap_chars_threshold']}+ cap (clipped): "
          f"{r['scrapes_at_or_near_cap']} / {r['scrape_calls_total']} scrape calls "
          f"({100*r['scrapes_at_or_near_cap']/max(1,r['scrape_calls_total']):.1f}%)")
    print(f"  scrapes/episode : {_fmt(r['scrapes_per_episode'])}")
    print(f"  episode TOTAL   : {_fmt(r['per_episode_total_scrape_chars'])}")

    mr = r["menu_reach"]
    print("\n-- 2. Menu reach (found episodes) --")
    print(f"  found w/ menu   : {mr['n_found_with_menu']}")
    print(f"  item/section name match rate: overall {100*mr['overall_item_match_rate']:.1f}%"
          f"  |  mean per-episode {100*mr['mean_per_episode_match_rate']:.1f}%")
    print(f"  deepest pos, CONCAT  (chars): {_fmt(mr['deepest_matched_position_chars_concat'])}")
    print(f"  deepest off, PER-CALL(chars): {_fmt(mr['deepest_matched_offset_chars_percall'])}")
    print("  (per-call = smallest per-scrape cap that retains a string; the cap is"
          " per call,\n   so per-call is the accurate guardrail; concat is pessimistic.)")

    print("\n-- 3. Cap simulation --")
    print("  %menus covered: PER-CALL is the real guardrail (cap is per scrape call);"
          "\n  CONCAT is the pessimistic literal-item-3a number.")
    has_tok = any("tokens" in v for v in r["cap_simulation"].values())
    if has_tok:
        print(f"  {'cap':>7} | {'%menus cov':>10} | {'%menus cov':>10} | {'%eps under':>11} | "
              f"{'p95 tok':>8} | {'max tok':>8}")
        print(f"  {'':>7} | {'PER-CALL':>10} | {'CONCAT':>10} | {'32768':>11} | "
              f"{'':>8} | {'':>8}")
        print("  " + "-" * 68)
        for c in CANDIDATE_CAPS:
            cs = r["cap_simulation"][str(c)]
            tk = cs.get("tokens", {})
            print(f"  {c:>7} | {100*cs['menu_fully_covered_frac_percall']:>9.2f}% | "
                  f"{100*cs['menu_fully_covered_frac_concat']:>9.2f}% | "
                  f"{100*tk.get('frac_under_maxlen',0):>10.2f}% | "
                  f"{tk.get('p95','-'):>8} | {tk.get('max','-'):>8}")
    else:
        print(f"  {'cap':>7} | {'%menus PER-CALL':>15} | {'%menus CONCAT':>14}")
        for c in CANDIDATE_CAPS:
            cs = r["cap_simulation"][str(c)]
            print(f"  {c:>7} | {100*cs['menu_fully_covered_frac_percall']:>14.2f}% | "
                  f"{100*cs['menu_fully_covered_frac_concat']:>13.2f}%")


if __name__ == "__main__":
    main()
