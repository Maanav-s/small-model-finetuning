"""Compute menu-grounding for EVERY trace (+ the not-found -> found sibling map).

Grounding = the fraction of a trace's own menu item names that literally appear
in the text it scraped/searched (its OWN captured tool output). It is the
anti-hallucination signal, computed for every trace rather than only the
reject-triggering siblings:

  grounding high  -> the model really read those items off the page it scraped.
  grounding low   -> the model may have INVENTED items, or extracted a DIFFERENT
                     restaurant's page (same-name, wrong-city) -- worth a look.

The check is against each trace's OWN captured tool output (cached at scrape
time), so it is robust to the live site having changed since.

Also records, for EVERY trace, its "siblings" -- the other slices of the SAME
restaurant_id (the free trace and any dietary-conditioned slices). The review UI
shows them side-by-side so the reviewer can compare menus across slices, confirm
a menu is demonstrably extractable when one slice abstained, and catch identity
mismatches (a slice that scraped a same-name restaurant in a different city).

Output: data/review/grounding.json
  { "<trace filename>": {restaurant_id, restaurant_name, episode_input,
    restrictions, found, schema_valid, n_items, grounding (null if no menu),
    unmatched_items, source_url, siblings} }
Consumed by viz/review.py (the list %-grounded stat + the sibling panel). The
per-item "where in the scrape" match is recomputed live in the app.

  uv run python scripts/audit_grounding.py
  uv run python scripts/audit_grounding.py --threshold 0.5   # flag < 50% grounded
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRACES = REPO / "data" / "traces"
OUT = REPO / "data" / "review" / "grounding.json"

# Cap the stored miss list so a wildly-hallucinated menu can't bloat the file.
UNMATCHED_CAP = 25


def tool_text(trace: dict) -> str:
    """Concatenate every tool_result payload (search + scrape) in a trace, lowercased.

    Anthropic format: tool_result blocks live in user-role messages; `content` is
    a string or a list of {type:text, text:...} blocks.
    """
    parts = []
    for m in trace.get("messages", []):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            c = block.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(b.get("text", ""))
    return "\n".join(parts).lower()


def item_names(menu) -> list[str]:
    names = []
    for section in menu or []:
        for it in section.get("items", []) or []:
            n = (it.get("name") or "").strip()
            if n:
                names.append(n)
    return names


def grounding(names: list[str], haystack: str) -> tuple[float, list[str]]:
    """Fraction of item names that appear (case-insensitive substring) in the tool
    text, plus the list of items that did NOT match (the suspicious ones).

    Matches viz/review.py's `_match_menu_items` normalization exactly (lowercase,
    collapse runs of whitespace to one space, strip, substring) so the aggregate
    here and the per-item "where" the app renders always agree.
    """
    if not names:
        return 0.0, []
    misses = []
    for n in names:
        core = re.sub(r"\s+", " ", n.lower()).strip()
        if core and core in haystack:
            continue
        misses.append(n)
    matched = len(names) - len(misses)
    return matched / len(names), misses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="flag traces grounded below this fraction in the report (default 0.5)")
    args = ap.parse_args()
    if not TRACES.exists():
        sys.exit(f"no traces at {TRACES}")

    traces: dict[str, dict] = {}
    by_rid: dict[str, list[str]] = defaultdict(list)
    for p in sorted(TRACES.glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        traces[p.name] = t
        by_rid[t.get("restaurant_id")].append(p.name)

    out: dict[str, dict] = {}
    for fn, t in traces.items():
        fj = t.get("final_json") or {}
        found = bool(fj.get("found"))
        names = item_names(fj.get("menu"))
        if names:
            score, misses = grounding(names, tool_text(t))
            g = round(score, 3)
        else:
            g, misses = None, []
        rid = t.get("restaurant_id")
        # Every other slice of the same restaurant (free + dietary-conditioned).
        siblings = sorted(s for s in by_rid.get(rid, []) if s != fn)
        out[fn] = {
            "restaurant_id": rid,
            "restaurant_name": t.get("restaurant_name"),
            "episode_input": t.get("episode_input"),
            "restrictions": t.get("dietary_restrictions"),
            "found": found,
            "schema_valid": bool(t.get("schema_valid")),
            "n_items": len(names),
            "grounding": g,
            "unmatched_items": misses[:UNMATCHED_CAP],
            "source_url": fj.get("source_url"),
            "siblings": siblings,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    scored = [v for v in out.values() if v["grounding"] is not None]
    print(f"grounding computed for {len(scored)}/{len(out)} traces (those with a menu) -> {OUT.relative_to(REPO)}")
    if scored:
        gs = sorted(v["grounding"] for v in scored)
        print(f"grounding: min {gs[0]:.2f}  median {gs[len(gs)//2]:.2f}  max {gs[-1]:.2f}")
    flagged = sorted((v for v in scored if v["grounding"] < args.threshold),
                     key=lambda v: v["grounding"])
    print(f"\n-- SUSPICIOUS (grounding < {args.threshold}: may be hallucinated / wrong page) --")
    if not flagged:
        print("  none")
    for v in flagged:
        r = f" [{', '.join(v['restrictions'])}]" if v["restrictions"] else " [free]"
        print(f"  {v['grounding']:.0%}  {v['restaurant_name']}{r}  ({v['n_items']} items)  src={v['source_url']}")
        if v["unmatched_items"]:
            print(f"      unmatched: {v['unmatched_items'][:12]}")
    n_sib = sum(1 for v in out.values() if v["siblings"])
    print(f"\ntraces with >=1 sibling slice (same restaurant): {n_sib}")


if __name__ == "__main__":
    main()
