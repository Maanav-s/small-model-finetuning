"""WS-E: mine the corpus traces for query templates, the URL funnel, and outcomes.

Reads every teacher trace out of `corpus.sqlite` (via src/corpus.py) -- the v2
rebuild of the v1 script that walked `data/traces/*.json`. The only change is the
INPUT SOURCE (DB instead of loose files); every aggregate below is verbatim from
v1, because the DB stores the same Anthropic content-block `messages` shape. It
prints a readable text report:

  1. query TEMPLATES -- each query with the restaurant's name/city substituted
     by {name}/{city}, ranked by frequency (these drive the WS-C2 bulk warm);
  2. scraped-URL DOMAINS, with the delivery-app/aggregator subgroup flagged
     (the source-selection behavior context distillation must carry over);
  3. scrape MODES (direct vs browser, from the tool_use inputs);
  4. OUTCOME splits -- found-rate / mean tool calls by whether the episode ever
     scraped a delivery domain, plus the tool-calls-per-episode distribution;
  5. FALLBACK signals -- web.archive.org scrapes and "(scrape failed ...)"
     results, with the domains that fail most.

  uv run python scripts/analysis/analyze_queries.py                          # full report
  uv run python scripts/analysis/analyze_queries.py --model claude-sonnet-5  # one teacher
  uv run python scripts/analysis/analyze_queries.py --split sft              # one split
  uv run python scripts/analysis/analyze_queries.py --json out/agg.json      # + raw aggregates

Pure local and READ-ONLY on the corpus: it opens corpus.sqlite (create=False), reads
non-rejected traces, and writes nothing except the optional --json path (which is
refused inside data/). Re-run after a corpus rebuild to re-tune the WS-C2 warm.
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules live in src/ (nested-script path convention -- see CLAUDE.md /
# notes/v2_rebuild_plan.md §7).
sys.path.insert(0, str(REPO_ROOT / "src"))

from cache import norm_query  # noqa: E402
from corpus import VALID_SPLITS, open_corpus  # noqa: E402

# Delivery-app / aggregator hosts the teacher prompt steers away from
# (cf. _SOURCE_GUIDANCE in src/prompts.py). Matched as substrings of the
# hostname so subdomains ("order.ubereats.com") and ccTLDs count too.
DELIVERY_DOMAINS = (
    "doordash", "ubereats", "grubhub", "yelp", "deliveroo", "skipthedishes",
    "menupix", "restaurantguru", "tripadvisor", "zomato",
)

SCRAPE_FAILURE_PREFIX = "(scrape failed"
ARCHIVE_DOMAIN = "web.archive.org"

# Leading name tokens too generic to templatize on their own ("The Cheesecake
# Factory": a lone "the" in a query is not a name mention).
_STOPWORDS = frozenset({"the", "a", "an", "and", "of"})


# ---------------------------------------------------------------------------
# Templatization -- substitute {name}/{city} into a normalized query.
# ---------------------------------------------------------------------------
def _canon(token: str) -> str:
    """Canonical token for matching: lowercase, punctuation stripped -- so
    "Joe's" in the restaurant name matches both "joe's" and "joes" in a query."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _replace_runs(tokens: list[str], target: list[str], placeholder: str,
                  partial: bool) -> list[str]:
    """Replace contiguous token runs matching a contiguous slice of `target`.

    Greedy longest-match, left to right. A run is accepted when it covers the
    whole target, or (partial only) spans >=2 target tokens, or is the single
    "distinctive" token (first non-stopword) of a multi-token target -- so
    "cheesecake factory", "joes pizza" and a lone "ssamjang" all collapse to
    {name} without a lone generic "pizza" doing the same.
    """
    if not target:
        return tokens
    distinctive = next((t for t in target if t not in _STOPWORDS), target[0])
    out, i = [], 0
    while i < len(tokens):
        best = 0
        canon_i = _canon(tokens[i])
        if canon_i and canon_i in target:
            # longest run of query tokens matching a contiguous target slice
            for start in (j for j, t in enumerate(target) if t == canon_i):
                length = 0
                while (i + length < len(tokens) and start + length < len(target)
                       and _canon(tokens[i + length]) == target[start + length]):
                    length += 1
                best = max(best, length)
        ok = best == len(target) or (partial and (
            best >= 2 or (best == 1 and canon_i == distinctive and canon_i not in _STOPWORDS)))
        if best and ok:
            if not (out and out[-1] == placeholder):  # collapse adjacent repeats
                out.append(placeholder)
            i += best
        else:
            out.append(tokens[i])
            i += 1
    return out


def templatize(query: str, restaurant_name: str, city: str) -> str:
    """Turn a raw search query into a template: "Ssamjang Atlanta menu" ->
    "{name} {city} menu". Name matching is case/punctuation-insensitive and
    accepts partial (multi-token or distinctive-token) mentions; the name is
    substituted first so a city word inside the name isn't clobbered."""
    tokens = norm_query(query).split()
    name_tokens = [c for c in (_canon(t) for t in restaurant_name.split()) if c]
    city_tokens = [c for c in (_canon(t) for t in city.split()) if c]
    tokens = _replace_runs(tokens, name_tokens, "{name}", partial=True)
    tokens = _replace_runs(tokens, city_tokens, "{city}", partial=False)
    return " ".join(tokens)


def city_of(trace: dict) -> str:
    """The trace's city. The DB joins it in directly; fall back to the city half of
    the episode input ("{name}, {city}") for defensiveness."""
    if trace.get("city"):
        return trace["city"]
    name = trace.get("restaurant_name") or ""
    episode_input = trace.get("episode_input") or ""
    if name and episode_input.startswith(f"{name}, "):
        return episode_input[len(name) + 2:]
    return episode_input.rsplit(",", 1)[1].strip() if "," in episode_input else ""


# ---------------------------------------------------------------------------
# Trace parsing -- tool calls + results out of the raw message list.
# ---------------------------------------------------------------------------
def domain_of(url: str) -> str:
    host = urlsplit(url.strip()).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def is_delivery(domain: str) -> bool:
    return any(marker in domain for marker in DELIVERY_DOMAINS)


def _result_text(content) -> str:
    """tool_result content is a string in our traces; tolerate the SDK's
    list-of-blocks form too."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def extract_calls(trace: dict) -> tuple[list[str], list[dict], int]:
    """(queries, scrapes, total tool calls) from the trace's message list, in
    order -- same walk as build_corpus.extract_tool_calls, but scrapes also
    carry the requested `mode` (absent -> the backend default "direct") and
    whether the paired tool_result was a "(scrape failed ...)" sentinel."""
    queries: list[str] = []
    scrapes: list[dict] = []  # {"url", "mode", "failed"}
    by_id: dict[str, dict] = {}  # tool_use_id -> scrape record, to pair results
    n_calls = 0
    for m in trace.get("messages") or []:
        if isinstance(m.get("content"), str):
            continue
        for block in m.get("content") or []:
            if not isinstance(block, dict):
                continue
            if m.get("role") == "assistant" and block.get("type") == "tool_use":
                n_calls += 1
                args = block.get("input") or {}
                if block.get("name") == "web_search":
                    queries.append(args.get("query", ""))
                elif block.get("name") == "scrape_url":
                    rec = {"url": args.get("url", ""),
                           "mode": args.get("mode", "direct"), "failed": False}
                    scrapes.append(rec)
                    if block.get("id"):
                        by_id[block["id"]] = rec
            elif block.get("type") == "tool_result":
                rec = by_id.get(block.get("tool_use_id"))
                if rec is not None:
                    rec["failed"] = _result_text(
                        block.get("content")).startswith(SCRAPE_FAILURE_PREFIX)
    if not queries and not scrapes and not trace.get("messages"):
        # Degenerate trace without messages: fall back to the extracted lists;
        # modes/failures are unknowable there.
        queries = list(trace.get("queries") or [])
        scrapes = [{"url": u, "mode": "unknown", "failed": False}
                   for u in trace.get("urls") or []]
        n_calls = len(queries) + len(scrapes)
    return queries, scrapes, n_calls


def load_traces(corpus_path: Path, model: str | None, split: str | None) -> list[dict]:
    """Non-rejected traces from corpus.sqlite, optionally filtered by teacher model
    id and/or split. Read-only (create=False)."""
    with open_corpus(corpus_path, create=False) as cx:
        traces = []
        for trace in cx.iter_traces(split=split, include_rejected=False):
            if model and trace.get("model") != model:
                continue
            traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def analyze(traces: list[dict]) -> dict:
    """All report aggregates as one plain-JSON dict (what --json emits)."""
    template_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    mode_counts: Counter = Counter()
    failure_domains: Counter = Counter()
    tool_call_dist: Counter = Counter()
    n_queries = n_templatized = n_scrapes = 0
    n_failures = n_archive = 0
    episodes = []  # per-episode outcome rows for the correlation split

    for trace in traces:
        queries, scrapes, n_calls = extract_calls(trace)
        city = city_of(trace)
        name = trace.get("restaurant_name") or ""
        for q in queries:
            template = templatize(q, name, city)
            template_counts[template] += 1
            n_queries += 1
            n_templatized += "{name}" in template
        touched_delivery = False
        for s in scrapes:
            n_scrapes += 1
            dom = domain_of(s["url"])
            domain_counts[dom] += 1
            mode_counts[s["mode"]] += 1
            touched_delivery |= is_delivery(dom)
            n_archive += dom == ARCHIVE_DOMAIN
            if s["failed"]:
                n_failures += 1
                failure_domains[dom] += 1
        tool_call_dist[n_calls] += 1
        final = trace.get("final_json") or {}
        episodes.append({
            "delivery": touched_delivery,
            "found": bool(final.get("found")),
            "schema_valid": bool(trace.get("schema_valid")),
            "tool_calls": n_calls,
        })

    def outcome_split(rows):
        n = len(rows)
        return {
            "episodes": n,
            "found_rate": sum(r["found"] for r in rows) / n if n else None,
            "schema_valid_rate": sum(r["schema_valid"] for r in rows) / n if n else None,
            "mean_tool_calls": sum(r["tool_calls"] for r in rows) / n if n else None,
        }

    delivery_scrapes = sum(c for d, c in domain_counts.items() if is_delivery(d))
    return {
        "n_traces": len(traces),
        "n_queries": n_queries,
        "n_scrapes": n_scrapes,
        "query_templates": [
            {"template": t, "count": c, "pct": 100 * c / n_queries}
            for t, c in template_counts.most_common()
        ],
        "template_coverage": {
            "templatized": n_templatized,
            "total": n_queries,
            "pct": 100 * n_templatized / n_queries if n_queries else None,
        },
        "domains": [
            {"domain": d, "count": c, "pct": 100 * c / n_scrapes,
             "delivery": is_delivery(d)}
            for d, c in domain_counts.most_common()
        ],
        "delivery_share": {
            "scrapes": delivery_scrapes,
            "pct": 100 * delivery_scrapes / n_scrapes if n_scrapes else None,
        },
        "scrape_modes": dict(mode_counts),
        "outcomes": {
            "scraped_delivery": outcome_split([e for e in episodes if e["delivery"]]),
            "no_delivery": outcome_split([e for e in episodes if not e["delivery"]]),
            "all": outcome_split(episodes),
        },
        "tool_call_distribution": {str(k): v for k, v in sorted(tool_call_dist.items())},
        "fallbacks": {
            "archive_org_scrapes": n_archive,
            "scrape_failures": n_failures,
            "failure_domains": [
                {"domain": d, "count": c} for d, c in failure_domains.most_common()
            ],
        },
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _pct(x) -> str:
    return "-" if x is None else f"{x:.1f}%"


def render_report(agg: dict, *, top: int = 25) -> str:
    lines = []
    out = lines.append

    out("\n== 1. Query templates ==")
    out(f"{agg['n_queries']} queries across {agg['n_traces']} traces "
        f"({agg['n_queries'] / max(1, agg['n_traces']):.2f}/episode)")
    for row in agg["query_templates"][:top]:
        out(f"  {row['count']:4d}  {row['pct']:5.1f}%  {row['template']}")
    rest = agg["query_templates"][top:]
    if rest:
        out(f"  ... {len(rest)} more templates ({sum(r['count'] for r in rest)} queries)")
    cov = agg["template_coverage"]
    out(f"coverage: {cov['templatized']}/{cov['total']} queries "
        f"({_pct(cov['pct'])}) contain {{name}}; the rest are NOT covered by "
        f"name-based templates")

    out("\n== 2. Scraped-URL domains ==")
    out(f"{agg['n_scrapes']} scrape calls")
    for row in agg["domains"][:top]:
        flag = "  [delivery/aggregator]" if row["delivery"] else ""
        out(f"  {row['count']:4d}  {row['pct']:5.1f}%  {row['domain']}{flag}")
    rest = agg["domains"][top:]
    if rest:
        out(f"  ... {len(rest)} more domains ({sum(r['count'] for r in rest)} scrapes)")
    share = agg["delivery_share"]
    delivery_rows = [r for r in agg["domains"] if r["delivery"]]
    out(f"delivery/aggregator subgroup: {share['scrapes']} scrapes "
        f"({_pct(share['pct'])} of all scrapes) across {len(delivery_rows)} domains")
    for row in delivery_rows:
        out(f"    {row['count']:4d}  {row['domain']}")

    out("\n== 3. Scrape modes ==")
    for mode, count in sorted(agg["scrape_modes"].items(), key=lambda kv: -kv[1]):
        out(f"  {count:4d}  {100 * count / max(1, agg['n_scrapes']):5.1f}%  {mode}")

    out("\n== 4. Outcomes (split by delivery-domain scraping) ==")
    for label, key in (("scraped a delivery domain", "scraped_delivery"),
                       ("never scraped one", "no_delivery"),
                       ("all episodes", "all")):
        o = agg["outcomes"][key]
        found = _pct(None if o["found_rate"] is None else 100 * o["found_rate"])
        valid = _pct(None if o["schema_valid_rate"] is None else 100 * o["schema_valid_rate"])
        calls = "-" if o["mean_tool_calls"] is None else f"{o['mean_tool_calls']:.2f}"
        out(f"  {label:28s} n={o['episodes']:<4d} found={found}"
            f"  schema_valid={valid}  mean tool calls={calls}")
    out("  tool calls / episode:")
    dist = agg["tool_call_distribution"]
    peak = max(dist.values(), default=1)
    for k, v in dist.items():
        out(f"    {k:>3s}: {'#' * max(1, round(40 * v / peak))} {v}")

    out("\n== 5. Fallback signals ==")
    fb = agg["fallbacks"]
    out(f"  web.archive.org scrapes: {fb['archive_org_scrapes']}")
    out(f"  scrape failures (\"{SCRAPE_FAILURE_PREFIX} ...\"): {fb['scrape_failures']}")
    for row in fb["failure_domains"][:top]:
        out(f"    {row['count']:4d}  {row['domain']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path,
                        default=REPO_ROOT / "data" / "corpus.sqlite",
                        help="corpus.sqlite (read-only; default data/corpus.sqlite)")
    parser.add_argument("--model", default=None,
                        help="only analyze traces from this teacher model id (e.g. claude-sonnet-5)")
    parser.add_argument("--split", default=None, choices=list(VALID_SPLITS),
                        help="only analyze traces whose restaurant is in this split (default: all)")
    parser.add_argument("--json", type=Path, default=None, metavar="PATH",
                        help="also write the raw aggregates as JSON to PATH "
                             "(for warm_cache.py / notebooks)")
    parser.add_argument("--top", type=int, default=25,
                        help="rows to show per ranking (default 25)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.json and (REPO_ROOT / "data") in args.json.resolve().parents:
        sys.exit("--json must not write into data/ (it is read-only for this tool)")
    if not args.corpus.is_file():
        sys.exit(f"no corpus at {args.corpus}")

    traces = load_traces(args.corpus, args.model, args.split)
    header = (f"===== WS-E query analysis: {len(traces)} traces from {args.corpus}"
              + (f" (model={args.model})" if args.model else "")
              + (f" (split={args.split})" if args.split else "")
              + " =====")
    print(header)
    if not traces:
        sys.exit("no traces matched")

    agg = analyze(traces)
    print(render_report(agg, top=args.top))

    if args.json:
        agg["generated_at"] = datetime.now(timezone.utc).isoformat()
        agg["corpus"] = str(args.corpus)
        agg["model_filter"] = args.model
        agg["split_filter"] = args.split
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"\naggregates written to {args.json}")


if __name__ == "__main__":
    main()
