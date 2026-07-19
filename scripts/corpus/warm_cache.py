"""Bulk programmatic cache warm over corpus.sqlite restaurants -- no teacher tokens.

Adapts v1 scripts/warm_cache.py to the v2 store: the selection is read from
corpus.sqlite (corpus.iter_restaurants, filtered by --splits) instead of
data/restaurants.jsonl. For every selected restaurant it issues one or more query
templates through the cached Brave search backend, parses the top-N result URLs
out of each (cached) response, and warms a scrape for each URL (BOTH modes -- mode
is part of the cache key, and the GRPO student may pick either). This is what makes
a frozen (canned) eval + GRPO cache possible beyond the URLs the teacher visited.

BREADTH for the big GRPO warm (plan §9.2): the student's exploration distribution,
not just the teacher's path, must be pre-cached, so:
  * --queries takes MULTIPLE templates ({name}/{city} placeholders), default the
    dominant "{name} {city} menu"; add e.g. "{name} {city}" and "{name} menu".
  * --urls-per-query deepens the URL funnel per search.
  * --modes both warms the auto-scroll "browser" render for EVERY URL, not only
    when the quick "direct" render comes back thin/failed (--modes auto, v1's
    default). `cache.sqlite` may grow to GBs -- that's the intended storage-for-GPU
    trade (plan §9.2).

Caching is NOT reimplemented: the same Cache.wrap over the same backend closures
with the same key fns (norm_query / norm_scrape) as tools.setup_tools, so an agent
later issuing an identical query/URL/mode hits the rows warmed here, and the stored
responses are the raw uncapped strings (MAX_TOOL_CHARS stays a read-time concern).

Idempotent / resumable BY CONSTRUCTION: every call goes through the cache under
miss_policy="live", where an already-stored ok/empty row is a hit that returns
without touching the network -- so re-running after an interrupt re-plays the
finished prefix in milliseconds. --offset / --limit exist only to shard or
smoke-test, not for resume.

  uv run python scripts/corpus/warm_cache.py --dry-run --limit 5    # plan only, no network
  uv run python scripts/corpus/warm_cache.py --splits grpo --limit 100
  uv run python scripts/corpus/warm_cache.py --splits grpo \\
      --queries "{name} {city} menu" "{name} {city}" "{name} menu" --urls-per-query 5 --modes both

Requires BRAVE_API_KEY (repo-root .env); scrape runs locally, no key.
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules live in src/ (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from backends import build_scrape, build_search, close_pool, has_search_key  # noqa: E402
from cache import CANNED, Cache, norm_query, norm_scrape, scrape_status  # noqa: E402
from corpus import VALID_SPLITS, open_corpus  # noqa: E402

# The default query template: the dominant WS-E-mined teacher pattern.
DEFAULT_QUERIES = ["{name} {city} menu"]

DIRECT_MODE, BROWSER_MODE = "direct", "browser"

# "direct" already internally escalates a client-rendered shell to a NO-SCROLL
# browser render (>= backends.DIRECT_MIN_CHARS on success), so a direct result
# below this larger bar is effectively empty or page chrome, not a menu -- the
# signal (under --modes auto) to ALSO warm the auto-scroll "browser" render.
WARM_BROWSER_IF_UNDER = 2000

# Obvious dead ends, not warmed (bot-walled aggregators / login-walled socials).
# Match by registrable suffix (subdomains included).
SKIP_DOMAINS = frozenset({
    "doordash.com", "ubereats.com", "grubhub.com", "seamless.com",
    "postmates.com", "yelp.com", "facebook.com", "instagram.com",
})

# Non-HTML payloads the markdown scrape can't do anything useful with.
SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")

# A cache hit returns in microseconds; anything slower did real network work.
NETWORK_CALL_MIN_S = 0.05

# Abort if this many restaurants fail in a row -- a dead Brave key/network
# should not grind through the whole selection (same pattern as build_corpus).
MAX_CONSECUTIVE_FAILURES = 5

# One search result renders as "[i] title\n    url\n    desc" (backends.
# _format_results); the URL is the line immediately after the [i] header.
_RESULT_URL_RE = re.compile(r"^\[\d+\][^\n]*\n\s+(\S+)", re.MULTILINE)


def parse_splits(text: str) -> list[str]:
    """Parse "sft,grpo,eval" into a validated list of splits."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in VALID_SPLITS:
            raise argparse.ArgumentTypeError(
                f"unknown split {part!r} (valid: {', '.join(VALID_SPLITS)})"
            )
        if part not in out:
            out.append(part)
    if not out:
        raise argparse.ArgumentTypeError("--splits parsed to nothing")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                        help="corpus.sqlite path (default data/corpus.sqlite)")
    parser.add_argument("--splits", type=parse_splits, default=None,
                        help="comma-separated splits to warm (default: all assigned splits "
                             "sft,grpo,eval; unmarked rows are always skipped)")
    parser.add_argument("--queries", nargs="+", default=None, metavar="TEMPLATE",
                        help='query templates with {name}/{city} placeholders (default '
                             '"{name} {city} menu"); pass several for breadth')
    parser.add_argument("--urls-per-query", type=int, default=3,
                        help="warm scrapes for the top N search-result URLs PER query (default 3)")
    parser.add_argument("--modes", choices=["auto", "both"], default="auto",
                        help="auto (default): warm 'browser' only when 'direct' is thin/failed; "
                             "both: warm the 'browser' render for EVERY URL (bigger, deeper cache)")
    parser.add_argument("--limit", type=int, default=None,
                        help="restaurant count (after --offset; default: all)")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many restaurants first (sharding/smoke only -- "
                             "resume is automatic via cache hits)")
    parser.add_argument("--workers", type=int, default=2,
                        help="thread-pool size (one pooled Chromium per worker; one shared "
                             "egress IP -- keep small, default 2)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="per-worker seconds between NETWORK scrapes (cache hits don't "
                             "sleep; default 1.0)")
    parser.add_argument("--cache-path", default=str(REPO_ROOT / "data" / "cache.sqlite"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned queries (and URLs already derivable from "
                             "cached searches) without any network calls")
    return parser.parse_args()


def load_selection(db: Path, splits: list[str], offset: int, limit: int | None) -> list[dict]:
    """Restaurants across the chosen splits, in the deterministic sorted-by-id order
    so --offset/--limit shards are stable. Unmarked rows are excluded (no split)."""
    rows: list[dict] = []
    with open_corpus(db, create=False) as cx:
        for split in splits:
            rows.extend(cx.iter_restaurants(split=split))
    rows.sort(key=lambda r: r["restaurant_id"])
    rows = rows[offset:]
    return rows[:limit] if limit else rows


def render_queries(templates: list[str], row: dict) -> list[str]:
    """Fill {name}/{city} in each template for one restaurant (blank/dupes dropped)."""
    out, seen = [], set()
    for tmpl in templates:
        q = " ".join(tmpl.format(name=row.get("name", ""), city=row.get("city", "")).split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def extract_urls(search_response: str, top_n: int) -> tuple[list[str], list[str]]:
    """Top-N result URLs from a formatted search response -> (keep, skipped).

    Dead ends (SKIP_DOMAINS / SKIP_EXTENSIONS / non-http) land in `skipped`.
    """
    keep, skipped = [], []
    for url in _RESULT_URL_RE.findall(search_response or ""):
        if len(keep) >= top_n:
            break
        parts = urlsplit(url)
        host = parts.netloc.lower().removeprefix("www.")
        is_dead = (
            parts.scheme not in ("http", "https")
            or any(host == d or host.endswith("." + d) for d in SKIP_DOMAINS)
            or parts.path.lower().endswith(SKIP_EXTENSIONS)
        )
        (skipped if is_dead else keep).append(url)
    return keep, skipped


def _scrape(scrape_fn, url: str, mode: str, sleep_s: float) -> str:
    """One cached scrape + a politeness sleep only if it actually hit the network."""
    t0 = time.monotonic()
    result = scrape_fn(url, mode)  # cached; each (url, mode) is a distinct key
    if sleep_s and time.monotonic() - t0 > NETWORK_CALL_MIN_S:
        time.sleep(sleep_s)
    return result


def warm_one(row: dict, search_fn, scrape_fn, queries: list[str], urls_per_query: int,
             sleep_s: float, warm_both: bool) -> dict:
    """Warm one restaurant across every query template.

    For each rendered query: 1 cached search + up to urls_per_query direct scrapes.
    Each direct scrape escalates to a browser render when warm_both (--modes both)
    OR the direct result is thin/failed (WARM_BROWSER_IF_UNDER). URLs seen under an
    earlier template are not re-scraped (the cache would hit anyway; skipping keeps
    the counters honest). Returns a summary dict; the caller aggregates.
    """
    rendered = render_queries(queries, row)
    n_direct = n_browser = scrape_errors = 0
    n_urls = n_skipped = n_no_results = 0
    seen_urls: set[str] = set()

    for query in rendered:
        response = search_fn(query)  # cached: hit is free, miss fetches+stores
        urls, skipped = extract_urls(response, urls_per_query)
        n_skipped += len(skipped)
        if not urls and not skipped:
            n_no_results += 1
        for url in urls:
            if url in seen_urls:
                continue  # already warmed under an earlier template this restaurant
            seen_urls.add(url)
            n_urls += 1
            result = _scrape(scrape_fn, url, DIRECT_MODE, sleep_s)
            n_direct += 1
            direct_failed = scrape_status(result) == "error"
            if direct_failed:
                scrape_errors += 1
            if warm_both or direct_failed or len(result) < WARM_BROWSER_IF_UNDER:
                bresult = _scrape(scrape_fn, url, BROWSER_MODE, sleep_s)
                n_browser += 1
                if scrape_status(bresult) == "error":
                    scrape_errors += 1

    return {
        "rid": row["restaurant_id"], "name": row["name"],
        "queries": len(rendered), "urls": n_urls, "urls_skipped": n_skipped,
        "no_results": n_no_results,
        "scrape_direct": n_direct, "scrape_browser": n_browser,
        "scrape_errors": scrape_errors,
    }


def _no_network(*args, **kwargs):
    """Guard fn for --dry-run: the canned policy never calls through, so any
    call here is a bug (and would be a network call)."""
    raise AssertionError("dry-run must not call a backend")


def dry_run(selection: list[dict], cache_path: str, queries: list[str],
            urls_per_query: int, warm_both: bool) -> None:
    """Print the plan without touching the network.

    Uses a canned-policy view of the cache: already-warmed searches replay their
    recorded response (so their URL plan prints), absent ones return the canned
    constant (URLs unknowable until a live run fetches the search).
    """
    peek = Cache(cache_path, miss_policy="canned")
    search_fn = peek.wrap("search", _no_network, key_fn=norm_query, provider="brave")
    browser_note = "direct + browser" if warm_both else "direct, + browser only if direct is thin"
    for row in selection:
        print(f"[dry-run] {row['name']}, {row['city']} ({row['restaurant_id']})")
        for query in render_queries(queries, row):
            response = search_fn(query)
            print(f"  query: {query!r}")
            if response == CANNED["search"]:
                print("    urls: (search not cached yet -- known after a live run fetches it)")
                continue
            urls, skipped = extract_urls(response, urls_per_query)
            for url in urls:
                print(f"    scrape: {url} ({browser_note})")
            for url in skipped:
                print(f"    skip (dead end): {url}")
    peek.close()


def main():
    args = parse_args()
    splits = args.splits or list(VALID_SPLITS)
    queries = args.queries or list(DEFAULT_QUERIES)
    warm_both = args.modes == "both"
    selection = load_selection(args.db, splits, args.offset, args.limit)
    modes_note = "direct + browser (both)" if warm_both else "direct, + browser only when direct is thin"
    print(f"selection: {len(selection)} restaurants across splits {splits} "
          f"(offset {args.offset}, limit {args.limit}); {len(queries)} quer"
          f"{'y' if len(queries) == 1 else 'ies'} x <= {args.urls_per_query} urls; scrape: {modes_note}")

    if args.dry_run:
        dry_run(selection, args.cache_path, queries, args.urls_per_query, warm_both)
        return
    if not selection:
        print("nothing to do (empty selection -- are splits assigned?)")
        return
    if not has_search_key():
        sys.exit("BRAVE_API_KEY is required (repo-root .env)")
    cache = Cache(args.cache_path, miss_policy="live")

    # Exactly the setup_tools wiring: cache wraps the RAW backend closures with
    # the same key fns the agent runs use, so agent-issued identical queries /
    # url+mode pairs hit the rows warmed here.
    search_fn = cache.wrap("search", build_search(), key_fn=norm_query, provider="brave")
    scrape_fn = cache.wrap("scrape", build_scrape(), key_fn=norm_scrape,
                           status_fn=scrape_status, provider="local")

    t_start = time.monotonic()
    results, failures = [], []
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(warm_one, row, search_fn, scrape_fn, queries,
                        args.urls_per_query, args.sleep, warm_both): row
            for row in selection
        }
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                summary = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad restaurant must not kill the run
                failures.append((row["restaurant_id"], row["name"], repr(exc)))
                consecutive_failures += 1
                print(f"[{i}/{len(selection)}] FAILED {row['name']!r}: {exc!r}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"aborting: {consecutive_failures} consecutive failures")
                    for f in futures:
                        f.cancel()
                    break
                continue
            consecutive_failures = 0
            results.append(summary)
            print(f"[{i}/{len(selection)}] {summary['name']!r}: urls={summary['urls']} "
                  f"(direct={summary['scrape_direct']} browser={summary['scrape_browser']}) "
                  f"skipped={summary['urls_skipped']} scrape_errors={summary['scrape_errors']}"
                  f"{'  (some searches had no results)' if summary['no_results'] else ''}")

    elapsed = time.monotonic() - t_start
    stats = cache.stats()
    print("\n===== cache warm summary =====")
    print(f"restaurants: {len(results)} warmed, {len(failures)} failed, "
          f"{len(selection) - len(results) - len(failures)} not attempted "
          f"({elapsed:.1f}s, {elapsed / max(1, len(results)):.1f}s/restaurant)")
    if results:
        n_direct = sum(r["scrape_direct"] for r in results)
        n_browser = sum(r["scrape_browser"] for r in results)
        print(f"searches issued: {sum(r['queries'] for r in results)}  "
              f"urls planned: {sum(r['urls'] for r in results)}  "
              f"dead ends skipped: {sum(r['urls_skipped'] for r in results)}  "
              f"searches with no results: {sum(r['no_results'] for r in results)}")
        print(f"scrape calls: {n_direct} direct + {n_browser} browser = {n_direct + n_browser} total; "
              f"{100 * n_browser / max(1, n_direct):.0f}% browser rate")
        print(f"scrape calls returning a failure sentinel: "
              f"{sum(r['scrape_errors'] for r in results)} (stored as 'error'; a re-run re-fetches them)")
    print(f"cache: {stats['writes']} entries warmed (writes), {stats['hits']} already cached (hits), "
          f"{stats['misses']} misses")
    for rid, name, err in failures:
        print(f"  FAILED {rid} {name!r}: {err}")
    cache.close()
    close_pool()


if __name__ == "__main__":
    main()
