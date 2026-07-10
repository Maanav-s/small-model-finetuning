"""WS-C2: bulk programmatic cache warm over ALL restaurants -- no Anthropic tokens.

For every row in data/restaurants.jsonl (train + eval), issue the dominant
pilot-mined query pattern "{name} {city} menu" through the cached Brave search
backend, parse the top-N result URLs out of the (cached) response, and warm a
scrape for each URL in BOTH modes ("direct" and "browser" -- mode is part of the
cache key, and the GRPO student may pick either). This is what makes the frozen
(canned) eval + GRPO cache possible beyond the URLs the teacher happened to
visit (notes/phase2_plan.md, WS-C2).

Caching is NOT reimplemented here: the same `Cache.wrap` over the same backend
closures with the same key fns (`norm_query` / `norm_scrape`) as
tools.setup_tools, so an agent later issuing an identical query/URL/mode hits
the rows this script warms, and the stored responses are the raw uncapped
strings (MAX_TOOL_CHARS stays a read-time concern in tools.py).

Idempotent / resumable BY CONSTRUCTION: every call goes through the cache under
miss_policy="live", where an already-stored ok/empty row is a hit that returns
without touching the network (verified in tests/test_cache.py) -- so
re-running after an interrupt re-plays the finished prefix in milliseconds and
only pays for what's missing. 'error' rows (transient scrape failures) are
deliberately misses under "live", so a re-run also self-heals them. --offset /
--limit exist only to shard or smoke-test, not for resume.

Politeness: everything egresses one IP, so keep --workers small (each worker
thread also lazily launches its own pooled Chromium -- see backends.py) and
leave the per-worker --sleep between *network* scrapes on (cache hits don't
sleep, keeping warm re-runs fast).

  uv run python scripts/warm_cache.py --dry-run --limit 5   # plan only, no network
  uv run python scripts/warm_cache.py --limit 100           # partial warm
  uv run python scripts/warm_cache.py                       # all 3500 (WS-C2 proper)

Requires BRAVE_API_KEY (repo-root .env); scrape runs locally, no key.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
# Shared modules live in src/ (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from backends import build_scrape, build_search, close_pool, has_search_key  # noqa: E402
from cache import CANNED, Cache, norm_query, norm_scrape, scrape_status  # noqa: E402

# Warm the quick "direct" render for every kept URL; escalate to the auto-scroll
# "browser" render ONLY when direct comes back failed or too thin to hold a full
# menu. This mirrors the agent's own rule (try direct first, escalate on
# empty/missing -- prompts._TEACHER_GUIDANCE), so we warm the mode the agent will
# actually request: a URL that yields a full menu in "direct" is never re-requested
# in "browser", so blindly warming both ~doubles the slow renders for paths that
# never run. The two modes remain distinct cache entries (norm_scrape keys on
# url+mode). Trade-off: a scroll-lazy menu whose direct render clears the bar but
# is still partial won't get its browser entry warmed here -- the plan defers
# measuring the student's per-mode miss rate and back-filling those (WS-C2 / Part 4).
DIRECT_MODE, BROWSER_MODE = "direct", "browser"

# "direct" already internally escalates a client-rendered shell to a NO-SCROLL
# browser render (>= backends.DIRECT_MIN_CHARS==600 on success), so a direct result
# below this larger bar is effectively empty or just page chrome, not a menu --
# the signal to also warm the auto-scroll "browser" render. Heuristic, tunable;
# an 'error' sentinel always escalates regardless of length.
WARM_BROWSER_IF_UNDER = 2000

# Obvious dead ends, not warmed. The bot-walled aggregators bounce off headless
# Chromium (30s timeout -> an 'error' row that "live" would just re-fetch next
# run), and the login-walled socials serve no menu to an anonymous client. A
# frozen-cache miss on these returns the canned "(page not available)" -- the
# same signal to the student as the failure row warming them would store, minus
# the wasted timeouts. Match by registrable suffix (subdomains included).
SKIP_DOMAINS = frozenset({
    "doordash.com", "ubereats.com", "grubhub.com", "seamless.com",
    "postmates.com", "yelp.com", "facebook.com", "instagram.com",
})

# Non-HTML payloads the markdown scrape can't do anything useful with.
SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")

# A cache hit returns in microseconds; anything slower did real network work.
# Used to sleep only after genuine fetches, so warm re-runs stay fast.
NETWORK_CALL_MIN_S = 0.05

# Abort if this many restaurants fail in a row -- a dead Brave key/network
# should not grind through the whole selection (same pattern as build_corpus).
MAX_CONSECUTIVE_FAILURES = 5

# One search result renders as "[i] title\n    url\n    desc" (backends.
# _format_results); the URL is the line immediately after the [i] header.
_RESULT_URL_RE = re.compile(r"^\[\d+\][^\n]*\n\s+(\S+)", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="restaurant count (after --offset; default: all)")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many restaurants first (sharding/smoke only -- "
                             "resume is automatic via cache hits)")
    parser.add_argument("--workers", type=int, default=2,
                        help="thread-pool size (one pooled Chromium per worker; one shared "
                             "egress IP -- keep small, default 2)")
    parser.add_argument("--top-n", type=int, default=3,
                        help="warm scrapes for the top N search-result URLs (default 3)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="per-worker seconds between NETWORK scrapes (cache hits don't "
                             "sleep; default 1.0)")
    parser.add_argument("--cache-path", default=str(REPO_ROOT / "data" / "cache.sqlite"))
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned queries (and URLs already derivable from "
                             "cached searches) without any network calls")
    return parser.parse_args()


def load_selection(data_dir: Path, offset: int, limit: int | None) -> list[dict]:
    """ALL restaurants (train + eval -- the warm covers both splits), in the
    deterministic sorted-by-id order so --offset/--limit shards are stable."""
    rows = [json.loads(line) for line in open(data_dir / "restaurants.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["restaurant_id"])
    rows = rows[offset:]
    return rows[:limit] if limit else rows


def build_query(row: dict) -> str:
    """The dominant WS-E-mined teacher query pattern: "{name} {city} menu"."""
    return " ".join(part for part in (row.get("name"), row.get("city"), "menu") if part)


def extract_urls(search_response: str, top_n: int) -> tuple[list[str], list[str]]:
    """Top-N result URLs from a formatted search response -> (keep, skipped).

    Parses the "[i] title / url / desc" layout of backends._format_results;
    the no-result strings ("(no search results)", the canned "(no results)")
    contain no [i] headers so they naturally yield nothing. Dead ends
    (SKIP_DOMAINS / SKIP_EXTENSIONS / non-http) land in `skipped`.
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


def warm_one(row: dict, search_fn, scrape_fn, top_n: int, sleep_s: float) -> dict:
    """Warm one restaurant: 1 cached search + up to top_n direct scrapes, each
    escalated to a browser render only when the direct result is thin/failed
    (WARM_BROWSER_IF_UNDER) -- the mode the agent would actually request next.

    Returns a summary dict; counters are aggregated by the caller (no shared state).
    """
    query = build_query(row)
    response = search_fn(query)  # cached: hit is free, miss fetches+stores
    urls, skipped = extract_urls(response, top_n)

    n_direct = n_browser = scrape_errors = 0
    for url in urls:
        result = _scrape(scrape_fn, url, DIRECT_MODE, sleep_s)
        n_direct += 1
        direct_failed = scrape_status(result) == "error"
        if direct_failed:
            scrape_errors += 1
        # Escalate to the auto-scroll render only when direct is failed or too
        # thin to be a menu -- otherwise the browser entry is one the agent never
        # asks for (it keeps the good direct result).
        if direct_failed or len(result) < WARM_BROWSER_IF_UNDER:
            bresult = _scrape(scrape_fn, url, BROWSER_MODE, sleep_s)
            n_browser += 1
            if scrape_status(bresult) == "error":
                scrape_errors += 1
    return {
        "rid": row["restaurant_id"], "name": row["name"], "query": query,
        "urls": len(urls), "urls_skipped": len(skipped),
        "no_results": not urls and not skipped,
        "scrape_direct": n_direct, "scrape_browser": n_browser,
        "scrape_errors": scrape_errors,
    }


def dry_run(selection: list[dict], cache_path: str, top_n: int) -> None:
    """Print the plan without touching the network.

    Uses a canned-policy view of the cache: already-warmed searches replay
    their recorded response (so their URL plan prints), absent ones return the
    canned constant (URLs unknowable until a live run fetches the search).
    """
    peek = Cache(cache_path, miss_policy="canned")
    search_fn = peek.wrap("search", _no_network, key_fn=norm_query, provider="brave")
    for row in selection:
        query = build_query(row)
        response = search_fn(query)
        print(f"[dry-run] {row['name']}, {row['city']} ({row['restaurant_id']})")
        print(f"  query: {query!r}")
        if response == CANNED["search"]:
            print("  urls: (search not cached yet -- known after a live run fetches it)")
            continue
        urls, skipped = extract_urls(response, top_n)
        for url in urls:
            print(f"  scrape: {url} (direct, + browser only if direct is thin)")
        for url in skipped:
            print(f"  skip (dead end): {url}")
    peek.close()


def _no_network(*args, **kwargs):
    """Guard fn for --dry-run: the canned policy never calls through, so any
    call here is a bug (and would be a network call)."""
    raise AssertionError("dry-run must not call a backend")


def main():
    args = parse_args()
    selection = load_selection(args.data_dir, args.offset, args.limit)
    print(f"selection: {len(selection)} restaurants (offset {args.offset}, limit {args.limit}); "
          f"1 search + <= {args.top_n} urls (direct, + browser only when direct is thin)")

    if args.dry_run:
        dry_run(selection, args.cache_path, args.top_n)
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
            pool.submit(warm_one, row, search_fn, scrape_fn, args.top_n, args.sleep): row
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
                  f"{'  (no search results)' if summary['no_results'] else ''}")

    elapsed = time.monotonic() - t_start
    stats = cache.stats()
    print("\n===== cache warm summary =====")
    print(f"restaurants: {len(results)} warmed, {len(failures)} failed, "
          f"{len(selection) - len(results) - len(failures)} not attempted "
          f"({elapsed:.1f}s, {elapsed / max(1, len(results)):.1f}s/restaurant)")
    if results:
        n_direct = sum(r["scrape_direct"] for r in results)
        n_browser = sum(r["scrape_browser"] for r in results)
        print(f"urls planned: {sum(r['urls'] for r in results)}  "
              f"dead ends skipped: {sum(r['urls_skipped'] for r in results)}  "
              f"searches with no results: {sum(r['no_results'] for r in results)}")
        print(f"scrape calls: {n_direct} direct + {n_browser} browser (escalated on thin/failed "
              f"direct) = {n_direct + n_browser} total; "
              f"{100 * n_browser / max(1, n_direct):.0f}% escalation rate")
        print(f"scrape calls returning a failure sentinel: "
              f"{sum(r['scrape_errors'] for r in results)} (stored as 'error'; a re-run re-fetches them)")
    # writes = entries actually warmed this run; hits = already cached (a fully
    # warm re-run is ~all hits -- that's the resumability check).
    print(f"cache: {stats['writes']} entries warmed (writes), {stats['hits']} already cached (hits), "
          f"{stats['misses']} misses")
    for rid, name, err in failures:
        print(f"  FAILED {rid} {name!r}: {err}")
    cache.close()
    close_pool()


if __name__ == "__main__":
    main()
