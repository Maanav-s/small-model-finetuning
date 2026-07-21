"""Unit tests for scripts/corpus/warm_cache.py (v2: warm over corpus.sqlite restaurants).

No network anywhere: search/scrape are fakes, and the dry-run tests monkeypatch the
backend builders to raise so any call through them fails loudly. The v2 change is the
selection SOURCE: restaurants come from corpus.sqlite (iter_restaurants, filtered by
--splits) instead of data/restaurants.jsonl, and queries are rendered from --queries
TEMPLATES (render_queries) rather than a single build_query. The caching assertions
(Cache.wrap counting fakes; warm_one direct->browser escalation) are UNCHANGED.

Run: uv run python -m pytest tests/test_warm_cache.py -q
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/, the
# script under test in scripts/corpus/ (the v2 location -- NOT scripts/).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "corpus"))

import warm_cache  # noqa: E402
from backends import is_cacheable  # noqa: E402
from cache import Cache, norm_query, norm_scrape, scrape_status  # noqa: E402
from corpus import open_corpus  # noqa: E402

# The one query template the tests warm with (the v2 default).
QUERIES = ["{name} {city} menu"]


def brave_response(*urls: str) -> str:
    """A canned search response in backends._format_results' exact layout."""
    return "\n".join(
        f"[{i}] Result {i} - Menu\n    {url}\n    Some description text."
        for i, url in enumerate(urls, 1)
    )


# ---------------------------------------------------------------------------
# URL extraction from a formatted Brave search response
# ---------------------------------------------------------------------------
class TestExtractUrls:
    def test_top_n_urls_in_order(self):
        resp = brave_response(
            "https://pagliacci.com/menu",
            "https://www.seriouspieseattle.com",
            "https://order.square.site/store/x",
            "https://a-fourth-result.com",
        )
        keep, skipped = warm_cache.extract_urls(resp, top_n=3)
        assert keep == [
            "https://pagliacci.com/menu",
            "https://www.seriouspieseattle.com",
            "https://order.square.site/store/x",
        ]
        assert skipped == []

    def test_dead_ends_skipped_not_counted_against_top_n(self):
        resp = brave_response(
            "https://www.doordash.com/store/x",       # bot-walled aggregator
            "https://menu.example.com/menu.pdf",      # non-HTML payload
            "https://www.yelp.com/biz/x",             # subdomain match via www strip
            "https://own-site.com/menu",
        )
        keep, skipped = warm_cache.extract_urls(resp, top_n=3)
        assert keep == ["https://own-site.com/menu"]
        assert skipped == [
            "https://www.doordash.com/store/x",
            "https://menu.example.com/menu.pdf",
            "https://www.yelp.com/biz/x",
        ]

    def test_subdomains_of_skip_domains_are_skipped(self):
        keep, _ = warm_cache.extract_urls(
            brave_response("https://order.ubereats.com/x"), top_n=3
        )
        assert keep == []

    def test_lookalike_domain_is_kept(self):
        # "notyelp.com" must not be caught by the yelp.com suffix rule.
        keep, skipped = warm_cache.extract_urls(
            brave_response("https://notyelp.com/menu"), top_n=3
        )
        assert keep == ["https://notyelp.com/menu"]
        assert skipped == []

    def test_no_result_strings_yield_nothing(self):
        for resp in ("(no search results)", "(no results)", "", None):
            assert warm_cache.extract_urls(resp, top_n=3) == ([], [])


# ---------------------------------------------------------------------------
# Query templates (render_queries: {name}/{city} placeholders)
# ---------------------------------------------------------------------------
class TestRenderQueries:
    def test_dominant_pattern(self):
        row = {"name": "Ssamjang", "city": "Atlanta"}
        assert warm_cache.render_queries(QUERIES, row) == ["Ssamjang Atlanta menu"]

    def test_missing_city_collapses_cleanly(self):
        assert warm_cache.render_queries(QUERIES, {"name": "Ssamjang", "city": ""}) == \
            ["Ssamjang menu"]

    def test_multiple_templates_dedup_and_order(self):
        row = {"name": "Ssamjang", "city": "Atlanta"}
        out = warm_cache.render_queries(
            ["{name} {city} menu", "{name} menu", "{name} {city} menu"], row
        )
        assert out == ["Ssamjang Atlanta menu", "Ssamjang menu"]  # dupe dropped


# ---------------------------------------------------------------------------
# warm_one: cached search -> direct scrape per URL, browser only when direct is
# thin/failed; keys match setup_tools'
# ---------------------------------------------------------------------------
_FULL = "# full menu " + "x" * warm_cache.WARM_BROWSER_IF_UNDER  # clears the bar
_THIN = "# tiny"                                                 # under the bar


class TestWarmOne:
    def test_full_direct_result_skips_browser(self):
        cache = Cache(":memory:", miss_policy="live")
        search_calls, scrape_calls = [], []

        def fake_search(query):
            search_calls.append(query)
            return brave_response("https://a.com/menu", "https://b.com")

        def fake_scrape(url, mode="direct"):
            scrape_calls.append((url, mode))
            return _FULL

        search_fn = cache.wrap("search", fake_search, key_fn=norm_query, provider="brave")
        scrape_fn = cache.wrap("scrape", fake_scrape, key_fn=norm_scrape,
                               status_fn=scrape_status, provider="local")
        row = {"restaurant_id": "abc", "name": "A", "city": "B"}
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, QUERIES,
                                      urls_per_query=3, sleep_s=0, warm_both=False)

        assert search_calls == ["A B menu"]
        # Direct cleared the bar on both URLs -> no browser escalation.
        assert scrape_calls == [("https://a.com/menu", "direct"), ("https://b.com", "direct")]
        assert summary["scrape_direct"] == 2 and summary["scrape_browser"] == 0
        assert summary["urls"] == 2
        assert summary["infra_errors"] == 0 and summary["site_errors"] == 0

        # Idempotency: a second pass is all cache hits -- zero new backend calls.
        warm_cache.warm_one(row, search_fn, scrape_fn, QUERIES,
                            urls_per_query=3, sleep_s=0, warm_both=False)
        assert len(search_calls) == 1 and len(scrape_calls) == 2
        # The warmed rows answer the SAME keys the agent's setup_tools wiring
        # would compute for an identical query / url+mode.
        assert cache._get("search", norm_query("a b   MENU")) is not None
        assert cache._get("scrape", norm_scrape("https://a.com/menu", "direct")) is not None

    def test_thin_direct_result_escalates_to_browser(self):
        cache = Cache(":memory:", miss_policy="live")
        scrape_calls = []

        def fake_scrape(url, mode="direct"):
            scrape_calls.append((url, mode))
            return _THIN

        search_fn = cache.wrap("search", lambda q: brave_response("https://a.com/menu"),
                               key_fn=norm_query)
        scrape_fn = cache.wrap("scrape", fake_scrape, key_fn=norm_scrape, status_fn=scrape_status)
        row = {"restaurant_id": "abc", "name": "A", "city": "B"}
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, QUERIES,
                                      urls_per_query=1, sleep_s=0, warm_both=False)

        # Thin direct -> escalate; both modes warmed as distinct keys.
        assert scrape_calls == [("https://a.com/menu", "direct"), ("https://a.com/menu", "browser")]
        assert summary["scrape_direct"] == 1 and summary["scrape_browser"] == 1
        assert cache._get("scrape", norm_scrape("https://a.com/menu", "browser")) is not None

    def test_direct_failure_escalates_and_counts_both(self):
        cache = Cache(":memory:", miss_policy="live")
        search_fn = cache.wrap(
            "search", lambda q: brave_response("https://a.com"), key_fn=norm_query
        )
        scrape_fn = cache.wrap(
            "scrape", lambda url, mode="direct": f"(scrape failed for {url} in {mode!r} mode: x)",
            key_fn=norm_scrape, status_fn=scrape_status,
        )
        row = {"restaurant_id": "abc", "name": "A", "city": "B"}
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, QUERIES,
                                      urls_per_query=1, sleep_s=0, warm_both=False)
        # Direct errored -> escalate to browser; browser also errored -> both counted.
        assert summary["scrape_direct"] == 1 and summary["scrape_browser"] == 1
        assert summary["scrape_calls"] == 2
        # A bare "x" reason matches no browser-stack marker -> site, not infra.
        assert summary["site_errors"] == 2 and summary["infra_errors"] == 0

    def test_infra_failure_counted_apart_and_does_not_escalate(self):
        """A dead LOCAL browser must not read as a site refusing us: the URL was
        never fetched, so a row for it would be a fabrication rather than a finding.
        Keeping the two apart is what lets main() abort a run whose browser is
        broken -- one undifferentiated counter let a 100%-infra run go 2418 rows.

        And it must NOT escalate to "browser": the escalation target IS the broken
        component ("direct" already falls back to a browser render internally), so
        trying doubles both the wasted seconds and the failures."""
        cache = Cache(":memory:", miss_policy="live")
        scrape_calls = []

        def fake_scrape(url, mode="direct"):
            scrape_calls.append((url, mode))
            return (f"(scrape failed for {url} in {mode!r} mode: BrowserType.launch: "
                    f"Executable doesn't exist at /nope/chrome-headless-shell.exe)")

        search_fn = cache.wrap(
            "search", lambda q: brave_response("https://a.com"), key_fn=norm_query
        )
        scrape_fn = cache.wrap("scrape", fake_scrape, key_fn=norm_scrape,
                               status_fn=scrape_status)
        row = {"restaurant_id": "abc", "name": "A", "city": "B"}
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, QUERIES,
                                      urls_per_query=1, sleep_s=0, warm_both=False)
        assert scrape_calls == [("https://a.com", "direct")]  # no browser retry
        assert summary["infra_errors"] == 1 and summary["site_errors"] == 0
        assert summary["scrape_browser"] == 0

    def test_infra_failures_are_not_cached(self):
        """store_if=is_cacheable: the run leaves NOTHING behind for a URL it never
        fetched, so a later pass re-fetches instead of replaying a local error."""
        cache = Cache(":memory:", miss_policy="live")
        search_fn = cache.wrap(
            "search", lambda q: brave_response("https://a.com"), key_fn=norm_query
        )
        scrape_fn = cache.wrap(
            "scrape",
            lambda url, mode="direct": (
                f"(scrape failed for {url} in {mode!r} mode: BrowserType.launch: "
                f"Executable doesn't exist at /nope/chrome-headless-shell.exe)"
            ),
            key_fn=norm_scrape, status_fn=scrape_status, store_if=is_cacheable,
        )
        row = {"restaurant_id": "abc", "name": "A", "city": "B"}
        warm_cache.warm_one(row, search_fn, scrape_fn, QUERIES,
                            urls_per_query=1, sleep_s=0, warm_both=False)
        assert cache._get("scrape", norm_scrape("https://a.com", "direct")) is None
        assert cache.stats()["writes"] == 1  # the search row only

    def test_modes_both_warms_browser_even_when_direct_is_full(self):
        cache = Cache(":memory:", miss_policy="live")
        scrape_calls = []

        def fake_scrape(url, mode="direct"):
            scrape_calls.append((url, mode))
            return _FULL  # clears the bar -> under 'auto' this would skip browser

        search_fn = cache.wrap("search", lambda q: brave_response("https://a.com/menu"),
                               key_fn=norm_query)
        scrape_fn = cache.wrap("scrape", fake_scrape, key_fn=norm_scrape, status_fn=scrape_status)
        row = {"restaurant_id": "abc", "name": "A", "city": "B"}
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, QUERIES,
                                      urls_per_query=1, sleep_s=0, warm_both=True)
        assert scrape_calls == [("https://a.com/menu", "direct"), ("https://a.com/menu", "browser")]
        assert summary["scrape_direct"] == 1 and summary["scrape_browser"] == 1


# ---------------------------------------------------------------------------
# Dry-run planning: no network, backends never even built; selection from the DB
# ---------------------------------------------------------------------------
@pytest.fixture
def corpus_path(tmp_path):
    """A temp corpus.sqlite with 2 assigned (sft) restaurants -- warm_cache reads
    iter_restaurants filtered by --splits (unmarked rows are always skipped)."""
    p = tmp_path / "corpus.sqlite"
    with open_corpus(p) as cx:
        cx.upsert_restaurants([
            {"name": "Ssamjang", "city": "Atlanta", "source": "osm", "split": "sft"},
            {"name": "Hamburger Mary's", "city": "Chicago", "source": "osm", "split": "sft"},
        ])
    return p


def _boom(*args, **kwargs):
    raise AssertionError("network backend touched during --dry-run")


class TestDryRun:
    def _run(self, monkeypatch, capsys, corpus_path, cache_path):
        # Any route to the network must fail the test.
        monkeypatch.setattr(warm_cache, "build_search", _boom)
        monkeypatch.setattr(warm_cache, "build_scrape", _boom)
        monkeypatch.setattr(sys, "argv", [
            "warm_cache.py", "--dry-run",
            "--db", str(corpus_path), "--cache-path", str(cache_path),
        ])
        warm_cache.main()
        return capsys.readouterr().out

    def test_cold_cache_prints_queries_and_unknown_urls(self, monkeypatch, capsys, corpus_path):
        out = self._run(monkeypatch, capsys, corpus_path, corpus_path.parent / "warm.sqlite")
        assert "'Ssamjang Atlanta menu'" in out
        assert "Hamburger Mary's Chicago menu" in out  # repr()'d with double quotes
        assert out.count("search not cached yet") == 2

    def test_warm_search_prints_its_scrape_plan(self, monkeypatch, capsys, corpus_path):
        # Pre-warm one search row exactly as a live run would store it.
        cache_path = corpus_path.parent / "warm.sqlite"
        pre = Cache(str(cache_path), miss_policy="live")
        pre.wrap("search", lambda q: brave_response(
            "https://ssamjang.com/menu", "https://www.doordash.com/store/ssamjang"
        ), key_fn=norm_query)("Ssamjang Atlanta menu")
        pre.close()

        out = self._run(monkeypatch, capsys, corpus_path, cache_path)
        assert "scrape: https://ssamjang.com/menu (direct, + browser only if direct is thin)" in out
        assert "skip (dead end): https://www.doordash.com/store/ssamjang" in out
        assert out.count("search not cached yet") == 1  # only the un-warmed row
