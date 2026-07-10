"""Unit tests for scripts/warm_cache.py (Phase 2 WS-C2).

No network anywhere: search/scrape are fakes, and the dry-run tests monkeypatch
the backend builders to raise so any call through them fails loudly.

Run: uv run python -m pytest tests/test_warm_cache.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/,
# the script under test in scripts/.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import warm_cache  # noqa: E402
from cache import Cache, norm_query, norm_scrape, scrape_status  # noqa: E402


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
# Query template
# ---------------------------------------------------------------------------
class TestBuildQuery:
    def test_dominant_pattern(self):
        row = {"name": "Ssamjang", "city": "Atlanta"}
        assert warm_cache.build_query(row) == "Ssamjang Atlanta menu"

    def test_missing_city_collapses_cleanly(self):
        assert warm_cache.build_query({"name": "Ssamjang", "city": ""}) == "Ssamjang menu"


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
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, top_n=3, sleep_s=0)

        assert search_calls == ["A B menu"]
        # Direct cleared the bar on both URLs -> no browser escalation.
        assert scrape_calls == [("https://a.com/menu", "direct"), ("https://b.com", "direct")]
        assert summary["scrape_direct"] == 2 and summary["scrape_browser"] == 0
        assert summary["urls"] == 2 and summary["scrape_errors"] == 0

        # Idempotency: a second pass is all cache hits -- zero new backend calls.
        warm_cache.warm_one(row, search_fn, scrape_fn, top_n=3, sleep_s=0)
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
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, top_n=1, sleep_s=0)

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
        summary = warm_cache.warm_one(row, search_fn, scrape_fn, top_n=1, sleep_s=0)
        # Direct errored -> escalate to browser; browser also errored -> both counted.
        assert summary["scrape_direct"] == 1 and summary["scrape_browser"] == 1
        assert summary["scrape_errors"] == 2


# ---------------------------------------------------------------------------
# Dry-run planning: no network, backends never even built
# ---------------------------------------------------------------------------
@pytest.fixture
def data_dir(tmp_path):
    rows = [
        {"restaurant_id": "id1", "name": "Ssamjang", "city": "Atlanta"},
        {"restaurant_id": "id2", "name": "Hamburger Mary's", "city": "Chicago"},
    ]
    (tmp_path / "restaurants.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def _boom(*args, **kwargs):
    raise AssertionError("network backend touched during --dry-run")


class TestDryRun:
    def _run(self, monkeypatch, capsys, data_dir, cache_path):
        # Any route to the network must fail the test.
        monkeypatch.setattr(warm_cache, "build_search", _boom)
        monkeypatch.setattr(warm_cache, "build_scrape", _boom)
        monkeypatch.setattr(sys, "argv", [
            "warm_cache.py", "--dry-run",
            "--data-dir", str(data_dir), "--cache-path", str(cache_path),
        ])
        warm_cache.main()
        return capsys.readouterr().out

    def test_cold_cache_prints_queries_and_unknown_urls(self, monkeypatch, capsys, data_dir):
        out = self._run(monkeypatch, capsys, data_dir, data_dir / "warm.sqlite")
        assert "'Ssamjang Atlanta menu'" in out
        assert "Hamburger Mary's Chicago menu" in out  # repr()'d with double quotes
        assert out.count("search not cached yet") == 2

    def test_warm_search_prints_its_scrape_plan(self, monkeypatch, capsys, data_dir):
        # Pre-warm one search row exactly as a live run would store it.
        cache_path = data_dir / "warm.sqlite"
        pre = Cache(str(cache_path), miss_policy="live")
        pre.wrap("search", lambda q: brave_response(
            "https://ssamjang.com/menu", "https://www.doordash.com/store/ssamjang"
        ), key_fn=norm_query)("Ssamjang Atlanta menu")
        pre.close()

        out = self._run(monkeypatch, capsys, data_dir, cache_path)
        assert "scrape: https://ssamjang.com/menu (direct, + browser only if direct is thin)" in out
        assert "skip (dead end): https://www.doordash.com/store/ssamjang" in out
        assert out.count("search not cached yet") == 1  # only the un-warmed row
