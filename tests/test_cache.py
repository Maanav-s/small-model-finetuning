"""Unit tests for src/cache.py (Phase 2 WS-A).

No network anywhere: every wrapped fn is a counting fake. Real-file tests use
pytest's tmp_path; everything else runs on :memory: sqlite.

Run: uv run python -m pytest tests/ -q
"""

import sys
import threading
from pathlib import Path

import pytest

# Shared modules live in src/ (flat imports, no packages) -- same convention as
# the entry scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cache import (  # noqa: E402
    CANNED,
    MIN_CONTENT_CHARS,
    MISS_POLICIES,
    Cache,
    CacheMiss,
    norm_query,
    norm_scrape,
    norm_url,
    scrape_status,
)


class CountingFn:
    """A fake backend closure that counts calls and returns a canned response."""

    def __init__(self, response="RESPONSE"):
        self.calls = 0
        self.response = response
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
        return self.response(*args, **kwargs) if callable(self.response) else self.response


# ---------------------------------------------------------------------------
# live policy: miss -> fetch + store; hit -> no fetch; normalized-key hits
# ---------------------------------------------------------------------------
class TestLivePolicy:
    def test_miss_fetches_and_stores_then_hit_skips_fetch(self):
        cache = Cache(":memory:", miss_policy="live")
        fn = CountingFn("brave results")
        wrapped = cache.wrap("search", fn, key_fn=norm_query, provider="brave")

        assert wrapped("Pagliacci Pizza menu") == "brave results"
        assert fn.calls == 1
        assert wrapped("Pagliacci Pizza menu") == "brave results"
        assert fn.calls == 1  # served from cache, no second fetch
        assert cache.stats() == {
            "hits": 1, "misses": 1, "writes": 1,
            "miss_policy": "live", "cache_version": 1,
        }

    def test_query_normalization_whitespace_and_case_hit_same_row(self):
        cache = Cache(":memory:", miss_policy="live")
        fn = CountingFn()
        wrapped = cache.wrap("search", fn, key_fn=norm_query)

        wrapped("Kashish  Kirkland   menu")
        wrapped("  kashish kirkland MENU ")
        wrapped("KASHISH\tKIRKLAND\nMENU")
        assert fn.calls == 1
        assert cache.stats()["hits"] == 2

    def test_url_normalization_tracking_fragment_hostcase_hit_same_row(self):
        cache = Cache(":memory:", miss_policy="live")
        fn = CountingFn("page md")
        wrapped = cache.wrap("scrape", fn, key_fn=norm_scrape, status_fn=scrape_status)

        wrapped("https://Example.COM/menu/?utm_source=x&fbclid=abc")
        wrapped("https://example.com/menu#section-2")
        wrapped("HTTPS://example.com/menu")
        assert fn.calls == 1
        assert cache.stats()["hits"] == 2

    def test_distinct_queries_are_distinct_entries(self):
        cache = Cache(":memory:", miss_policy="live")
        fn = CountingFn()
        wrapped = cache.wrap("search", fn, key_fn=norm_query)
        wrapped("query one")
        wrapped("query two")
        assert fn.calls == 2
        assert cache.stats()["misses"] == 2

    def test_unknown_namespace_rejected(self):
        cache = Cache(":memory:")
        with pytest.raises(ValueError):
            cache.wrap("nonsense", CountingFn(), key_fn=norm_query)

    def test_invalid_policy_rejected(self):
        assert set(MISS_POLICIES) == {"live", "canned", "error"}
        with pytest.raises(ValueError):
            Cache(":memory:", miss_policy="frozen")


# ---------------------------------------------------------------------------
# norm_url edge cases
# ---------------------------------------------------------------------------
class TestNormUrl:
    def test_blank_query_values_kept(self):
        # ?a=&b=1 must not silently collapse to ?b=1
        assert norm_url("https://x.com/p?a=&b=1") == "https://x.com/p?a=&b=1"

    def test_bare_ref_param_kept(self):
        # "ref" can select content (e.g. git refs) -- deliberately NOT dropped.
        assert "ref=main" in norm_url("https://x.com/repo?ref=main")

    def test_tracking_params_dropped(self):
        assert (
            norm_url("https://x.com/p?utm_source=tw&utm_medium=social&fbclid=z&gclid=g&q=1")
            == "https://x.com/p?q=1"
        )

    def test_trailing_slash_stripped_and_root_kept(self):
        assert norm_url("https://x.com/menu/") == norm_url("https://x.com/menu")
        assert norm_url("https://x.com/") == "https://x.com/"
        assert norm_url("https://x.com") == "https://x.com/"

    def test_param_order_canonicalized(self):
        assert norm_url("https://x.com/p?b=2&a=1") == norm_url("https://x.com/p?a=1&b=2")

    def test_fragment_dropped_scheme_host_lowercased_path_case_kept(self):
        assert norm_url("HTTPS://WWW.X.COM/Menu#top") == "https://www.x.com/Menu"


# ---------------------------------------------------------------------------
# scrape: mode in the key + status classification
# ---------------------------------------------------------------------------
class TestScrape:
    def test_direct_and_browser_are_distinct_entries(self):
        cache = Cache(":memory:", miss_policy="live")
        seen = []

        def fake_scrape(url, mode="direct"):
            seen.append(mode)
            return f"content via {mode}"

        wrapped = cache.wrap("scrape", fake_scrape, key_fn=norm_scrape, status_fn=scrape_status)
        assert wrapped("https://x.com/menu", "direct") == "content via direct"
        assert wrapped("https://x.com/menu", "browser") == "content via browser"
        assert seen == ["direct", "browser"]
        # each mode now hits its own row
        assert wrapped("https://x.com/menu", "direct") == "content via direct"
        assert wrapped("https://x.com/menu", "browser") == "content via browser"
        assert seen == ["direct", "browser"]
        assert cache.stats()["writes"] == 2

    def test_norm_scrape_keys_differ_by_mode_only(self):
        a = norm_scrape("https://x.com/menu", "direct")
        b = norm_scrape("https://x.com/menu", "browser")
        assert a != b
        assert a.split("\x00")[0] == b.split("\x00")[0]

    def test_norm_scrape_default_mode_matches_explicit_direct(self):
        assert norm_scrape("https://x.com/menu") == norm_scrape("https://x.com/menu", "direct")

    def test_scrape_kwarg_mode_hits_positional_mode_row(self):
        cache = Cache(":memory:", miss_policy="live")
        fn = CountingFn("md")
        wrapped = cache.wrap("scrape", fn, key_fn=norm_scrape, status_fn=scrape_status)
        wrapped("https://x.com/menu", mode="browser")
        wrapped("https://x.com/menu", "browser")
        assert fn.calls == 1

    @pytest.mark.parametrize(
        ("response", "status"),
        [
            ("(scrape failed for https://x.com in 'direct' mode: timeout)", "error"),
            ("(page returned no content)", "error"),
            ("(page not available)", "error"),
            ("", "empty"),
            ("   \n ", "empty"),
            (None, "empty"),
            # A bot-wall answers 200 with a token body rather than an error. It is
            # non-empty, so it used to classify 'ok' and count as coverage --
            # TripAdvisor does this in 15 characters. Too short to be a menu.
            ("Access Denied", "empty"),
            ("x" * (MIN_CONTENT_CHARS - 1), "empty"),
            ("x" * MIN_CONTENT_CHARS, "ok"),
            ("# Menu\n" + "- Margherita Pizza $12\n" * 12, "ok"),
        ],
    )
    def test_scrape_status_classification(self, response, status):
        assert scrape_status(response) == status


# ---------------------------------------------------------------------------
# error rows (negative caching)
# ---------------------------------------------------------------------------
class TestErrorRows:
    URL = "https://x.com/menu"
    FAIL = "(scrape failed for https://x.com/menu in 'direct' mode: net::ERR)"

    def _store_error_row(self, path):
        """Populate `path` with one stored-error scrape row via a live pass."""
        cache = Cache(path, miss_policy="live")
        fn = CountingFn(self.FAIL)
        wrapped = cache.wrap("scrape", fn, key_fn=norm_scrape, status_fn=scrape_status)
        assert wrapped(self.URL) == self.FAIL  # live still RETURNS the failure text
        assert cache.stats() == {
            "hits": 0, "misses": 1, "writes": 1,
            "miss_policy": "live", "cache_version": 1,
        }
        cache.close()

    def test_live_stores_error_row_then_refetches_it(self, tmp_path):
        path = str(tmp_path / "c.sqlite")
        self._store_error_row(path)

        # Second live pass: the stored 'error' row is NOT served -- re-fetched.
        cache = Cache(path, miss_policy="live")
        fn = CountingFn("# Menu recovered")
        wrapped = cache.wrap("scrape", fn, key_fn=norm_scrape, status_fn=scrape_status)
        assert wrapped(self.URL) == "# Menu recovered"
        assert fn.calls == 1
        assert cache.stats()["misses"] == 1 and cache.stats()["hits"] == 0

        # The healed 'ok' row is now served.
        assert wrapped(self.URL) == "# Menu recovered"
        assert fn.calls == 1
        cache.close()

    def test_canned_serves_stored_error_row_verbatim(self, tmp_path):
        path = str(tmp_path / "c.sqlite")
        self._store_error_row(path)

        cache = Cache(path, miss_policy="canned")
        fn = CountingFn("should never be called")
        wrapped = cache.wrap("scrape", fn, key_fn=norm_scrape, status_fn=scrape_status)
        assert wrapped(self.URL) == self.FAIL  # verbatim replay, NOT the canned constant
        assert fn.calls == 0
        assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 0
        cache.close()

    def test_error_policy_raises_on_stored_error_row(self, tmp_path):
        path = str(tmp_path / "c.sqlite")
        self._store_error_row(path)

        cache = Cache(path, miss_policy="error")
        fn = CountingFn()
        wrapped = cache.wrap("scrape", fn, key_fn=norm_scrape, status_fn=scrape_status)
        with pytest.raises(CacheMiss):
            wrapped(self.URL)
        assert fn.calls == 0
        cache.close()

    def test_error_policy_raises_on_absent_key(self):
        cache = Cache(":memory:", miss_policy="error")
        fn = CountingFn()
        wrapped = cache.wrap("search", fn, key_fn=norm_query)
        with pytest.raises(CacheMiss):
            wrapped("never stored")
        assert fn.calls == 0

    def test_error_policy_serves_stored_ok_row(self, tmp_path):
        path = str(tmp_path / "c.sqlite")
        live = Cache(path, miss_policy="live")
        live.wrap("search", CountingFn("results"), key_fn=norm_query)("q")
        live.close()

        cache = Cache(path, miss_policy="error")
        fn = CountingFn()
        assert cache.wrap("search", fn, key_fn=norm_query)("q") == "results"
        assert fn.calls == 0
        cache.close()

    def test_empty_ok_rows_served_under_live(self):
        # 'empty' is a valid negative-cache answer (page really had nothing);
        # only 'error' rows are re-fetched.
        cache = Cache(":memory:", miss_policy="live")
        fn = CountingFn("")
        wrapped = cache.wrap("search", fn, key_fn=norm_query)
        assert wrapped("q") == ""
        assert wrapped("q") == ""
        assert fn.calls == 1


# ---------------------------------------------------------------------------
# canned policy (frozen / GRPO)
# ---------------------------------------------------------------------------
class TestCannedPolicy:
    def test_absent_key_returns_canned_constant_without_calling_fn(self):
        cache = Cache(":memory:", miss_policy="canned")
        search_fn, scrape_fn = CountingFn(), CountingFn()
        search = cache.wrap("search", search_fn, key_fn=norm_query)
        scrape = cache.wrap("scrape", scrape_fn, key_fn=norm_scrape, status_fn=scrape_status)

        assert search("anything") == CANNED["search"] == "(no results)"
        assert scrape("https://x.com", "browser") == CANNED["scrape"] == "(page not available)"
        assert search_fn.calls == 0 and scrape_fn.calls == 0
        assert cache.stats()["misses"] == 2
        assert cache.stats()["writes"] == 0  # canned misses are never stored

    def test_recorded_rows_replayed_verbatim_any_status(self, tmp_path):
        path = str(tmp_path / "c.sqlite")
        live = Cache(path, miss_policy="live")
        scrape = live.wrap("scrape", CountingFn(lambda url, mode="direct": {
            "https://ok.com/": "# Menu",
            "https://empty.com/": "",
            "https://err.com/": "(scrape failed for https://err.com in 'direct' mode: x)",
        }[norm_url(url)]), key_fn=norm_scrape, status_fn=scrape_status)
        for u in ("https://ok.com", "https://empty.com", "https://err.com"):
            scrape(u)
        live.close()

        frozen = Cache(path, miss_policy="canned")
        fn = CountingFn("never")
        scrape = frozen.wrap("scrape", fn, key_fn=norm_scrape, status_fn=scrape_status)
        assert scrape("https://ok.com") == "# Menu"
        assert scrape("https://empty.com") == ""
        assert scrape("https://err.com") == "(scrape failed for https://err.com in 'direct' mode: x)"
        assert fn.calls == 0  # frozen: the wrapped fn is NEVER called
        assert frozen.stats() == {
            "hits": 3, "misses": 0, "writes": 0,
            "miss_policy": "canned", "cache_version": 1,
        }
        frozen.close()


# ---------------------------------------------------------------------------
# persistence + cache_version
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_rows_survive_close_and_reopen(self, tmp_path):
        path = str(tmp_path / "persist.sqlite")
        c1 = Cache(path, miss_policy="live")
        fn1 = CountingFn("stored once")
        c1.wrap("search", fn1, key_fn=norm_query)("some query")
        c1.close()

        c2 = Cache(path, miss_policy="live")
        fn2 = CountingFn("should not be fetched")
        assert c2.wrap("search", fn2, key_fn=norm_query)("Some  Query") == "stored once"
        assert fn2.calls == 0
        assert c2.stats()["hits"] == 1
        c2.close()

    def test_cache_version_bump_changes_key(self, tmp_path):
        path = str(tmp_path / "versioned.sqlite")
        v1 = Cache(path, miss_policy="live", cache_version=1)
        v1.wrap("search", CountingFn("v1 shape"), key_fn=norm_query)("q")
        v1.close()

        # v2 must NOT serve the v1 row (different key hash) ...
        v2 = Cache(path, miss_policy="live", cache_version=2)
        fn = CountingFn("v2 shape")
        assert v2.wrap("search", fn, key_fn=norm_query)("q") == "v2 shape"
        assert fn.calls == 1
        v2.close()

        # ... and the v1 row is untouched (never mutate rows in place).
        v1b = Cache(path, miss_policy="canned", cache_version=1)
        assert v1b.wrap("search", CountingFn(), key_fn=norm_query)("q") == "v1 shape"
        v1b.close()


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------
class TestConcurrency:
    N_THREADS = 8
    OPS_PER_THREAD = 50

    def test_threaded_mixed_reads_writes_consistent(self, tmp_path):
        path = str(tmp_path / "conc.sqlite")
        cache = Cache(path, miss_policy="live")
        search_fn = CountingFn(lambda q: f"results for {q}")
        scrape_fn = CountingFn(lambda url, mode="direct": f"md for {url} via {mode}")
        search = cache.wrap("search", search_fn, key_fn=norm_query)
        scrape = cache.wrap("scrape", scrape_fn, key_fn=norm_scrape, status_fn=scrape_status)

        errors = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid):
            try:
                barrier.wait()  # maximize interleaving
                for i in range(self.OPS_PER_THREAD):
                    if i % 2 == 0:
                        # small key space -> plenty of cross-thread hit/write races
                        assert search(f"query {i % 5}") == f"results for query {i % 5}"
                    else:
                        mode = "browser" if i % 4 == 1 else "direct"
                        url = f"https://x.com/page{i % 5}"
                        assert scrape(url, mode) == f"md for {url} via {mode}"
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"worker exceptions: {errors!r}"
        s = cache.stats()
        total_calls = self.N_THREADS * self.OPS_PER_THREAD
        assert s["hits"] + s["misses"] == total_calls
        # every miss under live is fetched+stored exactly once per miss
        assert s["writes"] == s["misses"] == search_fn.calls + scrape_fn.calls
        # 5 search keys + 5 urls x up to 2 modes; racing double-fetches are benign
        # (INSERT OR REPLACE) but each one is counted as a miss+write.
        assert s["misses"] >= 15
        cache.close()


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------
class TestStats:
    def test_counters_track_every_outcome(self):
        cache = Cache(":memory:", miss_policy="live", cache_version=7)
        fn = CountingFn()
        wrapped = cache.wrap("search", fn, key_fn=norm_query)
        wrapped("a")            # miss + write
        wrapped("a")            # hit
        wrapped("b")            # miss + write
        wrapped("a")            # hit
        wrapped("b")            # hit
        assert cache.stats() == {
            "hits": 3, "misses": 2, "writes": 2,
            "miss_policy": "live", "cache_version": 7,
        }

    def test_canned_misses_counted_not_written(self):
        cache = Cache(":memory:", miss_policy="canned")
        wrapped = cache.wrap("search", CountingFn(), key_fn=norm_query)
        wrapped("a"), wrapped("a")
        assert cache.stats()["misses"] == 2
        assert cache.stats()["writes"] == 0
