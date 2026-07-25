"""Regression tests for the scrape backend's failure contract (no network)."""

import os
import sys
from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backends  # noqa: E402
from backends import (  # noqa: E402
    BINARY_URL_RESULT,
    BLOCKED_SITE_RESULT,
    BrowserDeadError,
    build_scrape,
    is_cacheable,
    preflight_browser,
    skip_reason,
)
from cache import MIN_CONTENT_CHARS, scrape_status  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_backend_state(monkeypatch):
    """Isolate the process-wide backend state (infra streak, throttle ledger) per
    test, and zero the politeness interval so tests never sleep."""
    monkeypatch.setattr(backends, "_infra_streak", 0)
    monkeypatch.setattr(backends, "_LAST_FETCH", {})
    monkeypatch.setattr(backends, "DOMAIN_MIN_INTERVAL_S", 0)


# Deep enough to blow the interpreter recursion limit inside BeautifulSoup /
# markdownify -- the shape that killed 10/99 pilot episodes (see build_corpus).
DEEP_HTML = "<div>" * 3000 + "menu" + "</div>" * 3000


def test_deeply_nested_page_returns_sentinel_not_raise(monkeypatch):
    monkeypatch.setattr(
        backends, "_render_pooled", lambda url, *, wait, scroll: DEEP_HTML
    )
    scrape = build_scrape()
    out = scrape("http://deep.example.test/menu", mode="browser")
    assert out.startswith("(scrape failed"), out
    assert "recursion" in out.lower()


# ---------------------------------------------------------------------------
# "Executable doesn't exist" is Playwright's message for fs.accessSync() THROWING
# -- so it covers both a missing install (permanent) and a momentarily unreadable
# binary (transient). Only the path inside it tells them apart.
# ---------------------------------------------------------------------------
def _exe_error(path: str) -> str:
    return (f"BrowserType.launch: Executable doesn't exist at {path}\n"
            f"+---------------------------------------+\n"
            f"| Looks like Playwright was just installed |\n"
            f"|     playwright install                  |\n"
            f"+---------------------------------------+")


def _install_tree(root, *, exe: bool, installed: bool = True) -> str:
    """Playwright's real on-disk layout under `root`; returns the exe path.

        <root>/chromium_headless_shell-1228/
            INSTALLATION_COMPLETE
            chrome-headless-shell-win64/chrome-headless-shell.exe

    exe=False, installed=True  -> the LOCKED case (tree intact, file unreadable)
    exe=False, installed=False -> the genuinely-missing case (nothing on disk)
    """
    install = root / "chromium_headless_shell-1228"
    browser_dir = install / "chrome-headless-shell-win64"
    if installed:
        browser_dir.mkdir(parents=True, exist_ok=True)
        (install / "INSTALLATION_COMPLETE").write_text("")
    if exe:
        browser_dir.mkdir(parents=True, exist_ok=True)
        (browser_dir / "chrome-headless-shell.exe").write_bytes(b"MZ")
    return str(browser_dir / "chrome-headless-shell.exe")


class TestExecutablePresent:
    def test_absent_install_reads_as_missing(self, tmp_path):
        path = _install_tree(tmp_path, exe=False, installed=False)
        assert backends._executable_present(_exe_error(path)) is False

    def test_present_exe_reads_as_transient(self, tmp_path):
        path = _install_tree(tmp_path, exe=True)
        assert backends._executable_present(_exe_error(path)) is True

    def test_locked_exe_in_an_INTACT_install_reads_as_transient(self, tmp_path):
        """THE REGRESSION THIS GUARDS: os.path.exists() swallows OSError just like
        canAccessFile swallows accessSync, so a locked binary looks absent to both.
        Trusting exists() alone reported "Chromium is not installed" for a 203 MB
        file that launched seconds later -- and skipped the retries that would have
        ridden the lock out. An intact install tree means locked, never missing."""
        path = _install_tree(tmp_path, exe=False, installed=True)
        assert not os.path.exists(path)          # exactly what exists() would say
        assert backends._executable_present(_exe_error(path)) is True

    def test_unrelated_error_is_not_this_case(self):
        assert backends._executable_present("Page.goto: Timeout 30000ms exceeded.") is None
        assert backends._executable_present("") is None


class TestScrapeErrorHint:
    def test_missing_install_keeps_the_actionable_line(self, tmp_path):
        """Playwright puts 'run playwright install' in an ASCII box BELOW line 1,
        so taking splitlines()[0] alone hid the only useful part of the message."""
        path = _install_tree(tmp_path, exe=False, installed=False)
        out = backends._scrape_error("https://x.test", "browser",
                                     PlaywrightError(_exe_error(path)))
        assert "playwright install chromium" in out
        assert out.count("\n") == 0  # still one line: it goes into the model's context

    def test_transient_failure_gets_no_install_hint(self, tmp_path):
        path = _install_tree(tmp_path, exe=True)
        out = backends._scrape_error("https://x.test", "browser",
                                     PlaywrightError(_exe_error(path)))
        assert "playwright install" not in out

    def test_locked_binary_gets_no_install_hint(self, tmp_path):
        path = _install_tree(tmp_path, exe=False, installed=True)
        out = backends._scrape_error("https://x.test", "browser",
                                     PlaywrightError(_exe_error(path)))
        assert "playwright install" not in out

    def test_site_failure_is_unchanged(self):
        out = backends._scrape_error("https://x.test", "direct",
                                     PlaywrightError("Page.goto: Timeout 30000ms exceeded."))
        assert out == "(scrape failed for https://x.test in 'direct' mode: " \
                      "Page.goto: Timeout 30000ms exceeded.)"


class TestLaunchRetry:
    """Retry only what retrying can fix."""

    def _count_launches(self, monkeypatch, error_path):
        attempts = []

        class FakePW:
            def __init__(self):
                self.chromium = self
            def launch(self, **kwargs):
                attempts.append(1)
                raise PlaywrightError(_exe_error(error_path))
            def stop(self):
                pass

        monkeypatch.setattr(backends, "sync_playwright", lambda: type(
            "Ctx", (), {"start": staticmethod(FakePW)})())
        monkeypatch.setattr(backends, "_POOL", type("L", (), {})())
        monkeypatch.setattr(backends, "BROWSER_LAUNCH_BACKOFF_S", 0)  # no real sleeping
        with pytest.raises(PlaywrightError):
            backends._pooled_browser()
        return len(attempts)

    def test_missing_install_fails_immediately(self, monkeypatch, tmp_path):
        # Retrying cannot install a browser; it only multiplies the per-URL cost.
        path = _install_tree(tmp_path, exe=False, installed=False)
        assert self._count_launches(monkeypatch, path) == 1

    def test_unreadable_binary_is_retried(self, monkeypatch, tmp_path):
        path = _install_tree(tmp_path, exe=True)
        assert self._count_launches(monkeypatch, path) == backends.BROWSER_LAUNCH_ATTEMPTS

    def test_locked_binary_in_an_intact_install_is_retried(self, monkeypatch, tmp_path):
        """The lock is exactly what retrying exists for -- it must not be
        misclassified as a missing install and skipped."""
        path = _install_tree(tmp_path, exe=False, installed=True)
        assert self._count_launches(monkeypatch, path) == backends.BROWSER_LAUNCH_ATTEMPTS


class TestIsCacheable:
    def test_infra_failure_is_not_cacheable(self):
        assert not is_cacheable("(scrape failed for u in 'browser' mode: "
                                "BrowserType.launch: Executable doesn't exist at /nope)")
        assert not is_cacheable("(scrape failed for u in 'direct' mode: "
                                "It looks like you are using Playwright Sync API "
                                "inside the asyncio loop.)")

    def test_site_failure_and_real_content_are_cacheable(self):
        # A site refusing us is a FACT ABOUT THE WEB -- the correct permanent answer.
        assert is_cacheable("(scrape failed for u in 'direct' mode: "
                            "Page.goto: Timeout 30000ms exceeded.)")
        assert is_cacheable("(page returned no content)")
        assert is_cacheable("# Real menu\n- Pad Thai $12")


def _no_network(monkeypatch):
    """Make ANY network path fail the test loudly (requests fetch + browser render)."""
    def boom(*args, **kwargs):
        raise AssertionError("network path touched")
    monkeypatch.setattr(backends.requests.Session, "get", boom)
    monkeypatch.setattr(backends, "_render_pooled", boom)


class FakeResp:
    """Minimal requests.Response stand-in for the direct path."""

    def __init__(self, text="", ctype="text/html"):
        self.text = text
        self.headers = {"content-type": ctype}

    def raise_for_status(self):
        pass


class TestSkipReason:
    def test_blocked_domains_and_subdomains(self):
        assert skip_reason("https://www.doordash.com/store/x") == BLOCKED_SITE_RESULT
        assert skip_reason("https://order.ubereats.com/x") == BLOCKED_SITE_RESULT

    def test_binary_extensions(self):
        assert skip_reason("https://menu.example.com/menu.pdf") == BINARY_URL_RESULT

    def test_lookalike_domain_and_real_urls_pass(self):
        assert skip_reason("https://notyelp.com/menu") is None
        assert skip_reason("https://pagliacci.com/menu") is None

    def test_sentinel_contract_permanent_negative(self):
        """The sentinels MUST classify 'empty' (cached once, never re-fetched under
        live, replayed under canned) and be storable -- the whole design leans on
        cache.MIN_CONTENT_CHARS and the failure-marker prefixes not matching."""
        for sentinel in (BLOCKED_SITE_RESULT, BINARY_URL_RESULT,
                         backends._non_html_result("application/pdf")):
            assert len(sentinel) < MIN_CONTENT_CHARS
            assert scrape_status(sentinel) == "empty"
            assert is_cacheable(sentinel)


class TestDeadEndsAnswerInstantly:
    """A known dead-end URL must never reach the network -- no 30-45s bot-wall
    timeout, no 'error' row that live policy re-fetches every pass."""

    def test_blocked_domain_returns_sentinel_without_network(self, monkeypatch):
        _no_network(monkeypatch)
        scrape = build_scrape()
        assert scrape("https://www.yelp.com/biz/some-place") == BLOCKED_SITE_RESULT
        assert scrape("https://www.yelp.com/biz/some-place", mode="browser") == BLOCKED_SITE_RESULT

    def test_binary_extension_returns_sentinel_without_network(self, monkeypatch):
        _no_network(monkeypatch)
        assert build_scrape()("https://a.com/menu.pdf") == BINARY_URL_RESULT


class TestContentTypeGuard:
    def test_non_html_payload_returns_sentinel_and_does_not_escalate(self, monkeypatch):
        """An extension-less URL serving a PDF must not be markdownified (one PDF
        made a 14M-char row) NOR handed to the browser (which can't read it either)."""
        monkeypatch.setattr(backends.requests.Session, "get",
                            lambda self, url, timeout=None: FakeResp("%PDF-1.7 ...",
                                                                     "application/pdf"))
        def no_render(*args, **kwargs):
            raise AssertionError("browser render attempted for a binary payload")
        monkeypatch.setattr(backends, "_render_pooled", no_render)
        out = build_scrape()("https://a.com/menu")  # no .pdf extension
        assert "application/pdf" in out
        assert scrape_status(out) == "empty"

    def test_html_with_charset_param_passes(self, monkeypatch):
        big = "<p>" + "menu item $9 " * 200 + "</p>"
        monkeypatch.setattr(backends.requests.Session, "get",
                            lambda self, url, timeout=None: FakeResp(
                                big, "text/html; charset=utf-8"))
        out = build_scrape()("https://a.com/menu")
        assert "menu item $9" in out


class TestSlimAtSource:
    def test_scrape_output_is_slimmed(self, monkeypatch):
        html = ("<p>" + "Real menu: Pizza $10 " * 30 + "</p>"
                "<img src='https://a.com/logo.png' alt='logo'><ul><li></li></ul>")
        monkeypatch.setattr(backends, "_render_pooled", lambda url, *, wait, scroll: html)
        out = build_scrape()("https://a.com/menu", mode="browser")
        assert "![" not in out and "logo.png" not in out
        assert "Real menu: Pizza $10" in out


class TestInfraBreaker:
    """INFRA_STREAK_ABORT consecutive infra-failed scrapes must raise instead of
    grinding out sentinels -- a run whose local browser died mid-run has nothing
    left to produce, and only warm_cache used to notice (per-restaurant counter);
    build_corpus billed a teacher pod through it and TRL's GRPO loop would train
    on the failure text."""

    def _dead_browser(self, monkeypatch):
        def boom(url, *, wait, scroll):
            raise PlaywrightError("BrowserType.launch: browser has crashed")
        monkeypatch.setattr(backends, "_render_pooled", boom)

    def test_raises_browser_dead_after_streak(self, monkeypatch):
        monkeypatch.setattr(backends, "INFRA_STREAK_ABORT", 3)
        self._dead_browser(monkeypatch)
        scrape = build_scrape()
        for i in range(2):
            out = scrape(f"https://site{i}.test/menu", mode="browser")
            assert out.startswith("(scrape failed")  # sentinel until the streak trips
        with pytest.raises(BrowserDeadError):
            scrape("https://site9.test/menu", mode="browser")

    def test_site_failure_resets_the_streak(self, monkeypatch):
        """A nav timeout means the browser WORKED and the site refused -- it must
        not accumulate toward the abort, or aggregator-heavy selections would
        false-trip it."""
        monkeypatch.setattr(backends, "INFRA_STREAK_ABORT", 3)
        calls = {"n": 0}

        def flaky(url, *, wait, scroll):
            calls["n"] += 1
            if calls["n"] == 3:
                raise PlaywrightError("Page.goto: Timeout 30000ms exceeded.")
            raise PlaywrightError("BrowserType.launch: browser has crashed")
        monkeypatch.setattr(backends, "_render_pooled", flaky)
        scrape = build_scrape()
        for i in range(5):  # infra, infra, SITE (resets), infra, infra -- never 3 in a row
            out = scrape(f"https://site{i}.test/menu", mode="browser")
            assert out.startswith("(scrape failed")

    def test_successful_render_resets_the_streak(self, monkeypatch):
        monkeypatch.setattr(backends, "INFRA_STREAK_ABORT", 2)
        calls = {"n": 0}

        def flaky(url, *, wait, scroll):
            calls["n"] += 1
            if calls["n"] == 2:
                return "<p>" + "menu " * 300 + "</p>"
            raise PlaywrightError("BrowserType.launch: browser has crashed")
        monkeypatch.setattr(backends, "_render_pooled", flaky)
        scrape = build_scrape()
        assert scrape("https://a.test/x", mode="browser").startswith("(scrape failed")
        assert "menu" in scrape("https://b.test/x", mode="browser")  # resets
        assert scrape("https://c.test/x", mode="browser").startswith("(scrape failed")


class TestThrottle:
    def test_same_host_waits_out_the_interval(self, monkeypatch):
        import time as time_mod
        monkeypatch.setattr(backends, "DOMAIN_MIN_INTERVAL_S", 5.0)
        backends._LAST_FETCH["h.test"] = time_mod.monotonic()
        slept = []

        def fake_sleep(s):
            slept.append(s)
            backends._LAST_FETCH["h.test"] = -1e9  # release so the loop exits
        monkeypatch.setattr(backends.time, "sleep", fake_sleep)
        backends._throttle("h.test")
        assert slept and 4.0 < slept[0] <= 5.0

    def test_distinct_hosts_do_not_wait(self, monkeypatch):
        import time as time_mod
        monkeypatch.setattr(backends, "DOMAIN_MIN_INTERVAL_S", 5.0)
        backends._LAST_FETCH["a.test"] = time_mod.monotonic()

        def no_sleep(s):
            raise AssertionError("throttled a fetch to a DIFFERENT host")
        monkeypatch.setattr(backends.time, "sleep", no_sleep)
        backends._throttle("b.test")  # must return immediately


class TestPreflight:
    def test_returns_none_when_the_browser_launches(self, monkeypatch):
        monkeypatch.setattr(backends, "_pooled_browser", lambda: object())
        assert preflight_browser() is None

    def test_reports_a_missing_install_actionably(self, monkeypatch, tmp_path):
        path = _install_tree(tmp_path, exe=False, installed=False)

        def boom():
            raise PlaywrightError(_exe_error(path))
        monkeypatch.setattr(backends, "_pooled_browser", boom)
        reason = preflight_browser()
        assert reason is not None and "playwright install chromium" in reason

    def test_reports_a_LOCK_without_sending_you_to_reinstall(self, monkeypatch, tmp_path):
        """Same Playwright message, opposite advice: a 203 MB reinstall fixes
        nothing when the browser is installed and merely unreadable."""
        path = _install_tree(tmp_path, exe=False, installed=True)

        def boom():
            raise PlaywrightError(_exe_error(path))
        monkeypatch.setattr(backends, "_pooled_browser", boom)
        reason = preflight_browser()
        assert "do NOT reinstall" in reason
        assert "playwright install chromium" not in reason

    def test_reports_other_failures_without_raising(self, monkeypatch):
        def boom():
            raise RuntimeError("Sync API inside the asyncio loop")
        monkeypatch.setattr(backends, "_pooled_browser", boom)
        assert "asyncio loop" in preflight_browser()
