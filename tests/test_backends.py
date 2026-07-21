"""Regression tests for the scrape backend's failure contract (no network)."""

import os
import sys
from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backends  # noqa: E402
from backends import build_scrape, is_cacheable, preflight_browser  # noqa: E402

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
