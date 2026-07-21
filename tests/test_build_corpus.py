"""Regression tests for build_corpus's pre-run guards (no network, no teacher).

The one behaviour guarded here is the browser preflight, and it is guarded because
the FUNCTION existing was never the hard part -- backends.preflight_browser() was
written, tested, and wired into warm_cache.py, while build_corpus.py (the script
that runs against a metered teacher pod) silently did not call it.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "claude"))
sys.path.insert(0, str(REPO_ROOT / "src" / "serving"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "corpus"))

import build_corpus  # noqa: E402
from corpus import open_corpus  # noqa: E402


@pytest.fixture
def corpus_path(tmp_path):
    """A temp corpus.sqlite with 2 sft restaurants and no traces, so `todo` is
    non-empty and main() reaches the guards instead of returning 'nothing to do'."""
    p = tmp_path / "corpus.sqlite"
    with open_corpus(p) as cx:
        cx.upsert_restaurants([
            {"name": "Ssamjang", "city": "Atlanta", "source": "osm", "split": "sft"},
            {"name": "Hamburger Mary's", "city": "Chicago", "source": "osm", "split": "sft"},
        ])
    return p


def _argv(corpus_path, *extra):
    return ["build_corpus.py", "--teacher", "vllm", "--teacher-model", "teacher",
            "--db", str(corpus_path),
            "--cache-path", str(corpus_path.parent / "cache.sqlite"), *extra]


class TestBrowserPreflight:
    """A broken browser must stop the run BEFORE the first episode.

    Nothing downstream can catch it: scrape returns a sentinel string rather than
    raising, so every episode "succeeds" with found=false and the consecutive-
    failure guard -- which only counts episodes that RAISE -- never fires. An
    unguarded run therefore bills a teacher pod for hours to write junk traces.
    """

    def test_refuses_to_start_when_the_browser_is_broken(
        self, monkeypatch, capsys, corpus_path
    ):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        monkeypatch.setattr(build_corpus, "preflight_browser",
                            lambda: "BrowserType.launch: Executable doesn't exist at /nope")
        # Reaching either of these means the guard let the run through.
        monkeypatch.setattr(build_corpus, "setup_tools", _boom)
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url: object())
        monkeypatch.setattr(sys, "argv", _argv(corpus_path))

        with pytest.raises(SystemExit) as exc:
            build_corpus.main()

        message = str(exc.value)
        assert "preflight failed" in message
        assert "Executable doesn't exist" in message  # the actionable cause survives

    def test_healthy_browser_proceeds_past_the_guard(
        self, monkeypatch, corpus_path
    ):
        """The guard must be a gate, not a wall -- a launching browser lets the run
        continue (here to a sentinel raised from the very next call)."""
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        monkeypatch.setattr(build_corpus, "preflight_browser", lambda: None)
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url: object())
        monkeypatch.setattr(build_corpus, "setup_tools", _boom)
        monkeypatch.setattr(sys, "argv", _argv(corpus_path))

        with pytest.raises(AssertionError, match="got past the preflight"):
            build_corpus.main()

    def test_list_plans_without_launching_a_browser(
        self, monkeypatch, capsys, corpus_path
    ):
        """--list is plan-only and documented as needing no API; it must not require
        a working browser either."""
        monkeypatch.setattr(build_corpus, "preflight_browser", _boom)
        monkeypatch.setattr(sys, "argv", _argv(corpus_path, "--list"))

        build_corpus.main()

        assert "Ssamjang" in capsys.readouterr().out

    def test_canned_policy_needs_no_browser(self, monkeypatch, corpus_path):
        """miss_policy='canned' replays recorded rows and never launches Chromium,
        so demanding a browser there would block frozen/offline replay runs."""
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        monkeypatch.setattr(build_corpus, "preflight_browser", _boom)
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url: object())
        monkeypatch.setattr(build_corpus, "setup_tools", _boom)
        monkeypatch.setattr(sys, "argv", _argv(corpus_path, "--cache-policy", "canned"))

        with pytest.raises(AssertionError, match="got past the preflight"):
            build_corpus.main()


def _boom(*args, **kwargs):
    raise AssertionError("got past the preflight")
