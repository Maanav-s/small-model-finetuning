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
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url, **kw: object())
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
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url, **kw: object())
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
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url, **kw: object())
        monkeypatch.setattr(build_corpus, "setup_tools", _boom)
        monkeypatch.setattr(sys, "argv", _argv(corpus_path, "--cache-policy", "canned"))

        with pytest.raises(AssertionError, match="got past the preflight"):
            build_corpus.main()


class TestNoJsonEpisodeIsNotFatal:
    """An episode whose teacher emits no parseable JSON must be recorded as a
    failed episode, NOT crash the whole build.

    traces.final_json is NOT NULL by design (a menu-less trace has no training
    target), and the main-thread write of that None sat OUTSIDE the per-episode
    try/except -- so on 2026-07-21 one such episode aborted a 2252-episode build
    with sqlite3.IntegrityError after ~50 good traces were already written. The
    surviving traces were fine; the run was not resumable-clean because it died
    mid-loop. This locks the write inside the guard and the None into the failure
    path.
    """

    def _drive(self, monkeypatch, corpus_path, fake_run_one):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        monkeypatch.setattr(build_corpus, "preflight_browser", lambda: None)
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url, **kw: object())
        monkeypatch.setattr(build_corpus, "setup_tools", lambda *a, **k: ([], {}, "prompt"))
        monkeypatch.setattr(build_corpus, "run_one", fake_run_one)
        # 2 free episodes over the fixture's 2 restaurants, single worker so the
        # completion order is deterministic.
        monkeypatch.setattr(sys, "argv", _argv(corpus_path, "--limit", "2",
                                               "--conditioned-frac", "0", "--workers", "1"))
        build_corpus.main()

    def test_none_json_episode_is_failed_not_crashed(self, monkeypatch, corpus_path):
        def fake_run_one(client, episode, tools, registry, system_prompt, args, cache, teacher_model):
            rid = episode["row"]["restaurant_id"]
            if episode["row"]["name"] == "Ssamjang":
                return _no_json_trace(rid)              # the poison episode
            return _valid_trace(rid)                    # a normal one

        self._drive(monkeypatch, corpus_path, fake_run_one)  # must NOT raise

        # Exactly the good episode is stored; the menu-less one left no row, so a
        # re-run would retry it (idempotent recovery).
        with open_corpus(corpus_path) as cx:
            assert cx.trace_count() == 1

    def test_all_none_json_trips_the_consecutive_breaker(self, monkeypatch, corpus_path):
        """A systematically menu-less teacher must abort via the same breaker as any
        other failure -- not spin silently, and not write anything."""
        def fake_run_one(client, episode, tools, registry, system_prompt, args, cache, teacher_model):
            return _no_json_trace(episode["row"]["restaurant_id"])

        self._drive(monkeypatch, corpus_path, fake_run_one)
        with open_corpus(corpus_path) as cx:
            assert cx.trace_count() == 0


def _no_json_trace(rid: str) -> dict:
    """What run_one really returns when extract_json found nothing: a FULLY-formed
    trace whose only defect is final_json=None. Faithful on purpose -- with every
    other field present, the un-guarded write fails on exactly the traces.final_json
    NOT NULL constraint that aborted the real build, not on some incidental missing
    key. The guard must reject it on final_json alone."""
    t = _valid_trace(rid)
    t["final_json"] = None
    t["found"] = False
    t["schema_valid"] = False
    t["parse_error"] = "no JSON in final turn"
    return t


def _valid_trace(rid: str) -> dict:
    """A minimal but writable trace (write_trace + trace_summary both consume it)."""
    return {
        "restaurant_id": rid, "model": "teacher", "prompt_variant": "teacher",
        "dietary_restrictions": None, "trace_source": "teacher", "cache_version": 1,
        "messages": [{"role": "assistant", "content": "x"}], "queries": ["q"], "urls": ["u"],
        "final_json": {"found": True, "menu": [{"items": [{"name": "Pad Thai"}]}]},
        "found": True, "schema_valid": True, "grounding": 1.0, "unmatched_items": [],
        "captured_at": "2026-07-21T00:00:00+00:00",
    }


def _boom(*args, **kwargs):
    raise AssertionError("got past the preflight")
