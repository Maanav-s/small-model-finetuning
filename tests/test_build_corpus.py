"""Regression tests for build_corpus's pre-run guards (no network, no teacher).

The one behaviour guarded here is the browser preflight, and it is guarded because
the FUNCTION existing was never the hard part -- backends.preflight_browser() was
written, tested, and wired into warm_cache.py, while build_corpus.py (the script
that runs against a metered teacher pod) silently did not call it.
"""

import sys
import threading
import time
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


class TestCrashCancelsTheBacklog:
    """An UNHANDLED crash in the main loop must cancel the QUEUED episodes, not let
    the pool drain them.

    The per-episode try/except already turns a bad episode into a logged failure,
    and the consecutive-failure breaker already cancels on abort. The gap the
    2026-07-21 incident exposed was a crash the inner try does NOT catch -- there,
    a write raising OUTSIDE it. That fell straight into the pool's __exit__ =
    shutdown(wait=True), which runs every already-submitted future to completion,
    so a crash ~50 episodes in generated ~1861 more over ~3h and discarded them all
    (the writer thread was dead). The finally-cancel closes that gap. Here the
    uncaught crash is trace_summary raising (it runs after the inner try); the rest
    of the backlog must be cancelled, so run_one is not called for all of it.
    """

    def test_unhandled_crash_cancels_the_queue(self, monkeypatch, tmp_path):
        p = tmp_path / "corpus.sqlite"
        with open_corpus(p) as cx:
            cx.upsert_restaurants([
                {"name": f"R{i}", "city": "C", "source": "osm", "split": "sft"}
                for i in range(60)
            ])

        calls = []
        lock = threading.Lock()

        def slow_run_one(client, episode, tools, registry, system_prompt, args, cache, teacher_model):
            with lock:
                calls.append(episode["row"]["restaurant_id"])
            time.sleep(0.5)   # keep the backlog QUEUED so cancel() can still reach it
            return _valid_trace(episode["row"]["restaurant_id"])

        def boom_summary(trace, row):
            raise RuntimeError("bug after the write")   # NOT caught by the per-episode try

        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        monkeypatch.setattr(build_corpus, "preflight_browser", lambda: None)
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url, **kw: object())
        monkeypatch.setattr(build_corpus, "setup_tools", lambda *a, **k: ([], {}, "prompt"))
        monkeypatch.setattr(build_corpus, "run_one", slow_run_one)
        monkeypatch.setattr(build_corpus, "trace_summary", boom_summary)
        # workers=2 + a 0.5s episode: only ~2 are running when the crash hits on the
        # first result, so the other ~58 are still queued and must be cancelled.
        monkeypatch.setattr(sys, "argv", ["build_corpus.py", "--teacher", "vllm",
                            "--teacher-model", "teacher", "--db", str(p),
                            "--cache-path", str(tmp_path / "cache.sqlite"),
                            "--limit", "60", "--conditioned-frac", "0", "--workers", "2"])

        with pytest.raises(RuntimeError, match="bug after the write"):
            build_corpus.main()

        # Without the finally-cancel, shutdown(wait=True) drains all 60; with it,
        # only the handful already running when the crash hit ever start.
        assert len(calls) < 30, f"backlog drained: run_one ran {len(calls)}/60 times"


class TestSyncEvery:
    """--sync-every N must checkpoint the corpus DB to S3 every N completed episodes
    (plus a final push for the tail), and never touch S3 when it is off.

    The push itself is mocked (no boto3, no network): what is under test is the
    build's CADENCE and its fail-fast preflight, not corpus_sync's upload -- that
    has its own tests. A FakeRemote stands in for the reachability preflight so
    main() gets past it without a bucket.
    """

    class _FakeRemote:
        """Stands in for corpus_sync.S3Remote's preflight head() -- object absent."""

        def __init__(self, *a, **k):
            pass

        def head(self, rel):
            return None

    def _drive_with_sync(self, monkeypatch, corpus_path, pushes, *extra):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setattr(build_corpus, "preflight_browser", lambda: None)
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url, **kw: object())
        monkeypatch.setattr(build_corpus, "setup_tools", lambda *a, **k: ([], {}, "prompt"))
        monkeypatch.setattr(build_corpus, "S3Remote", self._FakeRemote)  # preflight
        monkeypatch.setattr(build_corpus, "push_corpus_snapshot",
                            lambda db: pushes.append(str(db)))

        def fake_run_one(client, episode, tools, registry, system_prompt, args, cache, teacher_model):
            return _valid_trace(episode["row"]["restaurant_id"])

        monkeypatch.setattr(build_corpus, "run_one", fake_run_one)
        monkeypatch.setattr(sys, "argv", _argv(corpus_path, "--conditioned-frac", "0",
                                               "--workers", "1", *extra))
        build_corpus.main()

    def test_pushes_on_interval_plus_final(self, monkeypatch, tmp_path):
        p = tmp_path / "corpus.sqlite"
        with open_corpus(p) as cx:
            cx.upsert_restaurants([
                {"name": f"R{i}", "city": "C", "source": "osm", "split": "sft"}
                for i in range(4)
            ])
        pushes: list[str] = []
        self._drive_with_sync(monkeypatch, p, pushes, "--limit", "4", "--sync-every", "2")
        # Interval fires at 2 and 4 completed episodes, then one final tail push.
        assert len(pushes) == 3
        assert all(db == str(p) for db in pushes)

    def test_off_by_default_never_pushes(self, monkeypatch, tmp_path):
        p = tmp_path / "corpus.sqlite"
        with open_corpus(p) as cx:
            cx.upsert_restaurants([
                {"name": f"R{i}", "city": "C", "source": "osm", "split": "sft"}
                for i in range(3)
            ])
        pushes: list[str] = []
        self._drive_with_sync(monkeypatch, p, pushes, "--limit", "3")  # no --sync-every
        assert pushes == []

    def test_missing_bucket_fails_fast_before_any_episode(self, monkeypatch, corpus_path):
        """A --sync-every run with no S3_BUCKET must exit BEFORE spending on the
        teacher, not discover the misconfig 500 episodes in."""
        monkeypatch.delenv("S3_BUCKET", raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        monkeypatch.setattr(build_corpus, "preflight_browser", lambda: None)
        monkeypatch.setattr(build_corpus, "openai_build_client", lambda url, **kw: object())
        monkeypatch.setattr(build_corpus, "run_one", _boom)  # reaching an episode = failure
        monkeypatch.setattr(sys, "argv", _argv(corpus_path, "--sync-every", "2"))
        with pytest.raises(SystemExit) as exc:
            build_corpus.main()
        assert "S3_BUCKET" in str(exc.value)


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
