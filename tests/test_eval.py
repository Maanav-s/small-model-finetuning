"""Unit tests for scripts/eval/eval.py (v2 WS-G: the merged eval_split + eval_menu).

No network, no torch, no GPU: the Claude runner (eval.claude_run_episode) is
monkeypatched with a stub returning a canned (final_text, messages) pair, and dummy
BRAVE_API_KEY / ANTHROPIC_API_KEY let setup_tools build its (unused) closures and the
Anthropic client construct without hitting anything. Everything runs the --model
claude path so torch is never imported.

The v2 change vs the old test_eval_split: the eval PLAN and the REFERENCE set both
come from corpus.sqlite (iter_restaurants(split="eval") + iter_traces(split="eval"))
instead of restaurants.jsonl + splits.json + a --reference directory; candidate and
reference join on the TRACE ID (not a filename); and "findable" is DERIVED from the
reference trace's found flag (labels.jsonl retired).

Run: uv run python -m pytest tests/test_eval.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/, the
# script under test in scripts/eval/ (the v2 location). eval.py adds src/claude etc.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

import eval as ev  # noqa: E402  (scripts/eval/eval.py -- a module named eval, aliased)
from cache import CacheMiss  # noqa: E402
from corpus import open_corpus  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: a temp eval-split corpus + menu builders
# ---------------------------------------------------------------------------
def _menu(name, *item_names, found=True):
    """A schema-valid menu dict (contract 1.5 / MENU_SCHEMA)."""
    return {
        "found": found,
        "restaurant_name": name,
        "cuisine": "Test",
        "menu": [{"section": "Mains", "items": [{"name": n, "description": None, "price": 9.0}
                                                for n in item_names]}] if item_names else [],
        "source_url": "https://example.com/menu",
    }


def _menu_by_restaurant():
    # Every eval restaurant gets a 2-item menu so a conditioned reference (which
    # keeps the 2nd item) is well-defined regardless of which one the seeded plan
    # picks for the conditioned slice.
    return {
        "Alpha Cafe": _menu("Alpha Cafe", "Burger", "Fries"),
        "Bravo Bistro": _menu("Bravo Bistro", "Steak", "Salad"),
        "Charlie Grill": _menu("Charlie Grill", "Wings", "Ribs"),
    }


@pytest.fixture
def corpus_path(tmp_path):
    """corpus.sqlite with 3 eval-split restaurants (+1 sft row that must never plan)."""
    p = tmp_path / "corpus.sqlite"
    with open_corpus(p) as cx:
        cx.upsert_restaurants([
            {"name": "Alpha Cafe", "city": "Austin", "source": "osm", "split": "eval"},
            {"name": "Bravo Bistro", "city": "Boston", "source": "osm", "split": "eval"},
            {"name": "Charlie Grill", "city": "Chicago", "source": "osm", "split": "eval"},
            {"name": "Delta Diner", "city": "Denver", "source": "osm", "split": "sft"},
        ])
    return p


@pytest.fixture
def keys(monkeypatch):
    """Dummy keys so setup_tools + anthropic.Anthropic() construct without network."""
    monkeypatch.setenv("BRAVE_API_KEY", "test-brave")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")


@pytest.fixture(autouse=True)
def no_wandb(monkeypatch):
    """W&B logging is key-gated -- unset the key so a developer who happens to have
    WANDB_API_KEY exported doesn't start a real run (and hit the network) per test."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def _argv(candidate_dir, corpus_path, *extra):
    return [str(candidate_dir), "--model", "claude", "--corpus", str(corpus_path),
            "--cache-path", str(corpus_path.parent / "cache.sqlite"), *extra]


def _run(candidate_dir, corpus_path, *extra):
    ev.main(_argv(candidate_dir, corpus_path, *extra))


def _plan(corpus_path, limit, frac, seed=42, split="eval"):
    """The planned episodes, using the SAME planner eval.main runs -- so tests
    assert against the real seeded order instead of guessing which row is front."""
    with open_corpus(corpus_path, create=False) as cx:
        rows = ev.load_seeded_rows(cx, split, seed)
    episodes = ev.plan_episodes(rows, limit, frac)
    return [(ev.episode_trace_id(e["row"]["restaurant_id"], e["restrictions"]),
             e["row"]["name"], e["restrictions"]) for e in episodes]


def _make_stub(menus, calls=None):
    def stub(client, episode_input, tools, registry, system_prompt, model=None):
        if calls is not None:
            calls.append(episode_input)
        name = episode_input.split(",")[0].strip()
        return json.dumps(menus[name]), [{"role": "user", "content": episode_input}]
    return stub


# ---------------------------------------------------------------------------
# (a) --list plans the free + conditioned episodes (no API/GPU, no keys)
# ---------------------------------------------------------------------------
class TestList:
    def test_list_prints_planned_episodes(self, capsys, corpus_path, tmp_path):
        cand = tmp_path / "cand"
        # 3 eval rows, limit 3, frac 0.4 -> round(1.2)=1 conditioned + 2 free.
        _run(cand, corpus_path, "--list", "--limit", "3", "--conditioned-frac", "0.4")
        out = capsys.readouterr().out
        assert "2 free + 1 conditioned" in out
        # Every planned trace id (free <rid>, conditioned <rid>__<slug>) is listed.
        plan = _plan(corpus_path, 3, 0.4)
        for tid, _name, _r in plan:
            assert tid in out
        assert "__vegetarian" in out  # first DIETARY_POOL entry on the seeded front row
        # --list never touches the network/GPU: no candidate files written.
        assert not cand.exists() or list(cand.glob("*.json")) == []

    def test_list_requires_no_keys(self, capsys, corpus_path, tmp_path):
        # No `keys` fixture here: --list must work with no API keys at all.
        _run(tmp_path / "cand", corpus_path, "--list", "--limit", "2", "--conditioned-frac", "0.0")
        assert "2 free + 0 conditioned" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (b) candidate traces written with the right names + idempotent skip
# ---------------------------------------------------------------------------
class TestCandidateWrite:
    def test_writes_and_is_idempotent(self, monkeypatch, capsys, corpus_path, tmp_path, keys):
        cand = tmp_path / "cand"
        calls = []
        monkeypatch.setattr(ev, "claude_run_episode", _make_stub(_menu_by_restaurant(), calls))
        _run(cand, corpus_path, "--limit", "3", "--conditioned-frac", "0.4", "--workers", "1")

        plan = _plan(corpus_path, 3, 0.4)
        expected = sorted(f"{tid}.json" for tid, _, _ in plan)
        files = sorted(p.name for p in cand.glob("*.json"))
        # 2 free + 1 conditioned = 3 candidate files (no .tmp left behind).
        assert files == expected
        cond_tid = next(tid for tid, _, r in plan if r)  # the conditioned episode
        assert cond_tid.endswith("__vegetarian")  # DIETARY_POOL[0]
        trace = json.loads((cand / f"{cond_tid}.json").read_text())
        assert trace["dietary_restrictions"] == ["vegetarian"]
        assert trace["prompt_variant"] == "student"
        assert trace["schema_valid"] is True
        assert trace["final_json"]["found"] is True
        assert trace["model"]  # a non-empty model label
        n_first = len(calls)
        assert n_first == 3

        # Second pass: every candidate already on disk -> runner never called again.
        _run(cand, corpus_path, "--limit", "3", "--conditioned-frac", "0.4", "--workers", "1")
        assert len(calls) == n_first  # no new runner calls
        assert "all candidates already exist" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (c) paired scoring joins on trace id, references read from the DB
# ---------------------------------------------------------------------------
class TestPairedScoring:
    def test_join_on_trace_id_free_and_conditioned(self, monkeypatch, capsys, corpus_path, tmp_path, keys):
        cand = tmp_path / "cand"
        menus = _menu_by_restaurant()  # full 2-item menu per restaurant
        plan = _plan(corpus_path, 3, 0.4)
        cond_tid, _cond_name, _ = next((t, n, r) for t, n, r in plan if r)

        # Reference = teacher eval traces written straight into the DB (NOT a
        # --reference dir). The conditioned reference shares the restaurant's id but
        # a distinct trace id -- the collision the trace-id join (not rid-join) avoids.
        with open_corpus(corpus_path) as cx:
            for tid, name, restrictions in plan:
                if restrictions:
                    # vegetarian reference keeps only the SECOND item of the full menu.
                    full = menus[name]["menu"][0]["items"]
                    ref_menu = _menu(name, full[1]["name"])
                else:
                    ref_menu = menus[name]
                cx.write_trace({
                    "restaurant_id": tid.split("__")[0],
                    "dietary_restrictions": restrictions or None,
                    "model": "claude-sonnet-5",
                    "prompt_variant": "teacher",
                    "found": ref_menu["found"],
                    "schema_valid": True,
                    "final_json": ref_menu,
                    "messages": [{"role": "user", "content": f"{name}, ref"}],
                    "captured_at": "2026-07-18T00:00:00Z",
                })

        # Candidate stub returns each restaurant's FULL menu regardless of the
        # restriction -> the conditioned episode over-includes (precision < 1).
        def stub(client, episode_input, tools, registry, system_prompt, model=None):
            name = episode_input.split(",")[0].strip()
            return json.dumps(menus[name]), []
        monkeypatch.setattr(ev, "claude_run_episode", stub)

        _run(cand, corpus_path, "--limit", "3", "--conditioned-frac", "0.4",
             "--workers", "1", "--json", str(tmp_path / "report.json"))
        out = capsys.readouterr().out
        assert "candidate vs reference" in out
        assert "joined on trace_id: 3" in out
        for block in ("--- all (n=3) ---", "--- free (n=2) ---", "--- conditioned (n=1) ---"):
            assert block in out

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["mode"] == "paired"
        assert report["aggregate"]["free"]["n_episodes"] == 2
        assert report["aggregate"]["conditioned"]["n_episodes"] == 1
        # Free episodes are exact matches -> perfect F1; the conditioned one isn't.
        assert report["aggregate"]["free"]["f1_mean"] == 1.0
        cond_ep = report["episodes"][cond_tid]
        assert cond_ep["conditioned"] is True
        assert cond_ep["findable"] is True  # derived from the reference found flag
        # candidate has both items, reference kept one -> recall 1.0, precision 0.5.
        assert cond_ep["recall"] == 1.0 and cond_ep["precision"] == 0.5


# ---------------------------------------------------------------------------
# (d) a CacheMiss is recorded as a failed candidate, not a crash
# ---------------------------------------------------------------------------
class TestCacheMiss:
    def test_cache_miss_recorded_not_crashed(self, monkeypatch, capsys, corpus_path, tmp_path, keys):
        cand = tmp_path / "cand"

        def stub(client, episode_input, tools, registry, system_prompt, model=None):
            raise CacheMiss(f"scrape miss for {episode_input}")
        monkeypatch.setattr(ev, "claude_run_episode", stub)

        # Should NOT raise. Every episode becomes an empty candidate on disk.
        _run(cand, corpus_path, "--limit", "2", "--conditioned-frac", "0.0", "--workers", "1")
        out = capsys.readouterr().out
        assert "cache-misses (recorded empty): 2" in out

        files = sorted(cand.glob("*.json"))
        assert len(files) == 2
        trace = json.loads(files[0].read_text())
        assert trace["cache_miss"] is True
        assert trace["final_json"] is None
        assert trace["schema_valid"] is False

    def test_self_report_runs_over_empty_candidates(self, monkeypatch, capsys, corpus_path, tmp_path, keys):
        # Mixed: one good, one CacheMiss -> self-report still aggregates cleanly.
        cand = tmp_path / "cand"
        menus = _menu_by_restaurant()
        # Pick the SECOND free episode's restaurant so exactly one of two misses,
        # independent of the seeded order (avoids hardcoding which row is front).
        miss_name = _plan(corpus_path, 2, 0.0)[1][1]

        def stub(client, episode_input, tools, registry, system_prompt, model=None):
            name = episode_input.split(",")[0].strip()
            if name == miss_name:
                raise CacheMiss("miss")
            return json.dumps(menus[name]), []
        monkeypatch.setattr(ev, "claude_run_episode", stub)

        _run(cand, corpus_path, "--limit", "2", "--conditioned-frac", "0.0", "--workers", "1")
        out = capsys.readouterr().out
        assert "self-report (no reference)" in out
        assert "--- all (n=2) ---" in out
        assert "schema-valid:   50.0%" in out  # one valid, one empty


# ---------------------------------------------------------------------------
# (e) the report carries HOW it was produced: cache hit rate + run stats
# ---------------------------------------------------------------------------
class TestRunAndCacheStats:
    """A score without its cache hit rate can't be read honestly (a model that ran
    off the warmed distribution is partly being scored on the cache), so `cache` and
    `run` travel inside the report JSON -- not just the console."""

    def test_report_carries_cache_and_run_blocks(self, monkeypatch, corpus_path, tmp_path, keys):
        cand = tmp_path / "cand"
        monkeypatch.setattr(ev, "claude_run_episode", _make_stub(_menu_by_restaurant()))
        _run(cand, corpus_path, "--limit", "2", "--conditioned-frac", "0.0", "--workers", "1",
             "--json", str(tmp_path / "report.json"))

        report = json.loads((tmp_path / "report.json").read_text())
        cache, run = report["cache"], report["run"]
        # hit_rate is over LOOKUPS; the stub calls no tools, so there are none -> None
        # (an honest "unknown", never a misleading 0.0 or 1.0).
        assert set(cache) >= {"hits", "misses", "writes", "lookups", "hit_rate", "miss_policy"}
        assert cache["lookups"] == cache["hits"] + cache["misses"]
        assert cache["hit_rate"] is None
        assert run["n_completed"] == 2 and run["n_failed"] == 0
        assert run["n_planned"] == 2 and run["n_todo"] == 2 and run["workers"] == 1
        assert run["elapsed_s"] > 0 and run["failures"] == []

    def test_hit_rate_is_over_lookups(self):
        class FakeCache:
            def stats(self):
                return {"hits": 3, "misses": 1, "writes": 1, "miss_policy": "live",
                        "cache_version": 1}
        got = ev.cache_report(FakeCache())
        # 3 hits / 4 lookups -- writes are a SUBSET of misses and must not be counted.
        assert got["lookups"] == 4 and got["hit_rate"] == 0.75

    def test_scoring_only_pass_reports_null_stats(self, monkeypatch, corpus_path, tmp_path, keys):
        # Re-scoring existing candidates produces no run/cache facts; the report says
        # so with nulls rather than inventing a 100% hit rate.
        cand = tmp_path / "cand"
        monkeypatch.setattr(ev, "claude_run_episode", _make_stub(_menu_by_restaurant()))
        _run(cand, corpus_path, "--limit", "2", "--conditioned-frac", "0.0", "--workers", "1")
        _run(cand, corpus_path, "--limit", "2", "--conditioned-frac", "0.0", "--workers", "1",
             "--json", str(tmp_path / "report.json"))
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["run"] is None and report["cache"] is None


# ---------------------------------------------------------------------------
# (f) W&B is key-gated and never fatal
# ---------------------------------------------------------------------------
class TestWandb:
    def test_no_key_is_a_noop(self, monkeypatch, corpus_path, tmp_path):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        args = ev.parse_args(_argv(tmp_path / "cand", corpus_path))
        assert ev.init_wandb(args, 10) is None

    def test_disabled_by_flag_even_with_key(self, monkeypatch, corpus_path, tmp_path):
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        args = ev.parse_args(_argv(tmp_path / "cand", corpus_path, "--no-wandb"))
        assert ev.init_wandb(args, 10) is None

    def test_init_failure_degrades_to_console(self, monkeypatch, capsys, corpus_path, tmp_path):
        """A telemetry misconfiguration must not kill a metered eval on a paid pod."""
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        fake = type("FakeWandb", (), {"init": staticmethod(
            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))})
        monkeypatch.setitem(sys.modules, "wandb", fake)
        args = ev.parse_args(_argv(tmp_path / "cand", corpus_path))
        assert ev.init_wandb(args, 10) is None
        assert "init failed" in capsys.readouterr().out

    def test_log_helpers_tolerate_a_broken_run(self, capsys):
        class BrokenRun:
            def log(self, *a, **k):
                raise RuntimeError("network down")
            summary = property(lambda self: (_ for _ in ()).throw(RuntimeError("nope")))
        ev.wandb_log(None, {"x": 1})            # None run -> silent no-op
        ev.wandb_log(BrokenRun(), {"x": 1})     # broken run -> caught, not raised
        ev.wandb_summarize(BrokenRun(), {"aggregate": {"all": {"found_rate": 1.0}},
                                         "abstention": {}, "cache": {}, "run": {}})
        assert "log failed" in capsys.readouterr().out
