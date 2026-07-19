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
