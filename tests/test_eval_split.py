"""Unit tests for scripts/eval_split.py (Phase 2 WS-G, the run+score half).

No network, no torch, no GPU: the Claude runner is monkeypatched with a stub that
returns a canned (final_text, messages) pair, and dummy BRAVE_API_KEY /
ANTHROPIC_API_KEY env vars let setup_tools build its (unused) closures and the
Anthropic client construct without hitting anything. Everything runs the
--model claude path so torch is never imported.

Run: uv run python -m pytest tests/test_eval_split.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Flat-import, script-run convention (see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "claude"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_split  # noqa: E402
from cache import CacheMiss  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: a tiny eval-split data dir + menu builders
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


@pytest.fixture
def data_dir(tmp_path):
    """restaurants.jsonl + splits.json with 3 eval rows (+1 train row ignored)."""
    rows = [
        {"restaurant_id": "aaaa000000000001", "name": "Alpha Cafe", "city": "Austin", "country": "US"},
        {"restaurant_id": "bbbb000000000002", "name": "Bravo Bistro", "city": "Boston", "country": "US"},
        {"restaurant_id": "cccc000000000003", "name": "Charlie Grill", "city": "Chicago", "country": "US"},
        {"restaurant_id": "dddd000000000004", "name": "Delta Diner", "city": "Denver", "country": "US"},
    ]
    (tmp_path / "restaurants.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    splits = {"aaaa000000000001": "eval", "bbbb000000000002": "eval",
              "cccc000000000003": "eval", "dddd000000000004": "train"}
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return tmp_path


@pytest.fixture
def keys(monkeypatch):
    """Dummy keys so setup_tools + anthropic.Anthropic() construct without network."""
    monkeypatch.setenv("BRAVE_API_KEY", "test-brave")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")


def _argv(candidate_dir, data_dir, *extra):
    return ["eval_split.py", str(candidate_dir), "--model", "claude",
            "--data-dir", str(data_dir), "--cache-path", str(data_dir / "cache.sqlite"),
            *extra]


def _run(monkeypatch, candidate_dir, data_dir, *extra):
    monkeypatch.setattr(sys, "argv", _argv(candidate_dir, data_dir, *extra))
    eval_split.main()


def _plan(data_dir, limit, frac, seed=42, split="eval"):
    """The planned episodes, using the SAME planner eval_split runs -- so tests
    assert against the real seeded order instead of guessing which row is front."""
    rows = eval_split.load_seeded_rows(data_dir, split, seed)
    episodes = eval_split.plan_episodes(rows, limit, frac)
    return [(eval_split.episode_trace_name(e["row"]["restaurant_id"], e["restrictions"]),
             e["row"]["name"], e["restrictions"]) for e in episodes]


# ---------------------------------------------------------------------------
# (a) --list plans the free + conditioned episodes with correct filenames
# ---------------------------------------------------------------------------
class TestList:
    def test_list_prints_planned_episodes(self, monkeypatch, capsys, data_dir, tmp_path):
        cand = tmp_path / "cand"
        # 3 eval rows, limit 3, frac 0.4 -> round(1.2)=1 conditioned + 2 free.
        _run(monkeypatch, cand, data_dir, "--list", "--limit", "3", "--conditioned-frac", "0.4")
        out = capsys.readouterr().out
        assert "2 free + 1 conditioned" in out
        # Free episodes -> <rid>.json ; conditioned -> <rid>__<slug>.json.
        assert "aaaa000000000001.json" in out
        assert "__vegetarian.json" in out  # first DIETARY_POOL entry on the seeded front row
        # --list never touches the network/GPU: no candidate files written.
        assert not cand.exists() or list(cand.glob("*.json")) == []

    def test_list_requires_no_keys(self, monkeypatch, capsys, data_dir, tmp_path):
        # No `keys` fixture here: --list must work with no API keys at all.
        _run(monkeypatch, tmp_path / "cand", data_dir, "--list", "--limit", "2",
             "--conditioned-frac", "0.0")
        assert "2 free + 0 conditioned" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Stub runner: canned final_text keyed off the episode input
# ---------------------------------------------------------------------------
def _menu_by_restaurant():
    return {
        "Alpha Cafe": _menu("Alpha Cafe", "Burger", "Fries"),
        "Bravo Bistro": _menu("Bravo Bistro", "Steak", "Salad"),
        "Charlie Grill": _menu("Charlie Grill", "Wings"),
    }


def _make_stub(menus, calls=None):
    def stub(client, episode_input, tools, registry, system_prompt, model=None):
        if calls is not None:
            calls.append(episode_input)
        name = episode_input.split(",")[0].strip()
        return json.dumps(menus[name]), [{"role": "user", "content": episode_input}]
    return stub


# ---------------------------------------------------------------------------
# (b) candidate traces written with the right names + idempotent skip
# ---------------------------------------------------------------------------
class TestCandidateWrite:
    def test_writes_and_is_idempotent(self, monkeypatch, capsys, data_dir, tmp_path, keys):
        cand = tmp_path / "cand"
        calls = []
        monkeypatch.setattr(eval_split, "claude_run_episode", _make_stub(_menu_by_restaurant(), calls))
        _run(monkeypatch, cand, data_dir, "--limit", "3", "--conditioned-frac", "0.4",
             "--workers", "1")

        plan = _plan(data_dir, 3, 0.4)
        expected = sorted(fname for fname, _, _ in plan)
        files = sorted(p.name for p in cand.glob("*.json"))
        # 2 free + 1 conditioned = 3 candidate files (no .tmp left behind).
        assert files == expected
        cond_fname = next(fname for fname, _, r in plan if r)  # the conditioned episode
        assert cond_fname.endswith("__vegetarian.json")  # DIETARY_POOL[0]
        trace = json.loads((cand / cond_fname).read_text())
        assert trace["dietary_restrictions"] == ["vegetarian"]
        assert trace["prompt_variant"] == "student"
        assert trace["schema_valid"] is True
        assert trace["final_json"]["found"] is True
        assert trace["model"]  # a non-empty model label
        n_first = len(calls)
        assert n_first == 3

        # Second pass: every candidate already on disk -> runner never called again.
        _run(monkeypatch, cand, data_dir, "--limit", "3", "--conditioned-frac", "0.4",
             "--workers", "1")
        assert len(calls) == n_first  # no new runner calls
        assert "all candidates already exist" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (c) paired scoring joins on filename with free/conditioned breakdown
# ---------------------------------------------------------------------------
class TestPairedScoring:
    def test_join_on_filename_free_and_conditioned(self, monkeypatch, capsys, data_dir, tmp_path, keys):
        cand = tmp_path / "cand"
        ref = tmp_path / "ref"
        ref.mkdir()

        menus = _menu_by_restaurant()  # full 2-item menu per restaurant (Charlie has 1)
        plan = _plan(data_dir, 3, 0.4)
        cond_fname, cond_name, _ = next((f, n, r) for f, n, r in plan if r)

        # Reference traces at the SAME filenames the planner produces. The
        # conditioned reference shares the restaurant's id but a distinct filename
        # -- the collision the filename-join (not id-join) must avoid.
        for fname, name, restrictions in plan:
            if restrictions:
                # vegetarian reference keeps only the SECOND item of the full menu.
                full = menus[name]["menu"][0]["items"]
                ref_menu = _menu(name, full[1]["name"])
            else:
                ref_menu = menus[name]
            (ref / fname).write_text(json.dumps({
                "restaurant_id": fname.split("__")[0].split(".")[0],
                "final_json": ref_menu, "dietary_restrictions": restrictions or None,
            }), encoding="utf-8")

        # Candidate stub returns each restaurant's FULL menu regardless of the
        # restriction -> the conditioned episode over-includes (precision < 1).
        def stub(client, episode_input, tools, registry, system_prompt, model=None):
            name = episode_input.split(",")[0].strip()
            return json.dumps(menus[name]), []
        monkeypatch.setattr(eval_split, "claude_run_episode", stub)

        _run(monkeypatch, cand, data_dir, "--limit", "3", "--conditioned-frac", "0.4",
             "--workers", "1", "--reference", str(ref), "--json", str(tmp_path / "report.json"))
        out = capsys.readouterr().out
        assert "candidate vs reference" in out
        assert "joined on episode filename: 3" in out
        for block in ("--- all (n=3) ---", "--- free (n=2) ---", "--- conditioned (n=1) ---"):
            assert block in out

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["mode"] == "paired"
        assert report["aggregate"]["free"]["n_episodes"] == 2
        assert report["aggregate"]["conditioned"]["n_episodes"] == 1
        # Free episodes are exact matches -> perfect F1; the conditioned one isn't.
        assert report["aggregate"]["free"]["f1_mean"] == 1.0
        cond_ep = report["episodes"][cond_fname]
        assert cond_ep["conditioned"] is True
        # candidate has both items, reference kept one -> recall 1.0, precision 0.5.
        assert cond_ep["recall"] == 1.0 and cond_ep["precision"] == 0.5


# ---------------------------------------------------------------------------
# (d) a CacheMiss is recorded as a failed candidate, not a crash
# ---------------------------------------------------------------------------
class TestCacheMiss:
    def test_cache_miss_recorded_not_crashed(self, monkeypatch, capsys, data_dir, tmp_path, keys):
        cand = tmp_path / "cand"

        def stub(client, episode_input, tools, registry, system_prompt, model=None):
            raise CacheMiss(f"scrape miss for {episode_input}")
        monkeypatch.setattr(eval_split, "claude_run_episode", stub)

        # Should NOT raise. Every episode becomes an empty candidate on disk.
        _run(monkeypatch, cand, data_dir, "--limit", "2", "--conditioned-frac", "0.0",
             "--workers", "1")
        out = capsys.readouterr().out
        assert "cache-misses (recorded empty): 2" in out

        files = sorted(cand.glob("*.json"))
        assert len(files) == 2
        trace = json.loads(files[0].read_text())
        assert trace["cache_miss"] is True
        assert trace["final_json"] is None
        assert trace["schema_valid"] is False

    def test_self_report_runs_over_empty_candidates(self, monkeypatch, capsys, data_dir, tmp_path, keys):
        # Mixed: one good, one CacheMiss -> self-report still aggregates cleanly.
        cand = tmp_path / "cand"
        menus = _menu_by_restaurant()

        def stub(client, episode_input, tools, registry, system_prompt, model=None):
            name = episode_input.split(",")[0].strip()
            if name == "Bravo Bistro":
                raise CacheMiss("miss")
            return json.dumps(menus[name]), []
        monkeypatch.setattr(eval_split, "claude_run_episode", stub)

        _run(monkeypatch, cand, data_dir, "--limit", "2", "--conditioned-frac", "0.0", "--workers", "1")
        out = capsys.readouterr().out
        assert "self-report (no reference)" in out
        assert "--- all (n=2) ---" in out
        assert "schema-valid:   50.0%" in out  # one valid, one empty
