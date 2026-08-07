"""Unit tests for the two reporting-side eval scripts:

  * scripts/eval/dump_reference.py -- DB traces -> candidate files, so the TEACHER is
    scored by the same scorer as the students (it is the reference, so it can only be
    self-reported; pairing it against itself would print 1.000).
  * scripts/eval/summarize.py -- a directory of report.json files -> one Markdown
    comparison table (the committed record in results/<run-set>/README.md).

No network, no torch, no GPU.

Run: uv run python -m pytest tests/test_eval_reporting.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

import dump_reference as dr  # noqa: E402
import eval as ev  # noqa: E402
import summarize as sm  # noqa: E402
from corpus import open_corpus  # noqa: E402

STAMP = "2026-08-07T00:00:00+00:00"


def _menu(name, *items, found=True):
    return {
        "found": found, "restaurant_name": name, "cuisine": "Test",
        "menu": [{"section": "Mains",
                  "items": [{"name": i, "description": None, "price": 9.0} for i in items]}]
        if items else [],
        "source_url": "https://example.com/menu",
    }


@pytest.fixture
def corpus_with_traces(tmp_path):
    """An eval-split corpus holding one free + one conditioned teacher trace, plus a
    REJECTED one (the cleaning pass's flag) that must not be dumped by default."""
    p = tmp_path / "corpus.sqlite"
    with open_corpus(p) as cx:
        cx.upsert_restaurants([
            {"name": "Alpha Cafe", "city": "Austin", "source": "osm", "split": "eval"},
            {"name": "Bravo Bistro", "city": "Boston", "source": "osm", "split": "eval"},
        ])
        rows = list(cx.iter_restaurants(split="eval"))
        by_name = {r["name"]: r["restaurant_id"] for r in rows}
        cx.write_trace({
            "restaurant_id": by_name["Alpha Cafe"], "dietary_restrictions": None,
            "model": "teacher", "prompt_variant": "teacher", "found": 1, "schema_valid": 1,
            "grounding": 0.9, "final_json": _menu("Alpha Cafe", "Burger", "Fries"),
            "messages": [{"role": "user", "content": "Alpha Cafe, Austin"}],
            "queries": ["alpha cafe austin menu"], "urls": ["https://example.com/menu"],
            "captured_at": STAMP,
        })
        cx.write_trace({
            "restaurant_id": by_name["Alpha Cafe"], "dietary_restrictions": ["vegetarian"],
            "model": "teacher", "prompt_variant": "teacher", "found": 1, "schema_valid": 1,
            "final_json": _menu("Alpha Cafe", "Fries"),
            "messages": [], "queries": [], "urls": [], "captured_at": STAMP,
        })
        cx.write_trace({
            "restaurant_id": by_name["Bravo Bistro"], "dietary_restrictions": None,
            "model": "teacher", "prompt_variant": "teacher", "found": 0, "schema_valid": 1,
            "final_json": _menu("Bravo Bistro", found=False),
            "messages": [], "queries": [], "urls": [], "captured_at": STAMP,
            "rejected": 1, "reject_reason": "test",
        })
    return p


class TestDumpReference:
    def test_dumps_candidate_shaped_files(self, capsys, corpus_with_traces, tmp_path):
        out = tmp_path / "cand"
        dr.main([str(out), "--corpus", str(corpus_with_traces)])

        files = sorted(p.name for p in out.glob("*.json"))
        # free -> <rid>.json, conditioned -> <rid>__<slug>.json; the REJECTED trace is
        # excluded (it is not part of the reference eval.py would score against).
        assert len(files) == 2
        assert any(f.endswith("__vegetarian.json") for f in files)
        assert "wrote 2 candidate files" in capsys.readouterr().out

        # The dumped files are readable by the SAME loader eval.py scores with, and
        # key on trace_id (so free/conditioned slices of one restaurant never collide).
        loaded, unreadable = ev.load_candidates(out)
        assert unreadable == [] and len(loaded) == 2
        for tid, obj in loaded.items():
            assert obj["trace_id"] == tid
            assert ev.menu_of(obj)["found"] is True          # final_json is the scored menu
            assert obj["model"] == "teacher"
        cond_tid = next(t for t in loaded if "__" in t)
        assert ev.is_conditioned(cond_tid, loaded[cond_tid]) is True
        assert loaded[cond_tid]["dietary_restrictions"] == ["vegetarian"]

    def test_carries_reference_extras_but_not_the_trajectory(self, corpus_with_traces, tmp_path):
        out = tmp_path / "cand"
        dr.main([str(out), "--corpus", str(corpus_with_traces)])
        free = json.loads(next(p for p in out.glob("*.json") if "__" not in p.name)
                          .read_text(encoding="utf-8"))
        assert free["grounding"] == 0.9 and free["queries"] == ["alpha cafe austin menu"]
        # `messages` is the biggest column by far -- summarized, never copied.
        assert free["n_messages"] == 1 and "messages" not in free

    def test_include_rejected_opt_in(self, corpus_with_traces, tmp_path):
        out = tmp_path / "cand"
        dr.main([str(out), "--corpus", str(corpus_with_traces), "--include-rejected"])
        assert len(list(out.glob("*.json"))) == 3

    def test_empty_split_exits_with_guidance(self, corpus_with_traces, tmp_path):
        with pytest.raises(SystemExit) as exc:
            dr.main([str(tmp_path / "cand"), "--corpus", str(corpus_with_traces),
                     "--split", "grpo"])
        assert "build_corpus" in str(exc.value)


def _report(path, *, mode, label, found, hit_rate=None, **agg):
    payload = {
        "mode": mode, "model": "vllm", "model_id": label,
        "checkpoint": {"run_id": None, "md5": None},
        "cache_policy": "live",
        "aggregate": {"all": {"n_episodes": 10,
                              ("found_rate" if mode == "self-report" else "found_accuracy"): found,
                              "schema_valid_rate": 0.9, **agg},
                      "free": {"n_episodes": 6}, "conditioned": {"n_episodes": 4}},
        "cache": {"hits": 9, "misses": 1, "hit_rate": hit_rate} if hit_rate is not None else {},
        "run": {"n_failed": 0, "eps_per_min": 12.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestSummarize:
    def test_table_lists_teacher_first_then_students_by_score(self, corpus_with_traces, tmp_path):
        d = tmp_path / "run-set"
        d.mkdir()
        _report(d / "teacher.json", mode="self-report", label="Qwen3-235B", found=0.85,
                hit_rate=0.9)
        _report(d / "gemma-base.json", mode="paired", label="gemma-base", found=0.40,
                hit_rate=0.5, f1_mean=0.11)
        _report(d / "gemma-sft.json", mode="paired", label="gemma-sft", found=0.78,
                hit_rate=0.95, f1_mean=0.62)
        out = d / "README.md"
        sm.main([str(d), "-o", str(out)])

        table = out.read_text(encoding="utf-8")
        order = [table.index(m) for m in ("Qwen3-235B", "gemma-sft", "gemma-base")]
        # self-report (the reference/ceiling) first, then paired rows by descending score.
        assert order == sorted(order)
        assert "85.0%" in table and "0.620" in table
        # The cache hit rate is a first-class column: a score can't be read without it.
        assert "cache hit-rate" in table and "95.0%" in table
        # Metrics that don't exist in a mode render as '--', never as a fake 0.
        teacher_row = next(ln for ln in table.splitlines() if "Qwen3-235B" in ln)
        assert "| -- |" in teacher_row

    def test_slice_selects_the_aggregate(self, tmp_path):
        d = tmp_path / "rs"
        d.mkdir()
        _report(d / "m.json", mode="paired", label="m", found=0.5)
        sm.main([str(d), "-o", str(d / "free.md"), "--slice", "free"])
        assert "| 6 |" in (d / "free.md").read_text(encoding="utf-8")

    def test_ignores_non_report_json(self, tmp_path):
        d = tmp_path / "rs"
        d.mkdir()
        (d / "train.jsonl.meta.json").write_text('{"md5": "abc"}', encoding="utf-8")
        _report(d / "m.json", mode="paired", label="m", found=0.5)
        sm.main([str(d), "-o", str(d / "t.md")])
        assert len([ln for ln in (d / "t.md").read_text(encoding="utf-8").splitlines()
                    if ln.startswith("| ") and "model" not in ln]) == 1

    def test_no_reports_is_an_error(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(SystemExit):
            sm.main([str(d)])
