"""Unit tests for src/corpus.py (v2 corpus store).

No network; real-file sqlite via pytest tmp_path.

Run: uv run python -m pytest tests/test_corpus.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corpus import (  # noqa: E402
    Corpus,
    VALID_SPLITS,
    open_corpus,
    restaurant_id_for,
    restriction_slug,
    trace_id_for,
)


def _restaurants():
    return [
        {"name": "Ssamjang", "city": "Atlanta", "source": "osm", "is_chain": False},
        {"name": "Pagliacci", "city": "Seattle", "source": "osm", "is_chain": False},
        {"name": "Serious Pie", "city": "Seattle", "source": "osm", "is_chain": False},
        {"name": "McDonald's", "city": "Chicago", "source": "osm", "is_chain": True},
    ]


def _trace(rid, restrictions=None, *, found=True, menu=None):
    menu = menu if menu is not None else [{"section": "Mains", "items": [{"name": "Bulgogi"}]}]
    return {
        "restaurant_id": rid,
        "model": "claude-sonnet-5",
        "prompt_variant": "teacher",
        "dietary_restrictions": restrictions,
        "found": found,
        "schema_valid": True,
        "final_json": {"found": found, "menu": menu},
        "messages": [{"role": "user", "content": [{"type": "tool_result", "content": "Bulgogi $18"}]}],
        "queries": ["q"],
        "urls": ["u"],
        "cache_version": 1,
        "captured_at": "2026-07-18T00:00:00Z",
    }


def test_id_derivation_stable_and_normalized():
    a = restaurant_id_for("Ssamjang", "Atlanta")
    b = restaurant_id_for("  ssamjang ", "ATLANTA")
    assert a == b and len(a) == 16


def test_trace_id_free_vs_conditioned():
    rid = "abc123"
    assert trace_id_for(rid, None) == rid
    assert trace_id_for(rid, []) == rid
    assert trace_id_for(rid, ["vegetarian", "no peanuts"]) == f"{rid}__{restriction_slug(['vegetarian','no peanuts'])}"
    assert "__" in trace_id_for(rid, ["vegan"])


def test_upsert_and_iter(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        assert cx.upsert_restaurants(_restaurants()) == 4
        assert cx.count_by_split() == {"sft": 0, "grpo": 0, "eval": 0, "unmarked": 4}
        names = [r["name"] for r in cx.iter_restaurants()]
        assert len(names) == 4
        # ids are derived + is_chain round-trips as bool
        r = cx.get_restaurant(restaurant_id_for("McDonald's", "Chicago"))
        assert r["is_chain"] is True and r["split"] is None


def test_upsert_preserves_split_when_omitted(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants())
        cx.assign_splits(seed=1, fractions={"sft": 0.5, "grpo": 0.25, "eval": 0.25})
        rid = restaurant_id_for("Ssamjang", "Atlanta")
        before = cx.get_restaurant(rid)["split"]
        assert before in VALID_SPLITS
        # re-harvest same rows WITHOUT split -> split preserved
        cx.upsert_restaurants(_restaurants())
        assert cx.get_restaurant(rid)["split"] == before


def test_assign_splits_deterministic_and_covers(tmp_path):
    def assign(path):
        with open_corpus(path) as cx:
            cx.upsert_restaurants(_restaurants())
            counts = cx.assign_splits(seed=42, fractions={"sft": 0.5, "grpo": 0.25, "eval": 0.25})
            return counts, {r["restaurant_id"]: r["split"] for r in cx.iter_restaurants()}

    counts_a, a = assign(tmp_path / "c.sqlite")
    _, b = assign(tmp_path / "c2.sqlite")
    assert sum(counts_a.values()) == 4
    assert "unmarked" not in set(a.values())
    # same seed + same id set -> identical assignment
    assert a == b


def test_assign_splits_fill_null_only_by_default(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants()[:2])
        cx.assign_splits(seed=1, fractions={"sft": 1.0})
        assert cx.count_by_split()["sft"] == 2
        # add two more, re-run: only the new (unmarked) get assigned; existing untouched
        cx.upsert_restaurants(_restaurants()[2:])
        counts = cx.assign_splits(seed=1, fractions={"eval": 1.0})
        assert counts == {"eval": 2}
        assert cx.count_by_split() == {"sft": 2, "grpo": 0, "eval": 2, "unmarked": 0}


def test_assign_splits_bad_fractions(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants())
        with pytest.raises(ValueError):
            cx.assign_splits(seed=1, fractions={"sft": 0.6, "grpo": 0.6})
        with pytest.raises(ValueError):
            cx.assign_splits(seed=1, fractions={"bogus": 1.0})


def test_reassign_refused_when_traces_exist(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants())
        cx.assign_splits(seed=1, fractions={"sft": 0.5, "grpo": 0.25, "eval": 0.25})
        rid = next(cx.iter_restaurants())["restaurant_id"]
        cx.write_trace(_trace(rid))
        with pytest.raises(ValueError, match="leakage"):
            cx.assign_splits(seed=2, fractions={"sft": 0.5, "grpo": 0.25, "eval": 0.25}, reassign=True)
        # force overrides
        cx.assign_splits(seed=2, fractions={"sft": 0.5, "grpo": 0.25, "eval": 0.25},
                         reassign=True, force=True)


def test_write_and_read_trace_roundtrip(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants())
        rid = restaurant_id_for("Ssamjang", "Atlanta")
        tid = cx.write_trace(_trace(rid))
        assert tid == rid and cx.has_trace(tid)
        t = cx.get_trace(tid)
        assert t["restaurant_name"] == "Ssamjang"
        assert t["episode_input"] == "Ssamjang, Atlanta"
        assert t["found"] is True
        assert t["final_json"]["menu"][0]["items"][0]["name"] == "Bulgogi"
        assert t["dietary_restrictions"] is None


def test_conditioned_trace_and_siblings(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants())
        rid = restaurant_id_for("Ssamjang", "Atlanta")
        free = cx.write_trace(_trace(rid))
        cond = cx.write_trace(_trace(rid, ["vegetarian"]))
        assert cond != free and cond.endswith("__vegetarian")
        assert cx.siblings(rid, exclude=free) == [cond]
        assert set(cx.siblings(rid)) == {free, cond}


def test_iter_traces_split_and_rejected(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants())
        rids = [r["restaurant_id"] for r in cx.iter_restaurants()]
        cx.upsert_restaurants([{**r, "split": "sft"} for r in _restaurants()[:2]])
        cx.upsert_restaurants([{**r, "split": "eval"} for r in _restaurants()[2:]])
        for rid in rids:
            cx.write_trace(_trace(rid))
        sft_ids = [t["trace_id"] for t in cx.iter_traces(split="sft")]
        assert len(sft_ids) == 2
        # reject one; default iter drops it
        cx.set_rejected(sft_ids[0], True, "wrong city")
        assert len(list(cx.iter_traces(split="sft"))) == 1
        assert len(list(cx.iter_traces(split="sft", include_rejected=True))) == 2


def test_set_grounding(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants(_restaurants())
        rid = restaurant_id_for("Ssamjang", "Atlanta")
        cx.write_trace(_trace(rid))
        cx.set_grounding(rid, 0.83, ["Mystery Dish"])
        t = cx.get_trace(rid)
        assert t["grounding"] == 0.83 and t["unmatched_items"] == ["Mystery Dish"]


def test_review_decision_and_counts(tmp_path):
    with open_corpus(tmp_path / "c.sqlite") as cx:
        cx.upsert_restaurants([{**r, "split": "sft"} for r in _restaurants()])
        rids = [r["restaurant_id"] for r in cx.iter_restaurants()]
        for rid in rids:
            cx.write_trace(_trace(rid))
        # all start unreviewed
        assert cx.review_counts() == {"reviewed": 0, "unreviewed": 4, "kept": 0, "rejected": 0}
        cx.set_review_decision(rids[0], "reject", "wrong city")
        cx.set_review_decision(rids[1], "keep")
        t0 = cx.get_trace(rids[0])
        assert t0["rejected"] is True and t0["reviewed_at"] and t0["reject_reason"] == "wrong city"
        assert cx.get_trace(rids[1])["rejected"] is False and cx.get_trace(rids[1])["reviewed_at"]
        assert cx.review_counts() == {"reviewed": 2, "unreviewed": 2, "kept": 1, "rejected": 1}
        # rejected traces are dropped from the default iter (build_sft path)
        assert rids[0] not in {t["trace_id"] for t in cx.iter_traces(split="sft")}
        # undecided un-reviews
        cx.set_review_decision(rids[0], "undecided")
        t0 = cx.get_trace(rids[0])
        assert t0["rejected"] is False and t0["reviewed_at"] is None
        with pytest.raises(ValueError):
            cx.set_review_decision(rids[0], "bogus")


def test_reviewed_at_migration_on_old_db(tmp_path):
    # Simulate a corpus.sqlite created before the reviewed_at column existed.
    import sqlite3
    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    # The original v2 traces schema (has trace_source; the only column added since
    # is reviewed_at), so the idx_traces_source index still builds on open.
    con.executescript(
        "CREATE TABLE restaurants (restaurant_id TEXT PRIMARY KEY, name TEXT, city TEXT, "
        "source TEXT, is_chain INTEGER, split TEXT);"
        "CREATE TABLE traces (trace_id TEXT PRIMARY KEY, restaurant_id TEXT, model TEXT, "
        "trace_source TEXT DEFAULT 'teacher', prompt_variant TEXT, found INTEGER, "
        "schema_valid INTEGER, final_json TEXT, messages TEXT, rejected INTEGER DEFAULT 0, "
        "reject_reason TEXT, captured_at TEXT);"
    )
    con.commit()
    con.close()
    # Opening it should ADD the reviewed_at column, not crash.
    with open_corpus(p) as cx:
        cols = {r["name"] for r in cx._conn.execute("PRAGMA table_info(traces)")}
        assert "reviewed_at" in cols


def test_open_corpus_create_false_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        Corpus(tmp_path / "nope.sqlite", create=False)
