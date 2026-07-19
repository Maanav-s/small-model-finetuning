"""Unit tests for scripts/datasets/build_grpo.py (v2 GRPO dataset builder).

v2 build_grpo is TRACE-FREE: it reads grpo-split RESTAURANTS from corpus.sqlite
(iter_restaurants(split="grpo")), plans a seeded free+conditioned episode mix with
the shared src/episodes.plan_episodes, and renders each into a student-prompt rollout
seed via episode_to_row. There is NO teacher trace, NO reference, and NO eval-leak
guard (the DB split is disjoint by restaurant_id). So the v1 tests for
trace_to_grpo_row / load_eval_rids are DROPPED (those functions no longer exist).

Pure-local, zero network / no tokenizer.

Run: uv run python -m pytest tests/test_build_grpo.py -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/, the
# script under test in scripts/datasets/ (the v2 location).
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))

import build_grpo  # noqa: E402
from corpus import open_corpus, restaurant_id_for  # noqa: E402


# ---------------------------------------------------------------------------
# episode_to_row: one planned episode -> one student-agnostic prompt row
# ---------------------------------------------------------------------------
def _episode(name="Joe's Pizza", city="NYC", restrictions=None, rid="r1"):
    return {"row": {"restaurant_id": rid, "name": name, "city": city, "source": "osm"},
            "restrictions": restrictions or []}


def test_free_row_shape():
    row = build_grpo.episode_to_row(_episode())
    assert row["restaurant_id"] == "r1"
    # trace-free GRPO has no teacher outcome -> found is unknown (None).
    assert row["found"] is None
    assert row["dietary_restrictions"] is None
    # prompt is [system, user]; tools are NOT rendered here.
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][1]["content"] == "Joe's Pizza, NYC"


def test_student_prompt_used_and_dietary_preserved():
    row = build_grpo.episode_to_row(_episode(restrictions=["vegetarian"]))
    assert row["dietary_restrictions"] == ["vegetarian"]
    system = row["prompt"][0]["content"]
    # dietary restriction (target-defining) is present in the student prompt...
    assert "vegetarian" in system
    # ...but teacher-only source-selection guidance is not (student variant).
    assert "Source selection:" not in system


# ---------------------------------------------------------------------------
# main(): grpo-split restaurants only, seeded free/conditioned split
# ---------------------------------------------------------------------------
def _corpus_with_splits(path):
    with open_corpus(path) as cx:
        cx.upsert_restaurants(
            [{"name": f"Grpo{i}", "city": "Town", "source": "osm", "split": "grpo"}
             for i in range(5)]
            + [{"name": "EvalOne", "city": "Town", "source": "osm", "split": "eval"},
               {"name": "SftOne", "city": "Town", "source": "osm", "split": "sft"}]
        )
    return {restaurant_id_for(f"Grpo{i}", "Town") for i in range(5)}


def test_main_reads_grpo_split_only(tmp_path):
    corpus = tmp_path / "corpus.sqlite"
    grpo_rids = _corpus_with_splits(corpus)
    out = tmp_path / "grpo" / "train.jsonl"

    build_grpo.main(["--corpus", str(corpus), "--out", str(out), "--seed", "42"])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    # default: one free episode per grpo restaurant (5); eval/sft never surface.
    assert len(rows) == 5
    assert all(r["restaurant_id"] in grpo_rids for r in rows)
    assert all(r["found"] is None for r in rows)
    assert all([m["role"] for m in r["prompt"]] == ["system", "user"] for r in rows)
    assert all(r["dietary_restrictions"] is None for r in rows)  # pure free by default


def test_main_conditioned_frac_splits_free_and_conditioned(tmp_path):
    corpus = tmp_path / "corpus.sqlite"
    _corpus_with_splits(corpus)
    out = tmp_path / "grpo" / "train.jsonl"

    # limit 5, frac 0.4 -> round(5*0.4)=2 conditioned + 3 free.
    build_grpo.main(["--corpus", str(corpus), "--out", str(out),
                     "--conditioned-frac", "0.4", "--limit", "5", "--seed", "42"])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    free = [r for r in rows if r["dietary_restrictions"] is None]
    cond = [r for r in rows if r["dietary_restrictions"]]
    assert len(free) == 3 and len(cond) == 2
    # conditioned rows carry the restriction into the student system prompt.
    assert all("Source selection:" not in r["prompt"][0]["content"] for r in rows)
    for r in cond:
        assert r["dietary_restrictions"][0] in r["prompt"][0]["content"]

    # provenance sidecar records the free/conditioned split.
    meta = json.loads((out.with_name(out.name + ".meta.json")).read_text(encoding="utf-8"))
    assert meta["n_free"] == 3 and meta["n_conditioned"] == 2


def test_main_empty_grpo_split_writes_nothing(tmp_path):
    corpus = tmp_path / "corpus.sqlite"
    with open_corpus(corpus) as cx:
        cx.upsert_restaurants([{"name": "OnlyEval", "city": "Town", "source": "osm", "split": "eval"}])
    out = tmp_path / "grpo" / "train.jsonl"
    build_grpo.main(["--corpus", str(corpus), "--out", str(out)])
    assert out.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("bad_frac", ["1.5", "-0.1"])
def test_main_rejects_out_of_range_frac(tmp_path, bad_frac):
    corpus = tmp_path / "corpus.sqlite"
    _corpus_with_splits(corpus)
    out = tmp_path / "grpo" / "train.jsonl"
    with pytest.raises(ValueError):
        build_grpo.main(["--corpus", str(corpus), "--out", str(out),
                         "--conditioned-frac", bad_frac, "--limit", "5"])
