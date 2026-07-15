"""Unit tests for scripts/build_grpo.py (Phase 3 GRPO dataset builder).

Pure-local, zero network / no tokenizer: a synthetic teacher trace (Anthropic
content-block shape, as data/traces/*.json store) -> a GRPO row, plus the
end-to-end reward path (row's reference string scored by make_grpo_reward).

Run: uv run python -m pytest tests/test_build_grpo.py -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grpo import load_eval_rids, trace_to_grpo_row  # noqa: E402


def trace(final_obj, *, dietary=None, rid="r1", episode="Joe's Pizza, NYC"):
    """A minimal teacher trace: user turn + a final assistant answer (JSON text)."""
    return {
        "restaurant_id": rid,
        "dietary_restrictions": dietary,
        "model": "claude-sonnet-5",
        "messages": [
            {"role": "user", "content": episode},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(final_obj)}]},
        ],
    }


MENU = {
    "found": True, "restaurant_name": "Joe's Pizza", "cuisine": "Italian",
    "menu": [{"section": "Pizza", "items": [{"name": "Margherita", "description": None, "price": 12.0}]}],
    "source_url": "https://joes.example/menu",
}


def test_basic_row_shape():
    row = trace_to_grpo_row(trace(MENU))
    assert row["restaurant_id"] == "r1"
    assert row["found"] is True
    # prompt is [system, user]; tools are NOT rendered here.
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][1]["content"] == "Joe's Pizza, NYC"
    # reference is a compact JSON STRING that round-trips to the menu.
    assert isinstance(row["reference"], str)
    assert json.loads(row["reference"]) == MENU


def test_student_prompt_used_and_dietary_preserved():
    row = trace_to_grpo_row(trace(MENU, dietary=["vegetarian"]))
    system = row["prompt"][0]["content"]
    # dietary restriction (target-defining) is present in the student prompt...
    assert "vegetarian" in system
    # ...but teacher-only source-selection guidance is not (student variant).
    assert "Source selection:" not in system


def test_found_false_reference_kept():
    nf = {"found": False, "restaurant_name": "Ghost", "cuisine": None, "menu": [],
          "source_url": None, "notes": "no menu"}
    row = trace_to_grpo_row(trace(nf))
    assert row["found"] is False
    assert json.loads(row["reference"])["found"] is False


@pytest.mark.parametrize("bad", [
    {"restaurant_id": "x", "messages": []},                                  # no messages
    {"restaurant_id": "x", "messages": [{"role": "assistant", "content": "x"}]},  # first not user
])
def test_bad_traces_raise(bad):
    with pytest.raises(ValueError):
        trace_to_grpo_row(bad)


def test_unparseable_final_raises():
    t = trace(MENU)
    t["messages"][-1]["content"] = [{"type": "text", "text": "no json here"}]
    with pytest.raises(ValueError):
        trace_to_grpo_row(t)


def test_reference_is_analysis_only_metadata():
    # The reward is now teacher-free (grounding-based), so `reference` is retained
    # only as offline-analysis metadata -- it is NOT consumed by the reward. Assert
    # it still round-trips so eval/analysis can use it.
    row = trace_to_grpo_row(trace(MENU))
    assert json.loads(row["reference"]) == MENU


def test_load_eval_rids(tmp_path):
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"train": ["a", "b"], "eval": ["c", "d"]}), encoding="utf-8")
    assert load_eval_rids(p) == {"c", "d"}
    assert load_eval_rids(None) == set()
    assert load_eval_rids(tmp_path / "missing.json") == set()
