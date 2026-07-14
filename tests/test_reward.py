"""Unit tests for src/reward.py (Phase 3 GRPO reward).

Pure-local, zero network: candidates are literal final_json dicts or raw strings,
references are literal dicts. Verifies the reward's SHAPE (the gradient GRPO needs):
empty < valid-but-thin < complete, hallucination is penalised in reference mode,
and correct abstention scores high.

Run: uv run python -m pytest tests/test_reward.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reward import (  # noqa: E402
    DEFAULT_WEIGHTS,
    RewardWeights,
    make_grpo_reward,
    menu_reward,
)


def it(name, price=None, description=None):
    return {"name": name, "description": description, "price": price}


def menu(*sections, found=True, name="Testaurant"):
    return {
        "found": found,
        "restaurant_name": name,
        "cuisine": "Test",
        "menu": [{"section": s, "items": items} for s, items in sections],
        "source_url": "https://example.com/menu",
    }


def not_found(name="Testaurant"):
    return {"found": False, "restaurant_name": name, "cuisine": None, "menu": [],
            "source_url": None, "notes": "no menu found"}


REF = menu(("Mains", [it("Margherita Pizza", 12.0), it("Caesar Salad", 9.0),
                      it("Tiramisu", 7.0), it("Espresso", 3.0)]))


# ---------------------------------------------------------------------------
# Weights invariant
# ---------------------------------------------------------------------------
def test_weights_sum_to_one():
    assert DEFAULT_WEIGHTS.total() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The empty-output floor (the v1 failure the reward must punish)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, "", "   ", "I could not find the menu.", "{not json"])
def test_empty_or_unparseable_scores_zero(bad):
    assert menu_reward(bad, REF) == 0.0
    assert menu_reward(bad, None) == 0.0  # ref-free too


def test_schema_invalid_dict_scores_zero():
    # missing required "found"/"restaurant_name": not a valid menu.
    assert menu_reward({"menu": [{"section": "x", "items": [it("a")]}]}, REF) == 0.0


# ---------------------------------------------------------------------------
# Reference-based: shape + hallucination guard
# ---------------------------------------------------------------------------
def test_perfect_match_scores_one():
    assert menu_reward(REF, REF) == pytest.approx(1.0)


def test_raw_text_answer_is_parsed():
    import json
    assert menu_reward(json.dumps(REF), REF) == pytest.approx(1.0)


def test_partial_menu_between_empty_and_complete():
    half = menu(("Mains", [it("Margherita Pizza", 12.0), it("Caesar Salad", 9.0)]))
    r = menu_reward(half, REF)
    assert 0.0 < r < 1.0


def test_more_complete_scores_higher():
    one = menu(("Mains", [it("Margherita Pizza", 12.0)]))
    three = menu(("Mains", [it("Margherita Pizza", 12.0), it("Caesar Salad", 9.0),
                            it("Tiramisu", 7.0)]))
    assert menu_reward(three, REF) > menu_reward(one, REF)


def test_valid_thin_menu_beats_empty():
    # A valid menu with a single (even wrong) item must outrank the empty floor,
    # so GRPO pushes the model to COMMIT rather than return "".
    thin = menu(("Mains", [it("Something", 5.0)]))
    assert menu_reward(thin, REF) > menu_reward("", REF)


def test_hallucinated_padding_penalised_in_reference_mode():
    # Same 4 correct items, plus 20 invented ones: precision drops, so F1 drops,
    # so reward must be LOWER than the exact match. This is the anti-hacking check.
    padded_items = [it("Margherita Pizza", 12.0), it("Caesar Salad", 9.0),
                    it("Tiramisu", 7.0), it("Espresso", 3.0)]
    padded_items += [it(f"Invented Dish {i}", float(i)) for i in range(20)]
    padded = menu(("Mains", padded_items))
    assert menu_reward(padded, REF) < menu_reward(REF, REF)


def test_correct_abstention_scores_high():
    r = menu_reward(not_found(), not_found())
    # valid + found_correct + full content/price credit -> the max.
    assert r == pytest.approx(1.0)


def test_false_find_against_unfindable_scores_low():
    # Reference says not-found; candidate hallucinates a menu -> found wrong,
    # precision 0. Only the schema-valid floor should survive.
    r = menu_reward(menu(("Mains", [it("Ghost Dish", 5.0)])), not_found())
    assert r == pytest.approx(DEFAULT_WEIGHTS.valid)


def test_missed_findable_menu_scores_low():
    # Reference has a menu; candidate abstains -> found wrong, recall 0.
    r = menu_reward(not_found(), REF)
    assert r == pytest.approx(DEFAULT_WEIGHTS.valid)


# ---------------------------------------------------------------------------
# Reference-free mode
# ---------------------------------------------------------------------------
def test_reference_free_empty_is_zero_and_menu_is_positive():
    assert menu_reward(not_found(), None) < menu_reward(REF, None)
    assert menu_reward(REF, None) > 0.0


def test_reference_free_saturates():
    small = menu(("M", [it(f"D{i}") for i in range(5)]))
    big = menu(("M", [it(f"D{i}") for i in range(200)]))
    # more items -> higher, but bounded < 1 (saturation), so big isn't a perfect score.
    assert menu_reward(small, None) < menu_reward(big, None) < 1.0


# ---------------------------------------------------------------------------
# TRL adapter
# ---------------------------------------------------------------------------
def test_grpo_reward_func_batches():
    import json
    fn = make_grpo_reward()
    completions = [json.dumps(REF), "", json.dumps(not_found())]
    refs = [REF, REF, not_found()]
    out = fn(completions, reference=refs)
    assert out == [pytest.approx(1.0), 0.0, pytest.approx(1.0)]


def test_grpo_reward_func_reference_free_when_absent():
    import json
    fn = make_grpo_reward()
    out = fn([json.dumps(REF), ""])  # no reference kwarg
    assert out[0] > 0.0 and out[1] == 0.0


def test_grpo_reward_func_conversational_completions():
    import json
    fn = make_grpo_reward()
    convo = [[{"role": "assistant", "content": json.dumps(REF)}]]
    assert fn(convo, reference=[REF]) == [pytest.approx(1.0)]


def test_custom_weights():
    w = RewardWeights(valid=0.25, found=0.25, content=0.25, price=0.25)
    assert menu_reward("", REF, weights=w) == 0.0
    assert menu_reward(REF, REF, weights=w) == pytest.approx(1.0)
