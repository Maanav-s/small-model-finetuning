"""Unit tests for src/reward.py (Phase 3 GRPO reward -- PURE RL, teacher-free).

Pure-local: candidates are literal final_json dicts / strings, evidence is a
literal string standing in for the scraped tool output. Verifies the reward
ORDERING (the gradient GRPO follows): hallucination < empty < abstention <
grounded, and that grounding penalises invented items.

Run: uv run python -m pytest tests/test_reward.py -q
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reward import (  # noqa: E402
    DEFAULT_WEIGHTS,
    HALLUCINATION_PENALTY,
    RewardWeights,
    _evidence_from_completion,
    count_grounded,
    found_score,
    grounding_score,
    is_grounded,
    make_grpo_rewards,
    menu_reward,
    structure_score,
)


def it(name, price=None):
    return {"name": name, "description": None, "price": price}


def menu(*item_names, found=True, name="Testaurant"):
    return {
        "found": found, "restaurant_name": name, "cuisine": "Test",
        "menu": [{"section": "Menu", "items": [it(n) for n in item_names]}],
        "source_url": "https://example.com/menu",
    }


def not_found():
    return {"found": False, "restaurant_name": "Ghost", "cuisine": None, "menu": [],
            "source_url": None, "notes": "no menu found"}


# A scraped-page stand-in that CONTAINS these four dish names.
EVIDENCE = ("Welcome to Joe's. Our menu: Margherita Pizza $12, Caesar Salad $9, "
            "Tiramisu $7, Espresso $3. Open daily.")
REAL = ["Margherita Pizza", "Caesar Salad", "Tiramisu", "Espresso"]
FAKE = ["Dragon Roll", "Wagyu Skewer", "Truffle Fondue", "Golden Souffle"]


# ---------------------------------------------------------------------------
# Grounding primitives
# ---------------------------------------------------------------------------
def test_is_grounded_matches_normalized_substring():
    ev = "margherita pizza caesar salad"
    assert is_grounded("Margherita Pizza", ev) is True
    assert is_grounded("MARGHERITA  pizza!", ev) is True   # normalization
    assert is_grounded("Dragon Roll", ev) is False
    assert is_grounded("", ev) is False


def test_count_grounded():
    assert count_grounded([it(n) for n in REAL], EVIDENCE) == (4, 4)
    assert count_grounded([it(n) for n in FAKE], EVIDENCE) == (0, 4)
    assert count_grounded([it("Margherita Pizza"), it("Dragon Roll")], EVIDENCE) == (1, 2)
    assert count_grounded([it("x")], "") == (0, 1)  # no evidence -> ungrounded


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------
def test_structure_score():
    assert structure_score(menu(*REAL)) == 1.0
    assert structure_score(None) == 0.0
    assert structure_score({"menu": []}) == 0.0  # missing required found/name


def test_found_score():
    assert found_score(menu(*REAL)) == 1.0
    assert found_score(not_found()) == 0.0                       # abstained
    assert found_score(menu(found=True)) == 0.0                  # found but no items
    assert found_score(None) == 0.0


def test_grounding_fully_grounded_positive():
    assert grounding_score(menu(*REAL), EVIDENCE) > 0.0


def test_grounding_fully_hallucinated_is_max_penalty():
    assert grounding_score(menu(*FAKE), EVIDENCE) == pytest.approx(-HALLUCINATION_PENALTY)


def test_grounding_zero_when_nothing_to_grade():
    assert grounding_score(not_found(), EVIDENCE) == 0.0        # abstention
    assert grounding_score(None, EVIDENCE) == 0.0              # invalid


def test_grounding_more_real_items_scores_higher():
    two = menu("Margherita Pizza", "Caesar Salad")
    four = menu(*REAL)
    assert grounding_score(four, EVIDENCE) > grounding_score(two, EVIDENCE) > 0.0


def test_grounding_partial_between_hallucinated_and_grounded():
    partial = menu("Margherita Pizza", "Caesar Salad", "Dragon Roll", "Wagyu Skewer")
    g = grounding_score(partial, EVIDENCE)
    assert grounding_score(menu(*FAKE), EVIDENCE) < g < grounding_score(menu(*REAL), EVIDENCE)


# ---------------------------------------------------------------------------
# The reward ordering (what the gradient teaches)
# ---------------------------------------------------------------------------
def test_reward_ordering():
    grounded = menu_reward(menu(*REAL), EVIDENCE)
    abstain = menu_reward(not_found(), EVIDENCE)
    empty = menu_reward("", EVIDENCE)
    hallucinated = menu_reward(menu(*FAKE), EVIDENCE)
    # grounded complete  >  honest abstention  >  empty/invalid  >  confident hallucination
    assert grounded > abstain > empty > hallucinated
    assert empty == 0.0
    assert hallucinated < 0.0            # actively penalised, worse than saying nothing


def test_raw_text_answer_is_parsed():
    assert menu_reward(json.dumps(menu(*REAL)), EVIDENCE) == menu_reward(menu(*REAL), EVIDENCE)


def test_no_evidence_treats_menu_as_ungrounded():
    # A perfectly real menu with NO evidence supplied cannot be verified -> penalised.
    assert menu_reward(menu(*REAL), "") < menu_reward(menu(*REAL), EVIDENCE)


def test_breakdown_exposes_components():
    _r, comp = menu_reward(menu(*REAL), EVIDENCE, breakdown=True)
    assert comp["structure"] == 1.0 and comp["found"] == 1.0 and comp["grounding"] > 0
    assert comp["dietary"] == 0.0


# ---------------------------------------------------------------------------
# Evidence extraction fallback
# ---------------------------------------------------------------------------
def test_evidence_from_gemma_string_spans():
    completion = ("<|tool_response>Margherita Pizza and Caesar Salad<tool_response|> "
                  "some reasoning <|tool_response>Tiramisu<tool_response|>")
    ev = _evidence_from_completion(completion)
    assert "Margherita Pizza" in ev and "Tiramisu" in ev


def test_evidence_from_message_list_excludes_final_answer():
    convo = [
        {"role": "assistant", "tool_responses": [{"name": "scrape_url", "response": "Caesar Salad here"}]},
        {"role": "tool", "content": "Espresso listed"},
        {"role": "assistant", "content": json.dumps(menu(*REAL))},
    ]
    ev = _evidence_from_completion(convo)
    assert "Caesar Salad" in ev and "Espresso" in ev
    # the final answer must NOT be part of the evidence (no self-grounding)
    assert "source_url" not in ev


# ---------------------------------------------------------------------------
# TRL adapter: separate reward functions
# ---------------------------------------------------------------------------
def test_make_grpo_rewards_shape():
    funcs, weights = make_grpo_rewards()
    assert [f.__name__ for f in funcs] == ["structure_reward", "found_reward", "grounding_reward"]
    assert weights == [DEFAULT_WEIGHTS.structure, DEFAULT_WEIGHTS.found, DEFAULT_WEIGHTS.grounding]


def test_grpo_rewards_use_final_json_and_evidence_kwargs():
    funcs, _ = make_grpo_rewards()
    structure_reward, found_reward, grounding_reward = funcs
    completions = ["", "", ""]  # ignored: final_json supplied
    fj = [menu(*REAL), menu(*FAKE), not_found()]
    ev = [EVIDENCE, EVIDENCE, EVIDENCE]
    assert structure_reward(completions, final_json=fj, evidence=ev) == [1.0, 1.0, 1.0]
    assert found_reward(completions, final_json=fj, evidence=ev) == [1.0, 1.0, 0.0]
    g = grounding_reward(completions, final_json=fj, evidence=ev)
    assert g[0] > 0 and g[1] == pytest.approx(-HALLUCINATION_PENALTY) and g[2] == 0.0


def test_grpo_rewards_fallback_parse_from_completion():
    # No final_json/evidence kwargs: parse the menu + evidence out of the completion.
    _s, _f, grounding_reward = make_grpo_rewards()[0]
    completion = ("<|tool_response>Margherita Pizza, Caesar Salad, Tiramisu, Espresso<tool_response|>\n"
                  + json.dumps(menu(*REAL)))
    assert grounding_reward([completion])[0] > 0.0


def test_include_dietary_adds_zero_weight_slot():
    funcs, weights = make_grpo_rewards(include_dietary=True)
    assert [f.__name__ for f in funcs][-1] == "dietary_reward"
    assert weights[-1] == 0.0
    assert funcs[-1](["x"], final_json=[menu(*REAL)], dietary_restrictions=[["vegetarian"]]) == [0.0]


def test_custom_weights_as_list():
    w = RewardWeights(structure=0.1, found=0.3, grounding=0.5, dietary=0.1)
    assert w.as_list() == [0.1, 0.3, 0.5, 0.1]
    _funcs, weights = make_grpo_rewards(weights=w)
    assert weights == [0.1, 0.3, 0.5]
