"""Unit tests for src/eval_metrics.py (Phase 2 WS-G).

Pure-local, zero network: every menu is a literal dict shaped like a trace's
`final_json` (contract 1.5 / MENU_SCHEMA in src/schema.py).

Run: uv run python -m pytest tests/test_eval_metrics.py -q
"""

import sys
from pathlib import Path

import pytest

# Shared modules live in src/ (flat imports, no packages) -- same convention as
# the entry scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_metrics import (  # noqa: E402
    ITEM_MATCH_THRESHOLD,
    abstention_outcome,
    aggregate,
    aggregate_self_reports,
    is_schema_valid,
    match_items,
    name_similarity,
    normalize_name,
    prices_equal,
    score_episode,
    self_report,
)


def it(name, price=None, description=None):
    """One menu item dict."""
    return {"name": name, "description": description, "price": price}


def menu(*sections, found=True, name="Testaurant"):
    """A schema-valid menu dict from (section_name, [items]) pairs."""
    return {
        "found": found,
        "restaurant_name": name,
        "cuisine": "Test",
        "menu": [{"section": s, "items": items} for s, items in sections],
        "source_url": "https://example.com/menu",
    }


REF = menu(
    ("Pizza", [it("Margherita Pizza", 14.0), it("BBQ Chicken Pizza", 17.5)]),
    ("Salads", [it("Caesar Salad", 9.0), it("Greek Salad", 10.0)]),
)


# ---------------------------------------------------------------------------
# name normalization + similarity
# ---------------------------------------------------------------------------
class TestNameMatching:
    def test_normalize_case_whitespace_punctuation(self):
        assert normalize_name("  Mac & Cheese!! ") == "mac cheese"
        assert normalize_name("Po'Boy\t(Large)") == "po boy large"
        assert normalize_name(None) == ""

    def test_exact_after_normalization_is_1(self):
        assert name_similarity("CAESAR   salad", "Caesar Salad!") == 1.0

    def test_token_reorder_is_1(self):
        assert name_similarity("Pizza Margherita", "Margherita Pizza") == 1.0

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Ceasar Salad", "Caesar Salad"),          # typo
            ("BBQ Chicken Pizza", "Barbecue Chicken Pizza"),  # spell-out
            ("Spring Rolls (2)", "Spring Rolls"),      # count decoration
        ],
    )
    def test_renames_above_threshold(self, a, b):
        assert name_similarity(a, b) >= ITEM_MATCH_THRESHOLD

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Galbi Set", "Bulgogi Set"),   # shared generic word, different dish
            ("Pad Thai", "Pad See Ew"),
            ("Coke", "Sprite"),
        ],
    )
    def test_different_items_below_threshold(self, a, b):
        assert name_similarity(a, b) < ITEM_MATCH_THRESHOLD

    def test_single_generic_token_cannot_swallow_by_containment(self):
        # "Pizza" alone must not match every "<X> Pizza" via containment.
        assert name_similarity("Pizza", "BBQ Chicken Pizza") < ITEM_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# exact-match menus score 1.0
# ---------------------------------------------------------------------------
class TestExactMatch:
    def test_identical_menus_are_perfect(self):
        s = score_episode(REF, REF)
        assert s["schema_valid"] is True
        assert s["found_correct"] is True
        assert s["precision"] == s["recall"] == s["f1"] == 1.0
        assert s["price_agreement"] == 1.0
        assert s["section_count_delta"] == 0
        assert s["item_count_delta"] == 0
        assert s["n_matched"] == s["n_candidate_items"] == s["n_reference_items"] == 4

    def test_section_shuffle_still_perfect(self):
        # Matching is over the flattened item set; section layout only shows up
        # in the section-count delta.
        cand = menu(("Everything", [it("Greek Salad", 10.0), it("Margherita Pizza", 14.0),
                                    it("Caesar Salad", 9.0), it("BBQ Chicken Pizza", 17.5)]))
        s = score_episode(cand, REF)
        assert s["precision"] == s["recall"] == s["f1"] == 1.0
        assert s["price_agreement"] == 1.0
        assert s["section_count_delta"] == -1


# ---------------------------------------------------------------------------
# fuzzy renames are matched
# ---------------------------------------------------------------------------
class TestFuzzyRenames:
    def test_renamed_items_still_match(self):
        cand = menu(
            ("Pizza", [it("Pizza Margherita", 14.0), it("Barbecue Chicken Pizza", 17.5)]),
            ("Salads", [it("Ceasar Salad", 9.0), it("Greek Salad", 10.0)]),
        )
        s = score_episode(cand, REF)
        assert s["precision"] == s["recall"] == s["f1"] == 1.0
        assert s["n_matched"] == 4

    def test_matching_is_one_to_one(self):
        # Two candidate "Caesar Salad"s cannot both claim the single reference one.
        cand = menu(("Salads", [it("Caesar Salad", 9.0), it("Caesar Salad", 9.0)]))
        ref = menu(("Salads", [it("Caesar Salad", 9.0)]))
        s = score_episode(cand, ref)
        assert s["n_matched"] == 1
        assert s["precision"] == 0.5
        assert s["recall"] == 1.0

    def test_tie_broken_by_price_agreement(self):
        # Same-name items: the matcher should pair the price-agreeing ones.
        cand = [it("House Wine", 8.0)]
        ref = [it("House Wine", 12.0), it("House Wine", 8.0)]
        matches = match_items(cand, ref)
        assert matches == [(0, 1, 1.0)]


# ---------------------------------------------------------------------------
# missing items reduce recall; hallucinated items reduce precision
# ---------------------------------------------------------------------------
class TestPrecisionRecall:
    def test_missing_items_reduce_recall(self):
        cand = menu(("Pizza", [it("Margherita Pizza", 14.0), it("BBQ Chicken Pizza", 17.5)]))
        s = score_episode(cand, REF)
        assert s["precision"] == 1.0
        assert s["recall"] == 0.5
        assert s["f1"] == pytest.approx(2 / 3)
        assert s["item_count_delta"] == -2

    def test_hallucinated_items_reduce_precision(self):
        cand = menu(
            ("Pizza", [it("Margherita Pizza", 14.0), it("BBQ Chicken Pizza", 17.5)]),
            ("Salads", [it("Caesar Salad", 9.0), it("Greek Salad", 10.0)]),
            ("Desserts", [it("Chocolate Lava Cake", 8.0), it("Tiramisu", 7.5),
                          it("Gelato Trio", 6.0), it("Cannoli", 5.0)]),
        )
        s = score_episode(cand, REF)
        assert s["recall"] == 1.0
        assert s["precision"] == 0.5
        assert s["f1"] == pytest.approx(2 / 3)
        assert s["item_count_delta"] == 4

    def test_empty_candidate_menu_with_found_true(self):
        cand = menu(found=True)  # found a menu... with nothing in it
        s = score_episode(cand, REF)
        assert s["precision"] is None  # no claims -> precision N/A, not vacuous 1.0
        assert s["recall"] == 0.0
        assert s["f1"] == 0.0

    def test_both_empty_found_true_is_trivially_perfect(self):
        s = score_episode(menu(found=True), menu(found=True))
        assert s["precision"] == s["recall"] == s["f1"] == 1.0


# ---------------------------------------------------------------------------
# price agreement on matched items
# ---------------------------------------------------------------------------
class TestPriceAgreement:
    def test_price_mismatch_detected(self):
        cand = menu(("Salads", [it("Caesar Salad", 11.0), it("Greek Salad", 10.0)]))
        ref = menu(("Salads", [it("Caesar Salad", 9.0), it("Greek Salad", 10.0)]))
        s = score_episode(cand, ref)
        assert s["f1"] == 1.0  # names all match ...
        assert s["price_agreement"] == 0.5  # ... but one price is wrong

    def test_both_unknown_prices_agree(self):
        s = score_episode(menu(("S", [it("Soup")])), menu(("S", [it("Soup")])))
        assert s["price_agreement"] == 1.0

    def test_known_vs_unknown_disagrees(self):
        s = score_episode(menu(("S", [it("Soup", 5.0)])), menu(("S", [it("Soup")])))
        assert s["price_agreement"] == 0.0

    def test_tolerance_and_int_float(self):
        assert prices_equal(12, 12.0)
        assert prices_equal(9.99, 9.9949)
        assert not prices_equal(12.0, 12.5)

    def test_no_matches_means_price_na(self):
        s = score_episode(menu(("S", [it("Pad Thai", 15.0)])),
                          menu(("S", [it("Cheeseburger", 12.0)])))
        assert s["price_agreement"] is None
        assert s["f1"] == 0.0


# ---------------------------------------------------------------------------
# found / not-found combinations
# ---------------------------------------------------------------------------
class TestFoundCombinations:
    NOT_FOUND = {"found": False, "restaurant_name": "Testaurant", "cuisine": None,
                 "menu": [], "source_url": None, "notes": "no menu online"}

    def test_both_found_true(self):
        s = score_episode(REF, REF)
        assert s["found_correct"] is True

    def test_reference_found_candidate_abstains(self):
        s = score_episode(self.NOT_FOUND, REF)
        assert s["schema_valid"] is True
        assert s["found_correct"] is False
        assert s["recall"] == 0.0 and s["f1"] == 0.0
        assert s["precision"] is None  # abstention made no item claims

    def test_reference_not_found_candidate_hallucinates(self):
        s = score_episode(REF, self.NOT_FOUND)
        assert s["found_correct"] is False
        assert s["precision"] == 0.0 and s["f1"] == 0.0
        assert s["recall"] is None  # nothing to recall

    def test_both_not_found_is_correct_abstention(self):
        s = score_episode(self.NOT_FOUND, self.NOT_FOUND)
        assert s["found_correct"] is True
        assert s["precision"] is None and s["recall"] is None and s["f1"] is None
        assert s["price_agreement"] is None

    def test_abstention_outcomes_vs_labels(self):
        assert abstention_outcome(REF, findable=True) == "correct_find"
        assert abstention_outcome(self.NOT_FOUND, findable=True) == "false_abstain"
        assert abstention_outcome(REF, findable=False) == "false_find"
        assert abstention_outcome(self.NOT_FOUND, findable=False) == "correct_abstain"
        assert abstention_outcome(None, findable=False) == "correct_abstain"


# ---------------------------------------------------------------------------
# schema-invalid candidates
# ---------------------------------------------------------------------------
class TestSchemaInvalid:
    @pytest.mark.parametrize(
        "bad",
        [
            None,                                       # parse failure upstream
            "not even a dict",
            {"restaurant_name": "X", "menu": []},       # missing required `found`
            {"found": True, "restaurant_name": "X", "menu": "oops"},   # wrong type
            {"found": True, "restaurant_name": "X",
             "menu": [{"section": "S", "items": [{"description": "nameless"}]}]},
        ],
    )
    def test_invalid_candidate_scores_zero_against_found_reference(self, bad):
        s = score_episode(bad, REF)
        assert s["schema_valid"] is False
        assert s["found_correct"] is False
        assert s["recall"] == 0.0 and s["f1"] == 0.0
        assert s["found_candidate"] is None

    def test_invalid_candidate_against_not_found_reference(self):
        s = score_episode(None, TestFoundCombinations.NOT_FOUND)
        assert s["schema_valid"] is False
        assert s["found_correct"] is False
        assert s["f1"] is None  # no menu existed to miss

    def test_is_schema_valid_accepts_both_contract_shapes(self):
        assert is_schema_valid(REF)
        assert is_schema_valid(TestFoundCombinations.NOT_FOUND)


# ---------------------------------------------------------------------------
# aggregation + self-report
# ---------------------------------------------------------------------------
class TestAggregate:
    def test_means_skip_undefined_metrics(self):
        scores = [
            score_episode(REF, REF),                                    # perfect
            score_episode(TestFoundCombinations.NOT_FOUND,
                          TestFoundCombinations.NOT_FOUND),             # correct abstention
            score_episode(None, REF),                                   # invalid
        ]
        agg = aggregate(scores)
        assert agg["n_episodes"] == 3
        assert agg["schema_valid_rate"] == pytest.approx(2 / 3)
        assert agg["found_accuracy"] == pytest.approx(2 / 3)
        # precision defined only for the perfect episode; f1 for perfect + invalid.
        assert agg["precision_mean"] == 1.0 and agg["precision_n"] == 1
        assert agg["f1_mean"] == 0.5 and agg["f1_n"] == 2
        assert agg["price_agreement_mean"] == 1.0 and agg["price_agreement_n"] == 1
        assert agg["item_count_delta_n"] == 1

    def test_empty_aggregate(self):
        assert aggregate([]) == {"n_episodes": 0}

    def test_self_report_counts(self):
        r = self_report(REF)
        assert r == {"schema_valid": True, "found": True, "n_sections": 2,
                     "n_items": 4, "n_priced_items": 4, "price_coverage": 1.0}
        r = self_report(menu(("S", [it("Soup"), it("Stew", 8.0)])))
        assert r["price_coverage"] == 0.5
        assert self_report(None)["schema_valid"] is False

    def test_aggregate_self_reports(self):
        agg = aggregate_self_reports([self_report(REF), self_report(None)])
        assert agg["n_episodes"] == 2
        assert agg["schema_valid_rate"] == 0.5
        assert agg["found_rate"] == 0.5
        assert agg["mean_items"] == 2.0
        assert agg["price_coverage_n"] == 1
