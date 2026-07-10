"""Tests for schema.extract_json -- the shared answer parser.

extract_json gates schema_valid in the corpus builder, scores in the eval
harness, and will parse the GRPO reward's completions, so its recovery behavior
(esp. a leading narration around an otherwise-valid menu) is worth pinning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema import extract_json  # noqa: E402


def test_plain_json():
    obj, err = extract_json('{"found": true, "menu": []}')
    assert err is None and obj["found"] is True


def test_markdown_fenced_json():
    obj, err = extract_json('```json\n{"found": true, "menu": []}\n```')
    assert err is None and obj == {"found": True, "menu": []}


def test_leading_narration_is_recovered():
    """The Monterrey failure: a complete menu prefixed with a narration sentence."""
    text = (
        "Now I'll compile the JSON, excluding drinks and side add-ons but keeping "
        'actual food items.\n\n{"found": true, "restaurant_name": "Monterrey", '
        '"menu": [{"section": "Tacos", "items": [{"name": "Al Pastor", "price": 13.25}]}], '
        '"source_url": "https://example.test/menu"}'
    )
    obj, err = extract_json(text)
    assert err is None, err
    assert obj["found"] is True
    assert obj["menu"][0]["items"][0]["name"] == "Al Pastor"


def test_trailing_commentary_is_ignored():
    obj, err = extract_json('{"found": false, "menu": []}\n\nLet me know if you need more.')
    assert err is None and obj["found"] is False


def test_brace_inside_string_value_is_safe():
    """A '{' in a description must not confuse the object boundary."""
    text = 'Here you go:\n{"found": true, "note": "combo {a}", "menu": []}'
    obj, err = extract_json(text)
    assert err is None and obj["note"] == "combo {a}"


def test_genuinely_broken_returns_error():
    obj, err = extract_json("I could not find a menu for this restaurant, sorry.")
    assert obj is None and err  # no JSON object at all


def test_truncated_json_still_fails():
    """A cut-off object (no closing brace) is a real failure, not recoverable."""
    obj, err = extract_json('{"found": true, "menu": [{"name": "half')
    assert obj is None and err
