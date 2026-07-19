"""Unit tests for src/grounding.py (shared grounding normalizer + scorer).

Run: uv run python -m pytest tests/test_grounding.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grounding import (  # noqa: E402
    grounding_for_trace,
    item_names,
    normalize,
    score,
    tool_text,
)


def test_normalize():
    assert normalize("  Kimchi   Stew ") == "kimchi stew"
    assert normalize(None) == ""


def test_item_names():
    menu = [
        {"section": "Mains", "items": [{"name": "Bulgogi"}, {"name": " "}, {"name": "Bibimbap"}]},
        {"section": "Sides", "items": [{"name": "Kimchi"}]},
    ]
    assert item_names(menu) == ["Bulgogi", "Bibimbap", "Kimchi"]
    assert item_names(None) == []


def test_score_substring_and_misses():
    hay = "our menu: bulgogi $18, bibimbap $16, house kimchi".lower()
    frac, misses = score(["Bulgogi", "Bibimbap", "Kimchi", "Tteokbokki"], hay)
    assert misses == ["Tteokbokki"]
    assert abs(frac - 0.75) < 1e-9


def test_score_whitespace_collapse():
    frac, misses = score(["Kimchi   Stew"], "we serve kimchi stew daily")
    assert frac == 1.0 and misses == []


def test_tool_text_anthropic_shape():
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "name": "web_search", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "content": "Bulgogi and Bibimbap"},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "content": [{"type": "text", "text": "Kimchi Stew here"}]},
        ]},
    ]
    txt = tool_text(messages)
    assert "bulgogi" in txt and "kimchi stew" in txt


def test_tool_text_openai_shape():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "Bulgogi and Bibimbap"},
        {"role": "tool", "content": [{"type": "text", "text": "Kimchi Stew"}]},
    ]
    txt = tool_text(messages)
    assert "bulgogi" in txt and "kimchi stew" in txt


def test_grounding_for_trace_full_and_partial():
    trace = {
        "final_json": {"menu": [{"section": "M", "items": [{"name": "Bulgogi"}, {"name": "Ghost Dish"}]}]},
        "messages": [{"role": "tool", "content": "we sell bulgogi only"}],
    }
    g, misses = grounding_for_trace(trace)
    assert g == 0.5 and misses == ["Ghost Dish"]


def test_grounding_for_trace_no_menu_is_none():
    trace = {"final_json": {"found": False, "menu": []}, "messages": [{"role": "tool", "content": "x"}]}
    g, misses = grounding_for_trace(trace)
    assert g is None and misses == []


def test_grounding_unmatched_cap():
    items = [{"name": f"Dish {i}"} for i in range(50)]
    trace = {"final_json": {"menu": [{"section": "M", "items": items}]},
             "messages": [{"role": "tool", "content": "nothing matches"}]}
    g, misses = grounding_for_trace(trace)
    assert g == 0.0 and len(misses) == 25  # UNMATCHED_CAP
