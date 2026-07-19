"""Tests for scripts/datasets/build_sft.py (v2: corpus.sqlite sft traces -> SFT dataset).

Pure-logic unit tests (no tokenizer) cover the found=false ratio-guard math, the
tool-result slim+cap mapping by tool name, and the extract_json-based final-answer
cleaning -- all UNCHANGED from v1 (the student re-render pipeline is identical; only
the INPUT SOURCE moved to the DB). A small in-DB integration test drives main() over
a temp corpus.sqlite with a FAKE tokenizer to prove the v2 change: traces come from
`iter_traces(split="sft", include_rejected=False)`, so a rejected trace (marked via
corpus.set_rejected) and an eval-split trace are both excluded -- no reject_list.txt,
no data/traces files. A separate integration test loads the gated Gemma tokenizer and
round-trips one synthetic trace; it is skipped cleanly when the weights aren't there.

The v1 reject-list-file tests (load_reject_set / is_rejected) are DROPPED: rejection
is a DB field now (iter_traces filters it), so those functions no longer exist.

    uv run python -m pytest tests/test_build_sft.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/, the
# script under test in scripts/datasets/ (the v2 location -- NOT scripts/).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "datasets"))

import build_sft as bs  # noqa: E402
from corpus import open_corpus, restaurant_id_for  # noqa: E402
from tools import MAX_TOOL_CHARS  # noqa: E402  (src/ is on the path via build_sft's import)


# ---------------------------------------------------------------------------
# found=false ratio-guard math
# ---------------------------------------------------------------------------
def test_found_false_keep_all_when_under_cap():
    # 5 / (95 + 5) = 0.05 <= 0.2 -> keep all.
    assert bs.found_false_keep_count(95, 5, 0.2) == 5


def test_found_false_keep_none_when_no_false():
    assert bs.found_false_keep_count(100, 0, 0.2) == 0
    assert bs.found_false_keep_count(0, 0, 0.2) == 0


def test_found_false_downsamples_over_cap():
    # natural 50/100 = 0.5 > 0.2; keep k with k/(50+k) <= 0.2 -> k <= 12.5 -> 12.
    keep = bs.found_false_keep_count(50, 50, 0.2)
    assert keep == 12
    assert keep / (50 + keep) <= 0.2


def test_found_false_guard_disabled_at_one():
    assert bs.found_false_keep_count(10, 90, 1.0) == 90


def test_found_false_zero_cap_drops_all():
    assert bs.found_false_keep_count(10, 5, 0.0) == 0


def test_found_false_exact_boundary_kept():
    # 25 / (100 + 25) = 0.2 exactly -> not over the cap, keep all.
    assert bs.found_false_keep_count(100, 25, 0.2) == 25


# ---------------------------------------------------------------------------
# tool-result slim + cap mapping by tool name
# ---------------------------------------------------------------------------
def test_scrape_result_is_slimmed():
    md = "Margherita $10 ![pizza](https://x.com/p.png) more text"
    out = bs.transform_tool_result("scrape_url", md)
    assert "![pizza]" not in out          # markdown image stripped by _slim_scrape
    assert "Margherita $10" in out


def test_search_result_is_not_slimmed():
    # web_search results are the model's URL source -- only capped, never slimmed.
    md = "1. Joe's ![logo](https://x.com/l.png) joes.com"
    out = bs.transform_tool_result("web_search", md)
    assert "![logo]" in out


def test_tool_result_capped_to_max():
    big = "x" * (MAX_TOOL_CHARS + 5000)
    assert len(bs.transform_tool_result("web_search", big)) == MAX_TOOL_CHARS
    assert len(bs.transform_tool_result("scrape_url", big)) == MAX_TOOL_CHARS


def test_tool_result_transform_idempotent():
    md = "Menu ![x](y.png)\n\n\n\n\nItems"
    once = bs.transform_tool_result("scrape_url", md)
    twice = bs.transform_tool_result("scrape_url", once)
    assert once == twice


# ---------------------------------------------------------------------------
# extract_json-based final-answer cleaning
# ---------------------------------------------------------------------------
def test_clean_final_answer_strips_narration_and_compacts():
    raw = 'Here is the menu:\n{\n  "found": true,\n  "restaurant_name": "Joe"\n}\nDone.'
    result = bs.clean_final_answer(raw)
    assert result is not None
    compact, obj = result
    assert obj["found"] is True
    assert obj["restaurant_name"] == "Joe"
    # compact = no spaces after separators
    assert compact == '{"found":true,"restaurant_name":"Joe"}'


def test_clean_final_answer_strips_code_fence():
    raw = '```json\n{"found": false, "restaurant_name": "X", "menu": []}\n```'
    result = bs.clean_final_answer(raw)
    assert result is not None
    _compact, obj = result
    assert obj["found"] is False


def test_clean_final_answer_rejects_non_json():
    assert bs.clean_final_answer("I could not find a menu.") is None
    assert bs.clean_final_answer("") is None


def test_clean_final_answer_rejects_non_object():
    # a bare JSON array is valid JSON but not the menu object contract.
    assert bs.clean_final_answer("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# Message assembly (no tokenizer needed)
# ---------------------------------------------------------------------------
def _synthetic_trace(dietary=None):
    """A tiny two-tool-call trace in the stored Anthropic-block format."""
    final = (
        '{"found": true, "restaurant_name": "Joe\'s Pizza", "cuisine": "Pizza", '
        '"menu": [{"section": "Pizza", "items": [{"name": "Margherita", '
        '"description": null, "price": 10}]}], "source_url": "https://joespizza.com/menu"}'
    )
    return {
        "restaurant_id": "test123",
        "restaurant_name": "Joe's Pizza",
        "model": "claude-sonnet-5",
        "prompt_variant": "teacher",
        "dietary_restrictions": dietary,
        "cache_version": 1,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Joe's Pizza, New York"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "let me search"},
                {"type": "tool_use", "name": "web_search",
                 "input": {"query": "Joe's Pizza New York menu"}, "id": "tu1"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1",
                 "content": "1. Joe's Pizza - https://joespizza.com/menu"},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "scrape_url",
                 "input": {"url": "https://joespizza.com/menu", "mode": "direct"}, "id": "tu2"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu2",
                 "content": "# Menu\n Margherita $10 ![pizza](https://x.com/p.png)"},
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": final},
            ]},
        ],
    }


def test_build_gemma_messages_shape():
    messages, final_str, obj = bs.build_gemma_messages(_synthetic_trace())
    assert obj["found"] is True
    # system(student) -> user -> 2 bundled tool turns -> final
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "assistant", "assistant"]
    assert bs._TEACHER_ONLY_PHRASE not in messages[0]["content"]  # student prompt
    assert messages[1]["content"] == "Joe's Pizza, New York"
    # bundled shape: tool_calls + tool_responses, no reasoning / visible text
    turn = messages[2]
    assert set(turn) == {"role", "tool_calls", "tool_responses"}
    assert turn["tool_calls"][0]["function"]["name"] == "web_search"
    assert turn["tool_calls"][0]["function"]["arguments"]["query"] == "Joe's Pizza New York menu"
    # scrape result was slimmed (image stripped) in the second turn
    assert "![pizza]" not in messages[3]["tool_responses"][0]["response"]
    # final answer is the compact JSON string, action-only (no fences/narration)
    assert messages[4]["content"] == final_str
    assert final_str.startswith('{"found":true')


def test_build_gemma_messages_conditioned_keeps_restriction():
    messages, _s, _o = bs.build_gemma_messages(_synthetic_trace(dietary=["vegetarian"]))
    assert "vegetarian" in messages[0]["content"]          # restriction visible to student
    assert bs._TEACHER_ONLY_PHRASE not in messages[0]["content"]


def test_build_gemma_messages_skips_unparseable_final():
    trace = _synthetic_trace()
    trace["messages"][-1]["content"] = [{"type": "text", "text": "no menu found, sorry"}]
    with pytest.raises(ValueError, match="did not parse"):
        bs.build_gemma_messages(trace)


# ---------------------------------------------------------------------------
# In-DB integration: main() reads sft-split, NON-rejected traces from corpus.sqlite
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """A stand-in for the gated Gemma tokenizer so the DB-read path is testable
    without the real weights. apply_chat_template renders every load-bearing piece
    (system prompt content, tool-call names + string args, tool responses, and the
    final answer VERBATIM) into one string, so build_sft.verify_round_trip passes;
    tokenize=True returns a dummy input_ids list for token_length()."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                            tokenize=False, return_dict=False):
        parts = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                parts.append(content)
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                parts.append(fn["name"])
                for val in (fn.get("arguments") or {}).values():
                    parts.append(str(val))
            for tr in m.get("tool_responses") or []:
                parts.append(str(tr.get("name", "")))
                parts.append(str(tr.get("response", "")))
        text = "\n".join(parts)
        if tokenize:
            ids = list(range(max(1, len(text.split()))))
            return {"input_ids": ids} if return_dict else ids
        return text


_FINAL_JSON = ('{"found": true, "restaurant_name": "Joe\'s Pizza", "cuisine": "Pizza", '
               '"menu": [{"section": "Pizza", "items": [{"name": "Margherita", '
               '"description": null, "price": 10}]}], "source_url": "https://joespizza.com/menu"}')


def _write_trace(cx, name, city, *, split, dietary=None):
    """Upsert a restaurant into `split` and write one teacher trace for it."""
    cx.upsert_restaurants([{"name": name, "city": city, "source": "osm", "split": split}])
    rid = restaurant_id_for(name, city)
    cx.write_trace({
        "restaurant_id": rid,
        "model": "claude-sonnet-5",
        "prompt_variant": "teacher",
        "dietary_restrictions": dietary,
        "found": True,
        "schema_valid": True,
        "final_json": json.loads(_FINAL_JSON),
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"{name}, {city}"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "web_search",
                 "input": {"query": f"{name} {city} menu"}, "id": "tu1"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1",
                 "content": f"1. {name} - https://joespizza.com/menu"}]},
            {"role": "assistant", "content": [{"type": "text", "text": _FINAL_JSON}]},
        ],
        "queries": [f"{name} {city} menu"],
        "urls": ["https://joespizza.com/menu"],
        "cache_version": 1,
        "captured_at": "2026-07-18T00:00:00Z",
    })
    return rid


def test_main_reads_sft_split_and_excludes_rejected_and_eval(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.sqlite"
    with open_corpus(corpus) as cx:
        keep_rid = _write_trace(cx, "Keep Cafe", "Austin", split="sft")
        rej_rid = _write_trace(cx, "Reject Diner", "Boston", split="sft")
        _write_trace(cx, "Eval Grill", "Chicago", split="eval")   # wrong split -> excluded
        cx.set_rejected(rej_rid, True, "wrong city")               # rejected -> excluded

    out = tmp_path / "sft" / "gemma-4-e4b-it" / "train.jsonl"
    monkeypatch.setattr(bs, "load_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(sys, "argv", ["build_sft.py", "--corpus", str(corpus), "--out", str(out)])
    bs.main()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    # Only the one sft, non-rejected trace survives.
    assert [r["restaurant_id"] for r in rows] == [keep_rid]
    assert rej_rid not in {r["restaurant_id"] for r in rows}
    assert rows[0]["found"] is True
    assert rows[0]["meta"]["prompt_variant"] == "student"
    # provenance sidecar records the single kept example
    meta = json.loads((out.with_name(out.name + ".meta.json")).read_text(encoding="utf-8"))
    assert meta["n_examples"] == 1


# ---------------------------------------------------------------------------
# Tokenizer integration (gated -- skipped when the Gemma tokenizer is unavailable)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer():
    try:
        tok = bs.load_tokenizer()
    except Exception as e:  # noqa: BLE001 - gated model / offline -> skip, don't fail
        pytest.skip(f"Gemma tokenizer unavailable: {e}")
    if tok is None:
        pytest.skip("Gemma tokenizer unavailable")
    return tok


def test_round_trip_free(tokenizer):
    messages, final_str, _obj = bs.build_gemma_messages(_synthetic_trace())
    # must not raise
    bs.verify_round_trip(tokenizer, messages, final_str, None)
    rendered = tokenizer.apply_chat_template(
        messages, tools=bs.TOOLS, add_generation_prompt=False, tokenize=False
    )
    assert "web_search" in rendered
    assert "scrape_url" in rendered
    assert "Joe's Pizza New York menu" in rendered
    assert final_str in rendered
    assert bs._TEACHER_ONLY_PHRASE not in rendered
    assert bs.token_length(tokenizer, messages) > 0


def test_round_trip_conditioned(tokenizer):
    messages, final_str, _obj = bs.build_gemma_messages(_synthetic_trace(dietary=["vegetarian"]))
    bs.verify_round_trip(tokenizer, messages, final_str, ["vegetarian"])
    rendered = tokenizer.apply_chat_template(
        messages, tools=bs.TOOLS, add_generation_prompt=False, tokenize=False
    )
    assert "vegetarian" in rendered


def test_verify_round_trip_flags_missing_restriction(tokenizer):
    # A restriction the render can't contain must be caught (guards the swap).
    messages, final_str, _obj = bs.build_gemma_messages(_synthetic_trace())
    with pytest.raises(ValueError, match="restriction"):
        bs.verify_round_trip(tokenizer, messages, final_str, ["unicorn-only-diet-xyz"])
