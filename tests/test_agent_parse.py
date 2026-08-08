"""Regression tests for the Gemma turn parser (src/gemma/agent.py).

These pin the failure that cost a full eval run: a COMPLETE, valid final answer was
parsed to content='' purely because the text ended with Gemma's `<turn|>` marker, so
every finished episode looked like an empty/non-terminating one -- the v1 failure
signature, but caused by parsing rather than by the model.

No network, no torch, no GPU: `parse_response` is stubbed with the behaviour measured
on transformers 5.14.1 + the real Gemma-4 tokenizer.

Run: uv run python -m pytest tests/test_agent_parse.py -q
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "gemma"))

import agent as ag  # noqa: E402

ANSWER = '{"found": true, "restaurant_name": "X", "menu": []}'


class FakeTokenizer:
    """parse_response as transformers 5.14.1 actually behaves.

    Two measured properties: `prefix=` is REQUIRED, and a trailing `<turn|>` makes it
    return no content at all (the whole turn collapses into `thinking`).
    """

    def __init__(self, requires_prefix=True):
        self.requires_prefix = requires_prefix
        self.seen_prefix = None

    def parse_response(self, text, prefix=None):
        if self.requires_prefix:
            if prefix is None:
                raise ValueError("`parse_response` requires `prefix=` ...")
            self.seen_prefix = prefix
        elif prefix is not None:
            raise TypeError("parse_response() got an unexpected keyword 'prefix'")
        if text.rstrip().endswith("<turn|>"):
            return {"role": "assistant", "thinking": text}      # content swallowed
        if "<channel|>" in text:
            thinking, _, content = text.partition("<channel|>")
            return {"role": "assistant", "thinking": thinking, "content": content}
        return {"role": "assistant", "content": text}


class TestStripEndOfTurn:
    def test_strips_only_a_trailing_end_of_turn(self):
        assert ag.strip_end_of_turn("answer<turn|>") == "answer"
        assert ag.strip_end_of_turn("answer<turn|>\n") == "answer"
        assert ag.strip_end_of_turn("answer") == "answer"

    def test_never_strips_the_tool_call_marker(self):
        # The vLLM path re-appends <tool_call|> precisely so the parser sees a
        # complete call; eating it here would break every tool turn.
        text = '<|tool_call>call:web_search{query:<|"|>x<|"|>}<tool_call|>'
        assert ag.strip_end_of_turn(text) == text

    def test_leaves_an_interior_marker_alone(self):
        assert ag.strip_end_of_turn("a<turn|>b") == "a<turn|>b"


class TestParseTurn:
    def test_recovers_content_from_a_turn_terminated_answer(self):
        tok = FakeTokenizer()
        parsed = ag.parse_turn(tok, f"reasoning<channel|>{ANSWER}<turn|>\n", prefix="P")
        assert parsed.get("content") == ANSWER   # the whole point

    def test_passes_the_prefix_through(self):
        tok = FakeTokenizer()
        ag.parse_turn(tok, f"r<channel|>{ANSWER}", prefix="THE-PROMPT")
        assert tok.seen_prefix == "THE-PROMPT"

    def test_falls_back_for_transformers_without_prefix(self):
        tok = FakeTokenizer(requires_prefix=False)   # 5.10, the repo pin
        parsed = ag.parse_turn(tok, f"r<channel|>{ANSWER}<turn|>", prefix="P")
        assert parsed.get("content") == ANSWER


class TestRunEpisodeFinalAnswer:
    def _run(self, tok, raw_turn):
        return ag.run_episode(None, tok, "X, Y", tools=[], tool_registry={"web_search": lambda **k: ""},
                              system_prompt="sys",
                              vllm_generate=lambda prompt: raw_turn)

    def test_returns_the_answer_not_empty(self, monkeypatch):
        monkeypatch.setattr(ag, "render_prompt", lambda tok, m, t: "PROMPT")
        assert ANSWER in self._run(FakeTokenizer(), f"reasoning<channel|>{ANSWER}<turn|>\n")

    def test_falls_back_to_raw_text_when_parser_yields_no_content(self, monkeypatch):
        """Defence in depth: a turn with no tool call IS the answer, so it must never
        be discarded just because the parser surfaced nothing."""
        monkeypatch.setattr(ag, "render_prompt", lambda tok, m, t: "PROMPT")

        class NoContentTokenizer(FakeTokenizer):
            def parse_response(self, text, prefix=None):
                return {"role": "assistant", "thinking": text}   # never any content

        assert ANSWER in self._run(NoContentTokenizer(), f"reasoning<channel|>{ANSWER}")
