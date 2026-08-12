"""Unit tests for src/serving/openai_agent.py (the OpenAI-compatible vLLM runner).

Pure-local, no server / no `openai` package: run_episode takes an injected client,
so a scripted fake drives the tool loop. Covers to_openai_tools shape and the
agentic loop (tool execution, message threading, final answer).

Run: uv run python -m pytest tests/test_openai_agent.py -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "serving"))

import pytest  # noqa: E402
import openai_agent as oa  # noqa: E402  (module handle for monkeypatch + constants)
from openai_agent import run_episode, to_openai_tools  # noqa: E402


# --- tool callables (same shape as tools.build_model_tools) ------------------
def web_search(query: str) -> str:
    """Search the web for a restaurant's menu.

    Args:
        query: the search query.
    """
    return "SEARCH: joes.example/menu"


def scrape_url(url: str, mode: str = "direct") -> str:
    """Fetch a page as markdown.

    Args:
        url: the page URL.
        mode: direct or browser.
    """
    return "SCRAPE: Margherita Pizza $12"


TOOLS = [web_search, scrape_url]
REGISTRY = {f.__name__: f for f in TOOLS}


# --- scriptable fake OpenAI client -------------------------------------------
class _Fn:
    def __init__(self, name, arguments): self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id, self.type, self.function = id, "function", _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Resp:
    def __init__(self, msg): self.choices = [type("C", (), {"message": msg})()]


class FakeClient:
    """Returns scripted messages in order; records each create() kwargs."""
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                return _Resp(outer._scripted.pop(0))

        self.chat = type("Chat", (), {"completions": _Completions()})()


# ---------------------------------------------------------------------------
def test_to_openai_tools_shape():
    decls = to_openai_tools(TOOLS)
    assert [d["function"]["name"] for d in decls] == ["web_search", "scrape_url"]
    ws = decls[0]["function"]
    assert ws["parameters"]["properties"]["query"]["type"] == "string"
    assert ws["parameters"]["required"] == ["query"]           # query required
    scr = decls[1]["function"]
    assert scr["parameters"]["required"] == ["url"]            # mode has a default -> optional
    assert ws["description"].startswith("Search the web")      # from docstring


def test_run_episode_executes_tools_then_answers():
    final = '{"found": true, "restaurant_name": "Joe", "menu": []}'
    client = FakeClient([
        _Msg(tool_calls=[_ToolCall("c1", "web_search", '{"query": "Joe menu"}')]),
        _Msg(tool_calls=[_ToolCall("c2", "scrape_url", '{"url": "joes.example/menu"}')]),
        _Msg(content=final),
    ])
    text, messages = run_episode(client, "teacher", "Joe's, NYC", TOOLS, REGISTRY, "SYS")
    assert text == final
    # system + user + (assistant toolcall + tool result) x2 + final assistant
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    # tool results were threaded back with their call ids + registry output
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c1" and "SEARCH" in tool_msgs[0]["content"]
    assert "SCRAPE" in tool_msgs[1]["content"]


def test_run_episode_immediate_answer():
    client = FakeClient([_Msg(content='{"found": false}')])
    text, messages = run_episode(client, "teacher", "Ghost, Nowhere", TOOLS, REGISTRY, "SYS")
    assert text == '{"found": false}'
    assert len(client.calls) == 1
    assert client.calls[0]["tool_choice"] == "auto"


def test_budget_exhaustion_drops_tools_and_finalizes():
    # Model keeps calling tools; after max_tool_calls the loop injects the finalize
    # instruction and drops tools, and the model then answers.
    scripted = [_Msg(tool_calls=[_ToolCall(f"c{i}", "web_search", '{"query": "x"}')]) for i in range(2)]
    scripted.append(_Msg(content='{"found": true, "restaurant_name": "X", "menu": []}'))
    client = FakeClient(scripted)
    text, messages = run_episode(client, "teacher", "X", TOOLS, REGISTRY, "SYS", max_tool_calls=2)
    assert text.startswith("{")
    # the final create call dropped tools (out of budget)
    assert client.calls[-1]["tools"] is None and client.calls[-1]["tool_choice"] is None
    # and a finalize user turn was injected
    assert any(m["role"] == "user" and "SYS" not in m["content"] and m["content"] != "X"
               for m in messages)


# --- context-overflow handling (the max_tokens clamp + finalize-on-400) -------
class _RaisingClient:
    """Like FakeClient, but a scripted item that is an Exception is RAISED by
    create() -- to simulate a vLLM context-length 400 mid-loop. No base_url, so
    run_episode's _detect_max_model_len returns None (no clamp) unless patched."""
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                item = outer._scripted.pop(0)
                if isinstance(item, Exception):
                    raise item
                return _Resp(item)

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _ctx_400() -> Exception:
    """Stand-in for vLLM's context-length BadRequestError (matched by _is_context_overflow)."""
    return RuntimeError(
        "Error code: 400 - This model's maximum context length is 98304 tokens. However, "
        "you requested 16384 output tokens and your prompt contains at least 81921 input "
        "tokens ... Please reduce the length ... parameter=input_tokens"
    )


def test_run_episode_finalizes_on_context_overflow():
    # A tool turn 400s on length -> drop tools, inject the finalize instruction,
    # retry, and return that menu rather than raising.
    menu = '{"found": true, "menu": []}'
    client = _RaisingClient([_ctx_400(), _Msg(content=menu)])
    text, messages = run_episode(client, "teacher", "Joe's, NYC", TOOLS, REGISTRY, "SYS")
    assert text == menu
    assert client.calls[1]["tools"] is None                       # finalize retry dropped tools
    assert any(m.get("content") == oa.BUDGET_FINALIZE_INSTRUCTION for m in messages)


def test_run_episode_empty_when_prompt_itself_overflows():
    # Even the finalize retry 400s (accumulated prompt alone too big) -> graceful
    # empty return, never a raise that would kill the whole build.
    client = _RaisingClient([_ctx_400(), _ctx_400()])
    text, _ = run_episode(client, "teacher", "X", TOOLS, REGISTRY, "SYS")
    assert text == ""


def test_run_episode_reraises_non_context_error():
    # A non-context API error must propagate, not be silently finalized.
    client = _RaisingClient([RuntimeError("upstream 500: engine died")])
    with pytest.raises(RuntimeError, match="engine died"):
        run_episode(client, "teacher", "X", TOOLS, REGISTRY, "SYS")


class _CompletionsClient:
    """The RAW /v1/completions surface build_gemma_completions talks to.

    Separate from FakeClient, which doubles the chat/tools path -- the Gemma student
    deliberately does NOT use that one (no Gemma-4 tool parser in vLLM).
    """

    def __init__(self, text, *, stop_reason=None, finish_reason="stop"):
        self.calls = []
        outer = self

        class _C:
            def create(self, **kw):
                outer.calls.append(kw)
                choice = type("Ch", (), {"text": text, "stop_reason": stop_reason,
                                         "finish_reason": finish_reason})()
                return type("R", (), {"choices": [choice]})()

        self.completions = _C()


def test_gemma_completions_restores_the_stripped_stop_marker():
    """vLLM strips the stop string; parse_response needs it back or it sees no call."""
    client = _CompletionsClient("call:web_search{}", stop_reason="<tool_call|>")
    gen = oa.build_gemma_completions(client, "m", max_model_len=40960)
    assert gen("prompt").endswith("<tool_call|>")


def test_gemma_completions_warns_when_the_turn_is_cut_off(capsys):
    """A truncated turn has no closing channel, so it parses as pure `thinking` and
    the caller returns the raw reasoning as if it were the answer. Silent truncation
    reads to the user as 'the model emitted garbage'; say so instead."""
    client = _CompletionsClient("reasoning...", finish_reason="length")
    gen = oa.build_gemma_completions(client, "m", max_tokens=4096, max_model_len=40960)
    gen("prompt")
    assert "cut off" in capsys.readouterr().out


def test_gemma_completions_passes_the_requested_budget_through():
    client = _CompletionsClient("x", finish_reason="stop")
    gen = oa.build_gemma_completions(client, "m", max_tokens=16384, max_model_len=98304)
    gen("prompt")
    assert client.calls[0]["max_tokens"] == 16384


def test_output_budget_clamps_to_window(monkeypatch):
    # Known (small) window + large prompt -> requested max_tokens clamped below
    # MAX_TOKENS but never under the floor.
    monkeypatch.setattr(oa, "_detect_max_model_len", lambda client, model: 2000)
    client = FakeClient([_Msg(content='{"found": false}')])
    run_episode(client, "teacher", "X", TOOLS, REGISTRY, "S" * 6000)  # ~2000-token system
    asked = client.calls[0]["max_tokens"]
    assert oa._MIN_OUTPUT_TOKENS <= asked < oa.MAX_TOKENS
