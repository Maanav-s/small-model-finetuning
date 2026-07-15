"""Unit tests for src/vllm/openai_agent.py (the OpenAI-compatible vLLM runner).

Pure-local, no server / no `openai` package: run_episode takes an injected client,
so a scripted fake drives the tool loop. Covers to_openai_tools shape and the
agentic loop (tool execution, message threading, final answer).

Run: uv run python -m pytest tests/test_openai_agent.py -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vllm"))

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
