"""The agentic loop, OpenAI-compatible edition (vLLM-served models).

Drives any OpenAI-compatible chat endpoint -- a local vLLM server running a
Qwen3/Hermes-3 teacher, or any model whose vLLM tool-call parser surfaces
`tool_calls` -- through the SAME tools, system prompt, and JSON contract as the
Gemma (src/gemma/agent.py) and Claude (src/claude/claude_agent.py) loops. See
notes/vllm_inference.html.

The tool source is shared: setup_tools() in tools.py returns the same
`(tools, tool_registry, system_prompt)` regardless of which model drives it. The
only translation is the tool DECLARATION format -- to_openai_tools converts the
plain Python callables to OpenAI function-tool schema; the registry
(name -> callable -> str) is used as-is.

Like the other *_agent modules this is the reusable ENGINE: no CLI, no key
loading. The client is injected (build_client is a convenience for run_*.py /
eval_split). `openai` is imported lazily inside build_client so this module (and
to_openai_tools) import with no extra dependency.

NOTE ON THE GEMMA STUDENT: do NOT drive the fine-tuned Gemma student through this
chat/tool path -- vLLM ships no Gemma-4 tool-call parser, and re-templating risks
drift from the exact wire format it was SFT'd on. The student is served via the
raw /v1/completions path that keeps our own template + parser (see
src/gemma/agent.py generate_turn's vLLM branch). This module is for the teacher /
OpenAI-tool-parser models.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

# Shared modules (prompts/tools/schema) live in src/, the parent of this vllm/
# folder; put it on the path so the flat imports resolve (script-run convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import BUDGET_FINALIZE_INSTRUCTION  # noqa: E402

MAX_TOOL_CALLS = 8      # tool-call budget per episode (matches the other loops)
MAX_TOKENS = 16384      # the full menu JSON can be long

# Python annotation -> JSON Schema type, for converting the tool callables.
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _callable_to_openai(fn) -> dict:
    """Convert a plain Python tool function to an OpenAI function-tool declaration.

    Name from __name__, description from the docstring, parameters from the typed
    signature -- the same derivation as claude_agent._callable_to_anthropic, just a
    different envelope (`{"type":"function","function":{...,"parameters":...}}`).
    """
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        props[name] = {"type": _JSON_TYPES.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def to_openai_tools(tools: list) -> list[dict]:
    """Translate the setup_tools() callables into OpenAI tool declarations."""
    out = []
    for tool in tools:
        if not callable(tool):
            raise TypeError(f"Expected a callable tool, got: {tool!r}")
        out.append(_callable_to_openai(tool))
    return out


def build_client(base_url: str, api_key: str = "EMPTY"):
    """Construct an OpenAI client pointed at a (local vLLM) server. Lazy import so
    the module stays dependency-free until a client is actually needed."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


# Gemma student stop marker: generation stops right after a tool call so the model
# can't hallucinate its own tool response (matches agent.py's HF stop_strings).
_GEMMA_STOP = "<tool_call|>"


def build_gemma_completions(client, model: str, max_tokens: int = 4096,
                            stop: str = _GEMMA_STOP):
    """Return `generate(prompt_str) -> text` backed by vLLM /v1/completions, for the
    Gemma STUDENT (raw completions, NOT chat/tools -- vLLM has no Gemma tool parser).

    The prompt is rendered by the caller with the exact training chat template
    (agent.generate_turn), so training/inference stay byte-matched; vLLM only does
    fast batched decode. vLLM STRIPS the stop string from the returned text, so we
    re-append `<tool_call|>` when generation stopped on it -- otherwise
    tokenizer.parse_response wouldn't see a complete tool call.

    `skip_special_tokens=False` is LOAD-BEARING, not a tweak (verified on a served
    A100, 2026-07-16). Gemma's tool protocol IS special tokens (`<|tool_call>`,
    `<|"|>`, `<tool_call|>`), and vLLM's detokenizer defaults to skip_special_tokens
    =True, which deletes them. The failure is SILENT and looks like a bad model:

        skip_special_tokens=True  -> 'call:web_search{query:...}call:web_search{...}'
                                     stop_reason=None, finish='length'  (rambles to
                                     max_tokens; parse_response sees plain content,
                                     zero tool calls -> the agent loop never fires)
        skip_special_tokens=False -> '<|tool_call>call:web_search{query:<|"|>...<|"|>}'
                                     stop_reason='<tool_call|>', finish='stop'

    Note the stop string can't match either when the markers are stripped -- which is
    why this also breaks the re-append below. Same rule as the HF path (decode with
    skip_special_tokens=False); it just has to be requested over HTTP here.
    """
    def generate(prompt: str) -> str:
        comp = client.completions.create(
            model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0,
            stop=[stop],
            extra_body={"add_special_tokens": False, "skip_special_tokens": False},
        )
        choice = comp.choices[0]
        text = choice.text
        if getattr(choice, "stop_reason", None) == stop:
            text += stop  # restore the marker vLLM stripped, so parse_response is happy
        return text

    return generate


def _assistant_message(msg) -> dict:
    """Serialize an assistant response (content + any tool_calls) back into a
    plain message dict to echo into the next request. Uses getattr so it works
    with the SDK's pydantic objects and with test doubles alike."""
    out: dict = {"role": "assistant", "content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return out


def run_episode(
    client,
    model: str,
    restaurant_name: str,
    tools: list,
    tool_registry: dict,
    system_prompt: str,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, list[dict]]:
    """Run the tool-call loop for one restaurant; return (final_text, messages).

    Standard manual agentic loop over client.chat.completions.create: call the
    model, execute any tool_calls via the shared registry, feed tool results back
    (standalone role="tool" messages keyed by tool_call_id -- the OpenAI transport,
    no bundling), repeat until the model answers or the budget is spent. On the
    final (out-of-budget) turn, tools are dropped and BUDGET_FINALIZE_INSTRUCTION
    is injected so the model commits to JSON from what it gathered (a partial menu
    beats an empty reply -- matches the Gemma/Claude loops).
    """
    oai_tools = to_openai_tools(tools)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": restaurant_name},
    ]

    for step in range(max_tool_calls + 1):
        out_of_budget = step == max_tool_calls
        if out_of_budget:
            messages.append({"role": "user", "content": BUDGET_FINALIZE_INSTRUCTION})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            tools=None if out_of_budget else oai_tools,
            tool_choice=None if out_of_budget else "auto",
        )
        msg = resp.choices[0].message

        if not getattr(msg, "tool_calls", None):
            messages.append({"role": "assistant", "content": msg.content or ""})
            return (msg.content or "").strip(), messages

        messages.append(_assistant_message(msg))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  [step {step}] tool call: {name}({args})")
            if name not in tool_registry:
                out = f"Error: unknown tool {name!r}"
                print(f"  [warn] unknown tool {name!r}")
            else:
                try:
                    out = tool_registry[name](**args)
                    print(f"  [step {step}] -> {len(out)} chars returned")
                except Exception as e:  # noqa: BLE001 -- a bad call must not abort the episode
                    out = f"Error running tool {name!r}: {e}"
                    print(f"  [warn] {out}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": out}
            )

    return "", messages


if __name__ == "__main__":
    # Render-only demo: the OpenAI tool declarations built from the real
    # model-facing tools (over dummy backends). No server / key / network needed.
    from tools import build_model_tools  # noqa: E402

    tools, _ = build_model_tools(lambda query: "", lambda url, mode="direct": "")
    print(json.dumps(to_openai_tools(tools), indent=2))
