"""The agentic loop, Claude edition: restaurant name -> menu JSON.

A Claude baseline for the Gemma agent in gemma/agent.py — it runs the **same tools,
the same system prompt, and the same JSON contract** through Claude Sonnet via
the Anthropic Messages API, so the two models' outputs are directly comparable.

The tool source is shared: setup_tools() in tools.py returns the same
`(tools, tool_registry, system_prompt)` whether you're driving Gemma or Claude.
Both the offline stub and the live Firecrawl tools are plain Python callables, so
the only translation needed is the tool *declaration* format — Gemma's template
reads the callables directly, while the Anthropic API wants
`{"name", "description", "input_schema"}`. to_anthropic_tools does that; the
registry (name -> callable returning str) is used as-is.

Like agent.py this module is the reusable engine — no CLI or key loading. Drive
it from run_claude.py or a REPL. Adaptive thinking is on (the recommended
default for agentic work): thinking happens in thinking blocks, so the visible
text stays the schema-only JSON the system prompt asks for.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Shared modules (schema/prompts/tools) live in src/, the parent of this claude/
# folder; put it on the path so the flat imports (the __main__ demo here, and
# run_claude.py) resolve whether run directly or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

MODEL_ID = "claude-sonnet-4-6"        # default Claude baseline (Sonnet)
HAIKU_MODEL_ID = "claude-haiku-4-5"   # cheaper/faster comparison point (non-dated alias)
MAX_TOOL_CALLS = 4              # tool-call budget per episode (matches agent.py)
MAX_TOKENS = 16384            # the full menu JSON can be long (was 8192; output was truncating)

# Thinking config is model-dependent. Sonnet 4.6 supports adaptive thinking; Haiku
# 4.5 does NOT (verified via the Models API: thinking.types.adaptive.supported=false,
# enabled.supported=true) and 400s on an adaptive request, so it uses legacy extended
# thinking with a fixed budget. Newer models (Opus 4.7/4.8, Fable) instead reject
# `budget_tokens`, so adaptive is the safe default and only known non-adaptive models
# are listed here.
LEGACY_THINKING_MODELS = {
    "claude-haiku-4-5", "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5", "claude-sonnet-4-5-20250929",
}
THINK_BUDGET = 4096  # budget_tokens for the legacy path (must be < MAX_TOKENS, min 1024)


def thinking_config(model: str) -> dict:
    """Pick the thinking config a given Claude model accepts (adaptive vs budgeted)."""
    if model in LEGACY_THINKING_MODELS:
        return {"type": "enabled", "budget_tokens": THINK_BUDGET}
    return {"type": "adaptive"}

# Python annotation -> JSON Schema type, for converting the local web_search stub.
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _callable_to_anthropic(fn) -> dict:
    """Convert a plain Python tool function to an Anthropic tool declaration.

    Mirrors what transformers' apply_chat_template does for Gemma: name from
    __name__, description from the docstring, input schema from the typed
    signature. Used for every tool now that both sources are Python callables.
    """
    properties, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        properties[name] = {"type": _JSON_TYPES.get(param.annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def to_anthropic_tools(tools: list) -> list[dict]:
    """Translate the tool list from setup_tools() into Anthropic tool decls.

    Both tool sources (the offline stub and the live Firecrawl tools) are plain
    Python callables, so the same setup_tools() result drives Claude unchanged.
    """
    converted = []
    for tool in tools:
        if not callable(tool):
            raise TypeError(f"Expected a callable tool, got: {tool!r}")
        converted.append(_callable_to_anthropic(tool))
    return converted


def _final_text(response) -> str:
    """Join the text blocks of a Claude response (the schema-only JSON answer)."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def run_episode(
    client: anthropic.Anthropic,
    restaurant_name: str,
    tools: list,
    tool_registry: dict,
    system_prompt: str,
    model: str = MODEL_ID,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> str:
    """Run the tool-call loop for one restaurant; return the final answer text.

    Standard manual agentic loop: call the model, execute any tool_use blocks via
    the shared registry, feed the results back, repeat until Claude answers (or
    the tool-call budget is spent, after which one tool-free call forces a JSON
    answer rather than returning empty as agent.py does).
    """
    anthropic_tools = to_anthropic_tools(tools)
    messages: list[dict] = [{"role": "user", "content": restaurant_name}]

    for step in range(max_tool_calls + 1):
        # Budget spent: drop tools so the model must answer from what it gathered.
        out_of_budget = step == max_tool_calls
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=[] if out_of_budget else anthropic_tools,
            thinking=thinking_config(model),
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return _final_text(response)  # final answer

        if out_of_budget:
            # Shouldn't happen (no tools offered), but don't loop forever.
            print(f"  [warn] hit MAX_TOOL_CALLS={max_tool_calls} without a final answer")
            return _final_text(response)

        # Preserve the assistant turn verbatim (incl. thinking + tool_use blocks).
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            name, args = block.name, block.input
            print(f"  [step {step}] tool call: {name}({args})")
            if name not in tool_registry:
                print(f"  [warn] unknown tool {name!r}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: unknown tool {name!r}",
                    "is_error": True,
                })
                continue
            response_text = tool_registry[name](**args)
            print(f"  [step {step}] -> {len(response_text)} chars returned")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": response_text,
            })

        messages.append({"role": "user", "content": tool_results})

    return ""


if __name__ == "__main__":
    # Render-only demo: show the Anthropic tool declarations built from the stub
    # tool source. No API key / network needed.
    import json

    from tools import STUB_TOOLS

    print(json.dumps(to_anthropic_tools(STUB_TOOLS), indent=2))
