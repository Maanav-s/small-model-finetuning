"""The agentic loop, Claude edition: restaurant name -> menu JSON.

A Claude baseline for the Gemma agent in gemma/agent.py — it runs the **same tools,
the same system prompt, and the same JSON contract** through Claude Sonnet via
the Anthropic Messages API, so the two models' outputs are directly comparable.

The tool source is shared: setup_tools() in tools.py returns the same
`(tools, tool_registry, system_prompt, client)` whether you're driving Gemma or
Claude. The only translation needed is the tool *declaration* format — Gemma's
template takes Python callables or OpenAI-style function dicts, while the
Anthropic API wants `{"name", "description", "input_schema"}`. to_anthropic_tools
handles both shapes; the registry (name -> callable returning str) is used as-is.

Like agent.py this module is the reusable engine — no CLI or key loading. Drive
it from run_claude.py or a REPL. Adaptive thinking is on (the recommended
default for agentic work): thinking happens in thinking blocks, so the visible
text stays the schema-only JSON the system prompt asks for.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Shared modules (schema/prompts/tools/mcp_client) live in src/, the parent of
# this claude/ folder; put it on the path so the flat imports (the __main__ demo
# here, and run_claude.py) resolve whether run directly or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

MODEL_ID = "claude-sonnet-4-6"  # user-requested: the Claude baseline runs on Sonnet
MAX_TOOL_CALLS = 4              # tool-call budget per episode (matches agent.py)
MAX_TOKENS = 8192              # the full menu JSON can be long (matches MAX_NEW_TOKENS)

# Python annotation -> JSON Schema type, for converting the local web_search stub.
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _callable_to_anthropic(fn) -> dict:
    """Convert a plain Python tool function to an Anthropic tool declaration.

    Mirrors what transformers' apply_chat_template does for Gemma: name from
    __name__, description from the docstring, input schema from the typed
    signature. Used for the offline web_search stub (tools.TOOLS).
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


def _schema_to_anthropic(tool: dict) -> dict:
    """Convert an OpenAI-style {"type":"function","function":{...}} dict (the
    shape build_mcp_tools emits for Firecrawl) to an Anthropic tool declaration.
    """
    fn = tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def to_anthropic_tools(tools: list) -> list[dict]:
    """Translate the tool list from setup_tools() into Anthropic tool decls.

    Accepts either the stub path (Python callables) or the MCP path (function
    dicts), so the same setup_tools(use_mcp=...) result drives Claude unchanged.
    """
    converted = []
    for tool in tools:
        if callable(tool):
            converted.append(_callable_to_anthropic(tool))
        elif isinstance(tool, dict) and "function" in tool:
            converted.append(_schema_to_anthropic(tool))
        else:
            raise TypeError(f"Unrecognized tool declaration: {tool!r}")
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
            thinking={"type": "adaptive"},
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

    from tools import TOOLS

    print(json.dumps(to_anthropic_tools(TOOLS), indent=2))
