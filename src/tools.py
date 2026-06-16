"""Tool sources for the agent loop.

Two sources, selected by setup_tools():
  - the offline `web_search` stub (deterministic, returns sample_menu.md), and
  - the live Firecrawl MCP server (search/scrape over a local npx subprocess).

apply_chat_template(tools=...) accepts JSON-Schema dicts directly (not just
Python callables), so an MCP tool maps cleanly to the OpenAI-style
{"type": "function", "function": {...}} shape Gemma's template consumes — no
need to synthesize fake Python functions with docstrings.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp_client import MCPStdioClient
from prompts import MCP_SYSTEM_PROMPT, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Local stub tool
#
# Defined as a plain Python function: apply_chat_template reads the signature +
# Google-style docstring and converts it to the schema Gemma wants.
# ---------------------------------------------------------------------------
_SAMPLE_MENU = Path(__file__).with_name("sample_menu.md")


def web_search(query: str) -> str:
    """Search the web for a restaurant's menu information.

    Args:
        query: Search query, e.g. the restaurant name plus its city and the
            word "menu".
    """
    # TEMP STUB: ignores the query and returns a fixed sample menu, so the
    # agentic loop can be developed offline. The MCP path (setup_tools(use_mcp=
    # True)) replaces this with real Firecrawl search/scrape.
    return _SAMPLE_MENU.read_text(encoding="utf-8")


TOOLS = [web_search]
TOOL_REGISTRY = {fn.__name__: fn for fn in TOOLS}


# ---------------------------------------------------------------------------
# Firecrawl MCP tools
# ---------------------------------------------------------------------------
# Firecrawl exposes ~20 tools; for menu extraction we only want search (and an
# optional follow-up scrape of a specific page). Whitelisting keeps the prompt
# small and stops the model from reaching for crawl/extract/monitor tools.
DEFAULT_MCP_TOOL_ALLOWLIST = ("firecrawl_search", "firecrawl_scrape")

# Reduce-at-the-source: cap how much each Firecrawl call returns so big pages
# don't balloon the model's context (and the per-turn prefill that re-encodes
# it). Enforced in the dispatch wrapper, not the prompt, so it holds regardless
# of what args the model emits.
SEARCH_RESULT_LIMIT = 3

# Hard cap on a single tool result before it enters the message history. This
# bounds the worst case (one huge scraped page) that sets the peak context
# length -> peak VRAM, regardless of page size. A blind char cap can clip the
# tail of a very long menu; a hit is warned about (below) so it's never silent.
MAX_TOOL_CHARS = 8000


def _apply_arg_policy(name: str, kwargs: dict) -> dict:
    """Clamp/override a tool's args before dispatch to bound the returned text."""
    args = dict(kwargs)
    if name == "firecrawl_search":
        # Cap result count; honor a smaller value the model may have asked for.
        limit = args.get("limit")
        args["limit"] = (
            min(limit, SEARCH_RESULT_LIMIT)
            if isinstance(limit, int) and limit > 0
            else SEARCH_RESULT_LIMIT
        )
        # Drop scrapeOptions: with it, search scrapes every result to full
        # markdown and concatenates them (one call returned ~28k chars and OOM'd
        # the prefill). Keep search to lightweight snippets; the model can then
        # firecrawl_scrape a single chosen URL for the full page.
        args.pop("scrapeOptions", None)
    elif name == "firecrawl_scrape":
        # Force a single compact markdown payload (never raw HTML or the
        # json/jsonOptions extraction path, which bloats output and lets the
        # model author malformed schemas). Keep onlyMainContent off: it tends
        # to strip sidebars/sections that hold real menu items.
        args["formats"] = ["markdown"]
        args["onlyMainContent"] = False
        args.pop("jsonOptions", None)  # dead once we're not requesting json
    return args


def _dispatch(client, name: str, kwargs: dict) -> str:
    """Apply the arg policy, call the tool, and cap the returned text."""
    text = client.call_tool(name, _apply_arg_policy(name, kwargs))
    if len(text) > MAX_TOOL_CHARS:
        print(
            f"  [warn] {name} returned {len(text)} chars; truncating to "
            f"{MAX_TOOL_CHARS} (the tail is dropped - raise MAX_TOOL_CHARS or "
            f"use a chunk/lookup tool if this clips the menu)"
        )
        text = text[:MAX_TOOL_CHARS]
    return text


def _mcp_tool_to_schema(tool) -> dict:
    """Convert an `mcp.types.Tool` to the JSON-Schema dict the template expects."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def build_mcp_tools(client, allowlist=DEFAULT_MCP_TOOL_ALLOWLIST):
    """List the MCP server's tools and return (tools_schema, registry).

    `tools_schema` goes to apply_chat_template(tools=...); `registry` maps each
    tool name to a callable `(**kwargs) -> str` that dispatches to the server,
    matching the calling convention the loop uses for local tools.
    """
    tools_schema, registry = [], {}
    for tool in client.list_tools():
        if allowlist and tool.name not in allowlist:
            continue
        tools_schema.append(_mcp_tool_to_schema(tool))
        # bind `name` per-iteration so the closure captures the right tool
        registry[tool.name] = (
            lambda name: lambda **kwargs: _dispatch(client, name, kwargs)
        )(tool.name)
    if not tools_schema:
        available = [t.name for t in client.list_tools()]
        raise RuntimeError(
            f"No MCP tools matched the allowlist {allowlist!r}. "
            f"Server offers: {available}"
        )
    return tools_schema, registry


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def setup_tools(use_mcp: bool):
    """Pick the tool source: Firecrawl MCP server or the local stub.

    Returns (tools, tool_registry, system_prompt, mcp_client). mcp_client is None
    for the stub path; otherwise close() it when done.
    """
    if not use_mcp:
        return TOOLS, TOOL_REGISTRY, SYSTEM_PROMPT, None

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise SystemExit(
            "--mcp requires FIRECRAWL_API_KEY in the environment "
            "(get one at https://firecrawl.dev)."
        )
    print("Starting Firecrawl MCP server (npx -y firecrawl-mcp) ...")
    # Pass the full environment so npx resolves on PATH and the key is visible.
    client = MCPStdioClient("npx", ["-y", "firecrawl-mcp"], env={**os.environ})
    tools, registry = build_mcp_tools(client)
    print(f"MCP tools available: {[t['function']['name'] for t in tools]}")
    return tools, registry, MCP_SYSTEM_PROMPT, client
