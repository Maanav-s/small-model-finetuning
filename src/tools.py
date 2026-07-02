"""Tool sources for the agent loop.

Two sources, selected by setup_tools():
  - live web tools (`web_search` + `scrape_url`) whose *backends* are pluggable --
    pick a search provider and a scrape provider independently (see backends.py;
    firecrawl/tavily/brave/jina/browserless). This is the default, and
  - an offline `web_search` stub (deterministic, returns sample_menu.md) for
    developing the loop without a network/key (setup_tools(offline=True)).

Both sources expose tools as plain Python functions (typed signature + Google-
style docstring). apply_chat_template(tools=...) converts those to Gemma's
schema, and the Claude runner's to_anthropic_tools converts the same callables
to Anthropic decls -- so the agent loops are identical across sources.

The model only ever sees ONE search and ONE scrape tool, both named generically
(web_search / scrape_url) with fixed docstrings (build_model_tools), so the
*backend can be swapped underneath without the model noticing* -- this is how we
A/B providers on the same task. The generic names also keep vendor branding out
of the SFT/GRPO training data the tool calls get baked into later.
"""

from __future__ import annotations

from pathlib import Path

from backends import (
    DEFAULT_BACKEND,
    SCRAPE_BACKENDS,
    SEARCH_BACKENDS,
    build_backend,
)
from prompts import LIVE_SYSTEM_PROMPT, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Shared output bound
# ---------------------------------------------------------------------------
# Hard cap on a single tool result before it enters the message history, as a
# backstop against a pathologically huge page. Generation forces SDPA's O(seq)
# mem-efficient kernel (see generate_turn in agent.py), so a full menu page fits
# comfortably; a blind char cap can still clip the tail of a very long menu, so a
# hit is warned about (below) -- never silent.
MAX_TOOL_CHARS = 75000


def _cap(text: str, label: str) -> str:
    """Truncate an over-long tool result, warning so a clipped menu isn't silent."""
    if len(text) > MAX_TOOL_CHARS:
        print(
            f"  [warn] {label} returned {len(text)} chars; truncating to "
            f"{MAX_TOOL_CHARS} (the tail is dropped - raise MAX_TOOL_CHARS or use "
            f"a chunk/lookup tool if this clips the menu)"
        )
        text = text[:MAX_TOOL_CHARS]
    return text


# ---------------------------------------------------------------------------
# Offline stub tool
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
    # agentic loop can be developed offline. setup_tools(offline=False) replaces
    # this with the live Firecrawl web_search/scrape_url tools below.
    return _SAMPLE_MENU.read_text(encoding="utf-8")


STUB_TOOLS = [web_search]
STUB_REGISTRY = {fn.__name__: fn for fn in STUB_TOOLS}


# ---------------------------------------------------------------------------
# Live tools -- the model-facing wrappers
# ---------------------------------------------------------------------------
# The model is handed exactly two tools, named generically with fixed docstrings,
# so it sees the *same* tools no matter which provider backs them. The selected
# backend's search_fn/scrape_fn (from backends.py) do the actual network call;
# these wrappers add the MAX_TOOL_CHARS cap and nothing else.
def build_model_tools(search_fn, scrape_fn):
    """Wrap a backend's (search_fn, scrape_fn) as the model-facing tools.

    Returns (tools, registry) matching the stub's shape: `tools` is a list of
    plain functions (for apply_chat_template / to_anthropic_tools) and `registry`
    maps name -> callable(**kwargs) -> str. The docstrings here are what the model
    reads, so they stay vendor-neutral and identical across backends.
    """

    def web_search(query: str) -> str:
        """Search the web for a restaurant's menu information.

        Args:
            query: Search query, e.g. the restaurant name plus its city and the
                word "menu".
        """
        return _cap(search_fn(query), "web_search")

    def scrape_url(url: str) -> str:
        """Fetch the full contents of a web page as markdown.

        Use this on a promising URL returned by web_search to read the full menu
        page before writing the JSON.

        Args:
            url: The page URL to fetch (e.g. a result URL from web_search).
        """
        return _cap(scrape_fn(url), "scrape_url")

    tools = [web_search, scrape_url]
    registry = {fn.__name__: fn for fn in tools}
    return tools, registry


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def setup_tools(
    offline: bool = False,
    search_backend: str = DEFAULT_BACKEND,
    scrape_backend: str = DEFAULT_BACKEND,
):
    """Pick the tool source and return (tools, tool_registry, system_prompt).

    offline=False (default): live `web_search`/`scrape_url`, each backed by the
    chosen provider (see backends.py). The search and scrape providers are picked
    independently -- e.g. search via 'brave', scrape via 'jina' -- so any pair can
    be A/B tested on the same task. Only the selected providers' API keys are read.
    offline=True: the deterministic `web_search` stub that returns sample_menu.md,
    for developing the loop without a key or network.
    """
    if offline:
        return STUB_TOOLS, STUB_REGISTRY, SYSTEM_PROMPT

    if search_backend not in SEARCH_BACKENDS:
        raise SystemExit(
            f"{search_backend!r} has no web_search; pick --search-backend from "
            f"{SEARCH_BACKENDS}."
        )
    if scrape_backend not in SCRAPE_BACKENDS:
        raise SystemExit(
            f"{scrape_backend!r} has no scrape_url; pick --scrape-backend from "
            f"{SCRAPE_BACKENDS}."
        )

    # Build each requested provider once (the same backend may serve both roles).
    built: dict = {}

    def get(name):
        if name not in built:
            built[name] = build_backend(name)
        return built[name]

    search_fn = get(search_backend)[0]
    scrape_fn = get(scrape_backend)[1]
    tools, registry = build_model_tools(search_fn, scrape_fn)
    print(
        f"Live tools: web_search via {search_backend!r}, "
        f"scrape_url via {scrape_backend!r}"
    )
    return tools, registry, LIVE_SYSTEM_PROMPT
