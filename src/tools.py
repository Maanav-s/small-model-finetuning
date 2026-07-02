"""Tool sources for the agent loop.

Two sources, selected by setup_tools():
  - live web tools (`web_search` + `scrape_url`), backed by Brave (search) and
    Jina (scrape) -- see backends.py. This is the default, and
  - an offline `web_search` stub (deterministic, returns sample_menu.md) for
    developing the loop without a network/key (setup_tools(offline=True)).

Both sources expose tools as plain Python functions (typed signature + Google-
style docstring). apply_chat_template(tools=...) converts those to Gemma's
schema, and the Claude runner's to_anthropic_tools converts the same callables
to Anthropic decls -- so the agent loops are identical across sources.

The model only ever sees ONE search and ONE scrape tool, both named generically
(web_search / scrape_url) with fixed docstrings (build_model_tools), so the
*backend stays invisible to the model*. The generic names also keep vendor
branding out of the SFT/GRPO training data the tool calls get baked into later.
"""

from __future__ import annotations

from pathlib import Path

from backends import build_scrape, build_search
from prompts import build_system_prompt

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
    # this with the live web_search/scrape_url tools below (Brave + Jina).
    return _SAMPLE_MENU.read_text(encoding="utf-8")


STUB_TOOLS = [web_search]
STUB_REGISTRY = {fn.__name__: fn for fn in STUB_TOOLS}


# ---------------------------------------------------------------------------
# Live tools -- the model-facing wrappers
# ---------------------------------------------------------------------------
# The model is handed exactly two tools, named generically with fixed docstrings,
# so it never sees which provider backs them. The backend's search_fn/scrape_fn
# (from backends.py) do the actual network call; these wrappers add the
# MAX_TOOL_CHARS cap and nothing else.
def build_model_tools(search_fn, scrape_fn):
    """Wrap the backend's (search_fn, scrape_fn) as the model-facing tools.

    Returns (tools, registry) matching the stub's shape: `tools` is a list of
    plain functions (for apply_chat_template / to_anthropic_tools) and `registry`
    maps name -> callable(**kwargs) -> str. The docstrings here are what the model
    reads, so they stay vendor-neutral.
    """

    def web_search(query: str) -> str:
        """Search the web for a restaurant's menu information.

        Args:
            query: Search query, e.g. the restaurant name plus its city and the
                word "menu".
        """
        return _cap(search_fn(query), "web_search")

    def scrape_url(url: str, mode: str = "direct") -> str:
        """Fetch the full contents of a web page as markdown.

        Use this on a promising URL returned by web_search to read the full menu
        page before writing the JSON.

        Args:
            url: The page URL to fetch (e.g. a result URL from web_search).
            mode: How the page is fetched. "direct" (the default) does a plain,
                quick fetch of the page's HTML and works for most pages -- ALWAYS
                TRY "direct" FIRST. "browser" loads the page in a real browser that
                runs its JavaScript, which some pages need before their menu
                appears; it is slower, and some sites block automated browsers and
                return little. Neither mode is always better, so use "browser" only
                as a fallback when a "direct" fetch came back empty or clearly
                missing the menu, and keep whichever result actually has the menu.
        """
        return _cap(scrape_fn(url, mode), "scrape_url")

    tools = [web_search, scrape_url]
    registry = {fn.__name__: fn for fn in tools}
    return tools, registry


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def setup_tools(offline: bool = False, dietary_restrictions=None, variant: str = "teacher"):
    """Pick the tool source and return (tools, tool_registry, system_prompt).

    offline=False (default): live `web_search` (Brave) + `scrape_url` (Jina) --
    see backends.py; reads BRAVE_API_KEY and JINA_API_KEY.
    offline=True: the deterministic `web_search` stub that returns sample_menu.md,
    for developing the loop without a key or network.

    dietary_restrictions (None / str / list[str]): slotted into the system prompt
    so the model filters the menu to complying items; empty means no filtering.
    variant ("teacher" | "student"): system-prompt variant (see prompts.py) --
    "teacher" (default) carries the source-selection guidance, "student" omits it.
    """
    if offline:
        prompt = build_system_prompt(dietary_restrictions, live=False, variant=variant)
        return STUB_TOOLS, STUB_REGISTRY, prompt

    tools, registry = build_model_tools(build_search(), build_scrape())
    print("Live tools: web_search via Brave, scrape_url via Jina")
    return tools, registry, build_system_prompt(dietary_restrictions, live=True, variant=variant)
