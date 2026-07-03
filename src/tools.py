"""The model-facing web tools for the agent loop.

setup_tools() returns the live tools: `web_search` backed by Brave (search) and
`scrape_url` backed by a local headless Chromium (scrape) -- see backends.py.
(The old offline sample_menu.md stub and its --offline flag were removed once
the live backend was finalized.)

Tools are plain Python functions (typed signature + Google-style docstring).
apply_chat_template(tools=...) converts those to Gemma's schema, and the Claude
runner's to_anthropic_tools converts the same callables to Anthropic decls --
so the agent loops are identical across models.

The model only ever sees ONE search and ONE scrape tool, both named generically
(web_search / scrape_url) with fixed docstrings (build_model_tools), so the
*backend stays invisible to the model*. The generic names also keep vendor
branding out of the SFT/GRPO training data the tool calls get baked into later.

Caching (Phase 2): setup_tools(cache=Cache(...)) wraps the live backend closures
in the content-addressed SQLite cache (src/cache.py) BEFORE build_model_tools
applies the MAX_TOOL_CHARS cap, so the stored response is raw/uncapped and the
cap stays retunable without re-scraping. cache=None (default) = uncached.
"""

from __future__ import annotations

from backends import build_scrape, build_search
from cache import norm_query, norm_scrape, scrape_status
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
# The model-facing wrappers
# ---------------------------------------------------------------------------
# The model is handed exactly two tools, named generically with fixed docstrings,
# so it never sees which provider backs them. The backend's search_fn/scrape_fn
# (from backends.py) do the actual network call; these wrappers add the
# MAX_TOOL_CHARS cap and nothing else.
def build_model_tools(search_fn, scrape_fn):
    """Wrap the backend's (search_fn, scrape_fn) as the model-facing tools.

    Returns (tools, registry): `tools` is a list of plain functions (for
    apply_chat_template / to_anthropic_tools) and `registry` maps
    name -> callable(**kwargs) -> str. The docstrings here are what the model
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
# Setup
# ---------------------------------------------------------------------------
def setup_tools(dietary_restrictions=None, variant: str = "teacher", cache=None):
    """Build the live tools and return (tools, tool_registry, system_prompt).

    The tools are `web_search` (Brave; reads BRAVE_API_KEY) + `scrape_url` (local
    headless Chromium, no key) -- see backends.py.

    dietary_restrictions (None / str / list[str]): slotted into the system prompt
    so the model filters the menu to complying items; empty means no filtering.
    variant ("teacher" | "student"): system-prompt variant (see prompts.py) --
    "teacher" (default) carries the source-selection guidance, "student" omits it.
    cache (cache.Cache | None): when given, it wraps the BACKEND closures --
    BEFORE the MAX_TOOL_CHARS cap above -- so the RAW uncapped response is stored
    and the cap stays retunable without re-scraping. The cache's miss_policy
    (live/canned/error) decides what a miss does; see src/cache.py.
    """
    search_fn, scrape_fn = build_search(), build_scrape()
    if cache is not None:
        # scrape is 2-arg (url, mode); norm_scrape keys on BOTH so direct/browser
        # renders are distinct entries. scrape_status marks failure sentinels
        # 'error' so transient Chromium timeouts aren't frozen into the corpus.
        search_fn = cache.wrap("search", search_fn, key_fn=norm_query, provider="brave")
        scrape_fn = cache.wrap(
            "scrape", scrape_fn, key_fn=norm_scrape, status_fn=scrape_status, provider="local"
        )
    tools, registry = build_model_tools(search_fn, scrape_fn)
    cached = f", cached ({cache.miss_policy}) at {cache.path}" if cache is not None else ""
    print(f"Live tools: web_search via Brave, scrape_url via local Chromium{cached}")
    return tools, registry, build_system_prompt(dietary_restrictions, variant=variant)
