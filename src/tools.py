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

import asyncio
import functools
import re

from backends import build_scrape, build_search, is_cacheable
from cache import norm_query, norm_scrape, scrape_status
from prompts import build_system_prompt

# ---------------------------------------------------------------------------
# Shared output bound
# ---------------------------------------------------------------------------
# Hard cap on a single tool result before it enters the message history. Sized to
# bound CONTEXT GROWTH: the vLLM teacher accumulates EVERY tool result in one
# prompt, so at ~3.3 chars/token a 24K-char cap is ~7K tokens and even a full
# tool-call budget of scrapes stays inside the served window. An uncapped 75K-char
# page was ~19K tokens, and a few of those overflowed the context (the corpus
# build's context-length 400s -- see notes/experiments.md 2026-07-19). A typical
# menu page still fits intact (Pagliacci's full order page was ~16K chars); a blind
# char cap can clip the tail of an unusually long menu, so a hit is warned about
# (below) -- never silent. Retunable without re-scraping (the cache stores the raw
# uncapped response; see setup_tools). Paired with the output-budget clamp +
# overflow-finalize in serving/openai_agent.run_episode as the safety net.
MAX_TOOL_CHARS = 24000


# ---------------------------------------------------------------------------
# Scrape-result slimming (token cost) -- applied at READ time, after the cache,
# so the stored row keeps the full markdown and this stays retunable.
# ---------------------------------------------------------------------------
# What gets dropped is deliberately CONSERVATIVE: images, text-less links, and
# dead-end hrefs (tel:/mailto:/js/#fragments/image files) -- measured 10.0%
# smaller on the pilot's 401 cached pages. Navigable hrefs are KEPT: 17.7% of
# the teacher's scrape calls used a URL found only via in-page links (homepage
# -> menu page, own site -> ordering platform), so stripping all link targets
# (a further ~13%) would break real trajectories. Applied to scrape results
# only -- search results are WHERE the model gets URLs from.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_EMPTY_LINK_RE = re.compile(r"\[\s*\]\(\s*[^)]*\)")
_MD_DEAD_HREF_RE = re.compile(
    r"\[([^\]]+)\]\(\s*(?:tel:|mailto:|javascript:|#)[^)]*\)"
    r"|\[([^\]]+)\]\(\s*[^)]*\.(?:png|jpe?g|gif|webp|svg|ico)(?:\?[^)]*)?\s*\)",
    re.IGNORECASE,
)


def _slim_scrape(md: str) -> str:
    """Drop non-navigable markdown bulk from a scraped page (see block comment)."""
    md = _MD_IMAGE_RE.sub("", md)
    md = _MD_EMPTY_LINK_RE.sub("", md)
    md = _MD_DEAD_HREF_RE.sub(lambda m: m.group(1) or m.group(2), md)
    return re.sub(r"\n{4,}", "\n\n\n", md)


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
def _to_async(fn):
    """A blocking tool -> a coroutine that runs it in a worker thread.

    WHY (measured 2026-07-16, GRPO round 1): TRL's GRPO tool loop splits the declared
    tools by `inspect.iscoroutinefunction` (grpo_trainer.py ~1819):

        if name in sync_tool_dict:   results.append(sync_tool_dict[name](**args))  # inline, SERIAL
        elif name in async_tool_dict: async_coros.append(...)                      # asyncio.gather -> PARALLEL

    So SYNC tools are executed one at a time, blocking the whole loop. With live tools
    that is ~256 sequential network round-trips per step (32 completions x up to 8 calls),
    which measured >40 min/step with the GPU pinned at 0% -- ~100 h for a 150-step run.
    Making the tools coroutines lets TRL gather them, so the scrapes overlap.

    `asyncio.to_thread` is the right primitive because the backends are genuinely
    blocking (requests + SYNC Playwright) and cannot be made natively async. Its
    ThreadPoolExecutor reuses threads, which suits backends.py's THREAD-LOCAL Chromium
    pool exactly: each worker thread launches one browser and reuses it across calls
    (the same property that makes the viz server's threadpool safe -- see CLAUDE.md).

    functools.wraps keeps `__name__`/`__doc__`/`__annotations__`, so
    apply_chat_template still renders the SAME tool schema the model was SFT'd on --
    the docstring stays defined once, on the sync function, and cannot drift. (It also
    sets `__wrapped__`, so inspect.signature reports the real signature, while
    iscoroutinefunction still reports True -- it reads the code flags, not __wrapped__.)
    """
    @functools.wraps(fn)
    async def _async_tool(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)
    return _async_tool


def build_model_tools(search_fn, scrape_fn, async_tools: bool = False):
    """Wrap the backend's (search_fn, scrape_fn) as the model-facing tools.

    Returns (tools, registry): `tools` is a list of plain functions (for
    apply_chat_template / to_anthropic_tools) and `registry` maps
    name -> callable(**kwargs) -> str. The docstrings here are what the model
    reads, so they stay vendor-neutral.

    async_tools=True returns the SAME tools as coroutines (see _to_async) so TRL's
    GRPO loop runs them in parallel instead of serially. OPT-IN on purpose: the other
    callers (eval_split, gemma/agent.py, claude_agent) invoke the registry
    synchronously and would get a coroutine object instead of a string.
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
        return _cap(_slim_scrape(scrape_fn(url, mode)), "scrape_url")

    tools = [web_search, scrape_url]
    if async_tools:
        tools = [_to_async(fn) for fn in tools]
    registry = {fn.__name__: fn for fn in tools}
    return tools, registry


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_tools(dietary_restrictions=None, variant: str = "teacher", cache=None,
                async_tools: bool = False):
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
    async_tools (bool): return the tools as coroutines so TRL's GRPO loop executes them
    in PARALLEL rather than one-at-a-time (see _to_async -- this is the difference
    between ~40 min/step and a usable rollout rate on live tools). Only train_grpo.py
    wants this; the sync callers would get coroutines back.
    """
    search_fn, scrape_fn = build_search(), build_scrape()
    if cache is not None:
        # scrape is 2-arg (url, mode); norm_scrape keys on BOTH so direct/browser
        # renders are distinct entries. scrape_status marks failure sentinels
        # 'error' so transient Chromium timeouts aren't frozen into the corpus, and
        # is_cacheable drops local-browser failures before they're stored at all.
        search_fn = cache.wrap("search", search_fn, key_fn=norm_query, provider="brave")
        scrape_fn = cache.wrap(
            "scrape", scrape_fn, key_fn=norm_scrape, status_fn=scrape_status,
            provider="local", store_if=is_cacheable,
        )
    tools, registry = build_model_tools(search_fn, scrape_fn, async_tools=async_tools)
    cached = f", cached ({cache.miss_policy}) at {cache.path}" if cache is not None else ""
    mode = " [async: parallel tool calls]" if async_tools else ""
    print(f"Live tools: web_search via Brave, scrape_url via local Chromium{cached}{mode}")
    return tools, registry, build_system_prompt(dietary_restrictions, variant=variant)
