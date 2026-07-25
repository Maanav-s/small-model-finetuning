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

Caching (Phase 2): setup_tools(cache=Cache(...)) wraps the backend closures in the
content-addressed SQLite cache (src/cache.py) BEFORE build_model_tools applies the
MAX_TOOL_CHARS cap. The scrape backend slims its own output at the source
(backends._slim_scrape, baked into build_scrape) -- so the stored response is
SLIMMED (junk removed) but uncapped, and only the cap stays retunable without
re-scraping. cache=None (default) = uncached. search is not slimmed (the model
mines its results for URLs).
"""

from __future__ import annotations

import asyncio
import functools

from backends import build_scrape, build_search, is_cacheable
from cache import norm_query, norm_scrape, scrape_status
from prompts import build_system_prompt

# ---------------------------------------------------------------------------
# Shared output bound
# ---------------------------------------------------------------------------
# Hard cap on a single tool result before it enters the message history. At ~3.3
# chars/token a 24K-char cap is ~7K tokens; the vLLM teacher accumulates every tool
# result in one prompt.
#
# DO NOT RAISE without re-measuring the teacher end-to-end. A 100K cap was tried
# 2026-07-22 and REVERTED: it did NOT overflow the 131K window (episodes sat at
# ~70K tokens), but feeding Qwen3-235B two ~100K-char scrapes -- typically 400K-char
# junk pages (infinite-scroll / nav / script-as-text) clipped down to 100K -- made
# it return EMPTY output instead of JSON on ~11% of episodes (a lost trace, not a
# clipped menu). A/B on the SAME restaurants over the warm cache was unambiguous: at
# 100K, YORI / The Ritz / Katsuya all produced 0-char finals; at 24K all three
# produced valid JSON, and Katsuya recovered a full 28-item menu the 100K junk had
# buried (big noisy contexts also cut exploration short -- 5 tool calls vs 8). A
# follow-up 50K test 2026-07-23 -- run AFTER base64 data: URIs were stripped, to check
# whether that junk was the cause -- reproduced it anyway: on 15 worst-case big-page
# restaurants over the same warm cache, 50K gave 4/15 EMPTY finals vs 24K's 0/15, and
# 24K produced BETTER menus (Imperial Restaurant: 140 items at 24K vs a give-up at
# 50K). So the trigger is TOTAL accumulated context (a few capped scrapes = 100-150K
# chars), NOT per-scrape junk -- base64 stripping lets the 24K window carry more real
# menu text but buys ZERO headroom for a bigger cap. So 24K is not a context-window
# limit; it is where the teacher stays reliable. A blind char
# cap can still clip an unusually long REAL menu, so a hit is WARNED about (below),
# never silent. Retunable without re-scraping (the cache stores the raw response,
# bounded at cache.MAX_STORED_CHARS = 400K); the STUDENT's SFT cap is tuned
# separately at build time (analyze_tool_chars.py).
MAX_TOOL_CHARS = 24000


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
        # scrape_fn already returns slimmed markdown (build_scrape slims at the
        # source -- see backends._slim_scrape), so here we only apply the char cap.
        return _cap(scrape_fn(url, mode), "scrape_url")

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
    cache (cache.Cache | None): when given, it wraps the BACKEND closures BEFORE
    the MAX_TOOL_CHARS cap. build_scrape slims its own output at the source, so the
    stored response is SLIMMED but uncapped, and the cap stays retunable without
    re-scraping. The cache's miss_policy (live/canned/error) decides what a miss
    does; see src/cache.py.
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
