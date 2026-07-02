"""Search + scrape backends behind the model-facing web_search/scrape_url tools.

Finalized providers: **Brave for search, a local headless Chromium for scrape.**
The model only ever sees two generically-named tools (web_search, scrape_url) with
fixed docstrings (see build_model_tools in tools.py); this module holds the
implementations behind them, so vendor/engine names stay out of the SFT/GRPO
training data the tool calls get baked into later.

  role     transport                                   key
  ------   -----------------------------------------   -----------
  search   Brave REST (/res/v1/web/search)             BRAVE_API_KEY
  scrape   local: requests (direct) + Playwright        none
           (pooled Chromium, auto-scroll) for browser

  build_search() -> search(query: str)              -> str   formatted result list (see _format_results)
  build_scrape() -> scrape(url: str, mode="direct") -> str   page markdown

Scrape modes:
  "direct"  a plain `requests` GET (no browser/JS) -- instant and complete for
            server-rendered pages. If it returns a thin client-rendered shell
            (< DIRECT_MIN_CHARS), it escalates to a NO-SCROLL browser render so
            CSR sites (Square/Toast) still work without paying for a full scroll.
  "browser" a headless-Chromium render that waits for the network to settle then
            AUTO-SCROLLS to the bottom, so scroll-lazy-loaded menus render fully.

The browser path uses a THREAD-LOCAL pooled Chromium (launched once per thread,
reused) -- a sync Playwright browser is bound to its creating thread, so the pool
is per-thread; this is why it is safe from the sync agent loops and from the viz
server's sync /api/extract endpoint (FastAPI runs it in a threadpool worker with
no event loop). It does NOT beat bot-protection (DoorDash/Yelp detect headless
Chromium) or recover *virtualized* lists (recycled DOM nodes). Needs
`playwright install chromium` + its system libs.

The model-facing wrappers in tools.py apply the MAX_TOOL_CHARS cap, so the
functions here return un-capped strings.
"""

from __future__ import annotations

import atexit
import os
import threading

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

# Reduce-at-the-source: cap how many results a search returns so big pages don't
# balloon the model's context (and the per-turn prefill that re-encodes it).
SEARCH_RESULT_LIMIT = 3

# Network timeout for the Brave search REST call (seconds).
HTTP_TIMEOUT = 60

# Which env var holds the search API key. Scrape needs no key (it runs locally).
SEARCH_ENV = "BRAVE_API_KEY"


def has_search_key() -> bool:
    """True if the search (Brave) API key is present in the environment."""
    return bool(os.environ.get(SEARCH_ENV))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _format_results(results: list[dict]) -> str:
    """Render a normalized [{title, url, description}, ...] list as compact text."""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        url = r.get("url") or ""
        desc = r.get("description") or ""
        lines.append(f"[{i}] {title}\n    {url}\n    {desc}".rstrip())
    return "\n".join(lines) if lines else "(no search results)"


def _require_key(env_var: str, provider: str) -> str:
    """Fetch a provider's API key from the env or exit with an actionable message."""
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(
            f"The {provider} backend requires {env_var} in the environment (or the "
            f"repo-root .env). Pass --offline for the local stub."
        )
    return key


# ---------------------------------------------------------------------------
# Brave -- search (REST; X-Subscription-Token auth)
# ---------------------------------------------------------------------------
def build_search():
    """Build the Brave-backed web_search function."""
    api_key = _require_key(SEARCH_ENV, "brave")
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}

    def search(query: str) -> str:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params={"q": query, "count": SEARCH_RESULT_LIMIT},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # `web` can be absent when there are no web results -- guard it.
        results = (data.get("web") or {}).get("results", [])
        return _format_results(
            [
                {"title": r.get("title"), "url": r.get("url"), "description": r.get("description")}
                for r in results
            ]
        )

    return search


# ---------------------------------------------------------------------------
# Local scrape -- requests fast-path + pooled headless Chromium
# ---------------------------------------------------------------------------
# Chromium launch flags. --disable-http2 avoids net::ERR_HTTP2_PROTOCOL_ERROR from
# servers/CDNs whose HTTP/2 negotiation Chromium can't complete (seen on e.g.
# mcdonalds.com); Chromium falls back to HTTP/1.1, which those hosts serve fine.
# --no-sandbox keeps it launchable in minimal/rootless containers.
LAUNCH_ARGS = ["--disable-http2", "--no-sandbox"]

# Navigation + settle timeouts (ms). NAV covers the initial load; NETWORKIDLE is
# how long we wait for XHR/fetch traffic to stop before scrolling (SPAs may never
# fully idle, so a timeout here is non-fatal -- we scroll/serialize anyway).
NAV_TIMEOUT_MS = 30000
NETWORKIDLE_TIMEOUT_MS = 15000

# Plain-requests fetch timeout (seconds) for the "direct" fast path.
SCRAPE_HTTP_TIMEOUT = 30

# Below this many markdown chars, a `requests` "direct" fetch is treated as an
# empty client-rendered shell and re-fetched with a (no-scroll) browser render.
# A bare CSR shell is tens of chars; any real server-rendered page clears this.
DIRECT_MIN_CHARS = 600

# Auto-scroll loop: pause after each scroll to let lazy content load, and cap the
# rounds so a true infinite-scroll feed can't loop forever. Stop early once the
# page height is stable across two consecutive rounds.
SCROLL_PAUSE_MS = 600
SCROLL_MAX_ROUNDS = 40
SCROLL_STABLE_ROUNDS = 2

# A desktop UA + viewport: some SPAs serve a stripped mobile/no-JS shell to
# unknown clients. This does not defeat real bot-protection, just avoids the
# trivial "you look like a bot" downgrades.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)
VIEWPORT = {"width": 1280, "height": 900}


def _auto_scroll(page) -> None:
    """Scroll to the bottom until the page height stops growing.

    Triggers scroll-driven lazy loading (Square/Toast render menu categories as
    they enter the viewport). Stops after SCROLL_STABLE_ROUNDS rounds with no
    height increase, or SCROLL_MAX_ROUNDS total.

    Note: this does NOT recover items from a *virtualized* list (react-window and
    the like recycle DOM nodes, so scrolling down drops the rows above) -- that's
    the DoorDash-style failure mode auto-scroll can't fix.
    """
    last_height = 0
    stable = 0
    for _ in range(SCROLL_MAX_ROUNDS):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(SCROLL_PAUSE_MS)
        height = page.evaluate("document.body.scrollHeight")
        if height <= last_height:
            stable += 1
            if stable >= SCROLL_STABLE_ROUNDS:
                break
        else:
            stable = 0
        last_height = height


def _html_to_markdown(html: str) -> str:
    """Strip non-content tags, then convert the (rendered) HTML to markdown.

    Drops script/style/noscript/svg first so markdownify doesn't dump JS blobs or
    inline CSS into the model's context.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()
    return markdownify(str(soup)).strip()


def _render_on_page(page, url: str, *, wait: bool, scroll: bool) -> str:
    """Navigate `page` to `url` and return its serialized HTML.

    wait: after DOMContentLoaded, also wait for the network to go idle so a
    client-rendered page has a chance to paint (a timeout here is non-fatal).
    scroll: additionally auto-scroll to force lazy-loaded content in (implies the
    network wait).
    """
    page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    if wait or scroll:
        try:
            page.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT_MS)
        except PlaywrightTimeout:
            # SPAs that keep polling never reach networkidle; the DOM is usually
            # ready anyway, so serialize/scroll rather than give up.
            pass
    if scroll:
        _auto_scroll(page)
    return page.content()


def _scrape_error(url: str, mode: str, e: Exception) -> str:
    """Format a navigation/render failure as a readable, model-recoverable string.

    The agent loops call the scrape tool without a try/except, so a raised error
    would kill the whole episode. Returning a message instead lets the model try
    another URL or mode, exactly as it would on an empty page. Take only the first
    line -- Playwright appends a multi-line call log.
    """
    detail = str(e).splitlines()[0] if str(e) else type(e).__name__
    return f"(scrape failed for {url} in {mode!r} mode: {detail})"


# Thread-local browser pool: a sync Playwright browser is bound to the thread that
# created it, so each thread lazily launches its own Chromium and reuses it.
# _POOL_REGISTRY tracks every (playwright, browser) started so atexit can
# best-effort close them (cross-thread close may fail -- the OS reaps the child
# anyway, so failures are swallowed).
_POOL = threading.local()
_POOL_REGISTRY: list = []
_POOL_LOCK = threading.Lock()


def _pooled_browser():
    """Return this thread's Chromium, launching (and registering) it on first use."""
    browser = getattr(_POOL, "browser", None)
    if browser is not None and browser.is_connected():
        return browser
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
    _POOL.pw = pw
    _POOL.browser = browser
    with _POOL_LOCK:
        _POOL_REGISTRY.append((pw, browser))
    return browser


def _render_pooled(url: str, *, wait: bool, scroll: bool) -> str:
    """Render `url` on a fresh context of this thread's pooled browser.

    A new context per call keeps state isolated (cookies/storage) while reusing
    the expensive browser process; the context is always closed, the browser kept.
    """
    browser = _pooled_browser()
    context = browser.new_context(user_agent=USER_AGENT, viewport=VIEWPORT)
    try:
        page = context.new_page()
        return _render_on_page(page, url, wait=wait, scroll=scroll)
    finally:
        context.close()


def close_pool() -> None:
    """Best-effort close of every pooled browser (registered atexit)."""
    with _POOL_LOCK:
        for pw, browser in _POOL_REGISTRY:
            try:
                browser.close()
            except Exception:  # noqa: BLE001 - cross-thread close can raise; ignore
                pass
            try:
                pw.stop()
            except Exception:  # noqa: BLE001
                pass
        _POOL_REGISTRY.clear()


atexit.register(close_pool)


def build_scrape():
    """Build the local scrape function: requests for "direct", pooled browser scroll.

    mode="direct": a plain `requests` GET first (instant, free, complete for
    server-rendered pages). If that returns fewer than DIRECT_MIN_CHARS of markdown
    -- the tell-tale of a client-rendered shell -- escalate to a NO-SCROLL browser
    render. mode="browser": the pooled auto-scroll render. All failures return a
    readable message (see _scrape_error) instead of raising.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    def scrape(url: str, mode: str = "direct") -> str:
        try:
            if mode == "browser":
                html = _render_pooled(url, wait=False, scroll=True)
                return _html_to_markdown(html) or "(page returned no content)"

            # direct: cheap requests first ...
            md = ""
            try:
                resp = session.get(url, timeout=SCRAPE_HTTP_TIMEOUT)
                resp.raise_for_status()
                md = _html_to_markdown(resp.text)
            except requests.RequestException:
                md = ""  # fall through to the browser render below
            if len(md) >= DIRECT_MIN_CHARS:
                return md
            # ... escalate a thin/failed fetch to a no-scroll browser render.
            html = _render_pooled(url, wait=True, scroll=False)
            return _html_to_markdown(html) or md or "(page returned no content)"
        except PlaywrightError as e:
            return _scrape_error(url, mode, e)

    return scrape
