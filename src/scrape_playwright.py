"""Local, self-hosted scrape backends -- prototype alternatives to Jina.

Three builders here, all returning the same `scrape(url, mode="direct") -> str`
closure shape as backends.build_scrape(), so they drop into build_model_tools /
setup_tools unchanged (select with setup_tools(scrape_backend=...)):

  build_scrape_playwright()  -> "playwright": every call launches a fresh Chromium.
                                 The naive baseline. Its "direct" grabs the page
                                 immediately (no JS wait), so on a client-rendered
                                 SPA it serializes an empty shell -- kept as-is for
                                 A/B contrast.
  build_scrape_hybrid(jina)  -> "hybrid": Jina for "direct", Playwright for
                                 "browser". Still depends on Jina.
  build_scrape_local()       -> "local": FULLY Jina-free. "direct" tries a plain
                                 `requests` fetch first (instant for server-
                                 rendered pages) and escalates to a no-scroll
                                 browser render if that returns a thin CSR shell;
                                 "browser" does the auto-scroll render. Uses a
                                 POOLED browser (launched once per thread, reused)
                                 so the Chromium launch cost amortizes across calls.

The point of the browser path is AUTO-SCROLL. Many restaurant SPAs (Square, Toast,
BentoBox) lazy-load menu categories only as they scroll into view, so a one-shot
render captures only the first screen. The browser path waits for the network to
settle, then scrolls to the bottom until the page stops growing, so the full
lazy-loaded menu is in the DOM before we serialize it. It does NOT beat
bot-protection (DoorDash/Yelp detect headless Chromium too).

Why "local" direct still needs a browser: a bare `requests` fetch of a
client-rendered SPA (e.g. *.square.site) returns ~30 chars of empty shell -- Jina
"direct" gets ~15k because Jina RENDERS even in its quick mode. So requests alone
can't replace Jina-direct; the shell-detection fallback to a no-scroll browser
render is what makes the Jina-free direct path actually work on CSR sites.

SYNC Playwright API: safe from the sync agent loops, the scripts/scrape_ab.py A/B
tool, and the viz server (its /api/extract endpoint is a sync def, so FastAPI runs
it in a threadpool worker with no running event loop). The pool is thread-local
for exactly this reason -- a sync Playwright browser is bound to the thread that
created it, so each worker thread keeps its own and never shares across threads.
"""

from __future__ import annotations

import atexit
import threading

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

# Chromium launch flags. --disable-http2 avoids net::ERR_HTTP2_PROTOCOL_ERROR
# from servers/CDNs whose HTTP/2 negotiation Chromium can't complete (seen on
# e.g. mcdonalds.com); Chromium falls back to HTTP/1.1, which those hosts serve
# fine. --no-sandbox keeps it launchable in minimal/rootless containers.
LAUNCH_ARGS = ["--disable-http2", "--no-sandbox"]

# Navigation + settle timeouts (ms). NAV covers the initial load; NETWORKIDLE is
# how long we wait for XHR/fetch traffic to stop before scrolling (SPAs may never
# fully idle, so a timeout here is non-fatal -- we scroll/serialize anyway).
NAV_TIMEOUT_MS = 30000
NETWORKIDLE_TIMEOUT_MS = 15000

# Plain-requests fetch timeout (seconds) for the "local" direct fast path.
HTTP_TIMEOUT = 30

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
    """Strip non-content tags, then convert the rendered HTML to markdown.

    Drops script/style/noscript/svg first so markdownify doesn't dump JS blobs or
    inline CSS into the model's context; the result mirrors Jina's markdown output.
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
    network wait). wait=scroll=False is the naive "grab immediately" path, which
    yields an empty shell on a CSR SPA.
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


# ---------------------------------------------------------------------------
# Per-call Playwright ("playwright" backend) -- the naive baseline
# ---------------------------------------------------------------------------
def build_scrape_playwright():
    """Build a Playwright scrape that launches a fresh Chromium per call.

    mode="direct": load and serialize immediately (no JS wait) -- an empty shell
    on a CSR SPA, kept for A/B contrast. mode="browser": network-settle +
    auto-scroll. See build_scrape_local for the pooled, requests-fast-path version.
    """

    def scrape(url: str, mode: str = "direct") -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=LAUNCH_ARGS)
            try:
                context = browser.new_context(user_agent=USER_AGENT, viewport=VIEWPORT)
                page = context.new_page()
                html = _render_on_page(
                    page, url, wait=False, scroll=(mode == "browser")
                )
            except PlaywrightError as e:
                return _scrape_error(url, mode, e)
            finally:
                browser.close()
        return _html_to_markdown(html) or "(page returned no content)"

    return scrape


# ---------------------------------------------------------------------------
# Thread-local browser pool (shared by the "local" backend)
# ---------------------------------------------------------------------------
# A sync Playwright browser is bound to the thread that created it, so the pool is
# thread-local: each thread lazily launches its own Chromium on first use and
# reuses it across calls. _POOL_REGISTRY tracks every (playwright, browser) started
# so atexit can best-effort close them (cross-thread close may fail -- the OS reaps
# the child anyway, so failures are swallowed).
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


# ---------------------------------------------------------------------------
# Fully Jina-free ("local" backend): requests fast-path + pooled browser
# ---------------------------------------------------------------------------
def build_scrape_local():
    """Build a Jina-free scrape: requests for "direct", pooled browser for scroll.

    mode="direct": a plain `requests` GET first (instant, free, and complete for
    server-rendered pages). If that returns fewer than DIRECT_MIN_CHARS of markdown
    -- the tell-tale of a client-rendered shell -- escalate to a NO-SCROLL browser
    render (renders the CSR content without paying for the full auto-scroll).
    mode="browser": the pooled auto-scroll render. All failures return a readable
    message (see _scrape_error) instead of raising.
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
                resp = session.get(url, timeout=HTTP_TIMEOUT)
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


def build_scrape_hybrid(scrape_direct):
    """Compose a scrape that routes mode="direct" to Jina and "browser" to Playwright.

    Playwright's own naive "direct" is a browser with no JS-wait, so on a
    client-rendered SPA it serializes an empty shell -- worse than Jina's rendered
    "direct". Meanwhile Playwright's auto-scrolling "browser" captures lazy-loaded
    menus Jina's one-shot render misses. So take the best of each: Jina for the
    cheap direct path, a per-call Playwright for the escalation. `scrape_direct` is
    a Jina scrape(url, mode). (The fully Jina-free counterpart is build_scrape_local.)
    """
    scrape_browser = build_scrape_playwright()

    def scrape(url: str, mode: str = "direct") -> str:
        if mode == "browser":
            return scrape_browser(url, mode="browser")
        return scrape_direct(url, mode="direct")

    return scrape


if __name__ == "__main__":
    # Smoke test: scrape a URL (default a Square restaurant site) with the local
    # backend and print how much markdown came back. Args: [url] [mode].
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "https://kashishkirkland.square.site/"
    mode = sys.argv[2] if len(sys.argv) > 2 else "browser"
    out = build_scrape_local()(target, mode=mode)
    print(f"{target} (mode={mode}): {len(out)} chars")
    print(out[:1000])
