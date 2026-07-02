"""Playwright-backed scrape backend -- a prototype alternative to Jina.

Same interface as backends.build_scrape(): build_scrape_playwright() returns a
`scrape(url, mode="direct") -> str` closure, so it drops into build_model_tools /
setup_tools unchanged (select it with setup_tools(scrape_backend="playwright")).

The point of this prototype is the "browser" mode's AUTO-SCROLL. Many restaurant
SPAs (Square, Toast, BentoBox) lazy-load menu categories only as they scroll into
view, so a one-shot render -- Jina's browser engine, or mode="direct" here --
captures only the first screen. This renderer drives a real Chromium: it waits
for the network to settle, then scrolls to the bottom repeatedly until the page
stops growing, so the full lazy-loaded menu is in the DOM before we serialize it.

Trade-offs vs Jina:
  + we control scroll/wait, so scroll-driven lazy lists render fully;
  - we run Chromium locally (heavier; needs `playwright install chromium`);
  - it does NOT beat bot-protection -- DoorDash/Yelp detect headless Chromium too
    (often better than Jina's browser, which they block outright, but not reliably).

This uses the SYNC Playwright API, so it works from the sync agent loops
(gemma/agent.py, claude/claude_agent.py) and the scripts/scrape_ab.py A/B tool,
but NOT from an asyncio context (the viz server) -- that would need async
Playwright. Wiring it into viz is deferred until the A/B says it's worth it.
"""

from __future__ import annotations

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
# fully idle, so a timeout here is non-fatal -- we scroll anyway).
NAV_TIMEOUT_MS = 30000
NETWORKIDLE_TIMEOUT_MS = 15000

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


def build_scrape_playwright():
    """Build a Playwright-backed scrape(url, mode="direct") -> str.

    mode="direct": load the page and serialize immediately (no scroll) -- the
    quick path, comparable to Jina's plain fetch. mode="browser": wait for the
    network to settle, then auto-scroll to force lazy-loaded content in, before
    serializing -- the reason this backend exists.
    """

    def scrape(url: str, mode: str = "direct") -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=LAUNCH_ARGS)
            context = browser.new_context(user_agent=USER_AGENT, viewport=VIEWPORT)
            page = context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                if mode == "browser":
                    try:
                        page.wait_for_load_state(
                            "networkidle", timeout=NETWORKIDLE_TIMEOUT_MS
                        )
                    except PlaywrightTimeout:
                        # SPAs that keep polling never reach networkidle; the DOM
                        # is usually ready anyway, so scroll rather than give up.
                        pass
                    _auto_scroll(page)
                html = page.content()
            except PlaywrightError as e:
                # A navigation/render failure (broken HTTP/2 or TLS, DNS, a nav
                # timeout, a site that drops the connection) must NOT crash the
                # episode -- the agent loops call this tool without a try/except.
                # Return a readable message so the model recovers (try another URL
                # or the other mode), exactly as it would on an empty page. Take
                # only the first line: Playwright appends a multi-line call log.
                detail = str(e).splitlines()[0] if str(e) else type(e).__name__
                return f"(scrape failed for {url} in {mode!r} mode: {detail})"
            finally:
                browser.close()
        return _html_to_markdown(html) or "(page returned no content)"

    return scrape


def build_scrape_hybrid(scrape_direct):
    """Compose a scrape that routes mode="direct" to Jina and "browser" to Playwright.

    This is the config the A/B supports: Playwright's own "direct" is a browser
    with no JS-wait, so on a client-rendered SPA it serializes an empty shell
    (~30 chars) -- worse than Jina's server-HTML fetch. Meanwhile Playwright's
    auto-scrolling "browser" mode captures lazy-loaded menus that Jina's one-shot
    render misses. So take the best of each: Jina for the cheap direct path,
    Playwright for the escalation. `scrape_direct` is a Jina scrape(url, mode).
    """
    scrape_browser = build_scrape_playwright()

    def scrape(url: str, mode: str = "direct") -> str:
        if mode == "browser":
            return scrape_browser(url, mode="browser")
        return scrape_direct(url, mode="direct")

    return scrape


if __name__ == "__main__":
    # Smoke test: render a URL (default a Square restaurant site) in browser mode
    # and print how much markdown came back. Pass a URL as argv[1] to override.
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "https://kashishkirkland.square.site/"
    mode = sys.argv[2] if len(sys.argv) > 2 else "browser"
    out = build_scrape_playwright()(target, mode=mode)
    print(f"{target} (mode={mode}): {len(out)} chars")
    print(out[:1000])
