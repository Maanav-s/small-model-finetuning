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
import re
import threading
import time

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
            f"repo-root .env)."
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


# Inline data: URIs -- base64 images/fonts, mostly `<img src="data:...;base64,...">`
# -- are the DOMINANT scrape bulk. Measured 2026-07-22: a single inline image was
# 397,471 chars (99% of a 400K page); another page repeated the same base64 line 8x.
# The model can't see images, so this is pure noise -- and worse, MAX_STORED_CHARS
# can clip a data URI mid-string, removing the closing ")" that tools._slim_scrape's
# markdown-image regex needs, so the blob survives read-time slimming too. `[^\s)]*`
# matches the whole base64 token (no whitespace/paren inside it) AND runs to
# end-of-string when a clip took the closing paren. Scrubbed at the SOURCE here
# (shrinks the cached row) and again in tools._slim_scrape (cleans rows already
# cached before this landed).
DATA_URI_RE = re.compile(r"data:[^\s)]*", re.IGNORECASE)


def _html_to_markdown(html: str) -> str:
    """Strip non-content tags + inline data: URIs, then convert HTML to markdown.

    Drops script/style/noscript/svg first so markdownify doesn't dump JS blobs or
    inline CSS into the model's context, then scrubs base64 data: URIs (see
    DATA_URI_RE) -- the single largest source of scrape bulk.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()
    return DATA_URI_RE.sub("", markdownify(str(soup)).strip())


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


# Playwright reports a browser it cannot USE as one that does not EXIST. Its gate
# (registry executablePathOrDie -> canAccessFile) is:
#
#     function canAccessFile(file) {
#       try { fs.accessSync(file); return true; } catch (e) { return false; }
#     }
#
# so `Executable doesn't exist at <path>` means **accessSync THREW**, not that the
# path is absent -- and on Windows that includes a transient sharing violation
# while an AV scanner reads the 203 MB headless-shell binary. Verified 2026-07-20:
# a run failed every launch for 90 s with that message while the file was present,
# 203,034,112 bytes, readable, and launching normally 3 minutes later.
#
# The two causes need OPPOSITE handling -- a missing install is permanent (retry
# is wasted), an unreadable one is transient (retry is the entire fix) -- and the
# message alone cannot tell them apart. The path inside it can.
_EXE_MISSING_RE = re.compile(r"Executable doesn't exist at (.+)")


def _executable_present(message: str) -> bool | None:
    """For an 'Executable doesn't exist' error: is the binary actually on disk?

    Returns True (present -> the check failed transiently), False (genuinely
    missing -> the install is broken), or None if this isn't that error at all.

    os.path.exists() is NOT sufficient on its own: it swallows OSError exactly the
    way canAccessFile swallows accessSync, so a momentarily locked binary reads as
    absent to BOTH and a naive exists() turns a transient into a permanent verdict.
    Measured 2026-07-20: a preflight reported "Chromium is not installed" for a
    203,034,112-byte binary that stat'd, opened and launched fine seconds later.

    So a non-stat'ing exe is only believed absent when its INSTALL DIRECTORY is
    gone too -- Playwright drops an INSTALLATION_COMPLETE marker beside the browser
    when the download finishes, and an intact tree means the file is there and
    merely unreadable this instant.
    """
    match = _EXE_MISSING_RE.search(message or "")
    if not match:
        return None
    exe = match.group(1).strip()
    if os.path.exists(exe):
        return True
    browser_dir = os.path.dirname(exe)                 # .../chrome-headless-shell-win64
    install_root = os.path.dirname(browser_dir)        # .../chromium_headless_shell-1228
    if os.path.isdir(browser_dir) or os.path.exists(
        os.path.join(install_root, "INSTALLATION_COMPLETE")
    ):
        return True  # install intact -> locked, not missing
    return False


INSTALL_HINT = "Chromium is not installed: run `uv run playwright install chromium`"

# The install is fine and the binary is merely unreadable right now -- reinstalling
# is a 203 MB no-op. On Windows this is typically an on-access AV/indexer scan of a
# 203 MB executable; excluding the browser dir is the durable fix.
LOCK_HINT = (
    "the binary IS installed but could not be read just now (a scanner or indexer "
    "holding it) -- do NOT reinstall. Retry; if it recurs, exclude the Playwright "
    "browser directory from your antivirus."
)


def _scrape_error(url: str, mode: str, e: Exception) -> str:
    """Format a navigation/render failure as a readable, model-recoverable string.

    The agent loops call the scrape tool without a try/except, so a raised error
    would kill the whole episode. Returning a message instead lets the model try
    another URL or mode, exactly as it would on an empty page. Take only the first
    line -- Playwright appends a multi-line call log.
    """
    message = str(e)
    detail = message.splitlines()[0] if message else type(e).__name__
    # ... except that for a missing install, the line we'd drop is the ONLY
    # actionable one (Playwright puts "run playwright install" in an ASCII box
    # below the first line). Keeping [0] alone is what made a broken install read
    # as an inexplicable transient for two days. Re-add it, still single-line.
    if _executable_present(message) is False:
        detail = f"{detail} -- {INSTALL_HINT}"
    return f"(scrape failed for {url} in {mode!r} mode: {detail})"


# Markers of an INFRASTRUCTURE failure: the local browser stack broke, so the URL
# was never actually fetched. Distinct from a SITE failure (nav timeout, bad cert,
# empty body), where we did ask and the site refused. The difference is not
# cosmetic -- a site failure is a fact about the web and is the correct, permanent
# outcome for that URL; an infra failure is a fact about THIS machine, and any row
# it produces is a fabrication. Anything that bulk-populates the cache must count
# them apart: on 2026-07-20 a warm ran six hours at a 49% error rate that was 100%
# infra, wrote 2418 junk rows, and looked healthy the whole way because one
# undifferentiated counter made it indistinguishable from aggregators being
# aggregators. (Post-mortem: 26 of those were a launch failure and 2413 were the
# thread-poisoning CASCADE it triggered -- see _pooled_browser.)
_INFRA_FAILURE_MARKERS = (
    "Executable doesn't exist",
    "asyncio loop",
    "BrowserType.launch",
    "Target page, context or browser has been closed",
)


def is_infra_failure(response: str) -> bool:
    """True if a scrape failure sentinel reflects a broken local browser stack
    rather than the site refusing us. False for non-sentinel (successful) responses."""
    return any(marker in (response or "") for marker in _INFRA_FAILURE_MARKERS)


def is_cacheable(response: str) -> bool:
    """False for responses that are artifacts of THIS machine, not answers from the web.

    Pass as `store_if=is_cacheable` when wrapping scrape in a Cache. An infra
    failure means the URL was never fetched, so storing one records a finding that
    was never made -- and under miss_policy="canned" it would later be REPLAYED as
    though the page had really answered that way. Not storing it costs nothing: the
    caller still gets the sentinel (so it can count/abort), and the next pass simply
    re-fetches a key that was never written.
    """
    return not is_infra_failure(response)


# Thread-local browser pool: a sync Playwright browser is bound to the thread that
# created it, so each thread lazily launches its own Chromium and reuses it.
# _POOL_REGISTRY tracks every (playwright, browser) started so atexit can
# best-effort close them (cross-thread close may fail -- the OS reaps the child
# anyway, so failures are swallowed).
_POOL = threading.local()
_POOL_REGISTRY: list = []
_POOL_LOCK = threading.Lock()

# Chromium launch is retried, because "Executable doesn't exist" is also what
# Playwright says when the binary is present but momentarily unreadable (see
# _executable_present). Retries apply ONLY to that transient case -- a genuinely
# absent binary fails immediately, since three attempts and a backoff cannot
# install it and only multiply the wasted time per URL.
BROWSER_LAUNCH_ATTEMPTS = 3
BROWSER_LAUNCH_BACKOFF_S = 1.0


def _drop_thread_browser() -> None:
    """Tear down and forget this thread's pooled browser (used when it has died).

    A disconnected browser still has its playwright instance started; relaunching
    without stopping that first would stack a second event loop on this thread.
    """
    pw = getattr(_POOL, "pw", None)
    browser = getattr(_POOL, "browser", None)
    _POOL.pw = None
    _POOL.browser = None
    if browser is not None:
        try:
            browser.close()
        except Exception:  # noqa: BLE001 - already dead; nothing to salvage
            pass
    if pw is not None:
        try:
            pw.stop()
        except Exception:  # noqa: BLE001
            pass
        with _POOL_LOCK:
            _POOL_REGISTRY[:] = [(p, b) for p, b in _POOL_REGISTRY if p is not pw]


def _pooled_browser():
    """Return this thread's Chromium, launching (and registering) it on first use."""
    browser = getattr(_POOL, "browser", None)
    if browser is not None and browser.is_connected():
        return browser
    _drop_thread_browser()  # a dead browser must be stopped before relaunching
    for attempt in range(BROWSER_LAUNCH_ATTEMPTS):
        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
        except BaseException as exc:
            # launch() failed AFTER start() succeeded, so pw's event loop is already
            # running in THIS thread. It is in neither _POOL nor _POOL_REGISTRY, so
            # nothing else can ever stop it -- and every later sync_playwright().start()
            # on this thread would raise "Sync API inside the asyncio loop". Without
            # this stop(), ONE transient launch failure silently disables the browser
            # path for the whole run: measured 2026-07-20, two worker threads each hit
            # one launch failure and every subsequent scrape in that run failed.
            try:
                pw.stop()
            except Exception:  # noqa: BLE001 - best effort; re-raising the cause matters more
                pass
            # Retry only what retrying can fix. `_executable_present(...) is False`
            # means the binary really is absent -- no amount of backoff installs it,
            # and retrying just triples the time each URL takes to fail. Anything
            # else (including a present-but-unreadable binary) is worth another go.
            permanent = _executable_present(str(exc)) is False
            if (not permanent and isinstance(exc, PlaywrightError)
                    and attempt < BROWSER_LAUNCH_ATTEMPTS - 1):
                time.sleep(BROWSER_LAUNCH_BACKOFF_S * (attempt + 1))
                continue
            raise
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


def preflight_browser() -> str | None:
    """Launch this thread's browser now; return None on success or a one-line reason.

    For bulk populators (warm_cache, build_corpus): a browser that cannot launch
    turns EVERY scrape into an infra failure, and learning that on restaurant 1 is
    the difference between exiting cleanly and grinding through a whole selection
    producing nothing. Cheap to call -- the browser it launches is the pooled one
    this thread would have launched on its first scrape anyway.
    """
    try:
        _pooled_browser()
        return None
    except BaseException as exc:  # noqa: BLE001 - reported to the caller, not raised
        message = str(exc)
        detail = message.splitlines()[0] if message else type(exc).__name__
        # Same message, opposite advice -- telling someone to reinstall a browser
        # that IS installed sends them after a 203 MB red herring.
        present = _executable_present(message)
        if present is False:
            detail = f"{detail}\n  {INSTALL_HINT}"
        elif present is True:
            detail = f"{detail}\n  {LOCK_HINT}"
        return detail


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
        # RecursionError: BeautifulSoup/markdownify recurse over the DOM, and a
        # pathologically nested page blows the interpreter limit -- observed
        # killing 10/99 pilot episodes. Same contract as Playwright failures:
        # return the sentinel so the model can try another URL/mode.
        except (PlaywrightError, RecursionError) as e:
            return _scrape_error(url, mode, e)

    return scrape
