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

Baked-in protections (every consumer gets these -- they are properties of the
backend, not of any one caller's wiring):
  * SLIM: scrape output is passed through _slim_scrape at the source, so what a
    Cache stores IS what the agent sees (minus the read-time MAX_TOOL_CHARS cap).
  * DEAD ENDS: known bot-walled/login-walled domains (SKIP_DOMAINS), binary file
    extensions (SKIP_EXTENSIONS), and non-HTML Content-Types answer with an
    instant, deterministic sentinel instead of a 30-45s timeout -- see skip_reason.
  * THROTTLE: network fetches to the same host are spaced >= DOMAIN_MIN_INTERVAL_S
    apart, process-wide, so parallel rollouts don't hammer one site from one IP.
  * CIRCUIT BREAKER: INFRA_STREAK_ABORT consecutive infra-failed scrapes raise
    BrowserDeadError, so a run whose local browser died aborts instead of grinding.
  * RENDER WATCHDOG: every browser render runs under a RENDER_WATCHDOG_S hard
    wall-clock cap; a page whose renderer wedges (page.evaluate/content have no
    client-side timeout) gets its driver killed and fails as a site error instead
    of parking the worker thread forever.

The model-facing wrappers in tools.py apply the MAX_TOOL_CHARS cap, so the
functions here return un-capped strings.
"""

from __future__ import annotations

import atexit
import os
import re
import threading
import time
from urllib.parse import urlsplit

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
# can clip a data URI mid-string, removing the closing ")" that _slim_scrape's
# markdown-image regex needs, so the blob survives markdown slimming too. `[^\s)]*`
# matches the whole base64 token (no whitespace/paren inside it) AND runs to
# end-of-string when a clip took the closing paren. Scrubbed in _html_to_markdown
# (shrinks every conversion) and again in _slim_scrape (cleans rows cached before
# this landed, where a clip may have orphaned the blob).
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


# ---------------------------------------------------------------------------
# Scrape-result slimming (token cost) -- baked into build_scrape at the SOURCE, so
# every consumer (setup_tools' cache wrap, warm_cache's cache wrap, an uncached
# agent run) returns -- and therefore STORES -- the same slimmed body. It used to
# live in tools.setup_tools, which left warm_cache caching RAW rows that a later
# cache hit served raw to the agent; putting it here makes that drift impossible.
# clean_cache.py retro-applies the SAME transform to rows cached before this. The
# MAX_TOOL_CHARS *cap* stays a read-time step in tools.py so it remains retunable
# without re-scraping -- only the junk removal is baked in.
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

# A list line that is ONLY a bullet marker (markdownify emits "* " for an empty <li>,
# and image/link removal above can leave a bullet with nothing after it). Measured
# 2026-07-22: one menu page had 918 of these. Pure noise -- drop the whole line.
_MD_EMPTY_BULLET_RE = re.compile(r"(?m)^[ \t]*[-*+][ \t]*$\n?")


def _slim_scrape(md: str) -> str:
    """Drop non-navigable markdown bulk from a scraped page (see block comment).

    Iterated to a FIXED POINT: one stage's removal can expose a match for an earlier
    stage (e.g. dropping an image empties the bullet that held it; dropping a line
    merges two blank runs), so a single pass isn't idempotent -- a few real pages
    needed a second pass. That matters here because clean_cache.py bakes this same
    transform into the stored rows: if `slim(raw)` still had junk `slim` would strip
    on the next read, a cached hit (stored pre-slimmed) and a fresh fetch would
    disagree. Every stage only deletes or shortens, so the length strictly drops on
    any change and the loop terminates (typically after 2 passes: one real, one that
    confirms nothing else matches)."""
    prev = None
    while md != prev:
        prev = md
        # NUL bytes never belong in menu markdown -- they signal a binary blob
        # mangled through the HTML->markdown pipeline (one cached page was a binary
        # file rendered to 1.9M chars with 10.6K NULs). Beyond being junk, an embedded
        # NUL breaks SQLite's length()/substr() (both stop counting at the first NUL),
        # so a NUL-bearing row is invisible to cache.clip_oversized's SQL bound and
        # can't be trimmed until the NULs are gone -- strip them first.
        md = md.replace("\x00", "")
        # Base64 data: URIs (the dominant bulk; see DATA_URI_RE). New scrapes are
        # already scrubbed in _html_to_markdown, but rows CACHED before that landed
        # still carry the blob -- often a data URI clipped mid-string, which the image
        # regex below cannot match -- so strip it here too.
        md = DATA_URI_RE.sub("", md)
        md = _MD_IMAGE_RE.sub("", md)
        md = _MD_EMPTY_LINK_RE.sub("", md)
        md = _MD_DEAD_HREF_RE.sub(lambda m: m.group(1) or m.group(2), md)
        md = _MD_EMPTY_BULLET_RE.sub("", md)  # after image/link removal, which can empty a bullet
        md = re.sub(r"\n{3,}", "\n\n", md)  # collapse runs of blank lines to a single one
    return md


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
    # The driver (node) process died out from under a call -- a real crash is an
    # infra fact. A RENDER-WATCHDOG kill produces this too, but _render_pooled
    # rewrites that case into its own "render watchdog:" error BEFORE
    # classification, so a watchdog kill stays a (cacheable) site fact.
    "Connection closed while reading from the driver",
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


# ---------------------------------------------------------------------------
# Known dead ends -- answered instantly, with no network fetch (skip_reason)
# ---------------------------------------------------------------------------
# Bot-walled aggregators / login-walled socials / SEO-spam menu-mirror farms.
# Matched by registrable suffix (subdomains included). This list used to live only
# in scripts/corpus/warm_cache.py, which meant the warm skipped these but the LIVE
# tool paid a 30-45s bot-wall timeout per visit -- and, because a stored 'error'
# row is a MISS under miss_policy="live" (cache.py), paid it again on every later
# pass of the same URL. Answering here makes the dead end instant, cacheable as a
# permanent negative, and byte-identical between live and canned runs.
SKIP_DOMAINS = frozenset({
    # US delivery apps (headless-Chromium bot-walled)
    "doordash.com", "ubereats.com", "grubhub.com", "seamless.com", "postmates.com",
    # reservation / review aggregators (bot-walled; every render times out to an error)
    "yelp.com", "yelp.ca", "yelp.co.uk", "yelp.com.au",
    "opentable.com", "opentable.co.uk", "opentable.ca", "opentable.com.au",
    # login-walled socials
    "facebook.com", "instagram.com",
    # SEO-spam menu-mirror farms (auto-generated <slug>.<farm> subdomains, never a real menu)
    "restaurants-world.com", "restaurants-world.net",
    "menu-world.com", "menu-res.com", "res-menu.net",
})

# Non-HTML payloads the markdown scrape can't do anything useful with (a PDF
# markdownified as binary once produced a 14M-char row; see also the Content-Type
# guard in build_scrape, which catches the extension-less version of this).
SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")

# These sentinels are MODEL-FACING (they enter SFT/GRPO training data), so they say
# what the model should do next. Contract, relied on by cache.scrape_status:
#   * under cache.MIN_CONTENT_CHARS (200) and matching no failure/infra marker, so
#     they classify 'empty' -- a PERMANENT negative: stored once, a hit under
#     "live" (never re-fetched), replayed verbatim under "canned";
#   * NOT starting with "(scrape failed"/"(page returned no content)"/"(page not
#     available)" (would classify 'error' -> re-fetched every live pass) and
#     containing no _INFRA_FAILURE_MARKERS (would make them uncacheable).
BLOCKED_SITE_RESULT = ("(this site blocks automated access and returns no usable "
                       "content -- try a different search result)")
BINARY_URL_RESULT = ("(this link is a file download, not a readable web page -- "
                     "try a different search result)")


def _non_html_result(content_type: str) -> str:
    """Sentinel for a fetch that answered with a non-HTML payload (PDF, image...)."""
    return (f"(this page returned {content_type[:40]!r}, not readable web content "
            f"-- try a different search result)")


def _readable_content_type(ctype: str) -> bool:
    """Content-Types the HTML->markdown pipeline can do something useful with.

    Empty is allowed (some servers omit the header; the DIRECT_MIN_CHARS shell
    check and markdownify cope). text/plain and JSON are readable as-is."""
    return (not ctype or ctype.startswith("text/")
            or "html" in ctype or "xml" in ctype or "json" in ctype)


def skip_reason(url: str) -> str | None:
    """The instant sentinel for a known dead-end URL, or None to really fetch it.

    Checked by build_scrape before any network work. Exposed so bulk populators
    (warm_cache) can tell "this URL will answer instantly with a sentinel" apart
    from "this URL will cost a real fetch" when planning/counting."""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    if any(host == d or host.endswith("." + d) for d in SKIP_DOMAINS):
        return BLOCKED_SITE_RESULT
    if parts.path.lower().endswith(SKIP_EXTENSIONS):
        return BINARY_URL_RESULT
    return None


# ---------------------------------------------------------------------------
# Per-host politeness throttle (process-wide, across threads)
# ---------------------------------------------------------------------------
# Minimum spacing between NETWORK fetches to the same host. Cache hits never reach
# the backend, so this only paces genuine fetches. warm_cache used to be the only
# throttled caller (its --sleep); live GRPO rollouts run the tools in PARALLEL
# (tools._to_async -> asyncio.gather) from one egress IP, which is exactly the
# per-site rate-limit hazard CLAUDE.md flags -- so the pacing belongs here, where
# every caller inherits it.
DOMAIN_MIN_INTERVAL_S = 1.0
_LAST_FETCH: dict[str, float] = {}
_THROTTLE_LOCK = threading.Lock()


def _throttle(host: str) -> None:
    """Block until DOMAIN_MIN_INTERVAL_S has passed since the last fetch to `host`.

    Loop-and-recheck: several waiters can wake together, but only the one that
    claims the slot under the lock proceeds; the rest wait out a fresh interval."""
    if DOMAIN_MIN_INTERVAL_S <= 0:
        return
    while True:
        with _THROTTLE_LOCK:
            now = time.monotonic()
            ready_at = _LAST_FETCH.get(host, 0.0) + DOMAIN_MIN_INTERVAL_S
            if now >= ready_at:
                _LAST_FETCH[host] = now
                return
            wait = ready_at - now
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Infra circuit breaker (process-wide, across threads)
# ---------------------------------------------------------------------------
# Consecutive scrapes that failed on the LOCAL browser stack. A dead browser turns
# every scrape into an infra sentinel; per-script breakers (warm_cache's
# INFRA_ABORT_CONSECUTIVE, build_corpus's episode counter) can only notice after
# whole restaurants/episodes of work have been wasted. Raising stops the sync
# callers: the exception propagates out of the tool call, fails the episode, and
# trips the script's per-episode breaker. ONE caller needs its own companion:
# TRL's GRPO tool loop CATCHES tool exceptions (grpo_trainer.py ~1527 `except
# Exception`; async gathers with return_exceptions=True) and feeds {"error": ...}
# back as the tool message -- there, this raise surfaces as a SATURATED
# `tools/failure_rate` metric (every call raising instantly), and train_grpo.py's
# ToolFailureAbort callback is what turns that into a stop. Successful browser
# renders and SITE failures (the browser worked; the site refused) reset the
# streak, so only a browser failing every consecutive call can trip it. A browser
# broken FROM THE START is caught cheaper by preflight_browser; this catches one
# that dies MID-RUN.
INFRA_STREAK_ABORT = 15
_infra_streak = 0
_INFRA_LOCK = threading.Lock()


class BrowserDeadError(RuntimeError):
    """INFRA_STREAK_ABORT consecutive scrapes failed on the local browser stack."""


def _note_scrape_outcome(result: str | None) -> None:
    """Feed one scrape outcome to the breaker (None = a successful browser render).

    Raises BrowserDeadError once the streak reaches INFRA_STREAK_ABORT; the streak
    is left at the threshold, so every subsequent infra failure keeps raising until
    a render succeeds (i.e. the run stays dead until the browser recovers)."""
    global _infra_streak
    with _INFRA_LOCK:
        if result is None or not is_infra_failure(result):
            _infra_streak = 0
            return
        _infra_streak += 1
        streak = _infra_streak
    if streak >= INFRA_STREAK_ABORT:
        raise BrowserDeadError(
            f"{streak} consecutive scrape calls failed on the LOCAL browser stack "
            f"(latest: {result}) -- the browser is dead, not the sites; aborting "
            f"instead of grinding out infra sentinels"
        )


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

# Hard wall-clock cap on ONE render. goto/networkidle carry their own timeouts,
# but page.evaluate (the auto-scroll), page.content(), and context open/close are
# PROTOCOL calls with NO client-side timeout: a page whose renderer goes
# unresponsive (wedged main thread) never answers, the call blocks forever, and
# the worker thread is starved for the rest of the run. Measured 2026-07-25: a
# grpo warm sat 30+ minutes with all 3 workers parked in exactly this state --
# idle CPU, no sockets, no timeout ever coming. The budgeted worst LEGIT render
# is ~100s (NAV 30 + NETWORKIDLE 15 + 40 scroll rounds x 0.6 + big-DOM
# serialization), so 180s only ever fires on a genuinely wedged page.
RENDER_WATCHDOG_S = 180


def _driver_proc(pw):
    """The node driver subprocess behind a sync_playwright instance, or None.

    Private Playwright internals (verified against the pinned version), guarded so
    an upstream refactor degrades to 'no watchdog' instead of an AttributeError.
    The watchdog needs an OS-level kill handle precisely because the wedge is
    protocol-level: no Playwright API call can time it out, and the sync API is
    greenlet-bound to its thread, so another thread cannot close the browser
    politely. Killing the DRIVER tears down the pipe, which makes every pending
    call on this thread raise immediately -- and each thread has its own driver,
    so only the wedged worker is affected.
    """
    try:
        return pw._impl_obj._connection._transport._proc
    except AttributeError:
        return None


def _drop_thread_browser() -> None:
    """Tear down and forget this thread's pooled browser (used when it has died).

    A disconnected browser still has its playwright instance started; relaunching
    without stopping that first would stack a second event loop on this thread.
    """
    pw = getattr(_POOL, "pw", None)
    browser = getattr(_POOL, "browser", None)
    _POOL.pw = None
    _POOL.browser = None
    _POOL.driver_proc = None
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


def _forget_thread_browser() -> None:
    """Drop this thread's pooled browser WITHOUT talking to it (watchdog path).

    _drop_thread_browser's close()/stop() are PROTOCOL calls. Against a killed
    driver they do not fail -- they HANG, spinning in Playwright's sync wrapper
    (`while not task.done(): fiber.switch()`) on a reply that can never arrive.
    That spin is pure greenlet work, so it pins a core AND holds the GIL, which
    starves the main thread and makes the process unkillable by Ctrl+C. Measured
    2026-08-05: a warm froze exactly this way with all 3 workers inside
    new_context()/close() and every driver already dead.

    A killed driver needs no cleanup anyway -- the OS reaped it and its Chromium
    child with it -- so the correct move is to forget the references and
    deregister them (so close_pool() at exit doesn't try to close a zombie
    either). The next _pooled_browser() then sees browser=None and launches fresh.
    """
    pw = getattr(_POOL, "pw", None)
    _POOL.pw = None
    _POOL.browser = None
    _POOL.driver_proc = None
    if pw is not None:
        with _POOL_LOCK:
            _POOL_REGISTRY[:] = [(p, b) for p, b in _POOL_REGISTRY if p is not pw]


def _pooled_browser():
    """Return this thread's Chromium, launching (and registering) it on first use.

    NOTE: `is_connected()` is `return self._is_connected` -- a CACHED flag that
    Playwright only clears when it receives a close EVENT over the pipe. A driver
    we force-killed sends no such event, so this check cannot be relied on to
    notice one; the watchdog path must clear the pool itself
    (_forget_thread_browser). Reusing a zombie browser is what froze the
    2026-08-05 warm.
    """
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
        _POOL.driver_proc = _driver_proc(pw)  # the render watchdog's kill handle
        with _POOL_LOCK:
            _POOL_REGISTRY.append((pw, browser))
        return browser


def _render_pooled(url: str, *, wait: bool, scroll: bool) -> str:
    """Render `url` on a fresh context of this thread's pooled browser, under a
    hard RENDER_WATCHDOG_S wall-clock cap.

    A new context per call keeps state isolated (cookies/storage) while reusing
    the expensive browser process; the context is always closed, the browser kept.

    When the watchdog fires it kills this thread's driver (see _driver_proc for
    why that is the only reliable lever). Everything after that point must assume
    the driver is GONE, because a protocol call against a dead driver hangs rather
    than raising (see _forget_thread_browser): the context is NOT closed, the
    pooled browser is FORGOTTEN so the next call relaunches, and the failure is
    rewritten to a "render watchdog:" site error.
    """
    browser = _pooled_browser()
    proc = getattr(_POOL, "driver_proc", None)
    fired = threading.Event()

    def _kill_driver():
        fired.set()
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - already-dead driver; the raise below still happens
            pass

    timer = None
    if proc is not None:
        timer = threading.Timer(RENDER_WATCHDOG_S, _kill_driver)
        timer.daemon = True
        timer.start()
    try:
        context = browser.new_context(user_agent=USER_AGENT, viewport=VIEWPORT)
        try:
            page = context.new_page()
            return _render_on_page(page, url, wait=wait, scroll=scroll)
        finally:
            # Skip the close entirely once the watchdog has fired: close() is an
            # untimed protocol call, so against the just-killed driver it would
            # hang forever -- the precise freeze this watchdog exists to prevent.
            # Nothing leaks: the driver and its Chromium are already gone.
            # (Tiny race: the timer can fire between this check and the call. The
            # window is microseconds at the end of a 180s deadline, and the
            # outcome is only the pre-existing hang, not a new failure mode.)
            if not fired.is_set():
                context.close()
    except Exception as e:
        if fired.is_set():
            # The page, not our stack, is broken: a wedged renderer is a fact
            # about that URL. Rewrite the kill fallout ("Connection closed while
            # reading from the driver" -- an INFRA marker when it happens on its
            # own) into a site-class failure so it caches as a normal 'error' row
            # and does not feed the infra circuit breaker.
            raise PlaywrightError(
                f"render watchdog: the page did not respond within "
                f"{RENDER_WATCHDOG_S}s (wedged renderer); the browser was recycled"
            ) from e
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if fired.is_set():
            # Unconditional, not just on the error path: if the render happened to
            # finish in the instant the watchdog fired, the driver is dead all the
            # same, and leaving that browser in the pool is what hangs the NEXT
            # call. is_connected() cannot catch it (cached flag, no close event),
            # so clearing the pool here is the only thing standing between a
            # watchdog kill and a frozen worker.
            _forget_thread_browser()


def _render_tracked(url: str, host: str, *, wait: bool, scroll: bool) -> str:
    """A throttled pooled render that also feeds the infra circuit breaker.

    Success means the browser stack works -> reset the streak. A failure raises
    through to build_scrape's handler, which classifies it (infra vs site) and
    feeds THAT to the breaker -- so the reset only ever happens on real renders."""
    _throttle(host)
    html = _render_pooled(url, wait=wait, scroll=scroll)
    _note_scrape_outcome(None)
    return html


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
    readable message (see _scrape_error) instead of raising -- except a dead local
    browser, which raises BrowserDeadError once the infra streak trips (a broken
    machine must abort the run, not fail every URL in it).

    Baked in on every path: known dead ends answer instantly (skip_reason), a
    non-HTML Content-Type answers with a sentinel instead of markdownified binary,
    fetches to one host are spaced by _throttle, and successful output is slimmed
    (_slim_scrape) at the source -- so a Cache wrapping this closure stores exactly
    what the agent sees (minus the read-time MAX_TOOL_CHARS cap).
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    def fetch(url: str, mode: str, host: str) -> str:
        if mode == "browser":
            html = _render_tracked(url, host, wait=False, scroll=True)
            return _html_to_markdown(html) or "(page returned no content)"

        # direct: cheap requests first ...
        md = ""
        try:
            _throttle(host)
            resp = session.get(url, timeout=SCRAPE_HTTP_TIMEOUT)
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            if not _readable_content_type(ctype):
                # A binary payload (PDF/image on an extension-less URL) markdownifies
                # to enormous junk -- one PDF made a 14M-char row -- and a browser
                # render of it can do no better, so don't escalate: answer with the
                # permanent sentinel.
                return _non_html_result(ctype)
            md = _html_to_markdown(resp.text)
        except requests.RequestException:
            md = ""  # fall through to the browser render below
        if len(md) >= DIRECT_MIN_CHARS:
            return md
        # ... escalate a thin/failed fetch to a no-scroll browser render.
        html = _render_tracked(url, host, wait=True, scroll=False)
        return _html_to_markdown(html) or md or "(page returned no content)"

    def scrape(url: str, mode: str = "direct") -> str:
        reason = skip_reason(url)
        if reason is not None:
            return reason  # instant, deterministic, no network
        host = urlsplit(url).netloc.lower()
        try:
            return _slim_scrape(fetch(url, mode, host))
        # RecursionError: BeautifulSoup/markdownify recurse over the DOM, and a
        # pathologically nested page blows the interpreter limit -- observed
        # killing 10/99 pilot episodes. Same contract as Playwright failures:
        # return the sentinel so the model can try another URL/mode.
        except (PlaywrightError, RecursionError) as e:
            sentinel = _scrape_error(url, mode, e)
            _note_scrape_outcome(sentinel)  # raises BrowserDeadError past the streak
            return sentinel

    return scrape
