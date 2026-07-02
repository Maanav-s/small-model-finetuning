"""Pluggable search/scrape backends behind the model-facing web_search/scrape_url.

The model only ever sees TWO generically-named tools (web_search, scrape_url) with
fixed docstrings -- see build_model_tools in tools.py. This module holds the actual
provider implementations behind them, so we can A/B each provider on the *same*
task/prompt/schema without the model ever seeing a vendor name (important: the tool
names get baked into the SFT/GRPO training data later).

Each provider is a builder that reads its own API key from the environment and
returns a `(search_fn, scrape_fn)` pair; either may be None when a provider doesn't
offer that capability:

  backend      search  scrape   transport                      env var
  ----------   ------  ------   ----------------------------   -------------------
  firecrawl      x       x      firecrawl-py SDK               FIRECRAWL_API_KEY
  tavily         x       x      REST (/search, /extract)       TAVILY_API_KEY
  brave          x       -      REST (/res/v1/web/search)      BRAVE_API_KEY
  jina           x       x      REST (s.jina.ai, r.jina.ai)    JINA_API_KEY
  browserless    -       x      REST (/content -> HTML->md)    BROWSERLESS_API_KEY

  search_fn(query: str) -> str   formatted result list (see _format_results)
  scrape_fn(url: str)   -> str   page contents as markdown

The model-facing wrappers in tools.py apply the MAX_TOOL_CHARS cap, so the functions
here return un-capped strings. setup_tools(search_backend=..., scrape_backend=...)
picks one of each; build_backend() below is the single dispatch point.
"""

from __future__ import annotations

import os

import requests

# Reduce-at-the-source: cap how many results a search returns so big pages don't
# balloon the model's context (and the per-turn prefill that re-encodes it). Shared
# by every search backend via its provider-specific "limit"/"count"/"num" arg.
SEARCH_RESULT_LIMIT = 3

# Network timeout for the REST backends (seconds). Browser-rendered scrapes
# (Browserless) and reader scrapes (Jina) can be slow, so keep this generous.
HTTP_TIMEOUT = 60

# Default provider for both capabilities (the historical Firecrawl default).
DEFAULT_BACKEND = "firecrawl"

# Single source of truth: which env var holds each backend's API key. Used by
# _require_key (below) and by callers that want to report availability without
# trying to build a backend (e.g. the viz server's /api/backends).
BACKEND_ENV = {
    "firecrawl": "FIRECRAWL_API_KEY",
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_API_KEY",
    "jina": "JINA_API_KEY",
    "browserless": "BROWSERLESS_API_KEY",
}


def backend_has_key(name: str) -> bool:
    """True if the backend's API key is present in the environment."""
    return bool(os.environ.get(BACKEND_ENV.get(name, "")))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _format_results(results: list[dict]) -> str:
    """Render a normalized [{title, url, description}, ...] list as compact text.

    One shared format across every search backend so the model sees identical
    search output regardless of which provider produced it.
    """
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        url = r.get("url") or ""
        desc = r.get("description") or ""
        lines.append(f"[{i}] {title}\n    {url}\n    {desc}".rstrip())
    return "\n".join(lines) if lines else "(no search results)"


def _require_key(backend: str) -> str:
    """Fetch a backend's API key from the env or exit with an actionable message."""
    env_var = BACKEND_ENV[backend]
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(
            f"The {backend!r} backend requires {env_var} in the environment (or the "
            f"repo-root .env). Pass --offline for the local stub, or pick a backend "
            f"whose key you have set."
        )
    return key


def _bearer(key: str) -> dict:
    """Standard JSON + Bearer-auth headers (Tavily, Jina)."""
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    resp = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _get_json(url: str, headers: dict, params: dict) -> dict:
    resp = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Firecrawl -- search + scrape (firecrawl-py SDK)
# ---------------------------------------------------------------------------
def _build_firecrawl():
    api_key = _require_key("firecrawl")
    # Imported lazily so picking another backend doesn't pay the SDK import cost.
    from firecrawl import Firecrawl

    client = Firecrawl(api_key=api_key)

    def search(query: str) -> str:
        data = client.search(query, limit=SEARCH_RESULT_LIMIT)
        # `.web` entries are SearchResultWeb (url/title/description); getattr guards
        # the Document shape `search` can also return.
        results = getattr(data, "web", None) or []
        return _format_results(
            [
                {
                    "title": getattr(r, "title", None),
                    "url": getattr(r, "url", None),
                    "description": getattr(r, "description", None),
                }
                for r in results
            ]
        )

    def scrape(url: str) -> str:
        # Markdown only -- never the json/jsonOptions extraction path (the model
        # produces the JSON, not the tool). only_main_content stays off: it tends
        # to strip sidebars/sections that hold real menu items.
        doc = client.scrape(url, formats=["markdown"], only_main_content=False)
        return doc.markdown or "(page returned no markdown content)"

    return search, scrape


# ---------------------------------------------------------------------------
# Tavily -- search + extract (REST; Bearer auth)
# ---------------------------------------------------------------------------
def _build_tavily():
    api_key = _require_key("tavily")
    headers = _bearer(api_key)

    def search(query: str) -> str:
        data = _post_json(
            "https://api.tavily.com/search",
            headers,
            {"query": query, "max_results": SEARCH_RESULT_LIMIT, "search_depth": "basic"},
        )
        return _format_results(
            [
                # `content` is Tavily's per-result snippet.
                {"title": r.get("title"), "url": r.get("url"), "description": r.get("content")}
                for r in data.get("results", [])
            ]
        )

    def scrape(url: str) -> str:
        data = _post_json(
            "https://api.tavily.com/extract",
            headers,
            {"urls": url, "extract_depth": "basic", "format": "markdown"},
        )
        results = data.get("results") or []
        if results:
            return results[0].get("raw_content") or "(page returned no content)"
        failed = data.get("failed_results") or []
        if failed:
            return f"(extract failed: {failed[0].get('error')})"
        return "(page returned no content)"

    return search, scrape


# ---------------------------------------------------------------------------
# Brave -- search only (REST; X-Subscription-Token auth)
# ---------------------------------------------------------------------------
def _build_brave():
    api_key = _require_key("brave")
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}

    def search(query: str) -> str:
        data = _get_json(
            "https://api.search.brave.com/res/v1/web/search",
            headers,
            {"q": query, "count": SEARCH_RESULT_LIMIT},
        )
        # `web` can be absent when there are no web results -- guard it.
        results = (data.get("web") or {}).get("results", [])
        return _format_results(
            [
                {"title": r.get("title"), "url": r.get("url"), "description": r.get("description")}
                for r in results
            ]
        )

    return search, None


# ---------------------------------------------------------------------------
# Jina -- search (s.jina.ai) + scrape (r.jina.ai); Bearer auth
# ---------------------------------------------------------------------------
def _build_jina():
    api_key = _require_key("jina")

    def search(query: str) -> str:
        # Accept: application/json switches the body from raw markdown to the JSON
        # envelope; for Search the `data` field is a LIST of result objects.
        data = _post_json(
            "https://s.jina.ai/",
            {**_bearer(api_key), "Accept": "application/json"},
            {"q": query, "num": SEARCH_RESULT_LIMIT},
        )
        return _format_results(
            [
                {"title": r.get("title"), "url": r.get("url"), "description": r.get("description")}
                for r in (data.get("data") or [])
            ]
        )

    def scrape(url: str) -> str:
        # URL-prefix form: append the target URL verbatim. X-Return-Format forces
        # clean markdown; the body is markdown text (no JSON envelope here).
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={**_bearer(api_key), "X-Return-Format": "markdown"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.text or "(page returned no content)"

    return search, scrape


# ---------------------------------------------------------------------------
# Browserless -- scrape only (REST /content returns rendered HTML -> markdown)
# ---------------------------------------------------------------------------
def _build_browserless():
    api_key = _require_key("browserless")
    # Region base URL (production-sfo by default); override for lon/ams if needed.
    base = os.environ.get("BROWSERLESS_BASE_URL", "https://production-sfo.browserless.io")

    def scrape(url: str) -> str:
        # /content returns the fully-rendered DOM as raw HTML; Browserless has no
        # markdown endpoint, so we convert locally to match the other scrapers.
        resp = requests.post(
            f"{base}/content",
            params={"token": api_key},
            json={"url": url},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        from markdownify import markdownify as html_to_markdown

        return html_to_markdown(resp.text) or "(page returned no content)"

    return None, scrape


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_BUILDERS = {
    "firecrawl": _build_firecrawl,
    "tavily": _build_tavily,
    "brave": _build_brave,
    "jina": _build_jina,
    "browserless": _build_browserless,
}

# Which providers offer each capability (drives setup_tools validation + CLI choices).
SEARCH_BACKENDS = ["firecrawl", "tavily", "brave", "jina"]
SCRAPE_BACKENDS = ["firecrawl", "tavily", "jina", "browserless"]


def build_backend(name: str):
    """Build a provider's `(search_fn, scrape_fn)` pair (either may be None).

    Reads only that provider's API key, so selecting one backend never requires
    another's key. Raises SystemExit on an unknown name or a missing key.
    """
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown backend {name!r}; choose from {sorted(_BUILDERS)}."
        )
    return builder()
