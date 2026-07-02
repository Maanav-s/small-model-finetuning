"""Search + scrape backends behind the model-facing web_search/scrape_url tools.

Finalized providers: **Brave for search, Jina for scrape.** The model only ever
sees two generically-named tools (web_search, scrape_url) with fixed docstrings
(see build_model_tools in tools.py); this module holds the provider
implementations behind them, so vendor names stay out of the SFT/GRPO training
data the tool calls get baked into later.

  provider   role     transport                     env var
  --------   ------   ---------------------------   -------------
  brave      search   REST (/res/v1/web/search)     BRAVE_API_KEY
  jina       scrape   REST (r.jina.ai)              JINA_API_KEY

  build_search() -> search(query: str) -> str   formatted result list (see _format_results)
  build_scrape() -> scrape(url: str)   -> str   page contents as markdown

The model-facing wrappers in tools.py apply the MAX_TOOL_CHARS cap, so the
functions here return un-capped strings.
"""

from __future__ import annotations

import os

import requests

# Reduce-at-the-source: cap how many results a search returns so big pages don't
# balloon the model's context (and the per-turn prefill that re-encodes it).
SEARCH_RESULT_LIMIT = 3

# Network timeout for the REST calls (seconds). Jina reader scrapes can be slow,
# so keep this generous.
HTTP_TIMEOUT = 60

# Which env var holds each provider's API key.
SEARCH_ENV = "BRAVE_API_KEY"
SCRAPE_ENV = "JINA_API_KEY"


def has_search_key() -> bool:
    """True if the search (Brave) API key is present in the environment."""
    return bool(os.environ.get(SEARCH_ENV))


def has_scrape_key() -> bool:
    """True if the scrape (Jina) API key is present in the environment."""
    return bool(os.environ.get(SCRAPE_ENV))


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
# Jina -- scrape (r.jina.ai; Bearer auth)
# ---------------------------------------------------------------------------
def build_scrape():
    """Build the Jina-backed scrape_url function."""
    api_key = _require_key(SCRAPE_ENV, "jina")

    def scrape(url: str) -> str:
        # URL-prefix form: append the target URL verbatim. X-Return-Format forces
        # clean markdown; the body is markdown text (no JSON envelope here).
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {api_key}", "X-Return-Format": "markdown"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.text or "(page returned no content)"

    return scrape
