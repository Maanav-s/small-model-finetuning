"""A/B the two scrape backends (Jina vs local Playwright) on the same URL(s).

Prototype tool for deciding whether the Playwright auto-scroll renderer
(src/scrape_playwright.py) beats Jina (src/backends.py) on lazy-loaded menu
pages. For each URL it runs both backends in both modes and reports how much
each returned plus a couple of cheap coverage proxies (price markers, and how
many of the page's own menu-section anchors show up), so a bigger number that is
just chrome/boilerplate doesn't look like a win.

  uv run python scripts/scrape_ab.py                 # built-in test URLs
  uv run python scripts/scrape_ab.py <url> [<url>..] # your own

Needs JINA_API_KEY (repo-root .env) for the Jina side; the Playwright side needs
`playwright install chromium` (+ its system libs) but no key.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backends import build_scrape as build_scrape_jina  # noqa: E402
from scrape_playwright import build_scrape_playwright  # noqa: E402

# Default targets: a Square SPA that lazy-loads its menu (the case Playwright
# should win) and a bot-protected delivery app (the case neither wins reliably).
DEFAULT_URLS = [
    "https://kashishkirkland.square.site/",
    "https://www.doordash.com/store/kashish-cuisine-of-india-kirkland-25529836",
]

# Cheap coverage proxies (see module docstring): dollar-price markers, and the
# distinct menu-item price patterns like "$12.95".
_PRICE_RE = re.compile(r"\$\s?\d")


def _coverage(text: str) -> str:
    prices = len(_PRICE_RE.findall(text))
    return f"{len(text):>7} chars | {prices:>3} price markers"


def main(urls: list[str]) -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    jina = build_scrape_jina()
    play = build_scrape_playwright()
    backends = [("jina", jina), ("playwright", play)]

    for url in urls:
        print(f"\n=== {url} ===")
        for name, scrape in backends:
            for mode in ("direct", "browser"):
                try:
                    out = scrape(url, mode=mode)
                    print(f"  {name:>10} / {mode:<7}: {_coverage(out)}")
                except Exception as e:  # noqa: BLE001 - prototype: report, keep going
                    print(f"  {name:>10} / {mode:<7}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_URLS)
