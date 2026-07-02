"""A/B the scrape backends (jina | hybrid | local) on the same URL(s).

Prototype tool for deciding whether to keep Jina or move to a fully self-hosted
scrape. For each URL it runs each backend in both modes and reports elapsed time
plus two cheap coverage proxies (chars, and $-price markers), so a bigger number
that is just chrome/boilerplate doesn't look like a win.

  jina    Brave-free? no. Jina renders even in "direct".         (src/backends.py)
  hybrid  Jina "direct" + a per-call Playwright "browser".
  local   FULLY Jina-free: requests fast-path for "direct" (with
          a no-scroll browser fallback for CSR shells) + a POOLED
          Playwright browser for "browser".              (src/scrape_playwright.py)

  uv run python scripts/scrape_ab.py                 # built-in test URLs
  uv run python scripts/scrape_ab.py <url> [<url>..] # your own

Needs JINA_API_KEY (repo-root .env) for jina/hybrid; local needs only
`playwright install chromium` (+ its system libs), no key.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backends import build_scrape as build_scrape_jina  # noqa: E402
from scrape_playwright import (  # noqa: E402
    build_scrape_hybrid,
    build_scrape_local,
)

# Default targets: a Square SPA that lazy-loads its menu (the case Playwright
# should win) and a bot-protected delivery app (the case neither wins reliably).
DEFAULT_URLS = [
    "https://kashishkirkland.square.site/",
    "https://www.doordash.com/store/kashish-cuisine-of-india-kirkland-25529836",
]

# Cheap coverage proxies (see module docstring): dollar-price markers, and the
# distinct menu-item price patterns like "$12.95".
_PRICE_RE = re.compile(r"\$\s?\d")


def _coverage(text: str, secs: float) -> str:
    prices = len(_PRICE_RE.findall(text))
    return f"{len(text):>7} chars | {prices:>3} price markers | {secs:5.1f}s"


def main(urls: list[str]) -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    # jina/hybrid need the Jina key; local is key-free. Build the shared Jina
    # direct once and hand it to hybrid.
    jina = build_scrape_jina()
    backends = [
        ("jina", jina),
        ("hybrid", build_scrape_hybrid(jina)),
        ("local", build_scrape_local()),
    ]

    for url in urls:
        print(f"\n=== {url} ===")
        for name, scrape in backends:
            for mode in ("direct", "browser"):
                try:
                    t = time.time()
                    out = scrape(url, mode=mode)
                    print(f"  {name:>7} / {mode:<7}: {_coverage(out, time.time() - t)}")
                except Exception as e:  # noqa: BLE001 - prototype: report, keep going
                    print(f"  {name:>7} / {mode:<7}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_URLS)
