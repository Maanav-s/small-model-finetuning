"""Regression tests for the scrape backend's failure contract (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backends  # noqa: E402
from backends import build_scrape  # noqa: E402

# Deep enough to blow the interpreter recursion limit inside BeautifulSoup /
# markdownify -- the shape that killed 10/99 pilot episodes (see build_corpus).
DEEP_HTML = "<div>" * 3000 + "menu" + "</div>" * 3000


def test_deeply_nested_page_returns_sentinel_not_raise(monkeypatch):
    monkeypatch.setattr(
        backends, "_render_pooled", lambda url, *, wait, scroll: DEEP_HTML
    )
    scrape = build_scrape()
    out = scrape("http://deep.example.test/menu", mode="browser")
    assert out.startswith("(scrape failed"), out
    assert "recursion" in out.lower()
