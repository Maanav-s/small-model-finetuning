"""Unit tests for scripts/analyze_queries.py (Phase 2 WS-E).

No network anywhere: synthetic traces (contract 1.5 shapes) are written to a
pytest tmp_path traces dir and mined exactly like the real data/traces/.

Run: uv run python -m pytest tests/test_analyze_queries.py -q
"""

import json
import sys
from pathlib import Path

# The tool lives in scripts/ and its imports in src/ (flat imports, no
# packages) -- same convention as the entry scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze_queries import (  # noqa: E402
    analyze,
    domain_of,
    extract_calls,
    is_delivery,
    load_traces,
    templatize,
)


def make_trace(name, city, *, queries=(), scrapes=(), found=True,
               model="claude-sonnet-5"):
    """A minimal contract-1.5 trace. `scrapes` rows are (url, mode, result)
    triples rendered as paired tool_use/tool_result blocks, like the real
    Claude loop records them."""
    content, results = [], []
    for i, q in enumerate(queries):
        content.append({"type": "tool_use", "id": f"q{i}", "name": "web_search",
                        "input": {"query": q}})
        results.append({"type": "tool_result", "tool_use_id": f"q{i}",
                        "content": "[1] some result"})
    for i, (url, mode, result) in enumerate(scrapes):
        block_input = {"url": url}
        if mode is not None:  # mode arg omitted -> backend default "direct"
            block_input["mode"] = mode
        content.append({"type": "tool_use", "id": f"s{i}", "name": "scrape_url",
                        "input": block_input})
        results.append({"type": "tool_result", "tool_use_id": f"s{i}",
                        "content": result})
    return {
        "restaurant_id": "deadbeef",
        "restaurant_name": name,
        "episode_input": f"{name}, {city}",
        "model": model,
        "messages": [
            {"role": "user", "content": f"{name}, {city}"},
            {"role": "assistant", "content": content},
            {"role": "user", "content": results},
        ],
        "queries": list(queries),
        "urls": [s[0] for s in scrapes],
        "final_json": {"found": found, "menu": []},
        "schema_valid": True,
    }


# ---------------------------------------------------------------------------
# Templatization: exact / partial / punctuated name mentions, city handling
# ---------------------------------------------------------------------------
class TestTemplatize:
    def test_exact_name_and_city(self):
        assert templatize("Ssamjang Atlanta menu", "Ssamjang", "Atlanta") == \
            "{name} {city} menu"

    def test_partial_name_multi_token(self):
        # Dropping the leading "The" still reads as a name mention.
        assert templatize("cheesecake factory menu prices",
                          "The Cheesecake Factory", "Seattle") == \
            "{name} menu prices"

    def test_partial_name_distinctive_single_token(self):
        # The distinctive first token alone counts; a lone generic later
        # token ("bbq") must not.
        assert templatize("ssamjang menu", "Ssamjang Korean BBQ", "Atlanta") == \
            "{name} menu"
        assert templatize("best bbq atlanta", "Ssamjang Korean BBQ", "Atlanta") == \
            "best bbq {city}"

    def test_name_with_punctuation(self):
        # Apostrophes match with or without: "Joe's" ~ "joes" ~ "joe's".
        assert templatize("joes pizza menu", "Joe's Pizza", "New York") == \
            "{name} menu"
        assert templatize("Joe's Pizza New York menu", "Joe's Pizza", "New York") == \
            "{name} {city} menu"

    def test_lone_stopword_not_swallowed(self):
        assert templatize("the best menu in town", "The Cheesecake Factory",
                          "Seattle") == "the best menu in town"

    def test_multi_token_city(self):
        assert templatize("ssamjang new york menu", "Ssamjang", "New York") == \
            "{name} {city} menu"


# ---------------------------------------------------------------------------
# Delivery-domain flagging + the aggregate report over synthetic traces
# ---------------------------------------------------------------------------
class TestDeliveryFlagging:
    def test_is_delivery_matches_subdomains(self):
        assert is_delivery(domain_of("https://www.doordash.com/store/x"))
        assert is_delivery(domain_of("https://order.ubereats.com/x"))
        assert is_delivery(domain_of("https://www.tripadvisor.co.uk/r"))
        assert not is_delivery(domain_of("https://ssamjangbbq.com/menu"))
        assert not is_delivery(domain_of("https://web.archive.org/web/x"))

    def test_aggregates_over_traces_dir(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        t1 = make_trace("Ssamjang", "Atlanta",
                        queries=["Ssamjang Atlanta menu"],
                        scrapes=[("https://ssamjangbbq.com/menu", "direct", "menu md"),
                                 ("https://ssamjangbbq.com/menu", "browser", "menu md")],
                        found=True)
        t2 = make_trace("Joe's Pizza", "New York",
                        queries=["joes pizza new york menu"],
                        scrapes=[("https://www.doordash.com/store/joes", None,
                                  "(scrape failed: bot wall)"),
                                 ("https://web.archive.org/web/joespizza", "browser",
                                  "archived menu")],
                        found=False)
        (traces_dir / "t1.json").write_text(json.dumps(t1), encoding="utf-8")
        (traces_dir / "t2.json").write_text(json.dumps(t2), encoding="utf-8")
        (traces_dir / "torn.json").write_text("{not json", encoding="utf-8")

        traces, skipped = load_traces(traces_dir, model=None)
        assert len(traces) == 2 and skipped == 1
        agg = analyze(traces)

        templates = {r["template"]: r["count"] for r in agg["query_templates"]}
        assert templates == {"{name} {city} menu": 2}
        assert agg["template_coverage"]["pct"] == 100.0

        by_domain = {r["domain"]: r for r in agg["domains"]}
        assert by_domain["doordash.com"]["delivery"] is True
        assert by_domain["ssamjangbbq.com"]["delivery"] is False
        assert agg["delivery_share"]["scrapes"] == 1

        # absent mode arg counts as the backend default "direct"
        assert agg["scrape_modes"] == {"direct": 2, "browser": 2}

        # outcome split: the delivery-touching episode is the not-found one
        assert agg["outcomes"]["scraped_delivery"] == {
            "episodes": 1, "found_rate": 0.0, "schema_valid_rate": 1.0,
            "mean_tool_calls": 3.0}
        assert agg["outcomes"]["no_delivery"]["found_rate"] == 1.0

        # fallback signals: one archive.org scrape, one failed scrape
        assert agg["fallbacks"]["archive_org_scrapes"] == 1
        assert agg["fallbacks"]["scrape_failures"] == 1
        assert agg["fallbacks"]["failure_domains"] == [
            {"domain": "doordash.com", "count": 1}]

    def test_model_filter(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        t1 = make_trace("A", "B", queries=["A menu"], model="claude-sonnet-5")
        t2 = make_trace("A", "B", queries=["A menu"], model="claude-sonnet-4-6")
        (traces_dir / "a.json").write_text(json.dumps(t1), encoding="utf-8")
        (traces_dir / "b.json").write_text(json.dumps(t2), encoding="utf-8")
        traces, _ = load_traces(traces_dir, model="claude-sonnet-5")
        assert [t["model"] for t in traces] == ["claude-sonnet-5"]


class TestExtractCalls:
    def test_pairs_results_and_counts_calls(self):
        trace = make_trace("X", "Y", queries=["X menu"],
                           scrapes=[("https://x.com/a", "direct", "(scrape failed: 403)")])
        queries, scrapes, n_calls = extract_calls(trace)
        assert queries == ["X menu"]
        assert scrapes == [{"url": "https://x.com/a", "mode": "direct", "failed": True}]
        assert n_calls == 2
