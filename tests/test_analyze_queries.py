"""Unit tests for scripts/analysis/analyze_queries.py (v2: mine corpus.sqlite traces).

No network anywhere. The pure functions (templatize / domain_of / is_delivery /
extract_calls) are UNCHANGED from v1 and tested directly on synthetic trace dicts.
The aggregate report is now mined from an in-DB corpus fixture (traces written via
corpus.write_trace, read back via analyze_queries.load_traces) instead of a
data/traces/*.json directory -- that is the only v2 change (input SOURCE = the DB).

Run: uv run python -m pytest tests/test_analyze_queries.py -q
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Flat-import, script-run convention (see CLAUDE.md): shared modules in src/, the
# script under test in scripts/analysis/ (the v2 location -- NOT scripts/).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from analyze_queries import (  # noqa: E402
    analyze,
    domain_of,
    extract_calls,
    is_delivery,
    load_traces,
    templatize,
)
from corpus import open_corpus, restaurant_id_for  # noqa: E402


def _messages(name, city, queries, scrapes):
    """The Anthropic content-block message list a teacher loop records: a user
    episode-input turn, an assistant tool_use turn, and a user tool_result turn.
    `scrapes` rows are (url, mode, result) triples (mode=None omits the arg)."""
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
    return [
        {"role": "user", "content": f"{name}, {city}"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": results},
    ]


def make_trace(name, city, *, queries=(), scrapes=(), found=True,
               model="claude-sonnet-5"):
    """A minimal contract-1.5 trace dict (as extract_calls consumes it)."""
    return {
        "restaurant_id": "deadbeef",
        "restaurant_name": name,
        "city": city,
        "episode_input": f"{name}, {city}",
        "model": model,
        "messages": _messages(name, city, queries, scrapes),
        "queries": list(queries),
        "urls": [s[0] for s in scrapes],
        "final_json": {"found": found, "menu": []},
        "schema_valid": True,
    }


def _add_trace(cx, name, city, *, queries=(), scrapes=(), found=True,
               model="claude-sonnet-5", split="sft"):
    """Upsert a restaurant and write one teacher trace for it into the corpus."""
    cx.upsert_restaurants([{"name": name, "city": city, "source": "osm", "split": split}])
    rid = restaurant_id_for(name, city)
    cx.write_trace({
        "restaurant_id": rid,
        "model": model,
        "prompt_variant": "teacher",
        "dietary_restrictions": None,
        "found": found,
        "schema_valid": True,
        "final_json": {"found": found, "menu": []},
        "messages": _messages(name, city, queries, scrapes),
        "queries": list(queries),
        "urls": [s[0] for s in scrapes],
        "cache_version": 1,
        "captured_at": "2026-07-18T00:00:00Z",
    })
    return rid


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
# Delivery-domain flagging + the aggregate report over an in-DB corpus
# ---------------------------------------------------------------------------
class TestDeliveryFlagging:
    def test_is_delivery_matches_subdomains(self):
        assert is_delivery(domain_of("https://www.doordash.com/store/x"))
        assert is_delivery(domain_of("https://order.ubereats.com/x"))
        assert is_delivery(domain_of("https://www.tripadvisor.co.uk/r"))
        assert not is_delivery(domain_of("https://ssamjangbbq.com/menu"))
        assert not is_delivery(domain_of("https://web.archive.org/web/x"))

    def test_aggregates_over_corpus(self, tmp_path):
        corpus = tmp_path / "corpus.sqlite"
        with open_corpus(corpus) as cx:
            _add_trace(cx, "Ssamjang", "Atlanta",
                       queries=["Ssamjang Atlanta menu"],
                       scrapes=[("https://ssamjangbbq.com/menu", "direct", "menu md"),
                                ("https://ssamjangbbq.com/menu", "browser", "menu md")],
                       found=True)
            _add_trace(cx, "Joe's Pizza", "New York",
                       queries=["joes pizza new york menu"],
                       scrapes=[("https://www.doordash.com/store/joes", None,
                                 "(scrape failed: bot wall)"),
                                ("https://web.archive.org/web/joespizza", "browser",
                                 "archived menu")],
                       found=False)

        traces = load_traces(corpus, model=None, split=None)
        assert len(traces) == 2
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
        corpus = tmp_path / "corpus.sqlite"
        with open_corpus(corpus) as cx:
            _add_trace(cx, "Alpha", "Austin", queries=["Alpha menu"], model="claude-sonnet-5")
            _add_trace(cx, "Beta", "Boston", queries=["Beta menu"], model="claude-sonnet-4-6")
        traces = load_traces(corpus, model="claude-sonnet-5", split=None)
        assert [t["model"] for t in traces] == ["claude-sonnet-5"]


class TestExtractCalls:
    def test_pairs_results_and_counts_calls(self):
        trace = make_trace("X", "Y", queries=["X menu"],
                           scrapes=[("https://x.com/a", "direct", "(scrape failed: 403)")])
        queries, scrapes, n_calls = extract_calls(trace)
        assert queries == ["X menu"]
        assert scrapes == [{"url": "https://x.com/a", "mode": "direct", "failed": True}]
        assert n_calls == 2
