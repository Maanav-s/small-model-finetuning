"""Tests for the trace-review app (viz/review.py), v2: corpus.sqlite backed.

No network, no model, no torch -- the app only reads/writes a corpus.sqlite via
src/corpus.py. Each test builds a temp DB (open_corpus + upsert_restaurants +
write_trace), points the app at it (monkeypatch review.DB_PATH), and drives the
endpoints through fastapi.testclient.TestClient.

v1 -> v2 differences these tests bake in:
  * a trace is addressed by its **trace_id** (no '.json' filename);
  * grounding + unmatched_items are trace FIELDS, not a grounding.json index;
  * keep/reject persists to the DB (traces.rejected / reviewed_at), not files;
  * the reject_list.txt export is retired (no /api/review/export);
  * the progress aggregate is GLOBAL (corpus.review_counts), not scope-local.

    uv run python -m pytest tests/test_review.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# corpus.py lives in src/ (flat-import convention). Importing viz.review also puts
# src/ on the path, but add it up front so the fixture's `from corpus import ...`
# never depends on import order.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import viz.review as review  # noqa: E402
from corpus import open_corpus  # noqa: E402


# --- fixture builders ---------------------------------------------------------

def _menu(items: int) -> list:
    return [{"section": "Mains", "items": [
        {"name": f"Dish {i}", "description": "", "price": "$1"} for i in range(items)]}] if items else []


def _messages(text: str = "R" * 2000, *, url: str | None = None, tool: str = "web_search") -> list:
    """An Anthropic-block trace body with one tool_use + its tool_result."""
    tu = {"type": "tool_use", "id": "tu_1", "name": tool,
          "input": ({"url": url} if url else {"query": "q"})}
    return [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [tu]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": [{"type": "text", "text": text}]}]},
    ]


def _add_restaurant(cx, rid: str, *, split: str = "sft") -> None:
    # Explicit restaurant_id -> human-readable trace_ids ("aaa", "acme__vegan").
    cx.upsert_restaurants([{
        "restaurant_id": rid, "name": rid.replace("_", " ").title(),
        "city": "Seattle", "source": "osm", "split": split,
    }])


def _add_trace(cx, rid: str, *, found: bool, items: int = 0, dietary=None,
               grounding=None, unmatched=None, messages=None, source_url=None) -> str:
    """Write one teacher trace for restaurant `rid`. trace_id is derived from
    rid + dietary (free -> '<rid>', conditioned -> '<rid>__<slug>')."""
    fj = {"found": found, "menu": _menu(items), "notes": "the note",
          "source_url": source_url if found else None}
    return cx.write_trace({
        "restaurant_id": rid,
        "model": "claude-sonnet-5",
        "prompt_variant": "teacher",
        "dietary_restrictions": dietary,
        "found": found,
        "schema_valid": True,
        "grounding": grounding,
        "unmatched_items": unmatched,
        "final_json": fj,
        "messages": messages if messages is not None else _messages(),
        "queries": ["q1"],
        "urls": ["https://example.com"],
        "captured_at": "2026-07-18T00:00:00Z",
    })


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "corpus.sqlite"


@pytest.fixture
def client(db_path, monkeypatch):
    # two not-found + one found (found only visible in scope=all), distinct
    # restaurants (no siblings).
    with open_corpus(db_path) as cx:
        for rid in ("aaa", "bbb"):
            _add_restaurant(cx, rid)
            _add_trace(cx, rid, found=False)
        _add_restaurant(cx, "ccc")
        _add_trace(cx, "ccc", found=True, items=3, grounding=1.0, source_url="https://x.com")
    monkeypatch.setattr(review, "DB_PATH", db_path)
    return TestClient(review.app)


# --- list + detail ------------------------------------------------------------

def test_list_notfound_scope_excludes_found(client):
    d = client.get("/api/review/traces?scope=notfound").json()
    ids = [t["trace_id"] for t in d["traces"]]
    assert ids == ["aaa", "bbb"]  # not-found only, ccc excluded
    assert all(t["decision"] is None for t in d["traces"])
    assert all(t["grounding"] is None for t in d["traces"])  # not-found have no menu
    # aggregate is GLOBAL (all 3 traces), not scope-local
    assert d["aggregate"] == {"total": 3, "kept": 0, "rejected": 0, "undecided": 3}


def test_list_all_scope_includes_found_with_grounding(client):
    d = client.get("/api/review/traces?scope=all").json()
    ids = [t["trace_id"] for t in d["traces"]]
    # not-found first (aaa, bbb), then found (ccc)
    assert ids == ["aaa", "bbb", "ccc"]
    assert d["traces"][-1]["found"] is True
    assert d["traces"][-1]["n_items"] == 3
    assert d["traces"][-1]["grounding"] == 1.0  # from the stored field


def test_trace_detail_has_grounding_and_conversation(client):
    d = client.get("/api/review/trace/aaa").json()
    assert d["ok"] is True
    assert d["trace_id"] == "aaa"
    assert d["grounding"] is None                # not-found: no menu to ground
    assert d["notes"] == "the note"
    assert d["episode_input"] == "Aaa, Seattle"  # joined name + city
    # tool result preview truncated to 800 chars
    tr = [t for t in d["conversation"] if t["tool_results"]][0]["tool_results"][0]
    assert len(tr["preview"]) == 800
    assert tr["truncated"] is True


def test_missing_trace_returns_error(client):
    assert client.get("/api/review/trace/nope").json()["ok"] is False
    assert client.post("/api/review/decision",
                       json={"trace_id": "nope", "decision": "keep"}).json()["ok"] is False


# --- decisions (persist to the DB) --------------------------------------------

def test_decision_persists_and_updates_aggregate(client, db_path):
    r = client.post("/api/review/decision", json={"trace_id": "aaa", "decision": "reject"})
    d = r.json()
    assert d["ok"] and d["decision"] == "reject"
    assert d["aggregate"]["rejected"] == 1 and d["aggregate"]["undecided"] == 2
    # persisted via set_review_decision: re-read shows rejected + a reviewed_at stamp
    with open_corpus(db_path, create=False) as cx:
        t = cx.get_trace("aaa")
    assert t["rejected"] is True and t["reviewed_at"] is not None


def test_reject_reason_is_stored(client, db_path):
    client.post("/api/review/decision",
                json={"trace_id": "aaa", "decision": "reject", "reason": "wrong city"})
    with open_corpus(db_path, create=False) as cx:
        assert cx.get_trace("aaa")["reject_reason"] == "wrong city"


def test_undo_clears_decision(client, db_path):
    client.post("/api/review/decision", json={"trace_id": "aaa", "decision": "keep"})
    r = client.post("/api/review/decision", json={"trace_id": "aaa", "decision": "undo"})
    d = r.json()
    assert d["decision"] is None
    assert d["aggregate"]["kept"] == 0
    with open_corpus(db_path, create=False) as cx:
        t = cx.get_trace("aaa")
    assert t["reviewed_at"] is None and t["rejected"] is False


def test_invalid_decision_rejected(client):
    r = client.post("/api/review/decision", json={"trace_id": "aaa", "decision": "maybe"})
    assert r.json()["ok"] is False


def test_no_export_endpoint(client):
    # the reject_list.txt export is retired -- rejection is the DB field now.
    assert client.post("/api/review/export").status_code == 404
    assert client.get("/api/review/export").status_code == 404


# --- sibling-evidence panel (siblings = other slices of the same restaurant) --

@pytest.fixture
def sib_client(db_path, monkeypatch):
    """Two not-found slices, each vouched by a found sibling of the SAME restaurant:
    one well-grounded (0.9), one weakly-grounded (0.25). The sibling link is the
    shared restaurant_id (corpus.siblings), not an explicit map."""
    with open_corpus(db_path) as cx:
        _add_restaurant(cx, "good")
        _add_trace(cx, "good", found=True, items=3, grounding=0.9,
                   source_url="https://good.example.com")            # trace_id "good"
        _add_trace(cx, "good", found=False, dietary=["keto"])         # trace_id "good__keto"
        _add_restaurant(cx, "bad")
        _add_trace(cx, "bad", found=True, items=3, grounding=0.25,
                   unmatched=["Dish 0", "Dish 1"],
                   source_url="https://bad.example.com")             # trace_id "bad"
        _add_trace(cx, "bad", found=False, dietary=["keto"])          # trace_id "bad__keto"
    monkeypatch.setattr(review, "DB_PATH", db_path)
    return TestClient(review.app)


def test_sibling_panel_shows_menu_and_grounding(sib_client):
    d = sib_client.get("/api/review/trace/good__keto").json()
    assert d["ok"] is True
    assert "siblings" in d and len(d["siblings"]) == 1
    sib = d["siblings"][0]
    assert sib["sibling_trace_id"] == "good"
    assert sib["grounding"] == 0.9
    # the sibling trace's menu is inlined (sections -> items)
    names = [it["name"] for sec in sib["menu"] for it in sec["items"]]
    assert names == ["Dish 0", "Dish 1", "Dish 2"]


def test_weakly_grounded_sibling_reports_unmatched(sib_client):
    d = sib_client.get("/api/review/trace/bad__keto").json()
    sib = d["siblings"][0]
    assert sib["sibling_trace_id"] == "bad"
    assert sib["grounding"] == 0.25
    assert sib["unmatched_items"] == ["Dish 0", "Dish 1"]


@pytest.fixture
def pair_client(db_path, monkeypatch):
    """One restaurant with three slices: a found free slice, a found vegan slice,
    and a not-found keto slice. Every slice lists the other two as siblings."""
    with open_corpus(db_path) as cx:
        _add_restaurant(cx, "acme")
        _add_trace(cx, "acme", found=True, items=3, grounding=1.0,
                   source_url="https://acme.example.com")             # trace_id "acme"
        _add_trace(cx, "acme", found=True, items=1, dietary=["vegan"], grounding=1.0)  # acme__vegan
        _add_trace(cx, "acme", found=False, dietary=["keto"])         # trace_id "acme__keto"
    monkeypatch.setattr(review, "DB_PATH", db_path)
    return TestClient(review.app)


def test_found_trace_shows_its_siblings(pair_client):
    # a FOUND trace surfaces its sibling slices too (not just not-found traces)
    d = pair_client.get("/api/review/trace/acme").json()
    assert d["ok"] is True and d["found"] is True
    sibs = {s["sibling_trace_id"]: s for s in d["siblings"]}
    assert set(sibs) == {"acme__vegan", "acme__keto"}
    # found sibling sorts before the not-found one
    assert d["siblings"][0]["found"] is True
    assert d["siblings"][-1]["sibling_trace_id"] == "acme__keto"
    # the not-found sibling reports found=False and no grounding
    assert sibs["acme__keto"]["found"] is False
    assert sibs["acme__keto"]["grounding"] is None


# --- per-item "where in the scrape" highlight (shared grounding.normalize) -----

_SCRAPE = ("Welcome to Seoul Kitchen. Our famous Galbi Set is grilled short rib "
           "served with banchan. The Kimchi Jjigae is a spicy stew. " + "x" * 3000)
_SCRAPE_URL = "https://seoulkitchen.example.com/menu"


@pytest.fixture
def match_client(db_path, monkeypatch):
    """A not-found keto slice vouched by a found free slice whose scrape text
    contains SOME of its menu item names (so we can assert per-item matched/context/
    source_hint and the found trace's stored %-grounded)."""
    menu = [{"section": "Mains", "items": [
        {"name": "Galbi Set", "description": "", "price": "$30"},
        {"name": "Kimchi Jjigae", "description": "", "price": "$16"},
        {"name": "Phantom Dish", "description": "", "price": "$9"},
    ]}]
    with open_corpus(db_path) as cx:
        _add_restaurant(cx, "seoul")
        cx.write_trace({
            "restaurant_id": "seoul", "model": "claude-sonnet-5", "prompt_variant": "teacher",
            "dietary_restrictions": None, "found": True, "schema_valid": True,
            "grounding": 0.667, "unmatched_items": ["Phantom Dish"],
            "final_json": {"found": True, "menu": menu, "notes": "n", "source_url": _SCRAPE_URL},
            "messages": _messages(_SCRAPE, url=_SCRAPE_URL, tool="scrape_url"),
            "queries": ["q1"], "urls": [_SCRAPE_URL], "captured_at": "2026-07-18T00:00:00Z",
        })
        _add_trace(cx, "seoul", found=False, dietary=["keto"])        # trace_id "seoul__keto"
    monkeypatch.setattr(review, "DB_PATH", db_path)
    return TestClient(review.app)


def test_found_trace_detail_grounding_read_from_field(match_client):
    # grounding is a stored field now (2 of 3 items grounded -> 0.667).
    d = match_client.get("/api/review/trace/seoul").json()
    assert d["ok"] is True
    assert d["grounding"] == 0.667


def test_sibling_items_carry_matched_and_context(match_client):
    d = match_client.get("/api/review/trace/seoul__keto").json()
    assert d["ok"] is True
    items = d["siblings"][0]["items"]
    by_name = {it["name"]: it for it in items}
    # matched item: context present, marked span wraps the exact name, source_hint set
    galbi = by_name["Galbi Set"]
    assert galbi["matched"] is True
    assert galbi["context"] is not None
    assert "〈Galbi Set〉" in galbi["context"]
    assert galbi["source_hint"] == _SCRAPE_URL
    assert by_name["Kimchi Jjigae"]["matched"] is True
    # unmatched item: matched False, no context
    phantom = by_name["Phantom Dish"]
    assert phantom["matched"] is False
    assert phantom["context"] is None


def test_full_tool_result_endpoint_returns_untruncated(match_client):
    d = match_client.get("/api/review/trace/seoul").json()
    # locate the tool-result block's (turn, idx) address from the compact conversation
    tr = None
    for t in d["conversation"]:
        if t["tool_results"]:
            tr = t["tool_results"][0]
            break
    assert tr is not None and tr["truncated"] is True and tr["full_len"] > 800
    full = match_client.get(
        f"/api/review/toolresult/seoul?turn={tr['turn']}&idx={tr['idx']}").json()
    assert full["ok"] is True
    assert full["full_len"] == tr["full_len"]
    assert len(full["text"]) == tr["full_len"] > 800
    assert full["text"].startswith("Welcome to Seoul Kitchen")


def test_full_tool_result_endpoint_bogus_id_and_range_guarded(match_client):
    # A bogus trace_id simply misses in the DB (no filesystem, so no traversal risk)
    # -> {ok: False}, never a 500.
    assert match_client.get(
        "/api/review/toolresult/nope%5C..%5Csecret?turn=0&idx=0").json()["ok"] is False
    assert match_client.get(
        "/api/review/toolresult/missing?turn=0&idx=0").json()["ok"] is False
    # out-of-range turn/idx are rejected, not 500s
    assert match_client.get(
        "/api/review/toolresult/seoul?turn=99&idx=0").json()["ok"] is False
    assert match_client.get(
        "/api/review/toolresult/seoul?turn=2&idx=99").json()["ok"] is False


# --- deciding a sibling directly from the evidence panel ----------------------

def test_sibling_reject_persists_and_shows_in_detail(sib_client, db_path):
    # POST a reject for the SIBLING's trace_id; it persists in the DB and is
    # surfaced as the sibling's `decision` in the not-found's detail.siblings.
    r = sib_client.post("/api/review/decision",
                        json={"trace_id": "good", "decision": "reject"})
    assert r.json()["ok"] and r.json()["decision"] == "reject"
    with open_corpus(db_path, create=False) as cx:
        assert cx.get_trace("good")["rejected"] is True
    d = sib_client.get("/api/review/trace/good__keto").json()
    assert d["siblings"][0]["sibling_trace_id"] == "good"
    assert d["siblings"][0]["decision"] == "reject"


def test_decision_updates_global_review_counts(sib_client):
    # the progress aggregate now reflects corpus.review_counts() over ALL traces,
    # so a decision on any trace (in or out of the current scope) moves it.
    before = sib_client.get("/api/review/traces?scope=notfound").json()["aggregate"]
    assert before == {"total": 4, "kept": 0, "rejected": 0, "undecided": 4}
    r = sib_client.post("/api/review/decision",
                        json={"trace_id": "good", "decision": "reject"})
    assert r.json()["aggregate"] == {"total": 4, "kept": 0, "rejected": 1, "undecided": 3}
    after = sib_client.get("/api/review/traces?scope=all").json()["aggregate"]
    assert after == {"total": 4, "kept": 0, "rejected": 1, "undecided": 3}


def test_app_imports_without_gpu_stack():
    # Must run in a CLEAN interpreter. An absolute sys.modules assertion in the
    # shared pytest process is polluted by sibling test files (test_build_sft /
    # test_eval_split import transformers), which would make this fail for reasons
    # unrelated to viz.review. A fresh subprocess actually tests the claim: that
    # importing viz.review pulls in none of the GPU/model stack (corpus + grounding
    # are stdlib-only).
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c",
         "import viz.review, sys; "
         "print([m for m in ('torch', 'anthropic', 'transformers') if m in sys.modules])"],
        capture_output=True, text=True, cwd=repo,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
