"""Tests for the trace-review app (viz/review.py).

No network, no model, no torch -- the app only reads/writes JSON under DATA_DIR.
Each test points DATA_DIR at a tmp dir (monkeypatch) and drives the app through
fastapi.testclient.TestClient.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import viz.review as review


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _trace(name, found, *, items=0, schema_valid=True):
    menu = [{"section": "Mains", "items": [
        {"name": f"Dish {i}", "description": "", "price": "$1"} for i in range(items)]}] if items else []
    return {
        "restaurant_id": name.replace(".json", ""),
        "restaurant_name": name.replace(".json", "").title(),
        "episode_input": f"{name}, Seattle",
        "dietary_restrictions": None,
        "queries": ["q1"],
        "urls": ["https://example.com"],
        "final_json": {"found": found, "menu": menu,
                       "notes": "the note", "source_url": None if not found else "https://x.com"},
        "schema_valid": schema_valid,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": name}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "web_search", "input": {"query": "q"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "content": [{"type": "text", "text": "R" * 2000}]}]},
        ],
    }


def _ground_entry(trace, *, grounding=None, siblings=None, unmatched=None):
    """A grounding.json entry mirroring scripts/audit_grounding.py's output shape.

    Most fields derive from the trace; `grounding`, `siblings`, and `unmatched`
    are set explicitly so a test can pin the value the list/sibling panel reads.
    """
    fj = trace.get("final_json") or {}
    n_items = sum(len(s.get("items") or []) for s in (fj.get("menu") or []))
    return {
        "restaurant_id": trace["restaurant_id"],
        "restaurant_name": trace["restaurant_name"],
        "episode_input": trace["episode_input"],
        "restrictions": trace.get("dietary_restrictions"),
        "found": bool(fj.get("found")),
        "schema_valid": bool(trace.get("schema_valid")),
        "n_items": n_items,
        "grounding": grounding,
        "unmatched_items": unmatched or [],
        "source_url": fj.get("source_url"),
        "siblings": siblings or [],
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(review, "DATA_DIR", data)
    # two not-found + one found (only visible in scope=all)
    aaa = _trace("aaa.json", found=False)
    bbb = _trace("bbb.json", found=False)
    ccc = _trace("ccc.json", found=True, items=3)
    _write(data / "traces" / "aaa.json", aaa)
    _write(data / "traces" / "bbb.json", bbb)
    _write(data / "traces" / "ccc.json", ccc)
    _write(data / "review" / "grounding.json", {
        "aaa.json": _ground_entry(aaa),
        "bbb.json": _ground_entry(bbb),
        "ccc.json": _ground_entry(ccc, grounding=1.0),
    })
    return TestClient(review.app)


def test_list_notfound_scope_excludes_found(client):
    d = client.get("/api/review/traces?scope=notfound").json()
    names = [t["filename"] for t in d["traces"]]
    assert names == ["aaa.json", "bbb.json"]  # not-found only, ccc excluded
    assert all("suggestion" not in t for t in d["traces"])  # triage feature is gone
    assert all(t["decision"] is None for t in d["traces"])
    assert all(t["grounding"] is None for t in d["traces"])  # not-found have no menu
    assert d["aggregate"] == {"total": 2, "kept": 0, "rejected": 0, "undecided": 2}


def test_list_all_scope_includes_found_with_grounding(client):
    d = client.get("/api/review/traces?scope=all").json()
    names = [t["filename"] for t in d["traces"]]
    # not-found first (aaa,bbb), then found (ccc)
    assert names == ["aaa.json", "bbb.json", "ccc.json"]
    assert d["traces"][-1]["found"] is True
    assert d["traces"][-1]["n_items"] == 3
    assert d["traces"][-1]["grounding"] == 1.0  # from grounding.json


def test_list_falls_back_when_no_grounding_file(client, tmp_path):
    # Delete the precomputed index: the app still lists traces (grounding = null).
    (tmp_path / "data" / "review" / "grounding.json").unlink()
    d = client.get("/api/review/traces?scope=all").json()
    names = [t["filename"] for t in d["traces"]]
    assert names == ["aaa.json", "bbb.json", "ccc.json"]
    assert all(t["grounding"] is None for t in d["traces"])


def test_trace_detail_no_triage_has_grounding_and_conversation(client):
    d = client.get("/api/review/trace/aaa.json").json()
    assert d["ok"] is True
    assert "triage" not in d                     # triage feature removed
    assert d["grounding"] is None                # not-found: no menu to ground
    assert d["notes"] == "the note"
    # tool result preview truncated to 800 chars
    tr = [t for t in d["conversation"] if t["tool_results"]][0]["tool_results"][0]
    assert len(tr["preview"]) == 800
    assert tr["truncated"] is True


def test_decision_persists_and_updates_aggregate(client, tmp_path):
    r = client.post("/api/review/decision", json={"filename": "aaa.json", "decision": "reject"})
    d = r.json()
    assert d["ok"] and d["decision"] == "reject"
    assert d["aggregate"]["rejected"] == 1 and d["aggregate"]["undecided"] == 1
    # persisted to decisions.json
    decisions = json.loads((tmp_path / "data" / "review" / "decisions.json").read_text())
    assert decisions["aaa.json"]["decision"] == "reject"
    assert "at" in decisions["aaa.json"]


def test_undo_clears_decision(client, tmp_path):
    client.post("/api/review/decision", json={"filename": "aaa.json", "decision": "keep"})
    r = client.post("/api/review/decision", json={"filename": "aaa.json", "decision": "undo"})
    d = r.json()
    assert d["decision"] is None
    assert d["aggregate"]["kept"] == 0
    decisions = json.loads((tmp_path / "data" / "review" / "decisions.json").read_text())
    assert "aaa.json" not in decisions


def test_invalid_decision_rejected(client):
    r = client.post("/api/review/decision", json={"filename": "aaa.json", "decision": "maybe"})
    assert r.json()["ok"] is False


def test_export_writes_reject_list_only_rejects(client, tmp_path):
    client.post("/api/review/decision", json={"filename": "aaa.json", "decision": "reject"})
    client.post("/api/review/decision", json={"filename": "bbb.json", "decision": "keep"})
    d = client.post("/api/review/export").json()
    assert d["ok"] and d["count"] == 1
    text = Path(d["path"]).read_text()
    lines = text.splitlines()
    assert lines[0].startswith("# generated") and "1 rejects" in lines[0]
    assert "aaa.json" in lines
    assert "bbb.json" not in text  # kept traces excluded


def test_missing_trace_returns_error(client):
    assert client.get("/api/review/trace/nope.json").json()["ok"] is False
    assert client.post("/api/review/decision",
                       json={"filename": "nope.json", "decision": "keep"}).json()["ok"] is False


# --- sibling-evidence panel ---------------------------------------------------

@pytest.fixture
def sib_client(tmp_path, monkeypatch):
    """Two not-found traces, each vouched by a found sibling: one well-grounded
    (0.9), one weakly-grounded (0.25). The sibling map + grounding come from
    grounding.json (scripts/audit_grounding.py output).
    """
    data = tmp_path / "data"
    monkeypatch.setattr(review, "DATA_DIR", data)
    nf_good = _trace("nf_good.json", found=False)
    nf_bad = _trace("nf_bad.json", found=False)
    sib_good = _trace("sib_good.json", found=True, items=3)
    sib_bad = _trace("sib_bad.json", found=True, items=3)
    for t in (nf_good, nf_bad, sib_good, sib_bad):
        # distinct restaurant_ids by default; the sibling map is explicit below.
        _write(data / "traces" / (t["restaurant_id"] + ".json"), t)
    _write(data / "review" / "grounding.json", {
        "nf_good.json": _ground_entry(nf_good, siblings=["sib_good.json"]),
        "nf_bad.json": _ground_entry(nf_bad, siblings=["sib_bad.json"]),
        "sib_good.json": _ground_entry(sib_good, grounding=0.9),
        "sib_bad.json": _ground_entry(sib_bad, grounding=0.25, unmatched=["Dish 0", "Dish 1"]),
    })
    return TestClient(review.app)


def test_sibling_panel_shows_menu_and_grounding(sib_client):
    d = sib_client.get("/api/review/trace/nf_good.json").json()
    assert d["ok"] is True
    assert "siblings" in d and len(d["siblings"]) == 1
    sib = d["siblings"][0]
    assert sib["sibling_file"] == "sib_good.json"
    assert sib["grounding"] == 0.9
    # the sibling trace's menu is inlined (sections -> items)
    names = [it["name"] for sec in sib["menu"] for it in sec["items"]]
    assert names == ["Dish 0", "Dish 1", "Dish 2"]


@pytest.fixture
def pair_client(tmp_path, monkeypatch):
    """One restaurant with three slices sharing a restaurant_id: a found free slice,
    a found vegan slice, and a not-found slice. Every slice lists the other two as
    siblings (audit_grounding.py now emits siblings for EVERY trace)."""
    data = tmp_path / "data"
    monkeypatch.setattr(review, "DATA_DIR", data)

    def _slice(fn, found, restr, items):
        t = _trace(fn, found=found, items=items)
        t["restaurant_id"] = "acme"          # shared id -> siblings of each other
        t["restaurant_name"] = "Acme"
        t["dietary_restrictions"] = restr
        return t

    free = _slice("acme.json", True, None, 3)
    vegan = _slice("acme__vegan.json", True, ["vegan"], 1)
    nf = _slice("acme__nf.json", False, ["keto"], 0)
    for fn, t in [("acme.json", free), ("acme__vegan.json", vegan), ("acme__nf.json", nf)]:
        _write(data / "traces" / fn, t)
    _write(data / "review" / "grounding.json", {
        "acme.json": _ground_entry(free, grounding=1.0, siblings=["acme__nf.json", "acme__vegan.json"]),
        "acme__vegan.json": _ground_entry(vegan, grounding=1.0, siblings=["acme.json", "acme__nf.json"]),
        "acme__nf.json": _ground_entry(nf, siblings=["acme.json", "acme__vegan.json"]),
    })
    return TestClient(review.app)


def test_found_trace_shows_its_siblings(pair_client):
    # a FOUND trace now surfaces its sibling slices too (not just not-found traces)
    d = pair_client.get("/api/review/trace/acme.json").json()
    assert d["ok"] is True and d["found"] is True
    sibs = {s["sibling_file"]: s for s in d["siblings"]}
    assert set(sibs) == {"acme__vegan.json", "acme__nf.json"}
    # found sibling sorts before the not-found one
    assert d["siblings"][0]["found"] is True
    assert d["siblings"][-1]["sibling_file"] == "acme__nf.json"
    # the not-found sibling reports found=False and no grounding
    assert sibs["acme__nf.json"]["found"] is False
    assert sibs["acme__nf.json"]["grounding"] is None


def test_weakly_grounded_sibling_reports_unmatched(sib_client):
    d = sib_client.get("/api/review/trace/nf_bad.json").json()
    sib = d["siblings"][0]
    assert sib["sibling_file"] == "sib_bad.json"
    assert sib["grounding"] == 0.25
    assert sib["unmatched_items"] == ["Dish 0", "Dish 1"]


def _trace_with_scrape(name, *, found, menu, scrape_url, scrape_text):
    """A trace whose single scrape tool_result carries `scrape_text` (so item-name
    matching / full-text expansion have real content to work against)."""
    return {
        "restaurant_id": name.replace(".json", ""),
        "restaurant_name": name.replace(".json", "").title(),
        "episode_input": f"{name}, Seattle",
        "dietary_restrictions": None,
        "queries": ["q1"],
        "urls": [scrape_url],
        "final_json": {"found": found, "menu": menu, "notes": "n",
                       "source_url": scrape_url if found else None},
        "schema_valid": True,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": name}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "scrape_url",
                 "input": {"url": scrape_url}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": [{"type": "text", "text": scrape_text}]}]},
        ],
    }


@pytest.fixture
def match_client(tmp_path, monkeypatch):
    """A not-found trace vouched by a found sibling whose scrape text contains SOME
    of its menu item names (so we can assert per-item matched/context/source_hint
    and the primary/found trace's live %-grounded)."""
    data = tmp_path / "data"
    monkeypatch.setattr(review, "DATA_DIR", data)
    scrape = ("Welcome to Seoul Kitchen. Our famous Galbi Set is grilled short rib "
              "served with banchan. The Kimchi Jjigae is a spicy stew. " + "x" * 3000)
    menu = [{"section": "Mains", "items": [
        {"name": "Galbi Set", "description": "", "price": "$30"},
        {"name": "Kimchi Jjigae", "description": "", "price": "$16"},
        {"name": "Phantom Dish", "description": "", "price": "$9"},
    ]}]
    nf = _trace("nf.json", found=False)
    sib = _trace_with_scrape("sib.json", found=True, menu=menu,
                             scrape_url="https://seoulkitchen.example.com/menu",
                             scrape_text=scrape)
    _write(data / "traces" / "nf.json", nf)
    _write(data / "traces" / "sib.json", sib)
    _write(data / "review" / "grounding.json", {
        "nf.json": _ground_entry(nf, siblings=["sib.json"]),
        "sib.json": _ground_entry(sib, grounding=0.667, unmatched=["Phantom Dish"]),
    })
    return TestClient(review.app)


def test_found_trace_detail_grounding_is_computed_live(match_client):
    # sib.json: 2 of its 3 menu items (Galbi Set, Kimchi Jjigae) appear in the
    # scrape, Phantom Dish does not -> live grounding 2/3.
    d = match_client.get("/api/review/trace/sib.json").json()
    assert d["ok"] is True
    assert abs(d["grounding"] - 2 / 3) < 0.01


def test_sibling_items_carry_matched_and_context(match_client):
    d = match_client.get("/api/review/trace/nf.json").json()
    assert d["ok"] is True
    items = d["siblings"][0]["items"]
    by_name = {it["name"]: it for it in items}
    # matched item: context present, marked span wraps the exact name, source_hint set
    galbi = by_name["Galbi Set"]
    assert galbi["matched"] is True
    assert galbi["context"] is not None
    assert "〈Galbi Set〉" in galbi["context"]
    assert galbi["source_hint"] == "https://seoulkitchen.example.com/menu"
    assert by_name["Kimchi Jjigae"]["matched"] is True
    # unmatched item: matched False, no context
    phantom = by_name["Phantom Dish"]
    assert phantom["matched"] is False
    assert phantom["context"] is None


def test_full_tool_result_endpoint_returns_untruncated(match_client):
    d = match_client.get("/api/review/trace/sib.json").json()
    # locate the tool-result block's (turn, idx) address from the compact conversation
    tr = None
    for t in d["conversation"]:
        if t["tool_results"]:
            tr = t["tool_results"][0]
            break
    assert tr is not None and tr["truncated"] is True and tr["full_len"] > 800
    full = match_client.get(
        f"/api/review/toolresult/sib.json?turn={tr['turn']}&idx={tr['idx']}").json()
    assert full["ok"] is True
    assert full["full_len"] == tr["full_len"]
    assert len(full["text"]) == tr["full_len"] > 800
    assert full["text"].startswith("Welcome to Seoul Kitchen")


def test_full_tool_result_endpoint_path_traversal_guarded(match_client):
    # a filename with a path separator is rejected by _load_trace's guard, same as
    # the other endpoints -- returns {ok: False}, never reads outside traces/.
    assert match_client.get(
        "/api/review/toolresult/nope%5C..%5Csecret.json?turn=0&idx=0").json()["ok"] is False
    assert match_client.get(
        "/api/review/toolresult/missing.json?turn=0&idx=0").json()["ok"] is False
    # out-of-range turn/idx are rejected, not 500s
    assert match_client.get(
        "/api/review/toolresult/sib.json?turn=99&idx=0").json()["ok"] is False
    assert match_client.get(
        "/api/review/toolresult/sib.json?turn=2&idx=99").json()["ok"] is False


# --- rejecting a sibling directly from the evidence panel ---------------------

def test_sibling_reject_persists_and_shows_in_detail(sib_client, tmp_path):
    # POST a reject for the SIBLING's filename; it persists under that key and is
    # surfaced as the sibling's `decision` in the not-found's detail.siblings.
    r = sib_client.post("/api/review/decision",
                        json={"filename": "sib_good.json", "decision": "reject"})
    assert r.json()["ok"] and r.json()["decision"] == "reject"
    decisions = json.loads((tmp_path / "data" / "review" / "decisions.json").read_text())
    assert decisions["sib_good.json"]["decision"] == "reject"
    d = sib_client.get("/api/review/trace/nf_good.json").json()
    assert d["siblings"][0]["sibling_file"] == "sib_good.json"
    assert d["siblings"][0]["decision"] == "reject"


def test_export_includes_rejected_sibling_out_of_scope(sib_client, tmp_path):
    # sib_good is found=true (NOT in the not-found scope), but a reject on it must
    # still be written to reject_list.txt (export iterates the decisions map).
    sib_client.post("/api/review/decision",
                    json={"filename": "sib_good.json", "decision": "reject"})
    d = sib_client.post("/api/review/export").json()
    assert d["ok"] and d["count"] == 1
    text = Path(d["path"]).read_text()
    assert "sib_good.json" in text.splitlines()
    # and it truly is out of the not-found scope
    lst = sib_client.get("/api/review/traces?scope=notfound").json()
    assert "sib_good.json" not in [t["filename"] for t in lst["traces"]]


def test_sibling_reject_leaves_notfound_aggregate_unchanged(sib_client):
    # rejecting an out-of-scope sibling must not corrupt the not-found progress.
    before = sib_client.get("/api/review/traces?scope=notfound").json()["aggregate"]
    r = sib_client.post("/api/review/decision",
                        json={"filename": "sib_good.json", "decision": "reject"})
    # the write's returned aggregate is over the not-found scope, unchanged
    assert r.json()["aggregate"] == before
    after = sib_client.get("/api/review/traces?scope=notfound").json()["aggregate"]
    assert after == before == {"total": 2, "kept": 0, "rejected": 0, "undecided": 2}


def test_app_imports_without_gpu_stack():
    # Must run in a CLEAN interpreter. An absolute sys.modules assertion in the
    # shared pytest process is polluted by sibling test files (test_build_sft /
    # test_eval_split import transformers), which would make this fail for reasons
    # unrelated to viz.review. A fresh subprocess actually tests the claim: that
    # importing viz.review pulls in none of the GPU/model stack.
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
