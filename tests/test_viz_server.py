"""Tests for the extraction endpoint (viz/server.py).

No network, no model, no GPU: the engine, the tools, and both agent loops are
stubbed, so what is actually under test is the SEAM -- the calls server.py makes
into src/ and how it handles what comes back.

That seam is exactly what rotted. Both of these shipped broken and neither was
caught by a test, because nothing here had one:

  * `build_system_prompt(..., live=True)` -- v2 deleted the offline stub and with
    it the `live` argument, so EVERY extraction raised TypeError. Note the tests
    below call the REAL build_system_prompt (only the agent loops are stubbed),
    which is what makes a signature drift fail here instead of in production.
  * `answer = run_claude_episode(...)` -- that loop returns (text, messages), so
    the tuple went to extract_json and every Claude extraction "failed to parse".

    uv run python -m pytest tests/test_viz_server.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import viz.server as server  # noqa: E402

MENU = {
    "found": True,
    "restaurant_name": "Test Cafe",
    "sections": [{"name": "Mains", "items": [{"name": "Soup", "description": "", "price": "$5"}]}],
}


@pytest.fixture
def client(monkeypatch):
    """A TestClient that never starts the lifespan (so no model is ever loaded).

    TestClient only runs lifespan when used as a context manager; constructing it
    plainly leaves _ENGINE to us.
    """
    monkeypatch.setattr(server, "_ENGINE", {"model": object(), "tokenizer": object(),
                                            "anthropic_client": None})
    monkeypatch.setattr(server, "_get_tools", lambda: ([], {}))
    return TestClient(server.app)


def _post(client, **kw):
    body = {"query": "Test Cafe", "agent": "gemma"}
    body.update(kw)
    return client.post("/api/extract", json=body).json()


# --- the seam into src/ -------------------------------------------------------

@pytest.mark.parametrize("variant", ["teacher", "student"])
def test_extraction_builds_a_real_prompt_and_returns_the_menu(client, monkeypatch, variant):
    """The regression for `live=True`: the REAL build_system_prompt must accept
    what the endpoint passes, for both variants and with restrictions set."""
    seen = {}

    def fake_episode(model, tokenizer, name, tools, registry, system_prompt,
                     vllm_generate=None):
        seen["prompt"] = system_prompt
        return json.dumps(MENU)

    monkeypatch.setattr(server, "run_gemma_episode", fake_episode)
    data = _post(client, prompt_variant=variant, dietary="vegetarian")

    assert data["ok"] is True, data
    assert data["menu"]["restaurant_name"] == "Test Cafe"
    # the dietary restriction reached the prompt the loop was handed
    assert "vegetarian" in seen["prompt"]
    assert data["prompt_variant"] == variant


def test_claude_episode_return_value_is_unpacked(client, monkeypatch):
    """The regression for the tuple bug: run_episode returns (text, messages)."""
    monkeypatch.setitem(server._ENGINE, "anthropic_client", object())
    monkeypatch.setattr(
        server, "run_claude_episode",
        lambda *a, **kw: (json.dumps(MENU), [{"role": "assistant", "content": "..."}]),
    )
    data = _post(client, agent="claude")

    assert data["ok"] is True, data          # a tuple here would fail to parse
    assert data["menu"]["sections"][0]["items"][0]["name"] == "Soup"


# --- provenance: a screenshot must name the weights that produced it ----------

def test_response_names_the_checkpoint(client, monkeypatch):
    monkeypatch.setattr(server, "_CHECKPOINT", "sft-adapter")
    monkeypatch.setattr(server, "run_gemma_episode", lambda *a, **kw: json.dumps(MENU))
    assert _post(client)["checkpoint"] == "sft-adapter"


def test_claude_responses_carry_no_checkpoint(client, monkeypatch):
    """The checkpoint describes the LOCAL weights; Claude has none of ours."""
    monkeypatch.setitem(server._ENGINE, "anthropic_client", object())
    monkeypatch.setattr(server, "run_claude_episode", lambda *a, **kw: (json.dumps(MENU), []))
    assert "checkpoint" not in _post(client, agent="claude")


def test_config_defaults_to_the_variant_the_checkpoint_was_trained_under(client, monkeypatch):
    """An adapter is a distilled student -> student prompt. Bare base -> teacher.

    Serving a student under the teacher prompt is a train/serve mismatch that
    silently under-reports the model, so this default is load-bearing.
    """
    monkeypatch.setattr(server, "_ADAPTER", "models/sft-adapter")
    assert client.get("/api/config").json()["default_variant"] == "student"

    monkeypatch.setattr(server, "_ADAPTER", "")
    assert client.get("/api/config").json()["default_variant"] == "teacher"


# --- the vLLM backend ---------------------------------------------------------

def test_vllm_generate_is_passed_through_to_the_loop(client, monkeypatch):
    """On the vLLM path `model` is None and generation is delegated -- the loop is
    otherwise identical, so a dropped vllm_generate would look like 'model is None'
    deep inside agent.generate_turn rather than a config error."""
    seen = {}

    def fake_episode(model, tokenizer, name, tools, registry, system_prompt, vllm_generate=None):
        seen["model"] = model
        seen["vllm_generate"] = vllm_generate
        return json.dumps(MENU)

    monkeypatch.setitem(server._ENGINE, "model", None)
    monkeypatch.setitem(server._ENGINE, "vllm_generate", lambda prompt: "x")
    monkeypatch.setattr(server, "run_gemma_episode", fake_episode)

    assert _post(client)["ok"] is True
    assert seen["model"] is None
    assert callable(seen["vllm_generate"])


def test_vllm_plus_adapter_is_refused(monkeypatch):
    """Both set = the adapter silently does nothing, and the UI would label the
    served model as the adapter's. Fail loudly at startup instead."""
    monkeypatch.setattr(server, "_VLLM_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setattr(server, "_ADAPTER", "models/sft-adapter")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        server._load_vllm_backend()


def test_config_reports_the_backend_and_still_defaults_to_student(client, monkeypatch):
    monkeypatch.setattr(server, "_VLLM_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setattr(server, "_ADAPTER", "")
    cfg = client.get("/api/config").json()
    # vLLM here always serves a MERGED student, so the student prompt is still right
    assert cfg["backend"] == "vllm"
    assert cfg["default_variant"] == "student"
    assert cfg["quantized"] is False


def test_adapter_on_a_4bit_base_is_refused(monkeypatch):
    """The ~32-point trap. VIZ_QUANTIZE defaults ON, so the obvious launch command
    would otherwise be the broken one."""
    monkeypatch.setattr(server, "_VLLM_URL", "")
    monkeypatch.setattr(server, "_ADAPTER", "models/sft-adapter")
    monkeypatch.setattr(server, "_QUANTIZE", True)
    monkeypatch.setattr(server, "load_model", lambda **kw: (object(), object()))
    with pytest.raises(SystemExit, match="4-bit"):
        server._load_local_backend()


# --- input validation ---------------------------------------------------------

def test_unknown_variant_is_rejected_before_any_work(client, monkeypatch):
    monkeypatch.setattr(server, "run_gemma_episode",
                        lambda *a, **kw: pytest.fail("episode should not run"))
    data = _post(client, prompt_variant="nonsense")
    assert data["ok"] is False and "nonsense" in data["error"]


def test_claude_without_a_key_fails_cleanly(client, monkeypatch):
    monkeypatch.setitem(server._ENGINE, "anthropic_client", None)
    data = _post(client, agent="claude")
    assert data["ok"] is False and "ANTHROPIC_API_KEY" in data["error"]


def test_unparseable_model_output_reports_the_raw_text(client, monkeypatch):
    monkeypatch.setattr(server, "run_gemma_episode", lambda *a, **kw: "not json at all")
    data = _post(client)
    assert data["ok"] is False and data["raw"] == "not json at all"
