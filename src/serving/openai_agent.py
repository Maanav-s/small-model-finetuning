"""The agentic loop, OpenAI-compatible edition (vLLM-served models).

Drives any OpenAI-compatible chat endpoint -- a local vLLM server running a
Qwen3/Hermes-3 teacher, or any model whose vLLM tool-call parser surfaces
`tool_calls` -- through the SAME tools, system prompt, and JSON contract as the
Gemma (src/gemma/agent.py) and Claude (src/claude/claude_agent.py) loops. See
notes/vllm_inference.html.

The tool source is shared: setup_tools() in tools.py returns the same
`(tools, tool_registry, system_prompt)` regardless of which model drives it. The
only translation is the tool DECLARATION format -- to_openai_tools converts the
plain Python callables to OpenAI function-tool schema; the registry
(name -> callable -> str) is used as-is.

Like the other *_agent modules this is the reusable ENGINE: no CLI, no key
loading. The client is injected (build_client is a convenience for run_*.py /
eval_split). `openai` is imported lazily inside build_client so this module (and
to_openai_tools) import with no extra dependency.

NOTE ON THE GEMMA STUDENT: do NOT drive the fine-tuned Gemma student through this
chat/tool path -- vLLM ships no Gemma-4 tool-call parser, and re-templating risks
drift from the exact wire format it was SFT'd on. The student is served via the
raw /v1/completions path that keeps our own template + parser (see
src/gemma/agent.py generate_turn's vLLM branch). This module is for the teacher /
OpenAI-tool-parser models.
"""

from __future__ import annotations

import inspect
import json
import sys
import urllib.request
from pathlib import Path

# Shared modules (prompts/tools/schema) live in src/, the parent of this vllm/
# folder; put it on the path so the flat imports resolve (script-run convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import BUDGET_FINALIZE_INSTRUCTION  # noqa: E402

MAX_TOOL_CALLS = 8      # tool-call budget per episode (matches the other loops)
MAX_TOKENS = 16384      # the full menu JSON can be long

# Output-budget clamp (see run_episode). The whole trajectory accumulates in ONE
# chat prompt, so a tool-heavy episode can approach the served context window and
# `prompt + max_tokens` would 400. The student's completions path pre-counts its
# own rendered prompt (build_gemma_completions); the chat path can't (the server
# applies the template + tool schema), so we OVER-estimate the prompt and clamp,
# with an overflow-400 catch as the hard backstop.
_CTX_MARGIN = 1024          # tokens kept free beyond the estimate
_MIN_OUTPUT_TOKENS = 512    # always request at least this; a non-fitting 400 is caught

# Python annotation -> JSON Schema type, for converting the tool callables.
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _callable_to_openai(fn) -> dict:
    """Convert a plain Python tool function to an OpenAI function-tool declaration.

    Name from __name__, description from the docstring, parameters from the typed
    signature -- the same derivation as claude_agent._callable_to_anthropic, just a
    different envelope (`{"type":"function","function":{...,"parameters":...}}`).
    """
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        props[name] = {"type": _JSON_TYPES.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def to_openai_tools(tools: list) -> list[dict]:
    """Translate the setup_tools() callables into OpenAI tool declarations."""
    out = []
    for tool in tools:
        if not callable(tool):
            raise TypeError(f"Expected a callable tool, got: {tool!r}")
        out.append(_callable_to_openai(tool))
    return out


def build_client(base_url: str, api_key: str = "EMPTY",
                 timeout: float = 300.0, max_retries: int = 1):
    """Construct an OpenAI client pointed at a (local vLLM) server. Lazy import so
    the module stays dependency-free until a client is actually needed.

    `timeout` bounds a SINGLE request's wall-clock (default 5 min); `max_retries`
    the retries after it. This is the wall-time guard the token clamp cannot be: a
    degenerate/runaway generation (measured ~12 tok/s grinding a length-capped output
    for ~22 min on one restaurant) now fails its worker in minutes instead of pinning
    it, and build_corpus just records a failed episode (idempotent -> a later run
    retries it). Legitimate generations finish well under the timeout (the 32-episode
    build averaged well under a minute of model time per episode)."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


# Gemma student stop marker: generation stops right after a tool call so the model
# can't hallucinate its own tool response (matches agent.py's HF stop_strings).
_GEMMA_STOP = "<tool_call|>"


def _detect_max_model_len(client, model: str):
    """Ask the vLLM server what context window it's serving (None if unavailable)."""
    try:
        url = str(client.base_url).rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=10) as f:
            for m in json.load(f).get("data", []):
                if m.get("id") == model and m.get("max_model_len"):
                    return int(m["max_model_len"])
    except Exception:  # noqa: BLE001 -- detection is best-effort; clamping just turns off
        return None
    return None


def build_gemma_completions(client, model: str, max_tokens: int = 4096,
                            stop: str = _GEMMA_STOP, tokenizer=None,
                            max_model_len: int | None = None):
    """Return `generate(prompt_str) -> text` backed by vLLM /v1/completions, for the
    Gemma STUDENT (raw completions, NOT chat/tools -- vLLM has no Gemma tool parser).

    The prompt is rendered by the caller with the exact training chat template
    (agent.generate_turn), so training/inference stay byte-matched; vLLM only does
    fast batched decode. vLLM STRIPS the stop string from the returned text, so we
    re-append `<tool_call|>` when generation stopped on it -- otherwise
    tokenizer.parse_response wouldn't see a complete tool call.

    `skip_special_tokens=False` is LOAD-BEARING, not a tweak (verified on a served
    A100, 2026-07-16). Gemma's tool protocol IS special tokens (`<|tool_call>`,
    `<|"|>`, `<tool_call|>`), and vLLM's detokenizer defaults to skip_special_tokens
    =True, which deletes them. The failure is SILENT and looks like a bad model:

        skip_special_tokens=True  -> 'call:web_search{query:...}call:web_search{...}'
                                     stop_reason=None, finish='length'  (rambles to
                                     max_tokens; parse_response sees plain content,
                                     zero tool calls -> the agent loop never fires)
        skip_special_tokens=False -> '<|tool_call>call:web_search{query:<|"|>...<|"|>}'
                                     stop_reason='<tool_call|>', finish='stop'

    Note the stop string can't match either when the markers are stripped -- which is
    why this also breaks the re-append below. Same rule as the HF path (decode with
    skip_special_tokens=False); it just has to be requested over HTTP here.

    max_tokens IS CLAMPED to what actually fits (pass `tokenizer`; max_model_len is
    auto-detected from the server). vLLM enforces `prompt + max_tokens <= max_model_len`
    and 400s otherwise -- a failure mode the HF path does NOT have, because
    transformers' max_new_tokens never checks the total. Measured 2026-07-16: an
    agentic episode whose context grew to 36,865 tokens + a fixed 4096 request = 40,961
    against a 40,960 window -> `BadRequestError`, which eval_split swallows as a FAILED
    episode. So long (tool-heavy) episodes would silently depress the score rather than
    error loudly -- exactly the episodes where the model had gathered the most.
    """
    if max_model_len is None:
        max_model_len = _detect_max_model_len(client, model)

    def generate(prompt: str) -> str:
        eff_max = max_tokens
        if tokenizer is not None and max_model_len:
            n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            # -8 margin: our count and the server's can differ by a token or two.
            eff_max = min(max_tokens, max_model_len - n_prompt - 8)
            if eff_max < 1:
                # Prompt alone fills the window; nothing useful can be generated.
                return ""
        comp = client.completions.create(
            model=model, prompt=prompt, max_tokens=eff_max, temperature=0.0,
            stop=[stop],
            extra_body={"add_special_tokens": False, "skip_special_tokens": False},
        )
        choice = comp.choices[0]
        text = choice.text
        if getattr(choice, "stop_reason", None) == stop:
            text += stop  # restore the marker vLLM stripped, so parse_response is happy
        elif getattr(choice, "finish_reason", None) == "length":
            # NOT cosmetic. A truncated turn has no closing `<channel|>`, so the whole
            # thing parses as `thinking` with no content, and the caller hands the raw
            # reasoning back as if it were the answer -- the user sees "not valid JSON"
            # and no hint that the model simply ran out of room. Measured 2026-08-12:
            # a dietary-conditioned episode over a large menu reasoned past 4096 tokens
            # (it annotates every dish) and needed 6159 to finish.
            print(f"  [warn] generation hit the {eff_max}-token budget and was cut off "
                  f"mid-turn; this turn is incomplete (raise max_tokens)", flush=True)
        return text

    return generate


def _assistant_message(msg) -> dict:
    """Serialize an assistant response (content + any tool_calls) back into a
    plain message dict to echo into the next request. Uses getattr so it works
    with the SDK's pydantic objects and with test doubles alike."""
    out: dict = {"role": "assistant", "content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return out


def _estimate_prompt_tokens(messages: list[dict], oai_tools: list) -> int:
    """Conservative (over-)estimate of the server-side prompt token count from raw
    character length. Dividing by 3 deliberately UNDER-counts chars-per-token (the
    real ratio is ~3.3-4), so the derived output clamp errs toward staying inside
    the window; the leftover risk is caught by _is_context_overflow."""
    chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                chars += len(part.get("text", "")) if isinstance(part, dict) else len(str(part))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            chars += len(fn.get("name", "")) + len(fn.get("arguments", "") or "")
    if oai_tools:
        chars += len(json.dumps(oai_tools))
    return chars // 3 + 512  # +512 for chat-template / role scaffolding


def _is_context_overflow(exc: Exception) -> bool:
    """True only for a context-length 400 (prompt+output exceeds max_model_len); any
    other API error must propagate rather than be silently finalized."""
    s = str(exc).lower()
    return ("maximum context length" in s or "context_length_exceeded" in s
            or "reduce the length" in s or "input_tokens" in s)


def run_episode(
    client,
    model: str,
    restaurant_name: str,
    tools: list,
    tool_registry: dict,
    system_prompt: str,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_tokens: int = MAX_TOKENS,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """Run the tool-call loop for one restaurant; return (final_text, messages).

    verbose=True (default) prints a per-tool-call / per-result trace -- useful for a
    single interactive episode (smoke_teacher). Pass verbose=False for a batched
    corpus build, where 16-32 concurrent episodes would otherwise interleave a flood
    of these lines; build_corpus drives its own aggregate progress instead.

    Standard manual agentic loop over client.chat.completions.create: call the
    model, execute any tool_calls via the shared registry, feed tool results back
    (standalone role="tool" messages keyed by tool_call_id -- the OpenAI transport,
    no bundling), repeat until the model answers or the budget is spent. On the
    final (out-of-budget) turn, tools are dropped and BUDGET_FINALIZE_INSTRUCTION
    is injected so the model commits to JSON from what it gathered (a partial menu
    beats an empty reply -- matches the Gemma/Claude loops).

    Context safety: each call's max_tokens is clamped to the room left in the served
    window (_output_budget), and if a call still 400s on length the loop finalizes
    early from what it has instead of raising -- so a tool-heavy episode degrades to
    a partial menu rather than a lost trace (the failure mode that cost ~14% of the
    first vLLM-teacher build; see notes/experiments.md 2026-07-19).
    """
    oai_tools = to_openai_tools(tools)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": restaurant_name},
    ]
    # Clamp each call's output budget to what's left in the context window (the
    # trajectory grows in one prompt). Detected once; None -> no clamp (unknown window).
    max_model_len = _detect_max_model_len(client, model)

    def _output_budget(active_tools: list) -> int:
        if not max_model_len:
            return max_tokens
        room = max_model_len - _estimate_prompt_tokens(messages, active_tools) - _CTX_MARGIN
        return max(_MIN_OUTPUT_TOKENS, min(max_tokens, room))

    def _complete(active_tools: list | None):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=_output_budget(active_tools or []),
            temperature=0.0,
            tools=active_tools,
            tool_choice="auto" if active_tools else None,
        )

    for step in range(max_tool_calls + 1):
        out_of_budget = step == max_tool_calls
        if out_of_budget:
            messages.append({"role": "user", "content": BUDGET_FINALIZE_INSTRUCTION})

        try:
            resp = _complete(None if out_of_budget else oai_tools)
        except Exception as exc:  # noqa: BLE001 -- only a context 400 is handled; others re-raise
            if not _is_context_overflow(exc):
                raise
            # Context too full for a normal turn: finalize gracefully. Drop tools,
            # tell the model to emit JSON from what it already gathered, retry with a
            # clamped budget. A partial menu beats a crashed episode (same intent as
            # the out-of-budget path and the student's completions clamp).
            if not (messages and messages[-1].get("content") == BUDGET_FINALIZE_INSTRUCTION):
                messages.append({"role": "user", "content": BUDGET_FINALIZE_INSTRUCTION})
            try:
                resp = _complete(None)
            except Exception as exc2:  # noqa: BLE001
                if not _is_context_overflow(exc2):
                    raise
                return "", messages  # prompt alone overflows -> yield empty, never crash
            content = (resp.choices[0].message.content or "").strip()
            messages.append({"role": "assistant", "content": content})
            return content, messages

        msg = resp.choices[0].message

        if not getattr(msg, "tool_calls", None):
            messages.append({"role": "assistant", "content": msg.content or ""})
            return (msg.content or "").strip(), messages

        messages.append(_assistant_message(msg))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if verbose:
                print(f"  [step {step}] tool call: {name}({args})")
            if name not in tool_registry:
                out = f"Error: unknown tool {name!r}"
                if verbose:
                    print(f"  [warn] unknown tool {name!r}")
            else:
                try:
                    out = tool_registry[name](**args)
                    if verbose:
                        print(f"  [step {step}] -> {len(out)} chars returned")
                except Exception as e:  # noqa: BLE001 -- a bad call must not abort the episode
                    out = f"Error running tool {name!r}: {e}"
                    if verbose:
                        print(f"  [warn] {out}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": out}
            )

    return "", messages


if __name__ == "__main__":
    # Render-only demo: the OpenAI tool declarations built from the real
    # model-facing tools (over dummy backends). No server / key / network needed.
    from tools import build_model_tools  # noqa: E402

    tools, _ = build_model_tools(lambda query: "", lambda url, mode="direct": "")
    print(json.dumps(to_openai_tools(tools), indent=2))
