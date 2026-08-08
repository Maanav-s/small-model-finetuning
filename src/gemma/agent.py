"""The agentic loop: restaurant name -> menu JSON.

Drives Gemma 4 through a tool-call loop using the canonical function-calling API
(`tokenizer.parse_response`, bundled `tool_responses`) per
ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4.

This module is the reusable engine — it takes `model`/`tokenizer`/`tools` as
arguments and has no CLI or model-loading side effects, so you can drive it from
run_agent.py (the CLI) or a REPL/notebook for fast iteration (load the model
once, re-run episodes as you edit prompts.py / tools.py). Loading lives in
model.py, tools in tools.py, prompts in prompts.py, the JSON contract in schema.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Shared modules (schema/prompts/tools) live in src/, the parent of
# this gemma/ folder; put it on the path so flat imports resolve whether this
# file is run directly or imported by run_agent.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch is imported LAZILY, inside the HF-generate branch of generate_turn -- it is
# the only thing here that needs it (one torch.no_grad()). The vLLM path is
# tokenizer-only by design (model may be None; see CLAUDE.md "vLLM serving"), so a
# module-scope import would force a ~2.5 GB CUDA torch into serving-only environments
# that never load HF weights -- and did: it crashed the pod's eval client venv with
# ModuleNotFoundError before episode 1.
from prompts import BUDGET_FINALIZE_INSTRUCTION, SYSTEM_PROMPT  # noqa: E402

MAX_TOOL_CALLS = 8          # tool-call budget per episode (matches claude_agent.py)
MAX_NEW_TOKENS = 4096       # the full menu JSON can be long


def build_messages(restaurant_name: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Assemble the initial chat history for one extraction episode."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": restaurant_name},
    ]


def render_prompt(tokenizer, messages: list[dict], tools: list) -> str:
    """The exact prompt string a turn is generated from.

    Shared by generate_turn (the vLLM path sends this string) and parse_turn (newer
    transformers needs it as `prefix=`), so the text we generate from and the text we
    parse against can never drift.
    """
    return tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
        enable_thinking=True, tokenize=False,
    )


def parse_turn(tokenizer, text: str, prefix: str) -> dict:
    """tokenizer.parse_response, across the transformers versions we run on.

    transformers >=5.11 requires `prefix=` (the prompt that preceded generation) for
    new-style response templates, because the template pre-writes part of the
    assistant message that the parser must see. Without it, it raises -- and since
    run_episode treats a parse failure as "the model emitted garbage", EVERY turn
    fell into the recovery path: the model burned its whole tool budget being told
    its (perfectly valid) tool calls were unparseable, then returned "". That is
    indistinguishable from the v1 non-termination failure, which is exactly how it
    nearly got misread as a broken checkpoint. transformers 5.10 (the repo pin) has
    no `prefix` kwarg at all, hence the TypeError fallback.
    """
    try:
        return tokenizer.parse_response(text, prefix=prefix)
    except TypeError:
        return tokenizer.parse_response(text)


def generate_turn(model, tokenizer, messages: list[dict], tools: list,
                  vllm_generate=None, prompt: str | None = None) -> str:
    """Render `messages`, generate one model turn, return the decoded text.

    Keeps special tokens (skip_special_tokens=False) so parse_response can see
    the <|tool_call> markers.

    vllm_generate: optional `callable(prompt_str) -> generated_text`. When given,
    generation is delegated to a vLLM /v1/completions server (fast, concurrent --
    for eval/GRPO on the SAME rendered template, so training/inference stay matched;
    see notes/vllm_inference.html and openai_agent.build_gemma_completions). When
    None (default), the local HF model.generate path below is used unchanged. The
    callable must reproduce the HF stop behaviour: stop at "<tool_call|>" and return
    text that still ENDS with that marker so parse_response sees a complete call.
    """
    if vllm_generate is not None:
        return vllm_generate(prompt if prompt is not None
                             else render_prompt(tokenizer, messages, tools))

    import torch  # local HF generate path only -- see the note at the imports

    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    # The long-context OOM fix lives in model.py's load_model (it forces SDPA's
    # mem-efficient kernel on Gemma 4's head_dim=512 global layers by making
    # transformers expand GQA KV heads instead of passing enable_gqa). With that
    # in place, default SDPA dispatch already picks the efficient kernel here, so
    # no sdpa_kernel(...) override is needed -- an explicit one was measured to be
    # a no-op (the GQA broadcast, not the kernel-priority, was the disqualifier).
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            # Stop right after a tool call so the model can't hallucinate its
            # own tool response; normal turns still stop at EOS.
            stop_strings=["<tool_call|>"],
            tokenizer=tokenizer,
        )
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=False)


def run_episode(
    model,
    tokenizer,
    restaurant_name: str,
    tools: list,
    tool_registry: dict,
    system_prompt: str,
    vllm_generate=None,
) -> str:
    """Run the tool-call loop for one restaurant; return the final answer text.

    vllm_generate (optional): delegate generation to a vLLM completions server
    instead of the local HF model (see generate_turn). `model` may be None on that
    path -- only the tokenizer (for rendering) is needed.
    """
    messages = build_messages(restaurant_name, system_prompt=system_prompt)

    for step in range(MAX_TOOL_CALLS + 1):
        out_of_budget = step == MAX_TOOL_CALLS
        if out_of_budget:
            # Budget spent: tell the model to answer from what it has instead of
            # looping to "" on exhaustion. Delivered as a synthesized tool response
            # (NOT a user turn) so the template stays prefix-preserving for GRPO --
            # the same carrier the parse-failure recovery below uses (and for the
            # same reason). Tools stay declared, so we can't hard-block another
            # call, but the instruction reliably makes the model commit.
            name = next(iter(tool_registry), "web_search")
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": {}}}
                ],
                "tool_responses": [{"name": name, "response": BUDGET_FINALIZE_INSTRUCTION}],
            })

        # Render ONCE and reuse: the string we generate from is exactly the `prefix`
        # the parser needs (see parse_turn), so they cannot drift apart.
        prompt = render_prompt(tokenizer, messages, tools)
        text = generate_turn(model, tokenizer, messages, tools,
                             vllm_generate=vllm_generate, prompt=prompt)
        try:
            # {role, [thinking], content?, tool_calls?}
            parsed = parse_turn(tokenizer, text, prompt)
        except (ValueError, TypeError) as e:
            # Gemma's parser raises when a tool call's JSON arguments are
            # malformed - e.g. the model degenerated into a recursive/truncated
            # `jsonOptions.schema` blob. Don't crash the episode: feed the error
            # back and let the model retry within the remaining tool-call budget.
            #
            # Deliver it as a TOOL response, never a user turn: Gemma's template
            # strips reasoning from assistant turns before the last user message,
            # so a mid-episode user turn rewrites earlier tokens and breaks the
            # prefix-preservation GRPO rollouts rely on. A bare {"role": "tool"}
            # message won't work either - the template only renders a tool
            # response paired with an *open* assistant tool_call, so one appended
            # after our bundled turns is silently dropped (and with greedy
            # decoding the model would just re-emit the same bad output). So
            # synthesize a minimal assistant tool_call to carry the error; this
            # renders, adds context, and keeps the prefix intact (verified).
            print(f"  [step {step}] [warn] could not parse model turn: {e}")
            m = re.search(r"call:([A-Za-z_]\w*)", text)
            name = m.group(1) if m else next(iter(tool_registry), "web_search")
            error_msg = (
                "Your previous reply could not be parsed as a valid tool call. "
                "Emit either a single, well-formed tool call or the final JSON "
                "answer - do not nest or truncate tool arguments."
            )
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": {}}}
                ],
                "tool_responses": [{"name": name, "response": error_msg}],
            })
            continue
        tool_calls = parsed.get("tool_calls")

        if not tool_calls:
            return (parsed.get("content") or "").strip()  # final answer

        if out_of_budget:
            # Model tried to call a tool with no budget left (despite the finalize
            # instruction); take whatever text it produced rather than running the
            # call and falling through to "".
            print(f"  [warn] hit MAX_TOOL_CALLS={MAX_TOOL_CALLS} without a final answer")
            return (parsed.get("content") or "").strip()

        # Execute every call the model made this turn, collecting responses.
        tool_responses = []
        for tc in tool_calls:
            fn = tc["function"]
            name, args = fn["name"], fn["arguments"]
            print(f"  [step {step}] tool call: {name}({args})")
            if name not in tool_registry:
                print(f"  [warn] unknown tool {name!r}")
                response = f"Error: unknown tool {name!r}"
            else:
                response = tool_registry[name](**args)
                print(f"  [step {step}] -> {len(response)} chars returned")
            tool_responses.append({"name": name, "response": response})

        # Append the assistant turn (its tool_calls) + the tool results, bundled
        # in one message as Gemma 4 expects.
        messages.append({**parsed, "tool_responses": tool_responses})

    # Only reached if the final (out-of-budget) turn failed to parse; the normal
    # exhaustion path returns the model's best-effort text above.
    return ""


if __name__ == "__main__":
    # Render-only demo: confirms the system prompt and tool schema coexist in the
    # prompt the model receives. Tokenizer-only, so no GPU / weights needed -- the
    # tools are the real model-facing declarations over dummy backends (no key).
    from transformers import AutoTokenizer

    from model import MODEL_ID
    from tools import build_model_tools

    tools, _ = build_model_tools(lambda query: "", lambda url, mode="direct": "")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rendered = tok.apply_chat_template(
        build_messages("Joe's Pizza, New York"),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    print(rendered)
