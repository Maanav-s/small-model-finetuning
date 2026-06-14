"""Phase 1 agentic loop: restaurant name -> menu JSON.

Drives Gemma 4 (MODEL_ID from agent.py) through a tool-call loop, using the
canonical function-calling API (`tokenizer.parse_response`, bundled
`tool_responses`) per ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4.

The `web_search` tool is currently stubbed (returns sample_menu.md) so the loop
is deterministic and offline. Run with:
  uv run python src/run_agent.py
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from agent import MODEL_ID, TOOL_REGISTRY, TOOLS, build_messages

MAX_TOOL_CALLS = 4          # tool-call budget per episode (plan: 2-3 expected)
MAX_NEW_TOKENS = 2560       # the full menu JSON can be long

# ---------------------------------------------------------------------------
# CLI: 4-bit quantization is opt-in. On a card with enough VRAM, load the
# weights in bf16 (default); pass --quantize on small cards (e.g. 6 GB).
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--quantize",
    action="store_true",
    help="Load the model in 4-bit (nf4). Off by default; use on low-VRAM GPUs.",
)
cli_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load the model, pinned to GPU 0 (see CLAUDE.md for why not "auto").
# ---------------------------------------------------------------------------
assert torch.cuda.is_available(), "CUDA not available - check the torch install"

load_kwargs = {"device_map": {"": 0}, "low_cpu_mem_usage": True}
if cli_args.quantize:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
else:
    load_kwargs["dtype"] = torch.bfloat16

print(f"Loading {MODEL_ID} ({'4-bit' if cli_args.quantize else 'bf16'}) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
model.eval()
print(f"Loaded on {model.device}, {torch.cuda.memory_allocated() / 1e9:.2f} GB VRAM")


def generate_turn(messages: list[dict]) -> str:
    """Render `messages`, generate one model turn, return the decoded text.

    Keeps special tokens (skip_special_tokens=False) so parse_response can see
    the <|tool_call> markers.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

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


def run_episode(restaurant_name: str) -> str:
    """Run the tool-call loop for one restaurant; return the final answer text."""
    messages = build_messages(restaurant_name)

    for step in range(MAX_TOOL_CALLS + 1):
        text = generate_turn(messages)
        parsed = tokenizer.parse_response(text)  # {role, [thinking], content?, tool_calls?}
        tool_calls = parsed.get("tool_calls")

        if not tool_calls:
            return (parsed.get("content") or "").strip()  # final answer

        # Execute every call the model made this turn, collecting responses.
        tool_responses = []
        for tc in tool_calls:
            fn = tc["function"]
            name, args = fn["name"], fn["arguments"]
            print(f"  [step {step}] tool call: {name}({args})")
            if name not in TOOL_REGISTRY:
                print(f"  [warn] unknown tool {name!r}")
                response = f"Error: unknown tool {name!r}"
            else:
                response = TOOL_REGISTRY[name](**args)
                print(f"  [step {step}] -> {len(response)} chars returned")
            tool_responses.append({"name": name, "response": response})

        # Append the assistant turn (its tool_calls) + the tool results, bundled
        # in one message as Gemma 4 expects.
        messages.append({**parsed, "tool_responses": tool_responses})

    print(f"  [warn] hit MAX_TOOL_CALLS={MAX_TOOL_CALLS} without a final answer")
    return ""


def extract_json(text: str):
    """Best-effort: strip markdown fences and parse the answer as JSON."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


if __name__ == "__main__":
    RESTAURANT = "Pagliacci Pizza, Seattle"
    print(f"\n=== Episode: {RESTAURANT} ===")
    answer = run_episode(RESTAURANT)

    print("\n=== RAW FINAL ANSWER ===")
    print(answer)

    parsed, err = extract_json(answer)
    print("\n=== SCHEMA CHECK ===")
    if parsed is None:
        print(f"INVALID JSON: {err}")
    else:
        sections = parsed.get("menu", [])
        n_items = sum(len(s.get("items", [])) for s in sections)
        print(f"Valid JSON. restaurant_name={parsed.get('restaurant_name')!r}, "
              f"cuisine={parsed.get('cuisine')!r}, "
              f"{len(sections)} sections, {n_items} items")
