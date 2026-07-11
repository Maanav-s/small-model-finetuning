"""WS-I: transform per-episode teacher traces -> a student-rendered SFT dataset.

The immutable capture files `data/traces/*.json` hold the TEACHER trajectory as
Anthropic content blocks (generated under the teacher system prompt). This script
is the one seam where the teacher->student context-distillation swap happens: it
re-renders each trajectory under the **student** system prompt through Gemma 4's
chat template and consolidates the lot into a single trainable
`data/sft/train.jsonl` (one example per line) that `scripts/train_sft.py` consumes.

Recipe (DECIDED -- see notes/phase2_plan.md Part 5, "Locked" + recipe 1):

  * ACTION-ONLY (recipe 1). All teacher reasoning/thinking is DROPPED. Each
    assistant turn keeps only its tool call(s) (bundled with their tool
    responses, the shape src/gemma/agent.py builds live); the final assistant
    turn is the cleaned final JSON answer -- no reasoning field, no visible
    rationale text.
  * STUDENT PROMPT, restriction preserved. The system message is
    build_system_prompt(dietary_restrictions=trace["dietary_restrictions"],
    variant="student") -- teacher guidance ABSENT, but the dietary restriction
    (a target-defining visible input, NOT distilled) PRESENT for conditioned
    traces.
  * KEEP found=false traces (anti-hallucination / abstention signal) under a
    ratio guard: --max-found-false-frac caps the found=false share; the excess is
    downsampled (seeded, deterministic) so abstention is never inflated.
  * CAP TOOL RESULTS to match inference: _slim_scrape (scrape only) + _cap /
    MAX_TOOL_CHARS from src/tools.py are applied to every tool RESULT before it
    enters the example, so training matches what the student sees at inference.

Every kept example is verified with a LOSSLESS ROUND-TRIP: the message list is
rendered via tokenizer.apply_chat_template and asserted to contain the same tool
calls (names + key args) and the same final JSON the source trace recorded, the
student prompt (teacher-only phrasing absent), and -- for conditioned traces --
the restriction text. A round-trip mismatch skips the trace (counted) rather than
writing a bad example.

  uv run python scripts/build_sft.py                 # data/traces/* -> data/sft/train.jsonl
  uv run python scripts/build_sft.py --limit 10      # quick smoke over the first 10 traces

Output contract (one JSON object per line in data/sft/train.jsonl):

  {
    "restaurant_id": "<rid>",
    "dietary_restrictions": null | ["vegetarian", "no peanuts"],
    "found": true | false,
    "messages": [ <Gemma-format message list> ],
    "meta": {"model": "<teacher model>", "cache_version": <int|str>,
             "prompt_variant": "student", "source_trace": "<filename>"}
  }

`messages` is EXACTLY the structure tokenizer.apply_chat_template(messages,
tools=TOOLS, tokenize=False) renders -- the shape src/gemma/agent.py builds live:
system(student prompt) -> user(episode input) -> zero+ bundled assistant tool-call
turns ({"role":"assistant","tool_calls":[...],"tool_responses":[...]}) -> final
assistant turn ({"role":"assistant","content": <compact final JSON string>}).
The tools are global/fixed and are NOT stored per row; train_sft.py imports the
two tool callables from src/tools.py and passes them to apply_chat_template.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Shared modules live in src/, the Gemma loader in src/gemma/ (flat-import,
# script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "gemma"))

from prompts import build_system_prompt  # noqa: E402
from schema import extract_json  # noqa: E402
from tools import (  # noqa: E402
    MAX_TOOL_CHARS,
    _cap,
    _slim_scrape,
    build_model_tools,
)

# The two model-facing tool callables (over dummy backends -- we only need their
# signatures + docstrings so apply_chat_template can render the declarations, and
# for round-trip verification). Global + fixed: train_sft.py rebuilds these the
# same way, so they are NOT stored per row.
TOOLS, _REGISTRY = build_model_tools(lambda query: "", lambda url, mode="direct": "")

# A phrase that appears ONLY in the teacher's _TEACHER_GUIDANCE block (source
# selection). Its ABSENCE from the rendered prompt proves the student variant.
_TEACHER_ONLY_PHRASE = "Source selection:"


# ---------------------------------------------------------------------------
# Pure-logic helpers (unit-testable without the tokenizer)
# ---------------------------------------------------------------------------
def transform_tool_result(name: str, text: str) -> str:
    """Cap a tool RESULT the way the student sees it at inference (tools.py).

    scrape_url -> slim + cap; anything else (web_search / error strings) -> cap.
    Idempotent: the traces already store the capped+slimmed model-facing text, so
    re-applying is a no-op there -- but we apply regardless so a raw result (or a
    retuned MAX_TOOL_CHARS) is handled the same way.
    """
    if name == "scrape_url":
        return _cap(_slim_scrape(text), "scrape_url")
    return _cap(text, "web_search")


def clean_final_answer(final_text: str):
    """Recover + compact the final JSON answer from the teacher's final turn.

    Uses schema.extract_json (drops leading narration / trailing commentary the
    teacher sometimes emits). Returns (compact_json_str, obj) or None if the final
    answer does not parse (the caller skips + counts that trace).
    """
    obj, _err = extract_json(final_text or "")
    if obj is None or not isinstance(obj, dict):
        return None
    compact = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return compact, obj


def found_false_keep_count(n_true: int, n_false: int, max_frac: float) -> int:
    """How many found=false examples to KEEP so the ratio guard holds.

    Returns n_false unchanged when the natural found=false share is already within
    max_frac; otherwise the largest count k with k / (n_true + k) <= max_frac,
    i.e. k <= n_true * max_frac / (1 - max_frac). max_frac >= 1 disables the guard.
    """
    total = n_true + n_false
    if n_false == 0 or total == 0:
        return n_false
    if max_frac >= 1.0:
        return n_false
    if n_false / total <= max_frac:
        return n_false
    if max_frac <= 0.0:
        return 0
    target = int(n_true * max_frac / (1.0 - max_frac))
    return min(n_false, target)


def load_reject_set(path: Path | None) -> set[str]:
    """Read a reject-list file: one restaurant_id or trace filename per line.

    Blank lines and `#` comments are ignored; a trailing `.json` is stripped so
    both `<rid>` and `<rid>.json` (and conditioned `<rid>__<slug>.json`) match.
    """
    if path is None:
        return set()
    reject: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reject.add(line[:-5] if line.endswith(".json") else line)
    return reject


def is_rejected(trace_filename: str, restaurant_id: str, reject: set[str]) -> bool:
    """True if this trace is on the reject list (by filename stem or restaurant id)."""
    if not reject:
        return False
    stem = trace_filename[:-5] if trace_filename.endswith(".json") else trace_filename
    return stem in reject or restaurant_id in reject


# ---------------------------------------------------------------------------
# Trace -> Gemma message list
# ---------------------------------------------------------------------------
def _text_of(content) -> str:
    """Join the text blocks of an Anthropic message content (str or block list)."""
    if isinstance(content, str):
        return content
    return "".join(
        b["text"] for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _tool_result_text(content) -> str:
    """The string body of a tool_result block (a str, or a list of text blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def build_gemma_messages(trace: dict) -> tuple[list[dict], str, dict]:
    """Re-render one trace as a Gemma-format message list under the STUDENT prompt.

    Returns (messages, final_answer_json_str, final_obj). Raises ValueError with a
    human-readable reason the caller turns into a per-trace skip + count.

    ACTION-ONLY: reasoning/thinking and pre-tool-call visible text are dropped;
    each assistant tool-call turn is the project's BUNDLED shape (its tool_calls
    AND their tool_responses in one message -- src/gemma/agent.py); the final turn
    is the cleaned compact JSON answer.
    """
    tmsgs = trace["messages"]
    if not tmsgs:
        raise ValueError("trace has no messages")
    last = tmsgs[-1]
    if last.get("role") != "assistant":
        raise ValueError("last message is not an assistant turn")

    # tool_use id -> tool name, so a tool_result (which carries only tool_use_id)
    # can be mapped to the right result transform (scrape slim+cap vs search cap).
    id_to_name: dict[str, str] = {}
    for m in tmsgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    id_to_name[b.get("id")] = b.get("name")

    cleaned = clean_final_answer(_text_of(last["content"]))
    if cleaned is None:
        raise ValueError("final answer did not parse as JSON")
    final_json_str, final_obj = cleaned

    # Student system prompt (teacher guidance dropped; dietary restriction kept).
    system_prompt = build_system_prompt(
        trace.get("dietary_restrictions"), variant="student"
    )
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    episode_input = _text_of(tmsgs[0]["content"]) if tmsgs[0].get("role") == "user" else None
    if not episode_input:
        raise ValueError("first message is not a user episode-input turn")
    messages.append({"role": "user", "content": episode_input})

    # Bundled assistant tool-call turns: every assistant turn EXCEPT the last that
    # issued tool calls, paired with the tool_results in the following user turn.
    for idx, m in enumerate(tmsgs[:-1]):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        tool_uses = [b for b in m["content"]
                     if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not tool_uses:
            continue  # a bare-text assistant turn before the end shouldn't exist; drop it

        results_by_id: dict[str, dict] = {}
        nxt = tmsgs[idx + 1] if idx + 1 < len(tmsgs) else None
        if nxt and nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
            for b in nxt["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results_by_id[b.get("tool_use_id")] = b

        tool_calls, tool_responses = [], []
        for tu in tool_uses:
            name = tu.get("name")
            args = tu.get("input") or {}
            tool_calls.append(
                {"type": "function", "function": {"name": name, "arguments": args}}
            )
            res = results_by_id.get(tu.get("id"))
            raw = _tool_result_text(res["content"]) if res is not None else ""
            tool_responses.append(
                {"name": name, "response": transform_tool_result(name, raw)}
            )
        messages.append(
            {"role": "assistant", "tool_calls": tool_calls, "tool_responses": tool_responses}
        )

    messages.append({"role": "assistant", "content": final_json_str})
    return messages, final_json_str, final_obj


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------
def verify_round_trip(tokenizer, messages: list[dict], final_json_str: str,
                      dietary_restrictions) -> None:
    """Render `messages` and assert the trajectory survived losslessly.

    Raises ValueError (a skip reason) if the rendered text is missing a tool call
    (name + key string args), the final JSON, the student prompt, or -- for a
    conditioned trace -- the restriction text; or if teacher-only guidance leaked.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=False, tokenize=False
    )

    # Student prompt present, teacher guidance absent.
    if _TEACHER_ONLY_PHRASE in rendered:
        raise ValueError(f"teacher-only guidance leaked into render ({_TEACHER_ONLY_PHRASE!r})")
    for restriction in (dietary_restrictions or []):
        if restriction not in rendered:
            raise ValueError(f"conditioned restriction {restriction!r} missing from render")

    # Every tool call (name + key string args) present.
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            fn = tc["function"]
            if fn["name"] not in rendered:
                raise ValueError(f"tool call name {fn['name']!r} missing from render")
            for val in (fn.get("arguments") or {}).values():
                if isinstance(val, str) and val and val not in rendered:
                    raise ValueError(
                        f"tool arg value {val[:40]!r} missing from render for {fn['name']}"
                    )

    # Final JSON answer present verbatim.
    if final_json_str not in rendered:
        raise ValueError("final JSON answer not found in render")


def token_length(tokenizer, messages: list[dict]) -> int:
    """Rendered token length (apply_chat_template(tokenize=True)) for max_seq_len sizing."""
    enc = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=False, tokenize=True, return_dict=True
    )
    return len(enc["input_ids"])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_tokenizer():
    """Load the Gemma tokenizer (tokenizer-only, no GPU). MODEL_ID from gemma/model.py.

    The model is gated, so this needs an HF login/cache (this dev box has it).
    Returns None if the tokenizer can't be loaded, so callers can decide.
    """
    from transformers import AutoTokenizer

    from model import MODEL_ID
    return AutoTokenizer.from_pretrained(MODEL_ID)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "data" / "traces")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "sft" / "train.jsonl")
    p.add_argument("--max-found-false-frac", type=float, default=0.2,
                   help="cap on the found=false share of the dataset; the excess is "
                        "downsampled (seeded). Default 0.2. >=1 disables the guard.")
    p.add_argument("--reject-list", type=Path, default=None,
                   help="optional file of restaurant_ids / trace filenames (one per line) to DROP")
    p.add_argument("--seed", type=int, default=42, help="downsample seed (default 42)")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N trace files (sorted) -- for quick tests")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = load_tokenizer()
    if tokenizer is None:
        sys.exit("could not load the Gemma tokenizer (gated -- needs an HF login/cache)")

    reject = load_reject_set(args.reject_list)
    trace_files = sorted(args.traces_dir.glob("*.json"))
    if args.limit is not None:
        trace_files = trace_files[: args.limit]
    if not trace_files:
        sys.exit(f"no traces found in {args.traces_dir}")

    kept: list[dict] = []          # {row, found, source_trace}
    skipped: dict[str, int] = {}   # reason -> count
    rejected = 0
    token_lengths: list[int] = []

    def skip(reason: str, name: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        print(f"  [skip] {name}: {reason}")

    for path in trace_files:
        name = path.name
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            skip(f"unreadable trace ({e})", name)
            continue

        rid = trace.get("restaurant_id", name[:-5])
        if is_rejected(name, rid, reject):
            rejected += 1
            print(f"  [reject] {name}: on reject list")
            continue

        try:
            messages, final_json_str, final_obj = build_gemma_messages(trace)
            verify_round_trip(
                tokenizer, messages, final_json_str, trace.get("dietary_restrictions")
            )
        except ValueError as e:
            skip(str(e), name)
            continue

        found = bool(final_obj.get("found"))
        row = {
            "restaurant_id": rid,
            "dietary_restrictions": trace.get("dietary_restrictions"),
            "found": found,
            "messages": messages,
            "meta": {
                "model": trace.get("model"),
                "cache_version": trace.get("cache_version"),
                "prompt_variant": "student",
                "source_trace": name,
            },
        }
        kept.append({"row": row, "found": found, "source_trace": name})
        token_lengths.append(token_length(tokenizer, messages))

    # --- found=false ratio guard (seeded downsample) ---------------------
    false_traces = sorted(k["source_trace"] for k in kept if not k["found"])
    n_true = sum(1 for k in kept if k["found"])
    n_false = len(false_traces)
    keep_false = found_false_keep_count(n_true, n_false, args.max_found_false_frac)
    dropped_false = n_false - keep_false
    if dropped_false > 0:
        shuffled = list(false_traces)
        random.Random(args.seed).shuffle(shuffled)
        drop_set = set(shuffled[keep_false:])
    else:
        drop_set = set()

    # --- write, preserving original (trace-sorted) order -----------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_written_true = n_written_false = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for k in kept:
            if k["source_trace"] in drop_set:
                continue
            fh.write(json.dumps(k["row"], ensure_ascii=False) + "\n")
            n_written += 1
            if k["found"]:
                n_written_true += 1
            else:
                n_written_false += 1

    # --- summary ---------------------------------------------------------
    print("\n===== build_sft summary =====")
    print(f"traces scanned : {len(trace_files)}")
    print(f"rejected (list): {rejected}")
    print(f"skipped        : {sum(skipped.values())}")
    for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {reason}")
    nat_false_frac = (n_false / len(kept)) if kept else 0.0
    print(f"round-tripped  : {len(kept)}  (found=true {n_true}, found=false {n_false}, "
          f"natural found=false frac {nat_false_frac:.3f})")
    if dropped_false > 0:
        print(f"downsampled    : dropped {dropped_false} found=false to hold the "
              f"<= {args.max_found_false_frac:.2f} guard (seed {args.seed})")
    written_false_frac = (n_written_false / n_written) if n_written else 0.0
    print(f"written        : {n_written} -> {args.out}")
    print(f"    found=true  : {n_written_true}")
    print(f"    found=false : {n_written_false}  (frac {written_false_frac:.3f})")
    if token_lengths:
        kept_lengths = [tl for k, tl in zip(kept, token_lengths)
                        if k["source_trace"] not in drop_set]
        srt = sorted(kept_lengths)
        mean = sum(srt) / len(srt)
        p50 = srt[len(srt) // 2]
        p95 = srt[min(len(srt) - 1, int(0.95 * len(srt)))]
        print(f"token length   : mean {mean:.0f}, min {srt[0]}, p50 {p50}, "
              f"p95 {p95}, max {srt[-1]}  (rendered, tools included)")


if __name__ == "__main__":
    main()
