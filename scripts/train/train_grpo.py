"""GRPO trainer: RL-finetune the Gemma menu student with TRL's tool-calling GRPO.

Phase 3. Continues from the SFT student (--model-path = the merged bf16 checkpoint;
defaults to the base model for a smoke) and optimizes it against the PURE-RL reward
in src/reward.py -- structure + found + grounding (faithfulness to the scraped
evidence), NO teacher reference (see reward.py / notes/experiments.md).

How the pieces line up with TRL 1.5.1's GRPOTrainer:
- **Rollout = TRL's native tool loop.** We pass the SAME web_search/scrape_url
  callables (tools=...) that eval uses; GRPOTrainer generates G completions per
  prompt, runs the tool calls itself, and APPENDS each tool result as a
  {"role":"tool","content":...} message INTO the completion (grpo_trainer.py
  ~1559). So the scraped evidence the grounding reward needs is already in
  `completions` -- make_grpo_rewards()'s reward callbacks read it out
  (reward._evidence_from_completion). No rollout_func needed.
- **Reward = make_grpo_rewards()** -> a list of reward functions (structure/found/
  grounding) + matching reward_weights; TRL sums them and logs each mean.
- **Tools must return REAL content** for grounding to work, so the cache miss policy
  is `live` (fetch on miss, hit when warm) -- NOT `canned` (a frozen miss returns a
  constant, leaving nothing to ground against). This makes rollouts scrape-bound
  (each step does G x tool-calls of live scraping); see notes/vllm_inference.html.
- **Dataset = scripts/datasets/build_grpo.py output** (prompt = student system+user;
  the reward is teacher-free so `reference` is ignored). Only `prompt` is fed to TRL.

Generation backend:
- Default (--use-vllm OFF) uses transformers generation -- correct but SLOW: GRPO does
  num_generations x batch agentic rollouts per step, and the HF path is single-GPU /
  not thread-safe (the same reason eval_split forces --workers 1; that made a 500-ep
  eval take 5.4 h). Reuses model.py's SDPA GQA patch so long contexts don't OOM.
- --use-vllm turns on TRL's vLLM path (colocate) for fast rollouts. The head_dim=512
  "serve gate" this used to warn about is a NON-ISSUE (vLLM handles Gemma-4's mixed
  heads natively; see CLAUDE.md "vLLM serving"). The real requirements are the CUDA-13
  host + the unified cu130 env + `merged-text` -- see CLAUDE.md's GRPO subsection.

On a successful run a `meta.json` lineage record (plan §6) is written next to the
adapter, with starting_checkpoint tying it back to the SFT run it continued from.

Run (CUDA-13 A100/H200, from repo root; NOTE the /opt/grpo env, not uv/.venv --
colocate needs vllm in-process with trl, which the repo's cu128 pin can't give):
  # smoke: does the loop run, rollouts parse tool calls, and rewards fire?
  /opt/grpo/bin/python scripts/train/train_grpo.py --data data/grpo/train.jsonl \
      --model-path /workspace/merged-text --use-vllm --vllm-mode colocate \
      --max-steps 2 --num-generations 4 --limit 8 --cache-policy canned \
      --output-dir /workspace/grpo-smoke
  # --model-path MUST be the backfilled text-only checkpoint (to_text_only.py --base):
  # TRL colocate loads vLLM from the PATH, so raw `merged` dies on AutoProcessor.

Dev box: --dry-run builds the dataset + resolved config + wires the reward and runs
it on one synthetic TRL-shaped completion, WITHOUT loading the model or training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# scripts/train/ is one level deeper than v1's scripts/, so parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "gemma"))

from run_meta import md5_file, write_run_meta  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Default student family (recorded in meta.json / the models/<family>/ layout).
DEFAULT_FAMILY = "gemma-4-e4b-it"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_grpo_dataset(path: str, limit: int | None):
    """build_grpo.py jsonl -> a HF Dataset with just the conversational `prompt`.

    The reward is teacher-free, so `reference`/`dietary_restrictions`/etc. are not
    needed at train time; keeping only `prompt` sidesteps pyarrow schema issues and
    makes clear nothing but the prompt drives the rollout.
    """
    from datasets import Dataset

    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        sys.exit(f"no rows loaded from {path}")
    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Model (mirror train_sft.load_base_model: bf16, SDPA GQA patch, no device_map)
# ---------------------------------------------------------------------------
def load_policy_model(model_path: str | None, attn: str = "sdpa"):
    """Load the policy: the merged SFT student (--model-path) or the base model.

    Reuses model.py's SDPA GQA patch (so Gemma 4's head_dim=512 layers use the
    mem-efficient kernel instead of OOMing on MATH during the long tool rollouts).
    No device_map -- accelerate/Trainer places it.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from model import MODEL_ID, _force_repeat_kv_for_efficient_sdpa

    if attn == "sdpa":
        _force_repeat_kv_for_efficient_sdpa()
    src = model_path or MODEL_ID
    print(f"Loading policy {src} (bf16, attn={attn}, no device_map) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, attn_implementation=attn
    )
    model.config.use_cache = False  # required with gradient_checkpointing
    # Gemma 4 is multimodal: max_position_embeddings lives on the text sub-config, but
    # TRL's tool loop reads config.max_position_embeddings at the top level -> expose it.
    if not hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = model.config.get_text_config().max_position_embeddings
    return model


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/grpo/train.jsonl",
                   help="scripts/datasets/build_grpo.py jsonl (student-agnostic prompts)")
    p.add_argument("--output-dir", default="/workspace/gemma-menu-grpo",
                   help="persistent (RunPod network volume) dir for the GRPO adapter + meta.json")
    p.add_argument("--model-path", default=None,
                   help="merged bf16 SFT student to start from (default: base model MODEL_ID / "
                        "GEMMA_MODEL_PATH). GRPO adds a fresh LoRA on top.")
    p.add_argument("--family", default=DEFAULT_FAMILY,
                   help=f"student family key recorded in meta.json / the models/<family>/ "
                        f"layout (default {DEFAULT_FAMILY})")
    p.add_argument("--run-id", default=None,
                   help="lineage run id recorded in meta.json (default: the --output-dir "
                        "basename). Use the models/<family>/grpo/<run-id>/ convention.")
    p.add_argument("--starting-checkpoint", default=None,
                   help="the SFT run-id this GRPO run initialized from (--model-path's lineage); "
                        "recorded in meta.json as starting_checkpoint (plan §6).")
    p.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    # GRPO / generation
    p.add_argument("--num-generations", type=int, default=4, help="G: completions per prompt (group size)")
    p.add_argument("--max-completion-length", type=int, default=8192,
                   help="cap on the rollout completion (incl. tool responses). Long scraped pages "
                        "make this memory-heavy -- keep small for smoke; raise on H200 for real runs.")
    p.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temperature (GRPO needs >0 for diversity)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="KL coeff to the ref model (0 = no KL, common for GRPO)")
    p.add_argument("--max-tool-calls", type=int, default=8, help="tool-call budget per rollout (TRL max_tool_calls)")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--per-device-train-batch-size", type=int, default=None,
                   help="(prompt,completion) pairs per device; defaults to --num-generations "
                        "(1 prompt/device/step). Must be a multiple of --num-generations.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="if >0, overrides epochs (use for smokes)")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--lr-scheduler-type", default="constant")
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-strategy", default="steps")
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--report-to", default="none",
                   help="'wandb' to log to Weights & Biases (needs `wandb login` or $WANDB_API_KEY "
                        "on the box, and `pip install wandb` -- it is NOT a repo dep). Set "
                        "$WANDB_PROJECT to name the project.")
    p.add_argument("--run-name", default=None,
                   help="run name for the tracker (wandb). Defaults to the output-dir name.")
    p.add_argument("--seed", type=int, default=42)
    # LoRA (scoped to language_model -- Gemma 4 is multimodal, see train_sft.py)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # tools / cache
    p.add_argument("--cache-policy", choices=["live", "canned", "error"], default="live",
                   help="grounding needs REAL scraped content -> 'live' (default). 'canned' would "
                        "starve grounding (a frozen miss returns a constant, nothing to ground on).")
    p.add_argument("--cache-path", default=str(ROOT / "data" / "cache.sqlite"))
    # vLLM
    p.add_argument("--use-vllm", action="store_true",
                   help="use TRL's vLLM rollout path (fast; colocate needs the unified cu130 env + "
                        "a merged-text --model-path -- see CLAUDE.md's GRPO subsection). Default OFF "
                        "= transformers gen.")
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"])
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--vllm-server-base-url", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="build dataset + config + wire/exercise the reward on a synthetic "
                        "TRL-shaped completion; do NOT load the model or train")
    return p


# ---------------------------------------------------------------------------
# Reward-plumbing self-check (runs in --dry-run; also the local proof the grounding
# evidence path matches TRL's tool-completion shape)
# ---------------------------------------------------------------------------
def _synthetic_tool_completion():
    """A completion shaped like TRL's conversational tool rollout: assistant
    tool-call turn -> {"role":"tool"} result (the scraped evidence) -> final
    assistant JSON answer. Grounding must read the tool message, NOT the answer."""
    menu = {
        "found": True, "restaurant_name": "Joe's", "cuisine": "Italian",
        "menu": [{"section": "Pizza", "items": [
            {"name": "Margherita Pizza", "description": None, "price": 12.0},
            {"name": "Marinara Pizza", "description": None, "price": 10.0}]}],
        "source_url": "https://joes.example/menu",
    }
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"type": "function", "function": {"name": "scrape_url", "arguments": {"url": "x"}}}]},
        {"role": "tool", "name": "scrape_url",
         "content": "Menu: Margherita Pizza $12, Marinara Pizza $10. Open daily."},
        {"role": "assistant", "content": json.dumps(menu)},
    ]


def _reward_selfcheck(reward_funcs, reward_weights) -> None:
    grounded = _synthetic_tool_completion()
    # a hallucinated variant: same shape, but the answer invents dishes absent from the tool msg
    hallucinated = list(grounded[:-1])
    bad_menu = json.loads(grounded[-1]["content"])
    bad_menu["menu"][0]["items"] = [
        {"name": "Dragon Roll", "description": None, "price": 9.0},
        {"name": "Wagyu Skewer", "description": None, "price": 20.0}]
    hallucinated = grounded[:-1] + [{"role": "assistant", "content": json.dumps(bad_menu)}]

    print("\n=== reward self-check (synthetic TRL-shaped completions) ===")
    for label, comp in (("grounded", grounded), ("hallucinated", hallucinated)):
        per_func = {f.__name__: f([comp])[0] for f in reward_funcs}
        total = sum(w * per_func[f.__name__] for f, w in zip(reward_funcs, reward_weights))
        print(f"  {label:13s} total={total:+.3f}  " +
              "  ".join(f"{k.replace('_reward','')}={v:+.3f}" for k, v in per_func.items()))
    print("  (grounded should be positive; hallucinated should be lower / negative)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_arg_parser().parse_args()
    # TRL's ACTUAL rule (grpo_config.py, num_generations): the EFFECTIVE batch
    # (num_processes * per_device_batch_size * gradient_accumulation_steps) must be
    # divisible by num_generations -- NOT per_device_batch_size itself. TRL generates the
    # whole generation batch and computes group-relative advantages over it, then chunks
    # only the forward/backward into per_device_bs micro-batches. So per_device_bs is a
    # pure MEMORY knob and a group may span accumulation steps.
    #
    # That matters a lot for Gemma 4: vocab_size=262144 makes the fp32 logits tensor
    # (bs x max_completion_length x vocab x 4B) enormous -- bs=4 @ 8192 is 34.4 GB, which
    # OOMs an 80 GB card in colocate (HF policy ~15 GB + vLLM's own weight copy + KV
    # cache are already resident). bs=2 halves it to 17.2 GB with NO algorithmic change
    # (G, max_completion_length untouched). Do NOT "fix" colocate OOM by cutting
    # --max-completion-length: that truncates the long tool-heavy rollouts specifically,
    # biasing training against the episodes that gathered the most evidence.
    per_device_bs = args.per_device_train_batch_size or args.num_generations
    _effective = per_device_bs * args.gradient_accumulation_steps  # x num_processes (1 here)
    if _effective % args.num_generations != 0:
        sys.exit(f"effective batch (per_device_bs {per_device_bs} x accum "
                 f"{args.gradient_accumulation_steps} = {_effective}) must be divisible by "
                 f"--num-generations ({args.num_generations})")

    from reward import make_grpo_rewards

    reward_funcs, reward_weights = make_grpo_rewards()
    train_ds = load_grpo_dataset(args.data, args.limit)

    print("\n=== resolved GRPO config ===")
    print(f"  policy:                {args.model_path or 'base MODEL_ID/GEMMA_MODEL_PATH'}")
    print(f"  dataset:               {len(train_ds)} prompts from {args.data}")
    print(f"  reward funcs:          {[f.__name__ for f in reward_funcs]}  weights={reward_weights}")
    print(f"  num_generations (G):   {args.num_generations}")
    print(f"  per_device_bs x accum: {per_device_bs} x {args.gradient_accumulation_steps}")
    print(f"  max_completion_length: {args.max_completion_length}")
    print(f"  temperature/top_p:     {args.temperature}/{args.top_p}   beta(KL): {args.beta}")
    print(f"  generation backend:    {'vLLM (' + args.vllm_mode + ')' if args.use_vllm else 'transformers'}")
    print(f"  tools:                 web_search/scrape_url, cache={args.cache_policy}@{args.cache_path}")
    print(f"  max_tool_calls:        {args.max_tool_calls}")
    print(f"  lr / epochs / steps:   {args.learning_rate} / {args.num_train_epochs} / {args.max_steps}")
    print(f"  output-dir:            {args.output_dir}")

    if args.dry_run:
        _reward_selfcheck(reward_funcs, reward_weights)
        print("\n[dry-run] built dataset + config + reward wiring only; not loading the model or training.")
        return

    # ---- real training ----
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # BRAVE_API_KEY for the live scrape/search tool backends

    from backends import preflight_browser
    from cache import Cache
    from peft import LoraConfig
    from tools import setup_tools
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from model import MODEL_ID

    # TRL's tool loop CATCHES tool exceptions (grpo_trainer.py ~1527 wraps sync calls
    # in `except Exception`; the async path gathers with return_exceptions=True) and
    # feeds `{"error": str(e)}` back as the tool message -- so backends'
    # BrowserDeadError CANNOT abort training from inside a tool call. What it does
    # instead is saturate TRL's `tools/failure_frequency` metric -- failed calls /
    # total calls, in [0,1] (grpo_trainer.py ~1804; the name is the same on 1.5.1
    # and 1.8.0 -- there is NO `tools/failure_rate`) -- (once the breaker trips,
    # every scrape raises instantly). This callback is the abort path: stop when the
    # failure rate stays saturated, before hundreds of steps train against
    # {"error": ...} rollouts. A healthy run sits near 0 -- scrape/site failures
    # return SENTINELS (never raise), so TRL-visible failures are only malformed
    # tool-call args and the breaker itself -- which puts the threshold far outside
    # normal noise. Stopping via `should_training_stop` is graceful: the in-flight
    # step finishes and the adapter save below still runs.
    TOOL_FAILURE_ABORT_RATE = 0.8
    TOOL_FAILURE_ABORT_LOGS = 3

    class ToolFailureAbort(TrainerCallback):
        def __init__(self):
            self.consecutive = 0

        def on_log(self, args_, state, control, logs=None, **kwargs):
            rate = (logs or {}).get("tools/failure_frequency")
            if rate is None:
                return
            self.consecutive = self.consecutive + 1 if rate >= TOOL_FAILURE_ABORT_RATE else 0
            if self.consecutive >= TOOL_FAILURE_ABORT_LOGS:
                print(f"\n[ABORT] tools/failure_frequency >= {TOOL_FAILURE_ABORT_RATE} for "
                      f"{self.consecutive} consecutive logging steps (step {state.global_step}): "
                      f"the tool stack is broken (dead local browser / BrowserDeadError), so "
                      f"rollouts are error text, not evidence. Stopping training; the adapter "
                      f"for the steps already trained is still saved.", flush=True)
                control.should_training_stop = True

    # Fail before the (expensive, GPU-holding) model load, not on rollout 1: under a
    # live/error cache policy the rollouts scrape through the local browser, and a
    # browser that cannot launch would turn every tool result into an infra sentinel
    # the reward then trains against. Same discipline as build_corpus/warm_cache; a
    # browser that dies MID-run is caught by backends' BrowserDeadError breaker.
    if args.cache_policy != "canned":  # canned replays only; it never launches a browser
        browser_error = preflight_browser()
        if browser_error:
            sys.exit(f"browser preflight failed, refusing to start:\n  {browser_error}")

    # Tools: real backends wrapped in the cache (live -> grounding sees real content).
    cache = Cache(args.cache_path, miss_policy=args.cache_policy)
    # async_tools=True is REQUIRED here, not a tuning knob: TRL runs sync tools inline
    # one-at-a-time and only asyncio.gather's coroutines, so sync tools serialize every
    # live scrape across the whole generation batch (measured >40 min/step, GPU at 0%).
    # See tools._to_async.
    tools, _registry, _sys_prompt = setup_tools(dietary_restrictions=None, variant="student",
                                                cache=cache, async_tools=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path or MODEL_ID)

    # Load the policy BEFORE building the LoRA config: how we scope LoRA depends on the
    # checkpoint's actual module tree, so inspect it rather than assume.
    model = load_policy_model(args.model_path, attn=args.attn)

    # LoRA scoping is checkpoint-shape-dependent (measured 2026-07-16):
    #  - MULTIMODAL (Gemma4ForConditionalGeneration -- the raw merged SFT ckpt): modules
    #    nest under `model.language_model.`, and the vision/audio towers use
    #    Gemma4ClippableLinear which PEFT CANNOT adapt -> must scope by regex to the
    #    language_model subtree (see train_sft.py).
    #  - TEXT-ONLY (Gemma4ForCausalLM -- to_text_only.py output, REQUIRED for --use-vllm
    #    since TRL colocate loads vLLM from the path): to_text_only DROPS the towers, so
    #    there is nothing to exclude and modules are plain `model.layers.N.`. The
    #    language_model regex then matches NOTHING and PEFT raises
    #    "Target modules ... not found in the base model". Target the names directly.
    _names = "|".join(re.escape(t) for t in args.lora_target_modules)
    _has_lm_nesting = any("language_model." in n for n, _ in model.named_modules())
    _targets = (rf".*language_model\..*\.({_names})$" if _has_lm_nesting
                else list(args.lora_target_modules))
    print(f"  LoRA targets:          {'language_model-scoped regex (multimodal)' if _has_lm_nesting else 'plain module names (text-only, no towers)'}")
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=_targets, bias="none", task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        max_tool_calling_iterations=args.max_tool_calls,
        reward_weights=reward_weights,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        log_completions=True,
        num_completions_to_print=2,
        report_to=args.report_to,
        run_name=args.run_name or Path(args.output_dir).name,
        seed=args.seed,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_base_url=args.vllm_server_base_url,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        tools=tools,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[ToolFailureAbort()],
    )
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.save_model(adapter_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(adapter_dir)
        # Lineage record next to the adapter (plan §6). starting_checkpoint ties this
        # GRPO run back to the SFT run it continued from (--model-path).
        prefix = os.environ.get("S3_PREFIX", "v2")
        meta_path = write_run_meta(
            args.output_dir,
            family=args.family,
            stage="grpo",
            run_id=args.run_id or Path(args.output_dir).name,
            base_ref={"path": f"{prefix}/models/{args.family}/base", "md5": None},
            starting_checkpoint=args.starting_checkpoint,
            dataset={"path": f"{prefix}/grpo/train.jsonl", "md5": md5_file(args.data)},
            # GRPO adds a fresh LoRA on a bf16 policy (load_policy_model) -- same bf16
            # serve-fidelity rule as SFT; not a training-time quantization.
            quant={"method": "lora", "quant_type": None, "compute_dtype": "bfloat16"},
            hyperparams={
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lr": args.learning_rate,
                "epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "num_generations": args.num_generations,
                "max_completion_length": args.max_completion_length,
                "beta": args.beta,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tool_calls": args.max_tool_calls,
                "per_device_train_batch_size": per_device_bs,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "seed": args.seed,
            },
            git_cwd=ROOT,
        )
        print(f"Saved run meta -> {meta_path}")
    print(f"Saved GRPO adapter -> {adapter_dir}")
    cache.close()


if __name__ == "__main__":
    main()
