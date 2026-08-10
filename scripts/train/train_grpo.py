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
import statistics
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


def split_probe(train_ds, probe_size: int):
    """Hold out a fixed probe set from the training prompts -> (train, probe).

    WHY A PROBE AT ALL (measured 2026-08-09 on the 25-step run): the training reward
    is logged over the prompts of the CURRENT step -- 2 restaurants -- so its
    step-to-step spread (sd 0.154) is dominated by which restaurants came up, not by
    the policy. Over 25 steps the mean drifted +0.068, i.e. 0.44 sd: indistinguishable
    from zero no matter what the policy did. The same 30 restaurants scored repeatedly
    hold that term FIXED, so a change in `eval_reward` is a change in the policy.

    Held out of training, not sampled from it: a prompt the policy has taken gradient
    steps on measures memorization as much as capability.

    STRIDE, not head/tail. build_grpo writes the restriction-FREE episodes first and
    the dietary-CONDITIONED ones after (see its docstring), so `rows[-30:]` would be
    an all-conditioned probe and `rows[:30]` an all-free one -- either would track a
    sub-population rather than the task. A stride keeps the file's free/conditioned mix.
    """
    n = len(train_ds)
    if probe_size <= 0 or probe_size >= n:
        return train_ds, None
    stride = n / probe_size
    probe_idx = sorted({min(n - 1, int(i * stride)) for i in range(probe_size)})
    keep_idx = [i for i in range(n) if i not in set(probe_idx)]
    return train_ds.select(keep_idx), train_ds.select(probe_idx)


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
# LoRA movement (live "is the policy actually changing?" metric)
# ---------------------------------------------------------------------------
# PEFT initializes lora_B to EXACTLY ZERO, so ||B|| is a direct odometer on training.
# The 2026-08-08 run only revealed its problem post-hoc, by pulling the adapter out of
# S3 (scripts/analysis/adapter_norms.py): 100 steps at lr 1e-6 moved median ||B|| to
# 0.0026 against the v2 SFT adapter's 0.41 -- 159x smaller, i.e. the policy was still
# ~its starting checkpoint, which is why reward was flat. That is knowable at step 20,
# not step 100, so it belongs on the live curve.
SFT_REFERENCE_B_NORM = 0.41  # v2 SFT adapter, same r=16/alpha=32, 258 modules


def lora_movement(model, scaling: float) -> dict[str, float]:
    """{metric: value} describing how far the LoRA has moved from its init.

    ||dW||_F is computed WITHOUT forming dW: with B (out x r) and A (r x in),
        ||B @ A||_F^2 = trace(A^T B^T B A) = trace((B^T B)(A A^T)) = sum(G_B * G_A)
    where both Gram matrices are only r x r. Forming dW would be a full
    out x in matrix per module (~134 MB fp32 for one Gemma MLP projection); this is
    two small matmuls instead, cheap enough to run every step.
    """
    import torch

    a_by, b_by = {}, {}
    for name, p in model.named_parameters():
        if ".lora_A" in name:
            a_by[name.replace(".lora_A", ".*")] = p
        elif ".lora_B" in name:
            b_by[name.replace(".lora_B", ".*")] = p

    b_norms, dw_norms = [], []
    with torch.no_grad():
        for key, b in b_by.items():
            a = a_by.get(key)
            if a is None:
                continue
            bf, af = b.detach().float(), a.detach().float()
            b_norms.append(bf.norm().item())
            g_b = bf.T @ bf                      # (r, r)
            g_a = af @ af.T                      # (r, r)
            dw_norms.append(scaling * (g_b * g_a).sum().clamp(min=0).sqrt().item())
    if not b_norms:
        return {}
    med = statistics.median(b_norms)
    return {
        "lora/b_norm_median": med,
        "lora/b_norm_max": max(b_norms),
        "lora/delta_w_norm_median": statistics.median(dw_norms),
        # >= ~0.05 means movement on the order of a real fine-tune; the flat 2026-08-08
        # run sat at 0.006 for its whole length.
        "lora/b_norm_vs_sft": med / SFT_REFERENCE_B_NORM,
    }


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
    # --- the fixed probe: the instrument the TRAINING reward curve cannot be ---------
    p.add_argument("--probe-size", type=int, default=30,
                   help="hold out N prompts (stride-sampled, so the free/conditioned mix is "
                        "preserved) and re-score them every --probe-every steps as TRL's "
                        "eval_dataset. Logged as eval_reward / eval_rewards/*. 0 disables. The "
                        "training reward averages only the CURRENT step's prompts, so at 2 "
                        "prompts/step its sd (0.154, measured 2026-08-09) swamps any policy "
                        "change; a fixed probe removes the restaurant term entirely.")
    p.add_argument("--probe-every", type=int, default=10,
                   help="run the probe every N optimizer steps (default 10)")
    p.add_argument("--probe-generations", type=int, default=2,
                   help="completions per probe prompt (TRL num_generations_eval). Deliberately "
                        "far below --num-generations: the probe wants a low-variance MEAN over "
                        "many distinct restaurants, not a within-group advantage, so samples are "
                        "better spent on prompts than on repeats. At G=16 an unset value would "
                        "make one probe 30x16=480 rollouts -- 15 training steps of work.")
    # GRPO / generation
    p.add_argument("--num-generations", type=int, default=8,
                   help="G: completions per prompt (group size). NOT the flat-curve culprit -- "
                        "the 2026-08-08 run's frac_reward_zero_std sat at 0.00-0.05 at G=8, so "
                        "groups were not degenerate.")
    p.add_argument("--max-completion-length", type=int, default=16384,
                   help="cap on the rollout completion (INCLUDING tool responses -- see "
                        "--max-tool-chars). Memory is LINEAR in this at ~3.7 MB/token (the "
                        "retained logits chain in the backward), so 16384 is the H200 141 GB "
                        "ceiling and 24576 needs a 180 GB B200. 8192 fits exactly ONE capped "
                        "scrape and is a SMOKE value only.")
    p.add_argument("--temperature", type=float, default=1.2,
                   help="rollout sampling temperature. Above 1.0 on purpose: the 2026-08-08 run "
                        "logged mean token entropy of 0.06 at temperature 1.0 -- a near-"
                        "deterministic policy, so the G samples in a group barely differ and the "
                        "reward spread within a group comes from environment luck rather than "
                        "policy variation. Watch `entropy` (should rise) against "
                        "`tools/failure_frequency` (malformed tool-call args if pushed too far).")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="KL coeff to the ref model (0 = no KL, common for GRPO)")
    p.add_argument("--max-tool-calls", type=int, default=6,
                   help="tool-call ITERATIONS per rollout (TRL max_tool_calling_iterations). "
                        "NOT the number of calls: Gemma emits several calls per turn, so the "
                        "2026-08-08 run averaged 15.5 calls (median 12, max 88) under a limit of "
                        "8 -- vs the teacher's 4.8 -- and that tool-output volume, not the "
                        "budget size, is what exhausted the completion window.")
    p.add_argument("--max-tool-chars", type=int, default=None,
                   help="read-time cap on each tool result (default: tools.GRPO_TOOL_CHARS = "
                        "16000, below the 24000 eval/corpus default). Tool text is appended INTO "
                        "the completion, so this competes directly with --max-completion-length. "
                        "Re-scoring the teacher corpus: 16000 keeps 94.3% of grounded items, "
                        "8000 keeps only 75% -- long pages are long because they carry menu.")
    # --- budget-truncation handling (both halves of the Path-A fix; see reward.py) ---
    p.add_argument("--mask-truncated-completions", action="store_true",
                   help="also drop budget-truncated rollouts' TOKENS from the loss (TRL/DAPO). "
                        "Usually redundant: --neutralize-truncated already gives them exactly "
                        "zero advantage, and TRL's masking only sees rollouts cut mid-stream, "
                        "not the dangling-tool-call abort (reward.py Path A).")
    p.add_argument("--no-neutralize-truncated", dest="neutralize_truncated",
                   action="store_false",
                   help="let truncated rollouts' 0.0 rewards enter their group's mean/std. "
                        "Default is to replace them with the group's terminated mean, so they "
                        "don't hand positive advantage to the sibling that answered early.")
    p.add_argument("--use-liger-kernel", action="store_true",
                   help="fused Liger GRPO loss: computes the loss from hidden states + lm_head "
                        "in chunks instead of materializing the bs x len x 262144 logits tensor "
                        "(the term that caps --max-completion-length at 16384 on an H200). "
                        "Needs `pip install liger-kernel` in the GRPO env; validate with a smoke "
                        "before a real run -- Gemma 4 coverage in Liger is unverified here.")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="1e-5, NOT TRL's 1e-6 default -- that default is for FULL fine-tuning; "
                        "only the low-rank factors move here. Measured 2026-08-09 on the "
                        "2026-08-08 run (100 steps @ 1e-6): median LoRA ||B|| reached 0.0026 "
                        "vs 0.41 for the v2 SFT adapter at the same r/alpha -- 159x smaller, "
                        "i.e. the policy was still ~its starting checkpoint, which is why the "
                        "reward curve was flat. Check with scripts/analysis/adapter_norms.py.")
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
    p.add_argument("--vllm-max-model-len", type=int, default=None,
                   help="cap the colocate engine's context (TRL vllm_max_model_length). "
                        "Gemma-4's default is 131072, and vLLM sizes its KV pool to serve "
                        "at least one request THAT long -- 2.18 GiB -- so a low "
                        "--vllm-gpu-memory-utilization fails the engine's own startup check "
                        "('estimated maximum model length is 75904') before it ever runs. A "
                        "rollout here needs prompt (~3K) + --max-completion-length, so "
                        "capping this near that sum makes the same KV budget hold ~4x more "
                        "concurrent sequences and frees the rest for the backward.")
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
    train_ds, probe_ds = split_probe(train_ds, args.probe_size)

    print("\n=== resolved GRPO config ===")
    print(f"  policy:                {args.model_path or 'base MODEL_ID/GEMMA_MODEL_PATH'}")
    print(f"  dataset:               {len(train_ds)} prompts from {args.data}")
    print(f"  probe (held out):      "
          + (f"{len(probe_ds)} prompts x {args.probe_generations} gens every "
             f"{args.probe_every} steps -> eval_reward"
             if probe_ds is not None else "disabled (--probe-size 0)"))
    print(f"  reward funcs:          {[f.__name__ for f in reward_funcs]}  weights={reward_weights}")
    print(f"  num_generations (G):   {args.num_generations}")
    print(f"  per_device_bs x accum: {per_device_bs} x {args.gradient_accumulation_steps}")
    print(f"  max_completion_length: {args.max_completion_length}")
    print(f"  temperature/top_p:     {args.temperature}/{args.top_p}   beta(KL): {args.beta}")
    print(f"  generation backend:    {'vLLM (' + args.vllm_mode + ')' if args.use_vllm else 'transformers'}")
    print(f"  tools:                 web_search/scrape_url, cache={args.cache_policy}@{args.cache_path}")
    print(f"  max_tool_calls:        {args.max_tool_calls} iterations "
          f"(Gemma emits several CALLS per iteration)")
    print(f"  max_tool_chars:        {args.max_tool_chars or 16000} "
          f"(eval/corpus use 24000; lower here -- tool text eats the completion budget)")
    print(f"  truncated rollouts:    mask={args.mask_truncated_completions} "
          f"neutralize={args.neutralize_truncated}   liger={args.use_liger_kernel}")
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

    # Feed lora_movement() into TRL's own metrics dict rather than calling wandb
    # directly: GRPOTrainer.log() averages self._metrics[mode] and merges it into `logs`
    # BEFORE dispatching to the reporters (grpo_trainer.py ~2648), so one append reaches
    # wandb, the console, AND trainer_state.json -- the last of which is what made the
    # 2026-08-08 post-mortem possible at all. Mutating `logs` from a callback's on_log
    # would NOT work: Trainer puts the reporting callbacks BEFORE user-supplied ones, so
    # wandb has already logged by the time ours runs.
    class LoraNormLogger(TrainerCallback):
        def __init__(self, scaling: float):
            self.scaling = scaling
            self.trainer = None  # set after the trainer exists (chicken-and-egg)

        def on_step_end(self, args_, state, control, model=None, **kwargs):
            if self.trainer is None or model is None:
                return
            try:
                stats = lora_movement(model, self.scaling)
            except Exception as e:  # never let a diagnostic kill a 15-hour run
                print(f"  [warn] lora_movement failed at step {state.global_step}: {e}",
                      flush=True)
                return
            metrics = self.trainer._metrics["train"]
            for k, v in stats.items():
                metrics[k].append(v)

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
    from tools import GRPO_TOOL_CHARS

    max_tool_chars = args.max_tool_chars or GRPO_TOOL_CHARS
    tools, _registry, _sys_prompt = setup_tools(dietary_restrictions=None, variant="student",
                                                cache=cache, async_tools=True,
                                                max_tool_chars=max_tool_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path or MODEL_ID)

    # ---- Do NOT "fix" eos to <turn|> here. It looks wrong on paper and is right in
    # practice (checked against the 2026-08-08 run's logs, 2026-08-09).
    # The paper argument: TRL detects truncation as `ids[-1] not in [tokenizer.eos_token_id,
    # pad_token_id]` (grpo_trainer.py 1785/1921), tokenizer.eos_token_id is 1 (<eos>) while
    # generation_config.eos_token_id is [1, 106, 50] and a Gemma TURN ends on 106 (<turn|>) --
    # which would flag every rollout truncated. It does not: under vLLM the rollouts end on
    # <eos>, and the run's own metrics prove the detector discriminates --
    # completions/clipped_ratio ranged 0.00-0.69 (mean 0.299) and tracked the independently
    # measured rate of answers cut mid-text (29.3% of 1680 logged completions), while
    # completions/mean_terminated_length stayed real (586-8074, never its zeros(1) fallback).
    # Setting eos to <turn|> would INVERT this: the ~71% of rollouts ending on <eos> would all
    # be flagged truncated, and with masking on that zeroes most of the batch.
    # Watch clipped_ratio: pinned at ~1.0 with mean_terminated_length at 0 is the signature
    # that this assumption has broken (e.g. a checkpoint whose generation_config differs).
    print(f"  eos/pad token ids:     {tokenizer.eos_token_id}/{tokenizer.pad_token_id} "
          f"(TRL's truncation test; watch completions/clipped_ratio for ~1.0)")
    # REBUILD the rewards with the tokenizer -- NOT optional for Gemma. TRL's
    # mid-episode parse_response loses the final answer's content (bundled tool
    # turns + a second thought span defeat the streaming parser; measured
    # 2026-08-08: 16/16 finals parsed to content='' -> every reward 0, zero
    # gradient). With the tokenizer, the reward decodes TRL's completion_ids kwarg
    # itself and reads menu + evidence off the raw wire text (reward.py path 2).
    # num_generations lets the reward neutralize budget-truncated rollouts against
    # their group (see reward.make_grpo_rewards) -- the half of the Path-A fix that
    # mask_truncated_completions cannot do, since TRL computes advantages over the
    # RAW group rewards before any masking.
    reward_funcs, reward_weights = make_grpo_rewards(
        tokenizer=tokenizer,
        num_generations=args.num_generations,
        neutralize_truncated=args.neutralize_truncated,
    )

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
        mask_truncated_completions=args.mask_truncated_completions,
        use_liger_kernel=args.use_liger_kernel,
        reward_weights=reward_weights,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        # The probe. TRL has no separate eval path -- prediction_step calls compute_loss,
        # so an eval_dataset produces REAL rollouts through the same tools and the same
        # reward funcs, logged under an `eval_` prefix (grpo_trainer.log). num_generations
        # _eval is what keeps it affordable; per_device_eval_batch_size must satisfy
        # (per_device_eval_batch_size * num_processes) % num_generations_eval == 0.
        **({"eval_strategy": "steps",
            "eval_steps": args.probe_every,
            "per_device_eval_batch_size": args.probe_generations,
            "num_generations_eval": args.probe_generations}
           if probe_ds is not None else {}),
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
        vllm_max_model_length=args.vllm_max_model_len,
        vllm_server_base_url=args.vllm_server_base_url,
    )

    lora_logger = LoraNormLogger(scaling=args.lora_alpha / args.lora_r)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=probe_ds,
        tools=tools,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[ToolFailureAbort(), lora_logger],
    )
    lora_logger.trainer = trainer  # _metrics only exists once the trainer is built
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
                "max_tool_chars": max_tool_chars,
                "mask_truncated_completions": args.mask_truncated_completions,
                "neutralize_truncated": args.neutralize_truncated,
                "use_liger_kernel": args.use_liger_kernel,
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
