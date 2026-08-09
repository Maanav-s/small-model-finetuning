"""READ-ONLY: did a LoRA adapter actually MOVE? (the first check on a flat RL run)

PEFT initializes `lora_B` to EXACTLY ZERO, so B is a direct odometer on training:
at step 0 every ||B|| is 0, and the effective weight delta is

    dW = (lora_alpha / r) * B @ A

Nothing else about a run tells you this as cheaply. A GRPO run whose reward curve
is flat has two very different explanations -- "the policy moved and the reward
didn't follow" vs "the policy never moved" -- and they call for opposite fixes
(reward/rollout debugging vs. just a bigger learning rate). ||B|| separates them
before you spend another GPU-hour. On a run that trained normally the B norms are
comfortably nonzero and dW/W lands around 1e-3..1e-2; a median ||B|| at ~1e-4 or
below means the adapter is still ~the identity and the reward curve is not
evidence about anything else yet.

Also prints the adapter's TARGET MASK (which modules PEFT actually adapted), since
a run can silently no-op by targeting the wrong module names -- Gemma 4's tree is
`model.language_model.*` when multimodal and plain `model.layers.*` after
to_text_only.py, and train_grpo.py picks the regex per checkpoint shape.

  uv run python scripts/analysis/adapter_norms.py /workspace/grpo-run/adapter
  uv run python scripts/analysis/adapter_norms.py s3://restaurant-menu-corpus/v2/models/gemma-4-e4b-it/grpo/<run>/adapter
  uv run python scripts/analysis/adapter_norms.py <adapter> --base /workspace/merged-text   # adds dW/W

An s3:// source is copied to a temp dir with `aws s3 cp` (needs live creds:
`aws sts get-caller-identity` must succeed) and deleted afterwards. Writes nothing
anywhere else -- with --base it only READS the base checkpoint's tensors.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def fetch_if_s3(src: str) -> tuple[Path, Path | None]:
    """(local_dir, tempdir_to_clean). Pulls only the two adapter files from S3."""
    if not src.startswith("s3://"):
        p = Path(src)
        if not p.is_dir():
            sys.exit(f"not a directory: {p}")
        return p, None
    tmp = Path(tempfile.mkdtemp(prefix="adapter-norms-"))
    base = src.rstrip("/")
    for name in ADAPTER_FILES:
        cmd = ["aws", "s3", "cp", f"{base}/{name}", str(tmp / name)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            if name == "adapter_config.json":
                shutil.rmtree(tmp, ignore_errors=True)
                sys.exit(f"failed to fetch {base}/{name}\n  {r.stderr.strip()}\n"
                         f"  (expired creds? run `aws sts get-caller-identity` to check)")
            print(f"  [warn] no {name} at {base} ({r.stderr.strip()[:80]})")
    return tmp, tmp


def load_adapter(d: Path):
    from safetensors.torch import load_file

    cfg_path = d / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    st = d / "adapter_model.safetensors"
    if not st.exists():
        sys.exit(f"no adapter_model.safetensors in {d}")
    return cfg, load_file(str(st))


def pair_key(name: str) -> str:
    """'...q_proj.lora_A.weight' -> '...q_proj' so A and B pair up."""
    for marker in (".lora_A", ".lora_B"):
        if marker in name:
            return name.split(marker)[0]
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", help="adapter dir, local path or s3:// URI")
    ap.add_argument("--base", default=None,
                    help="base checkpoint dir; enables the relative ||dW||/||W|| column "
                         "(reads the base safetensors shards, writes nothing)")
    ap.add_argument("--top", type=int, default=10, help="how many modules to list (default 10)")
    ap.add_argument("--zero-threshold", type=float, default=1e-4,
                    help="||B|| at or below this counts as 'did not move' (default 1e-4)")
    args = ap.parse_args()

    import torch

    d, tmp = fetch_if_s3(args.adapter)
    try:
        cfg, sd = load_adapter(d)

        r = cfg.get("r")
        alpha = cfg.get("lora_alpha")
        scaling = (alpha / r) if (r and alpha) else None
        print("=== adapter config ===")
        print(f"  r / alpha / scaling:   {r} / {alpha} / {scaling}")
        print(f"  dropout:               {cfg.get('lora_dropout')}")
        print(f"  task_type:             {cfg.get('task_type')}")
        tm = cfg.get("target_modules")
        print(f"  target modules (mask): {tm if isinstance(tm, str) else sorted(tm or [])}")
        print(f"  base_model_name_or_path: {cfg.get('base_model_name_or_path')}")

        # --- pair A/B per adapted module ---
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in sd.items():
            if ".lora_A" in k or ".lora_B" in k:
                pairs.setdefault(pair_key(k), {})["B" if ".lora_B" in k else "A"] = v
        if not pairs:
            sys.exit("no lora_A/lora_B tensors found -- is this a LoRA adapter?")

        rows = []
        for mod, ab in pairs.items():
            A, B = ab.get("A"), ab.get("B")
            if A is None or B is None:
                continue
            Af, Bf = A.float(), B.float()
            bn = Bf.norm().item()
            dw = (scaling or 1.0) * (Bf @ Af).norm().item()
            rows.append({"module": mod, "B": bn, "A": Af.norm().item(), "dW": dw})
        rows.sort(key=lambda x: x["B"], reverse=True)

        b_norms = [x["B"] for x in rows]
        dead = sum(1 for x in b_norms if x <= args.zero_threshold)
        exact_zero = sum(1 for x in b_norms if x == 0.0)

        print(f"\n=== movement ({len(rows)} adapted modules) ===")
        print(f"  ||B||  median / max:   {statistics.median(b_norms):.6g} / {max(b_norms):.6g}")
        print(f"  ||B||  min:            {min(b_norms):.6g}")
        print(f"  exactly zero:          {exact_zero}/{len(rows)}  (PEFT init -- never updated)")
        print(f"  <= {args.zero_threshold:g}:              {dead}/{len(rows)}")
        if scaling:
            dws = [x["dW"] for x in rows]
            print(f"  ||dW|| median / max:   {statistics.median(dws):.6g} / {max(dws):.6g}")

        # --- optional: dW relative to the base weights it perturbs ---
        if args.base:
            base_dir = Path(args.base)
            if not base_dir.is_dir():
                sys.exit(f"--base is not a directory: {base_dir}")
            from safetensors import safe_open

            index = base_dir / "model.safetensors.index.json"
            shard_of: dict[str, str] = {}
            if index.exists():
                shard_of = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            else:
                for f in sorted(base_dir.glob("*.safetensors")):
                    with safe_open(str(f), framework="pt") as fh:
                        for k in fh.keys():
                            shard_of[k] = f.name

            def base_name(mod: str) -> str:
                # 'base_model.model.model.layers.0.self_attn.q_proj' -> 'model.layers...weight'
                m = mod.replace("base_model.model.", "")
                return f"{m}.weight"

            ratios = []
            cache: dict[str, object] = {}
            for row in rows:
                bn = base_name(row["module"])
                shard = shard_of.get(bn)
                if shard is None:
                    continue
                fh = cache.get(shard)
                if fh is None:
                    fh = safe_open(str(base_dir / shard), framework="pt")
                    cache[shard] = fh
                w = fh.get_tensor(bn).float().norm().item()
                if w > 0:
                    ratios.append(row["dW"] / w)
                    row["rel"] = row["dW"] / w
            if ratios:
                print(f"  ||dW||/||W|| median:   {statistics.median(ratios):.3e}  "
                      f"(max {max(ratios):.3e}, n={len(ratios)})")
            else:
                print("  ||dW||/||W||:          no base tensors matched the adapter module names")

        print(f"\n=== top {args.top} modules by ||B|| ===")
        for row in rows[: args.top]:
            rel = f"  dW/W={row['rel']:.3e}" if "rel" in row else ""
            print(f"  {row['B']:.6g}  dW={row['dW']:.6g}{rel}  {row['module']}")

        # --- the verdict this script exists to deliver ---
        print("\n=== verdict ===")
        med = statistics.median(b_norms)
        if exact_zero == len(rows):
            print("  ADAPTER NEVER UPDATED -- every ||B|| is exactly 0 (PEFT init). No gradient")
            print("  reached the adapter: check grad_norm, that the optimizer saw the LoRA params,")
            print("  and that the target mask above matched real modules.")
        elif med <= args.zero_threshold:
            print(f"  ADAPTER BARELY MOVED -- median ||B|| = {med:.3g}. The policy is still")
            print("  ~the SFT checkpoint, so a flat reward curve is EXPECTED and says nothing")
            print("  about group size, LoRA rank, or truncation. Raise the learning rate")
            print("  (1e-6 is a full-finetune default; LoRA usually wants ~1e-5) before")
            print("  re-diagnosing anything else.")
        else:
            # Absolute ||B|| means little on its own -- calibrate against a run that
            # demonstrably changed behaviour. Measured 2026-08-09 on this project's
            # v2 adapters (same r=16/alpha=32, same 258 modules):
            #   SFT (v2 student, clearly changed behaviour): median ||B|| = 0.41
            #   GRPO 100 steps @ lr 1e-6:                    median ||B|| = 0.0026
            print(f"  ADAPTER MOVED -- median ||B|| = {med:.3g}.")
            print(f"  Reference: the v2 SFT adapter (same r/alpha, a run that clearly changed")
            print(f"  behaviour) sits at ~0.41. This run is {0.41 / med:.0f}x smaller than that.")
            if med < 0.41 / 20:
                print("  At that ratio the policy is still ~the checkpoint it started from, so a")
                print("  flat reward curve is EXPECTED and is not yet evidence about group size,")
                print("  LoRA rank, or truncation. Raise the learning rate first (1e-6 is a")
                print("  full-finetune default; LoRA usually wants ~1e-5).")
            else:
                print("  That is a real change, so a flat reward is a real signal: look at")
                print("  frac_reward_zero_std, the per-term reward means, and the abort rate.")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
