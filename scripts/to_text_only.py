"""Convert a merged Gemma-4 multimodal checkpoint -> a text-only Gemma4ForCausalLM.

Why: vLLM registers BOTH Gemma4ForConditionalGeneration (gemma4_mm) and
Gemma4ForCausalLM (gemma4). The multimodal path demands a processor
(preprocessor_config.json) that our merged SFT checkpoint never saved (train_sft
saves model+tokenizer only), so `vllm serve` dies on AutoProcessor. We never use
the vision/audio towers anyway, so serve the text-only class: no processor needed,
and the towers' weights are dropped (smaller, less VRAM).

THE KV-SHARED-LAYER BACKFILL (--base), measured 2026-07-16:
Gemma-4 E4B sets `num_kv_shared_layers=18`, so its last 18 of 42 layers reuse K/V
from an earlier layer instead of computing their own. transformers honors that and
never *instantiates* k_norm/k_proj/v_proj for layers 24-41 -- so loading the base
checkpoint silently DROPS those 54 tensors, and every checkpoint we save downstream
(train_sft's merge, and this script) lacks them. That is invisible under transformers
(the params don't exist, so nothing reads them; missing=0 unexpected=0) and our HF
eval was unaffected -- they are genuinely dead weights.

vLLM disagrees: its Gemma4 builds a FUSED qkv_proj + k_norm for EVERY layer, then
hard-fails the load:

    ValueError: Following weights were not initialized from checkpoint:
    {'model.layers.24..41.self_attn.k_norm.weight', ...}

The shared layers discard the K/V they compute, so the VALUES are irrelevant -- but
the tensors must EXIST or vLLM won't start. `--base <base_model_dir>` copies them
straight out of the base safetensors. Do NOT instead disable vLLM's check
(`enable_weights_track`): that leaves those params as uninitialized memory rather
than an honest error.

`--base` does NOT need the full 16 GB base. Those 54 tensors are 110 MB, pre-extracted to
`s3://restaurant-menu-corpus/v1/base-model/kv-shared-backfill/` -- point --base at a dir
holding just that file (the backfill only globs *.safetensors):

    aws s3 cp --recursive s3://restaurant-menu-corpus/v1/base-model/kv-shared-backfill/ /w/kv/
    python to_text_only.py /w/merged /w/merged-text --base /w/kv

Usage: python to_text_only.py <src_dir> <dst_dir> [--base <base_model_dir>]
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

args = [a for a in sys.argv[1:]]
base_dir = None
if "--base" in args:
    i = args.index("--base")
    base_dir = args[i + 1]
    del args[i:i + 2]
src, dst = args[0], args[1]

print(f"loading {src} (bf16) ...", flush=True)
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, low_cpu_mem_usage=True)
print("top-level type:", type(m).__name__)
present = [a for a in ("language_model", "model", "lm_head", "vision_tower", "audio_tower")
           if hasattr(m, a)]
print("attrs present:", present)

# Find the text CausalLM. Different transformers versions nest this differently:
# some expose m.language_model as a *ForCausalLM, others put the backbone at
# m.model.language_model with the head at m.lm_head.
cand = getattr(m, "language_model", None)
if cand is None and hasattr(m, "model"):
    cand = getattr(m.model, "language_model", None)
print("language_model type:", type(cand).__name__ if cand is not None else None)

if cand is not None and type(cand).__name__.endswith("ForCausalLM"):
    text_model = cand
    print("-> language_model IS a ForCausalLM; saving it directly")
else:
    # Build a Gemma4ForCausalLM from the text sub-config and transplant weights.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(src)
    text_cfg = cfg.get_text_config()
    print("-> building Gemma4ForCausalLM from text_config")
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING  # noqa
    text_model = AutoModelForCausalLM.from_config(text_cfg)
    text_model = text_model.to(torch.bfloat16)
    # copy weights: strip the multimodal prefixes
    sd = m.state_dict()
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model.language_model."):
            new_sd["model." + k[len("model.language_model."):]] = v
        elif k.startswith("language_model."):
            new_sd[k[len("language_model."):]] = v
        elif k == "lm_head.weight":
            new_sd[k] = v
    missing, unexpected = text_model.load_state_dict(new_sd, strict=False)
    print(f"   loaded; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("   first missing:", missing[:5])

n_params = sum(p.numel() for p in text_model.parameters())
print(f"text model params: {n_params/1e9:.2f}B  arch={type(text_model).__name__}")

print(f"saving -> {dst}", flush=True)
text_model.save_pretrained(dst, safe_serialization=True)
AutoTokenizer.from_pretrained(src).save_pretrained(dst)

if base_dir:
    # Backfill the KV-shared layers' k_norm/k_proj/v_proj (see module docstring).
    # These params don't exist on `text_model`, so save_pretrained can't write them --
    # they have to be read RAW from the base safetensors and injected after the save.
    import glob
    import json
    import os

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from transformers import AutoConfig as _AC
    tcfg = _AC.from_pretrained(src).get_text_config()
    n_shared = getattr(tcfg, "num_kv_shared_layers", 0)
    if not n_shared:
        print("no num_kv_shared_layers in config; nothing to backfill")
    else:
        first = tcfg.num_hidden_layers - n_shared
        want = {f"model.layers.{i}.self_attn.{p}.weight"
                for i in range(first, tcfg.num_hidden_layers)
                for p in ("k_norm", "k_proj", "v_proj")}
        print(f"backfilling {len(want)} KV-shared tensors (layers {first}..{tcfg.num_hidden_layers-1}) "
              f"from {base_dir}", flush=True)

        # map text-only name -> the base checkpoint's multimodal name
        def base_name(k: str) -> str:
            return k.replace("model.layers.", "model.language_model.layers.")

        found = {}
        for shard in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
            with safe_open(shard, "pt") as f:
                have = set(f.keys())
                for k in want:
                    if (bk := base_name(k)) in have:
                        found[k] = f.get_tensor(bk).to(torch.bfloat16)
        missing_bf = want - set(found)
        if missing_bf:
            sys.exit(f"backfill FAILED: base lacks {len(missing_bf)}, e.g. {sorted(missing_bf)[:3]}")

        out = sorted(glob.glob(os.path.join(dst, "*.safetensors")))
        if len(out) != 1:
            sys.exit(f"expected 1 output shard to patch, found {len(out)}")
        sd = load_file(out[0])
        sd.update(found)
        save_file(sd, out[0], metadata={"format": "pt"})
        # keep the weight index (if any) consistent with what we just wrote
        idx = os.path.join(dst, "model.safetensors.index.json")
        if os.path.exists(idx):
            j = json.load(open(idx))
            for k in found:
                j["weight_map"][k] = os.path.basename(out[0])
            json.dump(j, open(idx, "w"), indent=2)
        print(f"   backfilled {len(found)}; shard now has {len(sd)} tensors")

print("DONE_TEXT_ONLY")
