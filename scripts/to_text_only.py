"""Convert a merged Gemma-4 multimodal checkpoint -> a text-only Gemma4ForCausalLM.

Why: vLLM registers BOTH Gemma4ForConditionalGeneration (gemma4_mm) and
Gemma4ForCausalLM (gemma4). The multimodal path demands a processor
(preprocessor_config.json) that our merged SFT checkpoint never saved (train_sft
saves model+tokenizer only), so `vllm serve` dies on AutoProcessor. We never use
the vision/audio towers anyway, so serve the text-only class: no processor needed,
and the towers' weights are dropped (smaller, less VRAM).

Usage: python to_text_only.py <src_dir> <dst_dir>
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src, dst = sys.argv[1], sys.argv[2]

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
print("DONE_TEXT_ONLY")
