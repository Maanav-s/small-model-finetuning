"""Model loading, isolated from the loop.

Kept separate so the CLI, a REPL/notebook, and the future SFT/eval scripts all
share one loader — and so importing schema/prompts/tools stays GPU-free.
"""

from __future__ import annotations

import os

# Reduce CUDA allocator fragmentation. This matters on this model: Gemma 4 E4B's
# head_dim=512 global layers force SDPA onto the math backend (no flash kernel,
# see CLAUDE.md), whose large O(seq^2) attention buffers fragment the pool and
# leave "reserved but unallocated" gaps that trigger OOM. Must be set before
# torch initializes its CUDA allocator, hence before the import below.
# setdefault so an explicit `PYTORCH_CUDA_ALLOC_CONF` export still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Single source of truth for the model id. E4B fits comfortably in bf16 (~9 GB)
# on a 23 GB card and is far better at tool calling than E2B (which would skip
# the search call and answer from memory).
MODEL_ID = "google/gemma-4-E4B-it"


def load_model(quantize: bool = False, attn: str = "sdpa"):
    """Load Gemma 4 pinned to GPU 0; return (model, tokenizer).

    quantize=True  -> 4-bit nf4 (~3.5 GB, fast load — good for dev on this
                      15 GB-host-RAM box).
    quantize=False -> bf16 (full quality; heavier load here, see CLAUDE.md).
    See CLAUDE.md for why device_map={"": 0} and attn='sdpa' (not FlashAttention).
    """
    assert torch.cuda.is_available(), "CUDA not available - check the torch install"

    load_kwargs = {
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "attn_implementation": attn,
    }
    if quantize:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    print(f"Loading {MODEL_ID} ({'4-bit' if quantize else 'bf16'}, attn={attn}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    print(f"Loaded on {model.device}, {torch.cuda.memory_allocated() / 1e9:.2f} GB VRAM")
    return model, tokenizer
