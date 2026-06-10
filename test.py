"""Smoke test: load google/gemma-4-E4B-it in 4-bit and run one generation.

Requires (one-time):
  - `huggingface-cli login` with a token from an account that has
    accepted the Gemma license at huggingface.co/google/gemma-4-E4B-it
Run with:
  uv run python test.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"

# 4-bit quantization so the model fits in the RTX 4050's 6 GB of VRAM.
# bf16 compute dtype matches what the GPU reported as supported.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading tokenizer for {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

assert torch.cuda.is_available(), "CUDA not available — check the torch install"

print(f"Loading model {MODEL_ID} (4-bit) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    # Pin the whole model to GPU 0. "auto" can silently offload to CPU on a
    # small card, which makes inference unusably slow.
    device_map={"": 0},
)
model.eval()

print(f"Model loaded on: {model.device}")
print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Quick sanity generation using the chat template.
messages = [{"role": "user", "content": "Hello there, who are you and what is your purpose?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)

# Only decode the newly generated tokens, not the prompt.
prompt_len = inputs["input_ids"].shape[-1]
response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
print("\n--- Model output ---")
print(response)
