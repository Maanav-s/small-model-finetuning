#!/usr/bin/env bash
# Serve a self-hosted vLLM TEACHER for the corpus build, ON a RunPod pod (or any
# CUDA-13 host). This is the exact recipe validated 2026-07-19: Qwen3-30B-A3B on
# 1xA100 (smoke) and Qwen3-235B-A22B-Instruct-2507-FP8 on 4xH100 (TP=4).
#
# The client side is already done -- build_corpus.py --teacher vllm and
# scripts/infra/smoke_teacher.py drive an OpenAI-compatible endpoint via
# src/serving/openai_agent.py. This script only stands up that endpoint.
#
# Provision the host with scripts/infra/runpod_create.py (it pins allowedCudaVersions
# so the driver is >=580 / CUDA 13 -- see that file + CLAUDE.md "vLLM serving"):
#   uv run python scripts/infra/runpod_create.py --name qwen-teacher-235b \
#       --gpu "NVIDIA H100 80GB HBM3" --gpu-count 4 --disk 400
# then scp this script over (or clone the repo) and run it on the pod:
#   bash serve_teacher.sh Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 4      # 235B, 4xH100
#   bash serve_teacher.sh Qwen/Qwen3-30B-A3B-Instruct-2507 1            # 30B validation
#
# Why these flags (validated, not guessed):
#   --tool-call-parser hermes   Qwen3 tool calls are Hermes-format; this is what makes
#                               vLLM surface `tool_calls` for openai_agent.run_episode.
#                               (No --reasoning-parser: the Instruct-2507 variants have
#                               no <think> blocks. Add `--reasoning-parser deepseek_r1`
#                               ONLY if you serve a *-Thinking-2507 variant instead.)
#   CUDA GRAPHS ARE ON (no --enforce-eager) -- measured 2026-07-21, worth ~10x.
#                               This script used to pass --enforce-eager on the theory
#                               that the teacher is "scrape-bound, not compute-bound".
#                               That theory was wrong by an order of magnitude: eager
#                               decode ran 9.5 tok/s/request, graphs run 94.0 -- and
#                               aggregate went 151 -> 1172 tok/s at 16 concurrent,
#                               2066 at 32. On a 1500-episode corpus build that is the
#                               difference between ~6 h and well under an hour. Graph
#                               capture costs 38 s and 3.93 GiB; it is not close.
#   --compilation-config ...    ...but graphs only capture with the allreduce_rms
#   --disable-custom-all-reduce fusion OFF. With it on, capture dies partway through
#                               with `Cuda error custom_all_reduce.cuh:455 'an illegal
#                               memory access was encountered'`, the engine exits, and
#                               the GPUs stay pinned by orphaned VLLM:: workers until
#                               killed. That crash is what made --enforce-eager look
#                               load-bearing; it is really just this one fusion. The
#                               two flags below were validated TOGETHER -- if you want
#                               to know whether --disable-custom-all-reduce alone is
#                               also needed, test it, do not assume.
#   head_dim is a NON-ISSUE on vLLM (that saga is HF-only; see CLAUDE.md).
#
# BLOCK-SCALE FP8 BACKEND (the load-bearing env below -- validated 2026-07-19):
# Qwen3-*-FP8 uses block-scale FP8. On vLLM 0.25.1 the default block-scale linear GEMM
# is FlashInfer (VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=1), but the runpod/pytorch image
# has NO FlashInfer cubin (VLLM_HAS_FLASHINFER_CUBIN=False), NO nvcc, and deep_gemm is
# not installed -- so /v1/models comes up HEALTHY but the FIRST real request crashes the
# engine with `RuntimeError: Assertion failed: !cubin.empty() || isPathValid(path_)`
# (flashinfer fp8_blockscale_gemm_sm90). Forcing the two flags off routes the dense FP8
# GEMM to compiled-in CUTLASS (no cubin/nvcc needed); the MoE uses Triton (ptxas+ninja,
# both present). Harmless on bf16 models (e.g. the 30B validation), so always set.
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_USE_DEEP_GEMM=0
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
TP="${2:-4}"
MAX_LEN="${3:-131072}"                   # 128K: headroom for tool-heavy episodes that
                                         # accumulate several scrapes (Qwen3-2507 supports
                                         # 262144 natively). vLLM sizes the KV pool from
                                         # free VRAM, not from this cap, so raising it does
                                         # not pre-allocate; only very long sequences use the
                                         # tail. Lower it if a tighter GPU fit is KV-starved.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"     # 0.95 validated with graphs on 4xH100; lower if OOM on load.
                                         # KV is the concurrency limit, not compute: at 0.95 the pool is
                                         # 12.22 GiB/GPU = 272,704 tokens TOTAL, so episodes carrying big
                                         # scraped contexts bound how many run at once. Watch the server's
                                         # "GPU KV cache usage" line before raising --workers past ~16.
SERVED_NAME="${SERVED_NAME:-teacher}"    # must match build_corpus --teacher-model / smoke --model
VENV="${VENV:-/opt/vllm}"
LOG="${LOG:-/workspace/vllm.log}"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "=== creating venv + installing vllm (separate venv; do NOT reuse a torch env) ==="
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip
  "$VENV/bin/pip" install -q vllm
fi
echo "vllm $("$VENV/bin/vllm" --version) | model=$MODEL tp=$TP max_len=$MAX_LEN util=$GPU_MEM_UTIL"
mkdir -p "$(dirname "$LOG")"

# nohup + setsid-free background with stdin detached so it survives the ssh session.
# env PATH=$VENV/bin:$PATH is belt-and-suspenders for Ampere's TRITON_ATTN ninja shell-out
# (CLAUDE.md); harmless on Hopper.
nohup env PATH="$VENV/bin:$PATH" "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --disable-custom-all-reduce \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  --host 0.0.0.0 --port 8000 </dev/null > "$LOG" 2>&1 &

echo "serve launched pid $! -> $LOG"
echo "poll:  curl -s http://localhost:8000/v1/models | python3 -m json.tool"
echo "smoke: /opt/client/bin/python scripts/infra/smoke_teacher.py --model $SERVED_NAME"
