#!/usr/bin/env bash
# GRPO run on ONE big-VRAM RunPod pod (H200 141 GB / B200 180 GB), vLLM colocate.
#
# Run it PHASE BY PHASE (each phase is idempotent and safe to re-run):
#
#   bash scripts/infra/grpo_pod.sh setup       # /opt/grpo env, chromium, corpus + warm cache + dataset
#   bash scripts/infra/grpo_pod.sh prep-model  # pull base + SFT merged, build merged-text
#   bash scripts/infra/grpo_pod.sh smoke       # 2 steps, canned cache -- the gate before spending
#   bash scripts/infra/grpo_pod.sh train       # the real run, in tmux, with a checkpoint->S3 pusher
#   bash scripts/infra/grpo_pod.sh watch       # tail the log
#   bash scripts/infra/grpo_pod.sh norms       # ||B|| on the latest checkpoint (the flat-curve check)
#   bash scripts/infra/grpo_pod.sh finish      # final push of adapter + cache to S3
#
# WHY ONE UNIFIED /opt/grpo ENV AND NOT THE eval_pod SPLIT: eval talks to a vLLM
# *server* over HTTP, so vllm can live in its own venv. TRL's GRPO **colocate** mode
# imports vllm IN-PROCESS alongside the trainer, so vllm + torch + trl must share one
# env. Build it vLLM-first so it pins the stack (torch 2.11+cu130), then add the rest.
#
# WHY merged-text AND NOT merged: TRL colocate builds the vLLM engine from the model
# PATH on disk, so both `vllm serve` gotchas apply verbatim -- raw `merged` has no
# preprocessor_config.json (dies in AutoProcessor), and Gemma-4 E4B's num_kv_shared_layers
# =18 means transformers silently drops 54 k_norm/k_proj/v_proj tensors that vLLM builds
# and hard-requires. to_text_only.py --base backfills them. See CLAUDE.md.
#
# ENV YOU MUST EXPORT ON THE POD BEFORE `setup` (see .env.example / notes/S3_setup.md):
#   BRAVE_API_KEY   -- search backend (scrape is local, no key)
#   WANDB_API_KEY, WANDB_ENTITY -- omit and the run degrades to console-only logging
#   S3_BUCKET, S3_PREFIX=v2, AWS_DEFAULT_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# HF_TOKEN is NOT needed: every checkpoint (incl. its tokenizer) comes from S3.
set -euo pipefail

REPO="${REPO:-/workspace/small-model-finetuning}"
REPO_URL="${REPO_URL:-https://github.com/Maanav-s/small-model-finetuning.git}"
BRANCH="${BRANCH:-grpo-run2}"
WORK="${WORK:-/workspace}"
VENV="${VENV:-/opt/grpo}"
PY="$VENV/bin/python"
AWS="$VENV/bin/aws"          # awscli is pip-installed into $VENV (no apt candidate on this image)

# THE VENV'S bin/ MUST BE ON PATH, and it is not enough to call $VENV/bin/python by
# absolute path -- that does not activate the venv. vLLM JIT-compiles attention kernels
# and shells out to `ninja` BY NAME, so the engine dies LATE (after the 15 GB policy
# load and the vLLM weight load) with a bare `FileNotFoundError: 'ninja'` even though
# ninja is installed right next to the vllm binary. CLAUDE.md documents this for
# Ampere/TRITON_ATTN; measured 2026-08-09 it bites on Blackwell too, via a different
# JIT -- flashinfer's trtllm_gen_fmha module (jit/cpp_ext.py run_ninja).
# nvcc comes from the pip nvidia-cu13 wheel, not the image (which ships only CUDA 12.8),
# so CUDA_HOME has to point INTO site-packages or flashinfer's JIT finds the wrong one.
CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

# GRPO rollouts are VARIABLE length by nature -- every completion is a different size,
# and the probe adds a second, shorter distribution on top -- which is exactly the case
# the caching allocator fragments on. Measured 2026-08-10, the 2-GPU smoke at MAXLEN
# 24576 OOM'd on rank1 asking for 20.9 GiB with 3.6 GiB free while holding **51.3 GiB
# reserved but unallocated**: a third of the card was stranded in size-mismatched
# segments, not actually in use. expandable_segments lets the allocator grow one
# segment instead of rounding each new shape up into a fresh block.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

S3_MODELS="s3://${S3_BUCKET:-restaurant-menu-corpus}/${S3_PREFIX:-v2}/models/gemma-4-e4b-it"
KV_DIR="$WORK/kv"                # ONLY the 54 kv-shared tensors (105 MB), not the 16 GB base
MERGED="$WORK/merged"            # the SFT student, merged bf16
MERGED_TEXT="$WORK/merged-text"  # what BOTH the trainer and vLLM actually load

# ---- the run config ------------------------------------------------------------
RUN_NAME="${RUN_NAME:-gemma-menu-grpo-v2}"
OUT="${OUT:-$WORK/$RUN_NAME}"
NPROC="${NPROC:-2}"              # GPUs. DDP data-parallel: each rank runs its own policy
                                 # AND its own colocate vLLM, so per-GPU memory is
                                 # UNCHANGED by adding ranks -- the second GPU buys
                                 # rollout throughput, not headroom for a longer MAXLEN.
# G=16, doubled from 8. G was never the flat-curve cause (frac_reward_zero_std sat at
# 0.00-0.05 at G=8, so groups were not degenerate) -- this is about the QUALITY of the
# advantage estimate, whose noise falls as 1/sqrt(G). The second GPU pays for it:
# NPROC*BS*ACCUM = 2*1*16 = 32 completions/step = 2 prompts at G=16, so prompt
# diversity per step is held at the 1-GPU value instead of being halved.
G="${G:-16}"
# 24576, up from 16384. On an H200 141 GB this OOM'd in accelerator.backward() at ANY
# vllm util (~85 GB of retained logits chain + 15 GB policy); it fits on a 180 GB B200.
# 32768 does NOT fit anywhere available: the chain is LINEAR in length at ~3.7 MB/token,
# so 32768 -> ~120 GB + ~29 GB of everything-else + vLLM's share > 180.
MAXLEN="${MAXLEN:-24576}"
BS="${BS:-1}"                    # pure memory knob; the group may span accumulation steps
ACCUM="${ACCUM:-16}"             # NPROC*BS*ACCUM = 32 must be divisible by G
PROBE_SIZE="${PROBE_SIZE:-30}"   # held out of training; the only trend instrument that
PROBE_EVERY="${PROBE_EVERY:-10}" # is not swamped by which restaurants a step drew
PROBE_GENS="${PROBE_GENS:-2}"    # 30x2=60 rollouts/probe (~2 steps of work), not 30x16
LR="${LR:-1e-5}"                 # NOT 1e-6: see the commit that raised the default
TEMP="${TEMP:-1.2}"              # entropy was 0.06 at 1.0 -- a near-deterministic policy
TOOL_CALLS="${TOOL_CALLS:-6}"
TOOL_CHARS="${TOOL_CHARS:-16000}"
# 0.12, down from the 1-GPU run's 0.18. The 2-GPU smoke peaked at 182.0 of 183.4 GB --
# 1.3 GB of headroom, which over 150 steps is an OOM waiting for one unusually long
# batch. Margin is bought HERE rather than from MAXLEN because the KV cache is pure
# rollout concurrency (0.12 x 183 = ~22 GB still holds 16 sequences comfortably),
# whereas cutting MAXLEN would truncate the tool-heavy rollouts specifically -- biasing
# training against exactly the episodes that gathered the most evidence.
VLLM_UTIL="${VLLM_UTIL:-0.12}"
SAVE_STEPS="${SAVE_STEPS:-25}"
# CAP IT. A full epoch is ~436 steps; at the measured ~580 s/step that is 70 h, and a
# 2-GPU pod costs ~2x/hr. 150 steps (~25 h) is 300 prompts and 15 probe points -- enough
# to see a trend in eval_reward, which is the question this run exists to answer.
MAX_STEPS="${MAX_STEPS:-150}"
export WANDB_PROJECT="${WANDB_PROJECT:-menu-grpo}"

LOG="$WORK/grpo.log"

log() { echo -e "\n=== $* ===\n"; }

require_env() {
  local missing=()
  for v in "$@"; do [ -n "${!v:-}" ] || missing+=("$v"); done
  if [ ${#missing[@]} -gt 0 ]; then echo "missing required env: ${missing[*]}" >&2; exit 1; fi
}

# ---------------------------------------------------------------------------
phase_setup() {
  require_env BRAVE_API_KEY S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
  [ -n "${WANDB_API_KEY:-}" ] || echo "[warn] WANDB_API_KEY unset -- console-only logging"

  log "system packages"
  # NOT awscli from apt: the runpod/pytorch ubuntu2404 image has no candidate for it.
  # It goes into $VENV via pip below, and every call here uses $AWS, never a bare `aws`.
  apt-get update -qq && apt-get install -y -qq tmux git curl

  log "repo @ $BRANCH"
  [ -d "$REPO/.git" ] || git clone "$REPO_URL" "$REPO"
  cd "$REPO"
  git fetch --all --quiet
  git checkout "$BRANCH"
  git pull --ff-only
  git log --oneline -1

  log "GPU -- driver MUST be >= 580 / CUDA 13 or vLLM cannot load Gemma-4 at all"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  local drv
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  if [ "$drv" -lt 580 ]; then
    echo "driver $drv < 580: this host cannot run vLLM's cu130 torch. Destroy the pod and" >&2
    echo "recreate with runpod_create.py (allowedCudaVersions=['13.0'])." >&2
    exit 1
  fi

  log "unified GRPO venv ($VENV) -- vLLM FIRST so it pins torch/cu130, then trl+peft, then repo deps"
  [ -x "$PY" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip
  "$VENV/bin/pip" install -q vllm==0.25.1
  "$VENV/bin/pip" install -q "trl>=1.5.1" "peft>=0.19.1" "accelerate>=1.13.0" \
      "bitsandbytes>=0.49.2" jmespath datasets wandb boto3
  "$VENV/bin/pip" install -q requests python-dotenv playwright markdownify beautifulsoup4 \
      jsonschema jinja2 awscli
  "$VENV/bin/playwright" install --with-deps chromium
  "$PY" -c "import torch,vllm,trl,peft,transformers as t; \
      print('torch',torch.__version__,'vllm',vllm.__version__,'trl',trl.__version__, \
            'peft',peft.__version__,'transformers',t.__version__)"

  # ---- Blackwell only: make the pip CUDA toolchain self-consistent -------------
  # On sm_100 vLLM picks the FlashInfer attention backend, and its trtllm-gen FMHA
  # module is NOT in the flashinfer-cubin wheel -- it is JIT-compiled on first use, so
  # the toolchain has to actually work. Out of the box it does not, and it fails THREE
  # times in a row, each one only visible after the previous is fixed (measured
  # 2026-08-09 on a B200; each failure surfaced ~4 min in, after the 15 GB policy load):
  #   1. nvcc 13.2 against nvidia-cuda-runtime 13.0 headers ->
  #      cccl: "CUDA compiler and CUDA toolkit headers are incompatible"
  #   2. nvvm still 13.2 -> emits PTX ISA 9.2, ptxas 13.0 rejects it:
  #      "Unsupported .version 9.2; current version is '9.0'"
  #   3. the link step passes -L$CUDA_HOME/lib64 and -lcudart, but the wheel ships
  #      lib/ (not lib64/) and only libcudart.so.13 (no unversioned .so) -> ld: cannot
  #      find -lcudart
  # Pinning the whole nvcc set DOWN to 13.0 (rather than the runtime UP) keeps torch
  # linked against the libcudart it was built for. Harmless on Hopper, which serves
  # Gemma-4 on FA4 and never compiles any of this.
  if [ -d "$CUDA_HOME" ]; then
    log "CUDA toolchain consistency (Blackwell JIT path)"
    "$VENV/bin/pip" install -q "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" \
        "nvidia-cuda-crt==13.0.88"
    [ -e "$CUDA_HOME/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
    mkdir -p "$CUDA_HOME/lib/stubs"
    [ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
    "$CUDA_HOME/bin/nvcc" --version | tail -1
  fi

  log "pull the corpus + the fully-warmed cache from S3 (228 MB cache -- the grpo split is 100% covered)"
  "$PY" scripts/infra/corpus_sync.py pull --only corpus.sqlite --only cache.sqlite --only grpo

  log "browser preflight -- fail here, not on rollout 1 (a dead browser turns every tool result into a sentinel)"
  "$PY" - <<'PY'
import sys; sys.path.insert(0, "src")
from backends import preflight_browser
err = preflight_browser()
print("browser preflight:", err or "OK")
sys.exit(1 if err else 0)
PY

  wc -l "$REPO/data/grpo/train.jsonl"
  echo "setup OK"
}

# ---------------------------------------------------------------------------
phase_prep_model() {
  cd "$REPO"
  log "pull the kv-shared backfill (105 MB) + the SFT merged student (14.8 GB)"
  mkdir -p "$KV_DIR" "$MERGED"
  # NOT the 16 GB base: to_text_only's backfill just globs *.safetensors in --base, and
  # the 54 tensors it needs are pre-extracted into base/kv-backfill/.
  "$AWS" s3 sync --no-progress "$S3_MODELS/base/kv-backfill/" "$KV_DIR/"
  "$AWS" s3 sync --no-progress "$S3_MODELS/sft/gemma-menu-sft/merged/" "$MERGED/"
  du -sh "$KV_DIR" "$MERGED"

  log "merged -> merged-text (text-only Gemma4ForCausalLM + the 54 backfilled KV tensors)"
  # WITHOUT --base this writes a checkpoint that loads fine under transformers and dies
  # under vLLM: 'weights were not initialized from checkpoint: model.layers.24..41.
  # self_attn.k_norm.weight'. to_text_only's own "missing=0 unexpected=0" is a FALSE
  # all-clear -- it only proves transformers' expectations were met.
  [ -f "$MERGED_TEXT/model.safetensors" ] || \
    "$PY" scripts/train/to_text_only.py "$MERGED" "$MERGED_TEXT" --base "$KV_DIR"
  du -sh "$MERGED_TEXT"
  echo "prep-model OK"
}

# ---------------------------------------------------------------------------
phase_smoke() {
  cd "$REPO"
  log "smoke: 2 steps, canned cache, tiny group -- does the loop run and do the rewards fire?"
  # canned so the smoke costs no scraping and cannot be slowed by the network. It also
  # starves grounding by design (a frozen miss returns a constant), so judge the smoke
  # on "it ran and rewards are finite", NOT on the reward value.
  # Runs on ALL $NPROC GPUs and with the probe ON, because those are the two things a
  # smoke is for here: DDP + one colocate vLLM engine per rank, and the eval_dataset
  # path (TRL routes it through compute_loss, so a probe misconfig is a crash, not a
  # missing metric). --probe-every 1 forces a probe inside 2 steps.
  local smoke_run="$PY scripts/train/train_grpo.py"
  [ "$NPROC" -gt 1 ] && smoke_run="$VENV/bin/accelerate launch --num_processes $NPROC \
      --num_machines 1 --mixed_precision bf16 --dynamo_backend no scripts/train/train_grpo.py"
  $smoke_run \
      --data data/grpo/train.jsonl --model-path "$MERGED_TEXT" \
      --use-vllm --vllm-mode colocate --vllm-gpu-memory-utilization "$VLLM_UTIL" \
      --max-completion-length "$MAXLEN" \
      --num-generations 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
      --probe-size 4 --probe-every 1 --probe-generations 2 \
      --max-steps 2 --limit 16 --cache-policy canned \
      --output-dir "$WORK/grpo-smoke" --report-to none
  log "smoke checkpoint movement (||B|| should be nonzero after 2 steps at lr $LR)"
  "$PY" scripts/analysis/adapter_norms.py "$WORK/grpo-smoke/adapter" 2>/dev/null || \
    echo "(no adapter saved at 2 steps -- fine; the gate is that the loop ran)"
  echo "smoke OK -- 'train' is the expensive one"
}

# ---------------------------------------------------------------------------
# Checkpoints are pushed OUT of the pod on a timer, not at the end: a 15-hour run on a
# rented box that dies at hour 14 with everything on local disk is a total loss.
phase_pusher() {
  while true; do
    sleep 600
    "$AWS" s3 sync --no-progress "$OUT/" "$S3_MODELS/grpo/$RUN_NAME/" \
        --exclude "*/optimizer.pt" --exclude "*/rng_state*" --quiet || true
  done
}

phase_train() {
  cd "$REPO"
  require_env BRAVE_API_KEY
  log "GRPO: ${NPROC}xGPU G=$G len=$MAXLEN bs=$BS x accum=$ACCUM lr=$LR temp=$TEMP \
tools=$TOOL_CALLS/$TOOL_CHARS probe=${PROBE_SIZE}x${PROBE_GENS}/${PROBE_EVERY} steps=$MAX_STEPS"
  mkdir -p "$OUT"
  # TRL's rule is on the GLOBAL batch: NPROC * BS * ACCUM must be divisible by G.
  # train_grpo.py can only check its single-process half of that, so check it here where
  # NPROC is known -- a mismatch otherwise surfaces after both ranks have loaded 15 GB.
  if [ $(( NPROC * BS * ACCUM % G )) -ne 0 ]; then
    echo "global batch (NPROC $NPROC * BS $BS * ACCUM $ACCUM = $((NPROC*BS*ACCUM))) must be" >&2
    echo "divisible by G ($G)" >&2
    exit 1
  fi
  local runner="$PY scripts/train/train_grpo.py"
  if [ "$NPROC" -gt 1 ]; then
    # accelerate launch, NOT plain python: colocate puts a vLLM engine on every rank, and
    # DDP is what makes the ranks agree on the gradient. --num_processes must equal the
    # visible GPU count or ranks contend for GPU 0.
    runner="$VENV/bin/accelerate launch --num_processes $NPROC --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no scripts/train/train_grpo.py"
  fi
  tmux kill-session -t grpo 2>/dev/null || true
  # REDIRECT, do not `| tee` into the tmux pane. TRL prints a rich table of sampled
  # completions every logging step; piping that through tee to a pane whose fd wandb's
  # console_capture has left non-blocking kills the run outright -- measured 2026-08-09,
  # the first attempt died 11 min in, one step from its first checkpoint, with
  # `BlockingIOError: [Errno 11] write could not complete without blocking` raised
  # inside rich's _write_buffer. TERM=dumb also stops rich from drawing box art into
  # what is now a plain file.
  TERM=dumb tmux new-session -d -s grpo "cd $REPO && TERM=dumb $runner \
      --data data/grpo/train.jsonl --model-path '$MERGED_TEXT' \
      --starting-checkpoint gemma-menu-sft \
      --use-vllm --vllm-mode colocate --vllm-gpu-memory-utilization $VLLM_UTIL \
      --num-generations $G --max-completion-length $MAXLEN \
      --per-device-train-batch-size $BS --gradient-accumulation-steps $ACCUM \
      --learning-rate $LR --temperature $TEMP \
      --max-tool-calls $TOOL_CALLS --max-tool-chars $TOOL_CHARS \
      --cache-policy live --cache-path data/cache.sqlite \
      --probe-size $PROBE_SIZE --probe-every $PROBE_EVERY --probe-generations $PROBE_GENS \
      --save-steps $SAVE_STEPS --max-steps $MAX_STEPS \
      --output-dir '$OUT' --run-id '$RUN_NAME' --run-name '$RUN_NAME' --report-to wandb \
      > '$LOG' 2>&1"
  tmux new-session -d -s push "bash $REPO/scripts/infra/grpo_pod.sh pusher"
  echo "launched. tmux sessions: grpo (training), push (S3 every 10 min)"
  echo "  bash scripts/infra/grpo_pod.sh watch"
}

phase_watch() { tail -f "$LOG"; }

phase_norms() {
  cd "$REPO"
  local ck
  ck="$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
  [ -n "$ck" ] || { echo "no checkpoint yet under $OUT"; exit 1; }
  log "LoRA movement at $ck (the v2 SFT adapter, same r/alpha, sits at median ||B|| ~0.41)"
  "$PY" scripts/analysis/adapter_norms.py "$ck"
}

phase_finish() {
  cd "$REPO"
  log "final push: adapter + the cache the rollouts warmed"
  "$AWS" s3 sync --no-progress "$OUT/" "$S3_MODELS/grpo/$RUN_NAME/" --exclude "*/optimizer.pt" --exclude "*/rng_state*"
  "$PY" scripts/infra/corpus_sync.py push --only cache.sqlite
  echo "finish OK -- safe to destroy the pod"
}

case "${1:-}" in
  setup)      phase_setup ;;
  prep-model) phase_prep_model ;;
  smoke)      phase_smoke ;;
  train)      phase_train ;;
  pusher)     phase_pusher ;;
  watch)      phase_watch ;;
  norms)      phase_norms ;;
  finish)     phase_finish ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
