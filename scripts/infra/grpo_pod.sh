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

S3_MODELS="s3://${S3_BUCKET:-restaurant-menu-corpus}/${S3_PREFIX:-v2}/models/gemma-4-e4b-it"
KV_DIR="$WORK/kv"                # ONLY the 54 kv-shared tensors (105 MB), not the 16 GB base
MERGED="$WORK/merged"            # the SFT student, merged bf16
MERGED_TEXT="$WORK/merged-text"  # what BOTH the trainer and vLLM actually load

# ---- the run config ------------------------------------------------------------
RUN_NAME="${RUN_NAME:-gemma-menu-grpo-v2}"
OUT="${OUT:-$WORK/$RUN_NAME}"
# G=8: the 2026-08-08 run's frac_reward_zero_std sat at 0.00-0.05, so groups were NOT
# degenerate -- group size was never the problem and is unchanged here.
G="${G:-8}"
# 24576, up from 16384. On an H200 141 GB this OOM'd in accelerator.backward() at ANY
# vllm util (~85 GB of retained logits chain + 15 GB policy); it fits on a 180 GB B200.
# 32768 does NOT fit anywhere available: the chain is LINEAR in length at ~3.7 MB/token,
# so 32768 -> ~120 GB + ~29 GB of everything-else + vLLM's share > 180.
MAXLEN="${MAXLEN:-24576}"
BS="${BS:-1}"                    # pure memory knob; the group may span accumulation steps
ACCUM="${ACCUM:-16}"             # effective 16 = 2 prompts/step at G=8 -> ~451 steps/epoch
LR="${LR:-1e-5}"                 # NOT 1e-6: see the commit that raised the default
TEMP="${TEMP:-1.2}"              # entropy was 0.06 at 1.0 -- a near-deterministic policy
TOOL_CALLS="${TOOL_CALLS:-6}"
TOOL_CHARS="${TOOL_CHARS:-16000}"
VLLM_UTIL="${VLLM_UTIL:-0.18}"
SAVE_STEPS="${SAVE_STEPS:-25}"
MAX_STEPS="${MAX_STEPS:--1}"
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
  "$PY" scripts/train/train_grpo.py \
      --data data/grpo/train.jsonl --model-path "$MERGED_TEXT" \
      --use-vllm --vllm-mode colocate --vllm-gpu-memory-utilization "$VLLM_UTIL" \
      --max-completion-length "$MAXLEN" \
      --num-generations 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
      --max-steps 2 --limit 8 --cache-policy canned \
      --output-dir "$WORK/grpo-smoke" --report-to none
  log "smoke checkpoint movement (||B|| should be nonzero after 2 steps at lr $LR)"
  "$PY" scripts/analysis/adapter_norms.py "$WORK/grpo-smoke" 2>/dev/null || \
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
  log "GRPO: G=$G len=$MAXLEN bs=$BS x accum=$ACCUM lr=$LR temp=$TEMP tools=$TOOL_CALLS/$TOOL_CHARS"
  mkdir -p "$OUT"
  tmux kill-session -t grpo 2>/dev/null || true
  tmux new-session -d -s grpo "cd $REPO && $PY scripts/train/train_grpo.py \
      --data data/grpo/train.jsonl --model-path '$MERGED_TEXT' \
      --starting-checkpoint gemma-menu-sft \
      --use-vllm --vllm-mode colocate --vllm-gpu-memory-utilization $VLLM_UTIL \
      --num-generations $G --max-completion-length $MAXLEN \
      --per-device-train-batch-size $BS --gradient-accumulation-steps $ACCUM \
      --learning-rate $LR --temperature $TEMP \
      --max-tool-calls $TOOL_CALLS --max-tool-chars $TOOL_CHARS \
      --cache-policy live --cache-path data/cache.sqlite \
      --save-steps $SAVE_STEPS --max-steps $MAX_STEPS \
      --output-dir '$OUT' --run-id '$RUN_NAME' --run-name '$RUN_NAME' --report-to wandb \
      2>&1 | tee '$LOG'"
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
