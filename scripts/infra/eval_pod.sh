#!/usr/bin/env bash
# Three-model v2 eval on ONE 4xH100 RunPod pod: the Qwen3-235B TEACHER, the
# UNTRAINED Gemma-4 E4B base, and the SFT-distilled Gemma student -- all on the same
# seeded eval plan, so their per-episode trace ids line up and the numbers compare.
#
# Run it PHASE BY PHASE (each phase is idempotent and safe to re-run):
#
#   bash scripts/infra/eval_pod.sh setup          # venvs, deps, chromium, S3 pull
#   bash scripts/infra/eval_pod.sh serve-teacher  # vLLM: Qwen3-235B-FP8, TP=4 (all 4 GPUs)
#   bash scripts/infra/eval_pod.sh smoke          # one real episode -- the gate before spending
#   bash scripts/infra/eval_pod.sh run-teacher    # teacher over the eval split -> REFERENCE + its own report
#   bash scripts/infra/eval_pod.sh stop-teacher   # free the GPUs
#   bash scripts/infra/eval_pod.sh prep-gemma     # pull both checkpoints, convert to text-only
#   bash scripts/infra/eval_pod.sh serve-gemma    # base on GPU0, SFT on GPU1 (concurrently)
#   bash scripts/infra/eval_pod.sh smoke-gemma    # 2 real episodes per server -- the gate
#   bash scripts/infra/eval_pod.sh run-gemma      # BOTH evals in parallel, scored vs the reference
#   bash scripts/infra/eval_pod.sh finish         # summary table + push everything to S3
#
# WHY THE TEACHER RUNS FIRST AND WHY IT IS SELF-REPORTED: corpus.sqlite has ZERO
# eval-split traces, so there is no reference set yet. The teacher pass CREATES it
# (build_corpus writes traces into the DB) -- which is also why the teacher can only
# be *self-reported*: it is the reference, so pairing it against itself would print
# P=R=F1=1.000 and mean nothing. The two Gemmas are then PAIRED against it and get
# real precision/recall/F1 + abstention buckets. dump_reference.py exports the
# teacher's traces into candidate files so all three rows come out of the SAME
# scorer in the SAME shape.
#
# CACHE POLICY: everything runs `live`. The eval split IS fully warmed (601/601
# restaurants x 6 query templates), so most tool calls hit; `live` lets a model that
# strays off the warmed distribution actually fetch, which is what the shipped
# product does. The two Gemma runs get their OWN cache snapshot (cache-base.sqlite /
# cache-sft.sqlite) so they (a) cannot contend on one SQLite file while running
# concurrently, (b) start from byte-identical inputs, and (c) each report an honest,
# uncontaminated hit rate. Hit rates land in the report JSON and in W&B.
#
# ENV YOU MUST EXPORT ON THE POD BEFORE `setup` (see .env.example / notes/S3_setup.md):
#   BRAVE_API_KEY   -- search backend (scrape is local, no key)
#   WANDB_API_KEY   -- omit and every run silently degrades to console-only logging
#   S3_BUCKET, S3_PREFIX=v2, AWS_DEFAULT_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#                   -- use the SCOPED pod key, never root creds (notes/S3_setup.md)
# HF_TOKEN is NOT needed: both checkpoints (incl. their tokenizers) come from S3.
set -euo pipefail

REPO="${REPO:-/workspace/small-model-finetuning}"
REPO_URL="${REPO_URL:-https://github.com/Maanav-s/small-model-finetuning.git}"
BRANCH="${BRANCH:-eval-v2-three-model}"
WORK="${WORK:-/workspace}"
CLIENT_VENV="${CLIENT_VENV:-/opt/client}"     # repo deps (tools, eval, corpus) -- NO torch
VLLM_VENV="${VLLM_VENV:-/opt/vllm}"           # vllm + its own torch; created by serve_teacher.sh
PY="$CLIENT_VENV/bin/python"
VLLM_PY="$VLLM_VENV/bin/python"

# The eval plan. These MUST be identical across all three models or the trace ids
# (hence the candidate<->reference join) do not line up.
SPLIT=eval
LIMIT="${LIMIT:-500}"                 # 300 free + 200 conditioned at COND=0.4
COND="${COND:-0.4}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-16}"

TEACHER_HF="${TEACHER_HF:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"   # reporting label
SERVED_TEACHER="${SERVED_TEACHER:-teacher}"  # ON THE WIRE: must match serve_teacher.sh's
                                             # SERVED_NAME default and build_corpus's
                                             # --teacher-model below, or requests 404
TEACHER_TP="${TEACHER_TP:-4}"
GEMMA_MAX_LEN="${GEMMA_MAX_LEN:-98304}"       # v1 precedent: 500/500 episodes, 0 context-400s
GEMMA_UTIL="${GEMMA_UTIL:-0.85}"
# Per-turn generation budget. 4096 (the eval.py default) TRUNCATES this student: it
# emits a long think plus a 40-item menu in ONE turn, hits finish_reason=length, and
# returns empty content -- scoring identically to non-termination.
GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-12288}"
# Ports: NOT 8001 -- the RunPod pytorch image runs nginx there, so vLLM dies with
# `[Errno 98] Address already in use` while nginx keeps answering 200 on /v1/models.
GEMMA_BASE_PORT="${GEMMA_BASE_PORT:-8011}"
GEMMA_SFT_PORT="${GEMMA_SFT_PORT:-8012}"

# RUN_SET is STICKY: computed once, then persisted. Deriving it from `date` on every
# invocation splits a single run-set across two directories the moment a phase runs
# after UTC midnight -- the teacher's reference would sit in one dir and the students'
# candidates in another, and they would never join. Explicit $RUN_SET always wins.
RUN_SET_FILE="${RUN_SET_FILE:-$WORK/run_set.txt}"
if [ -z "${RUN_SET:-}" ] && [ -f "$RUN_SET_FILE" ]; then
  RUN_SET="$(cat "$RUN_SET_FILE")"
fi
RUN_SET="${RUN_SET:-eval${LIMIT}-$(date -u +%Y%m%d)}"
mkdir -p "$(dirname "$RUN_SET_FILE")" && echo "$RUN_SET" > "$RUN_SET_FILE"
EVAL_DIR="$REPO/data/eval/$RUN_SET"           # candidates + reports -> S3 (data/eval is a synced dir)
RESULTS="$REPO/results/$RUN_SET"              # the small report JSONs -> committed to the repo
export WANDB_PROJECT="${WANDB_PROJECT:-menu-eval}"

log() { echo -e "\n=== $* ===\n"; }

require_env() {
  local missing=()
  for v in "$@"; do [ -n "${!v:-}" ] || missing+=("$v"); done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "missing required env: ${missing[*]}" >&2
    exit 1
  fi
}

is_vllm() {  # is_vllm <port> -- true only if a REAL vLLM model list answers
  # NOT `curl -sf` alone: the RunPod image runs nginx, which answers 200 with its
  # default HTML on any path it owns (it holds 8001). A status-only check reports
  # that as a healthy model server, and the eval then talks to a web server.
  curl -sf --max-time 10 "http://localhost:$1/v1/models" 2>/dev/null \
    | grep -q '"object"[[:space:]]*:[[:space:]]*"list"'
}

wait_health() {  # wait_health <port> <label> [timeout_s]
  local port="$1" label="$2" timeout="${3:-3600}" waited=0
  echo "waiting for $label on :$port (timeout ${timeout}s) ..."
  until is_vllm "$port"; do
    sleep 10; waited=$((waited + 10))
    if [ "$waited" -ge "$timeout" ]; then
      echo "$label did not come up within ${timeout}s -- check the log" >&2
      exit 1
    fi
    [ $((waited % 60)) -eq 0 ] && echo "  ... ${waited}s"
  done
  # A healthy /v1/models is NOT proof it serves (CLAUDE.md: block-scale FP8 can 500
  # on the FIRST real request). That is what `smoke` is for.
  echo "$label is up after ${waited}s"
}

snapshot_cache() {  # snapshot_cache <dst> -- WAL-safe copy (VACUUM INTO, never cp)
  local dst="$1"
  rm -f "$dst"
  "$PY" - "$REPO/data/cache.sqlite" "$dst" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
con = sqlite3.connect(src)
con.execute("VACUUM INTO ?", (dst,))   # folds in the WAL; verifies cleanly
con.close()
print(f"snapshot {src} -> {dst}")
PY
}

# ---------------------------------------------------------------------------
phase_setup() {
  require_env BRAVE_API_KEY S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
  [ -n "${WANDB_API_KEY:-}" ] || echo "[warn] WANDB_API_KEY unset -- runs will log to console only"

  log "system packages"
  apt-get update -qq && apt-get install -y -qq tmux git curl

  log "repo @ $BRANCH"
  [ -d "$REPO/.git" ] || git clone "$REPO_URL" "$REPO"
  cd "$REPO"
  git fetch --all --quiet
  git checkout "$BRANCH"
  git pull --ff-only
  git log --oneline -1

  log "GPUs -- the driver MUST be >= 580 / CUDA 13 or vLLM cannot serve Gemma-4 at all"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

  log "client venv ($CLIENT_VENV) -- repo deps only, NO torch"
  # anthropic is required even for the gemma/vllm paths: eval.py imports
  # claude_agent at module scope and that module imports anthropic at its top.
  [ -x "$PY" ] || python3 -m venv "$CLIENT_VENV"
  "$CLIENT_VENV/bin/pip" install -q -U pip
  # jinja2 is NOT a hard dep of transformers 5.x, but apply_chat_template needs it --
  # and the Gemma student path renders with OUR template, so without it every episode
  # dies with `ImportError: apply_chat_template requires jinja2`.
  "$CLIENT_VENV/bin/pip" install -q \
      requests python-dotenv playwright markdownify beautifulsoup4 jsonschema \
      openai transformers jinja2 wandb boto3 anthropic
  "$CLIENT_VENV/bin/playwright" install --with-deps chromium

  log "pull the corpus + the warmed cache from S3"
  "$PY" scripts/infra/corpus_sync.py pull --only corpus.sqlite --only cache.sqlite

  mkdir -p "$EVAL_DIR/candidates" "$EVAL_DIR/reports" "$RESULTS"
  log "sanity: browser preflight + plan (no model needed)"
  "$PY" - <<'PY'
import sys; sys.path.insert(0, "src")
from backends import preflight_browser
err = preflight_browser()
print("browser preflight:", err or "OK")
sys.exit(1 if err else 0)
PY
  # Write the plan out, THEN head the file. Piping python straight into `head`
  # closes the pipe under it -> BrokenPipeError traceback, and `set -o pipefail`
  # turns that into a failed setup phase for a run that actually succeeded.
  "$PY" scripts/eval/eval.py "$EVAL_DIR/candidates/_plan" --model vllm --list \
      --limit "$LIMIT" --conditioned-frac "$COND" --seed "$SEED" > "$WORK/plan.txt"
  head -5 "$WORK/plan.txt"
  rmdir "$EVAL_DIR/candidates/_plan" 2>/dev/null || true
  echo "setup OK -- run-set '$RUN_SET'"
}

# ---------------------------------------------------------------------------
phase_serve_teacher() {
  cd "$REPO"
  log "serving $TEACHER_HF (TP=$TEACHER_TP) -- 235 GB download on a cold pod, ~3-5 min"
  bash scripts/infra/serve_teacher.sh "$TEACHER_HF" "$TEACHER_TP"
  wait_health 8000 "teacher" 3600
  echo "tail the log with: tail -f /workspace/vllm.log"
}

phase_smoke() {
  cd "$REPO"
  log "teacher smoke (ONE real episode -- /v1/models being healthy proves nothing)"
  "$PY" scripts/infra/smoke_teacher.py --model teacher
}

phase_stop_teacher() {
  log "stopping the teacher and freeing the GPUs"
  pkill -f "vllm serve" || true
  sleep 20
  nvidia-smi --query-gpu=index,memory.used --format=csv
  echo "if any GPU still shows GBs used, kill the orphaned VLLM:: workers by pid"
}

# ---------------------------------------------------------------------------
phase_run_teacher() {
  cd "$REPO"
  require_env BRAVE_API_KEY
  log "TEACHER over the $SPLIT split -> reference traces in corpus.sqlite"
  # --sync-every checkpoints the DB to S3 so a pod death can't lose more than 25
  # traces of metered teacher work.
  WANDB_NAME="teacher-reference-build" \
  "$PY" scripts/corpus/build_corpus.py \
      --split "$SPLIT" --limit "$LIMIT" --conditioned-frac "$COND" --seed "$SEED" \
      --teacher vllm --teacher-base-url http://localhost:8000/v1 --teacher-model "$SERVED_TEACHER" \
      --workers "$WORKERS" --cache-policy live --sync-every 25

  log "export the teacher's traces as candidate files (no inference)"
  "$PY" scripts/eval/dump_reference.py "$EVAL_DIR/candidates/teacher"

  log "score the teacher through the SAME scorer (self-report -- it IS the reference)"
  # --served-model-name is what goes ON THE WIRE (must be 'teacher', what the server
  # serves); --model-label is what lands in the report. Conflating them 404s any
  # episode this pass has to re-run because build_corpus failed it.
  "$PY" scripts/eval/eval.py "$EVAL_DIR/candidates/teacher" \
      --model vllm --served-model-name "$SERVED_TEACHER" --model-label "$TEACHER_HF" --self-report \
      --limit "$LIMIT" --conditioned-frac "$COND" --seed "$SEED" \
      --cache-policy live --wandb-name "teacher-qwen3-235b" \
      --json "$EVAL_DIR/reports/teacher-qwen3-235b.json"
}

# ---------------------------------------------------------------------------
phase_prep_gemma() {
  cd "$REPO"
  log "pull both checkpoints from S3 (~32 GB)"
  "$PY" scripts/infra/corpus_sync.py pull --only models

  local base="$REPO/data/models/gemma-4-e4b-it/base"
  local sft="$REPO/data/models/gemma-4-e4b-it/sft/gemma-menu-sft/merged"

  # to_text_only needs torch -> run it from the vLLM venv, not the client venv.
  # --base is NOT optional: Gemma-4 E4B's 18 kv-shared layers mean 54 k_norm/k_proj/
  # v_proj tensors are never instantiated by transformers, and vLLM hard-fails the
  # load without them (CLAUDE.md). The base checkpoint is its own backfill source.
  log "convert BASE -> text-only (Gemma4ForCausalLM, +54 kv-shared tensors)"
  [ -d "$WORK/base-text" ] || "$VLLM_PY" scripts/train/to_text_only.py "$base" "$WORK/base-text" --base "$base"
  log "convert SFT merged -> text-only"
  [ -d "$WORK/sft-text" ] || "$VLLM_PY" scripts/train/to_text_only.py "$sft" "$WORK/sft-text" --base "$base"
  ls -la "$WORK/base-text" "$WORK/sft-text"
}

serve_one_gemma() {  # serve_one_gemma <gpu> <ckpt_dir> <port> <logfile> <label>
  local gpu="$1" ckpt="$2" port="$3" logf="$4" label="$5"
  if is_vllm "$port"; then
    echo "$label already serving on :$port -- skipping launch (phase is idempotent)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup env PATH="$VLLM_VENV/bin:$PATH" "$VLLM_VENV/bin/vllm" serve "$ckpt" \
      --served-model-name gemma-menu --max-model-len "$GEMMA_MAX_LEN" \
      --gpu-memory-utilization "$GEMMA_UTIL" --host 0.0.0.0 --port "$port" \
      </dev/null > "$logf" 2>&1 &
  echo "$label launching on GPU$gpu :$port -> $logf"
}

phase_serve_gemma() {
  cd "$REPO"
  log "serving BOTH Gemmas concurrently: base GPU0::$GEMMA_BASE_PORT, SFT GPU1::$GEMMA_SFT_PORT"
  # One GPU each, so the two evals run in parallel on the pod we are already paying
  # for. Serve the merged bf16 text-only checkpoints -- never 4-bit (the v1 QLoRA
  # fidelity finding: 4-bit serving alone cost ~32 points of success rate).
  serve_one_gemma 0 "$WORK/base-text" "$GEMMA_BASE_PORT" "$WORK/vllm-base.log" "gemma-base"
  serve_one_gemma 1 "$WORK/sft-text"  "$GEMMA_SFT_PORT"  "$WORK/vllm-sft.log"  "gemma-sft"
  wait_health "$GEMMA_BASE_PORT" "gemma-base" 1800
  wait_health "$GEMMA_SFT_PORT" "gemma-sft" 1800
}

phase_smoke_gemma() {
  # The teacher gets scripts/infra/smoke_teacher.py before anything metered runs; the
  # students had no equivalent, and paid for it twice (a torch import and a missing
  # jinja2, each killing all 500 episodes at episode 1). Two real episodes per server,
  # into a throwaway candidate dir, exercising the SAME code path as the full run.
  cd "$REPO"
  require_env BRAVE_API_KEY
  local rc=0
  for spec in "gemma-base:$GEMMA_BASE_PORT:$WORK/base-text" "gemma-sft:$GEMMA_SFT_PORT:$WORK/sft-text"; do
    local name="${spec%%:*}" rest="${spec#*:}"
    local port="${rest%%:*}" ckpt="${rest#*:}"
    log "smoke: $name on :$port (2 episodes)"
    rm -rf "$WORK/smoke-$name"
    if "$PY" scripts/eval/eval.py "$WORK/smoke-$name" \
        --model gemma --gemma-vllm-base-url "http://localhost:$port/v1" \
        --served-model-name gemma-menu --model-path "$ckpt" \
        --limit 2 --conditioned-frac 0.5 --seed "$SEED" \
        --cache-policy live --cache-path "$REPO/data/cache.sqlite" \
        --workers 2 --gemma-max-tokens "$GEMMA_MAX_TOKENS" --no-wandb --self-report 2>&1 | tail -12; then
      local n_ok
      n_ok=$(grep -c '"schema_valid": true' "$WORK/smoke-$name"/*.json 2>/dev/null || echo 0)
      echo "[smoke] $name wrote $(ls "$WORK/smoke-$name" 2>/dev/null | wc -l) candidates, $n_ok schema-valid"
    else
      echo "[smoke] $name FAILED"; rc=1
    fi
  done
  [ "$rc" -eq 0 ] || { echo "gemma smoke failed -- refusing to start the metered run" >&2; exit 1; }
  echo "gemma smoke OK for both servers"
}

phase_run_gemma() {
  cd "$REPO"
  require_env BRAVE_API_KEY
  log "snapshot the cache once per model (no write contention, identical starting state)"
  snapshot_cache "$WORK/cache-base.sqlite"
  snapshot_cache "$WORK/cache-sft.sqlite"

  log "running BOTH Gemma evals in parallel"
  WANDB_NAME="gemma-base" "$PY" scripts/eval/eval.py "$EVAL_DIR/candidates/gemma-base" \
      --model gemma --gemma-vllm-base-url http://localhost:$GEMMA_BASE_PORT/v1 \
      --served-model-name gemma-menu --model-path "$WORK/base-text" \
      --limit "$LIMIT" --conditioned-frac "$COND" --seed "$SEED" \
      --cache-policy live --cache-path "$WORK/cache-base.sqlite" --workers "$WORKERS" \
      --gemma-max-tokens "$GEMMA_MAX_TOKENS" \
      --wandb-name "gemma-base" \
      --json "$EVAL_DIR/reports/gemma-base.json" > "$WORK/eval-base.log" 2>&1 &
  local pid_base=$!
  WANDB_NAME="gemma-sft" "$PY" scripts/eval/eval.py "$EVAL_DIR/candidates/gemma-sft" \
      --model gemma --gemma-vllm-base-url http://localhost:$GEMMA_SFT_PORT/v1 \
      --served-model-name gemma-menu --model-path "$WORK/sft-text" \
      --limit "$LIMIT" --conditioned-frac "$COND" --seed "$SEED" \
      --cache-policy live --cache-path "$WORK/cache-sft.sqlite" --workers "$WORKERS" \
      --gemma-max-tokens "$GEMMA_MAX_TOKENS" \
      --wandb-name "gemma-sft" \
      --json "$EVAL_DIR/reports/gemma-sft.json" > "$WORK/eval-sft.log" 2>&1 &
  local pid_sft=$!
  echo "base pid $pid_base -> $WORK/eval-base.log"
  echo "sft  pid $pid_sft  -> $WORK/eval-sft.log"
  echo "follow with: tail -f $WORK/eval-base.log $WORK/eval-sft.log"
  local rc=0
  wait "$pid_base" || { echo "[warn] gemma-base eval exited nonzero"; rc=1; }
  wait "$pid_sft"  || { echo "[warn] gemma-sft eval exited nonzero";  rc=1; }
  return "$rc"
}

# ---------------------------------------------------------------------------
phase_finish() {
  cd "$REPO"
  log "comparison table"
  mkdir -p "$RESULTS"
  cp "$EVAL_DIR/reports/"*.json "$RESULTS/" 2>/dev/null || true
  "$PY" scripts/eval/summarize.py "$RESULTS" -o "$RESULTS/README.md"
  cat "$RESULTS/README.md"
  for s in free conditioned; do
    "$PY" scripts/eval/summarize.py "$RESULTS" --slice "$s" >> "$RESULTS/README.md"
  done

  log "push corpus (now holding the eval reference), cache, candidates + reports"
  "$PY" scripts/infra/corpus_sync.py push --only corpus.sqlite --only cache.sqlite --only eval

  echo
  echo "COMMIT THESE FROM YOUR LAPTOP (the pod's repo clone has no push credentials):"
  echo "  results/$RUN_SET/*.json + results/$RUN_SET/README.md"
  echo "Candidates + reports are archived at s3://\$S3_BUCKET/\$S3_PREFIX/eval/$RUN_SET/"
}

case "${1:-}" in
  setup)          phase_setup ;;
  serve-teacher)  phase_serve_teacher ;;
  smoke)          phase_smoke ;;
  run-teacher)    phase_run_teacher ;;
  stop-teacher)   phase_stop_teacher ;;
  prep-gemma)     phase_prep_gemma ;;
  serve-gemma)    phase_serve_gemma ;;
  smoke-gemma)    phase_smoke_gemma ;;
  run-gemma)      phase_run_gemma ;;
  finish)         phase_finish ;;
  *) sed -n '2,40p' "$0"; exit 1 ;;
esac
