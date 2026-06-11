#!/usr/bin/env bash
# Free GPU VRAM by killing leftover CUDA processes.
#
# Interrupted runs (Ctrl-C, a killed `uv run`, an OOM mid-load) often leave an
# orphaned python holding the full model in VRAM, which then makes the next load
# OOM or crawl. This kills every process currently using the GPU.
#
# Usage (from the project root):
#   ./scripts/free_vram.sh          # kill all processes using the GPU
#   DRY_RUN=1 ./scripts/free_vram.sh   # just list them, kill nothing
set -euo pipefail

echo "=== VRAM before ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v '^$' || true)

if [[ -z "$pids" ]]; then
  echo "No processes are using the GPU. Nothing to do."
  exit 0
fi

echo "=== GPU processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN set - not killing anything."
  exit 0
fi

for pid in $pids; do
  echo "killing $pid"
  kill -9 "$pid" 2>/dev/null || echo "  (already gone)"
done

sleep 2
echo "=== VRAM after ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
