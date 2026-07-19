"""Per-adapter lineage record: `models/<family>/<stage>/<run>/meta.json` (plan §6).

Shared by scripts/train/train_sft.py and scripts/train/train_grpo.py so both
trainers emit the SAME provenance schema. Lives in src/ (the v2 shared layer,
already on the train scripts' sys.path) rather than duplicated per script -- a
bare sibling import from scripts/train/ is not reliable under accelerate/torchrun,
which don't put the launched script's own dir on sys.path.

A meta.json makes a ~140 MB adapter reproducible without the weights: which base +
which dataset (by md5), the quant recipe (load-bearing -- the v1 finding is that
serve-time dtype must match the training-time base dtype), the hyperparams, and
the git sha / timestamp. See notes/v2_rebuild_plan.md §6 for the field contract.

  from run_meta import md5_file, write_run_meta
  write_run_meta(
      output_dir, family="gemma-4-e4b-it", stage="sft", run_id="20260714-qlora-r16",
      base_ref={"path": "v2/models/gemma-4-e4b-it/base", "md5": None},
      dataset={"path": "v2/sft/gemma-4-e4b-it/train.jsonl", "md5": md5_file(local_jsonl)},
      quant={"method": "lora", "quant_type": None, "compute_dtype": "bfloat16"},
      hyperparams={"lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 3},
      git_cwd=REPO_ROOT,
  )
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def md5_file(path) -> str | None:
    """md5 hexdigest of a file (chunked), or None if it can't be read.

    Matches scripts/datasets/build_sft.py's `_file_md5`, so the dataset md5 recorded
    in an adapter's meta.json equals the md5 the dataset export's own .meta.json
    carries -- the two provenance records line up on the same content hash.
    """
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def git_sha(cwd) -> str | None:
    """`git rev-parse HEAD` run in `cwd`, or None if git isn't available / not a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_meta(
    dest_dir,
    *,
    family: str,
    stage: str,
    run_id: str,
    base_ref: dict,
    dataset: dict,
    quant: dict,
    hyperparams: dict,
    starting_checkpoint: str | None = None,
    eval_ref: dict | None = None,
    git_cwd=None,
) -> Path:
    """Write `<dest_dir>/meta.json` (the adapter's lineage record) and return its path.

    The caller supplies everything that can't be derived locally:
      base_ref / dataset  -- each a {"path": <s3-or-local>, "md5": <hex|None>} dict
      quant               -- {"method", "quant_type", "compute_dtype"}
      hyperparams         -- {"lora_r", "lora_alpha", "lr", "epochs", ...}
      stage               -- "sft" | "grpo"
      starting_checkpoint -- grpo: the sft run-id it initialized from; else None
      eval_ref            -- filled in by eval later; defaults to {"run": None, "success": None}
    `git_sha` (run in `git_cwd`, which should be the repo root) and `created_at`
    are filled in here. Field order follows plan §6.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "family": family,
        "stage": stage,
        "run_id": run_id,
        "base_ref": base_ref,
        "starting_checkpoint": starting_checkpoint,
        "dataset": dataset,
        "quant": quant,
        "hyperparams": hyperparams,
        "git_sha": git_sha(git_cwd or dest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval": eval_ref or {"run": None, "success": None},
    }
    path = dest / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
