# notes/

Planning specs and the experiment log. Start with [CLAUDE.md](../CLAUDE.md) for the
engineering constraints and [../README.md](../README.md) for the results; these files are
the depth behind both.

| file | what it is | still current? |
|---|---|---|
| [experiments.md](experiments.md) | **The append-only log of every training/eval run** — config, numbers, takeaway, where the artifacts live. Newest first. Read this before re-deriving anything. | ✅ **live** |
| [S3_setup.md](S3_setup.md) | Canonical reference for the bucket, the `v2/` object layout, `corpus_sync.py`, and IAM. | ✅ live |
| [runpod_sft.md](runpod_sft.md) | Setup + run guide for the LoRA SFT stage on a 2×H200 pod, with the hyperparameter table and its reasoning. | ✅ live |
| [vllm_inference.html](vllm_inference.html) | Design + economics of serving on vLLM (why a server, not in-process; cost per episode). | ✅ live |
| [v2_rebuild_plan.md](v2_rebuild_plan.md) | Spec for the `corpus.sqlite` data layer and the `scripts/<stage>/` restructure. **Built** — kept as the design reference. | 📎 historical spec |
| [phase2_plan.md](phase2_plan.md) | The v1 tool-cache + corpus-build plan. Its caching design and SFT recipe still apply; its data-file layout was superseded by v2. | 📎 historical spec |
| [runpod_training_handoff.html](runpod_training_handoff.html) | v1-era handoff doc for moving training to rented GPUs. | 📎 historical |

The per-run machine-readable evidence behind `experiments.md` lives in
[../results/](../results/); bulky artifacts (checkpoints, per-episode traces, the tool
cache) live in S3.
