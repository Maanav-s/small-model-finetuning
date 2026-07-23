# S3 bucket setup (corpus + model store)

Manual AWS setup is done. This is the **canonical** reference for the bucket, the
v2 object layout, the sync tool, and IAM/credentials. (The repo-root
[S3_SETUP.md](../S3_SETUP.md) is just a pointer to this file — edit this one.)

## Bucket

- Name: `restaurant-menu-corpus`
- Region: `us-west-2` (same region as the `ec2-devbox` training instance, for free same-region egress)
- Private: Block Public Access is ON (all four settings)
- Default encryption: SSE-S3 (AES256)
- Object ownership: `BucketOwnerEnforced` — ACLs are disabled, access is IAM-policy-only
- **Versioning: ENABLED (2026-07-16).** Was off until then, which meant every delete/overwrite was
  permanent — dangerous for `models/`, whose merged checkpoint costs a training run to re-derive.
  No lifecycle policy (nothing expires).

## Sync tool

[scripts/infra/corpus_sync.py](../scripts/infra/corpus_sync.py) mirrors the whole
`data/` artifact set to/from S3 — it is the v2 rebuild of the old
`scripts/cache_sync.py` (which only synced the cache). `data/` is git-ignored;
**S3 is its source of truth.**

```bash
uv run python scripts/infra/corpus_sync.py push                       # local data/ -> S3
uv run python scripts/infra/corpus_sync.py pull                       # S3 -> local data/
uv run python scripts/infra/corpus_sync.py push --only corpus.sqlite --only sft
uv run python scripts/infra/corpus_sync.py push --dry-run             # plan only, no network
```

- **WAL snapshotting.** `corpus.sqlite` and `cache.sqlite` both run in WAL mode, so
  their live state spans the `.sqlite` file plus `-wal`/`-shm` sidecars. `push`
  never uploads the bare db: it snapshots each via `VACUUM INTO` (folds in the WAL,
  runs `PRAGMA integrity_check`), uploads the clean snapshot, and deletes the temp.
  The snapshot step runs under `--dry-run` too, so it stays exercisable without a bucket.
- **Skip-unchanged guard.** Every upload stores the file's md5 as object metadata
  (`x-amz-meta-md5`); the guard compares that first (ETags stop being plain md5s for
  multipart uploads), then a single-part ETag, then size. `pull` never deletes a
  local file that is absent remotely — it logs it as kept.

## v2 object layout (under the `v2/` prefix)

| key (under `v2/`) | what |
|---|---|
| `corpus.sqlite` | **the single source of truth** — restaurants + the `sft`/`grpo`/`eval` split + teacher/DAgger traces + per-trace grounding + reject flags, all in one DB (via [src/corpus.py](../src/corpus.py)) |
| `cache.sqlite` | content-hash-keyed tool-call cache (search + scrape results) |
| `sft/<family>/train.jsonl` (+ `.meta.json`) | exported SFT dataset for a model family |
| `grpo/train.jsonl` (+ `.meta.json`) | exported GRPO dataset |
| `models/<family>/{base, sft/<run>, grpo/<run>}` | **base weights + adapters only** — `merged` / `merged-text` are regenerated locally, never stored |
| `eval/` | eval run outputs |

`<family>` is `gemma-4-e4b-it`. The `v2/` prefix replaces v1's loose files:
`restaurants.jsonl`, `splits.json`, `labels.jsonl`, and the `traces/` dir are all
folded into `corpus.sqlite`, and the old `data/review/reject_list.txt` is now the
`traces.rejected` DB field (set by the viz review tool — see [viz/README.md](../viz/README.md)).

## Credentials

boto3's default credential chain resolves in this order here: **EC2 instance profile
→ environment / `.env` keys.**

- **On the devbox — no static keys.** The `ec2-devbox` instance has an IAM instance
  profile (`menu-corpus-devbox-profile`, backed by role `menu-corpus-devbox-role`)
  granting `s3:ListBucket` on the bucket and `s3:GetObject` / `s3:PutObject` on its
  objects. boto3 picks it up from instance metadata — **do not** put
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env` on the devbox.
- **Local / laptop pushes — scoped IAM user.** Off the devbox there is no instance
  profile, so use the scoped IAM user `restaurant-menu-corpus-sync` (inline policy:
  `ListBucket` + `Get/PutObject` on `v1/*` and `v2/*` only). Put its static keys in
  `.env`'s `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — these **outrank** `~/.aws`,
  so the push runs at least privilege rather than as root.
- **On a RunPod pod — a separate scoped key.** A pod is not the devbox and has no
  instance profile. Use the scoped IAM user `menu-corpus-pod` (its key is v2-scoped);
  `scripts/corpus/build_corpus.py --sync-every` uses it to stream the growing corpus
  to S3 during a build. Keep the key off-repo (pass it at runtime — e.g. from the
  scratchpad) and **never** put root/full AWS creds on a pod.

Never commit a real key value; `.env` is git-ignored (commit
[.env.example](../.env.example) instead).

## Keyless alternative for pod pulls — presigned URLs

If you'd rather hand a pod no key at all, presign the specific objects from a box
that already has credentials and pass the pod read-only, expiring URLs instead. The
pod holds no key material; each URL is read-only, single-object, and expires:

```bash
# on the devbox / any box with creds. --region is REQUIRED (the bucket is us-west-2;
# a presign that defaults to us-east-1 returns PermanentRedirect).
aws s3 presign "s3://restaurant-menu-corpus/v2/models/<family>/base/model.safetensors" \
  --expires-in 10800 --region us-west-2
```

Pipe the resulting URLs to the pod over ssh **stdin** (never argv — a presigned URL
carries its signature in the query string, so it is a secret that would otherwise
land in shell history / process lists).

## .env keys

```
S3_BUCKET=restaurant-menu-corpus
S3_PREFIX=v2
AWS_DEFAULT_REGION=us-west-2
# Off the devbox only — scoped `restaurant-menu-corpus-sync` keys (leave blank on the devbox):
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
```

## Note on identity

The bucket/role were created using root AWS credentials on this machine (one-off
setup only) — not relevant to the training code, just flagging it's not an IAM user.
