# S3 bucket setup (Phase 2, Part 0.1 / 0.2)

Manual AWS setup is done. This is what exists and how WS-D (`scripts/cache_sync.py`) should talk to it.

## Bucket

- Name: `restaurant-menu-corpus`
- Region: `us-west-2` (same region as the `ec2-devbox` training instance, for free same-region egress)
- Private: Block Public Access is ON (all four settings)
- Default encryption: SSE-S3 (AES256)
- Object ownership: `BucketOwnerEnforced` — ACLs are disabled, access is IAM-policy-only
- **Versioning: ENABLED (2026-07-16).** Was off until then, which meant every delete/overwrite was
  permanent — dangerous for `models/`, whose merged checkpoint costs an H100 training run to
  re-derive. No lifecycle policy (nothing expires; at ~32 GB the whole bucket is ~$0.75/mo, so
  there is no cost case for pruning).

## What's in the bucket (surveyed 2026-07-16: 1034 objects, 32.40 GB)

| prefix | objs | size | notes |
|---|---|---|---|
| `v1/base-model/gemma-4-E4B-it/` | 6 | 16.02 GB | pinned base. **Not a faithful HF mirror** — no `preprocessor_config.json` (harmless: we serve text-only) |
| `v1/base-model/kv-shared-backfill/` | 1 | 110 MB | **the 54 dead KV-shared tensors** (layers 24–41 `k_norm`/`k_proj`/`v_proj`) for `to_text_only.py --base`. Lets a pod fix a merged checkpoint for vLLM **without** pulling the 16 GB base |
| `v1/models/gemma-menu-sft-20260714/merged/` | 6 | 15.88 GB | the v1 SFT student (bf16). **Missing 54 tensors** by construction — see experiments.md 2026-07-16 |
| `v1/models/gemma-menu-sft-20260714/adapter/` | 7 | 0.17 GB | LoRA (139 MB) + tokenizer |
| `v1/cache.sqlite` | 1 | 153.7 MB | frozen tool-call cache |
| `v1/traces/` | 1000 | 73.1 MB | teacher corpus |
| `v1/sft/train.jsonl` | 1 | 60.6 MB | 948 SFT examples |
| `v1/restaurants.jsonl`, `v1/splits.json` | 2 | 0.94 MB | corpus index + seeded splits |
| `v1/eval/2026071{4,5}/` | 11 | 1.3 MB | claude, gemma (4-bit full), gemma_bf16, gemma_bf16_newprompt |

Two model copies are **98.6% of the bytes**; everything else is rounding error.

### Integrity metadata (`x-amz-meta-md5`)

Convention: upload big artifacts with an `md5` metadata entry. Held by `sft/train.jsonl`
(`07dd4615…`), `base-model/model.safetensors` (`2ef04081…`), and `kv-shared-backfill`
(`eb53b4c4…`). **GAP: `models/…/merged/model.safetensors` has NO metadata** — the one artifact
you cannot re-derive is the one you cannot verify. Its ETag is `814e974e…-1894`, i.e. a
**multipart** ETag (1894 × 8 MB parts), which is *not* an md5, and it carries no
`ChecksumSHA256`/`CRC32C` either. Fill this in from the next pod that pulls it (it downloads the
whole file anyway — `md5sum` it there, then `copy-object --metadata-directive REPLACE`); versioning
now makes that copy-in-place safe.

## Credentials — no static keys

The devbox EC2 instance (`i-05cb4cfbe23ff2efd`) has an **IAM instance profile** attached:
`menu-corpus-devbox-profile`, backed by role `menu-corpus-devbox-role`.

The role's policy grants exactly:
- `s3:ListBucket` on `arn:aws:s3:::restaurant-menu-corpus`
- `s3:GetObject` / `s3:PutObject` on `arn:aws:s3:::restaurant-menu-corpus/*`

boto3 picks this up automatically via the EC2 instance metadata service — **do not add
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` to `.env`**. Just use a default boto3
session/client (`boto3.client("s3")`) with no explicit credentials, and it will resolve
the role on this box. Static keys would only be needed if this code ever runs somewhere
other than the devbox.

## Getting corpus/models onto a RunPod pod — use presigned URLs, not keys

A pod is not the devbox, so it has no instance profile. The tempting fix is to mint an
access key for the scoped `menu-corpus-pod` IAM user and drop it in `~/.aws/credentials`
on the pod — **don't**: that's exactly the long-lived static key this project bans, and the
secret is only shown once at creation (so it tends to get re-minted and left behind).

**Presign from a box that already has credentials, and hand the pod URLs instead.** The pod
holds no key material, each URL is read-only, single-object, and expires:

```bash
# on the devbox / any box with creds. --region is REQUIRED: the bucket is us-west-2, and a
# presign that defaults to us-east-1 returns PermanentRedirect ("must be addressed using the
# specified endpoint") -- the same cross-region bounce AWS_DEFAULT_REGION guards against.
for f in config.json tokenizer.json model.safetensors; do
  aws s3 presign "s3://restaurant-menu-corpus/v1/models/<ckpt>/merged/$f" \
    --expires-in 10800 --region us-west-2
done
```

Pipe the resulting `curl` script to the pod over ssh **stdin** (never argv — a presigned URL
carries its signature in the query string, so it is a secret and would otherwise land in
shell history / process lists). Measured 2026-07-16: 16 GB pulled in ~2 min.

Pushing results back needs `aws s3 presign` on a `put-object` (or just pull from the devbox
after the run). If you do create a key anyway, it needs the user's **explicit** say-so.

## What to put in `.env` / `.env.example`

```
S3_BUCKET=restaurant-menu-corpus
S3_PREFIX=v1
```

(`S3_PREFIX` is just a namespace under the bucket — pick something version-like since
the cache is a frozen, dated snapshot per Part 1 framing. `v1` is a reasonable default
if nothing else is decided yet.)

## Note on identity

The bucket/role were created using root AWS credentials on this machine (one-off setup
only) — not relevant to the training code, just flagging it's not an IAM user.