# S3 bucket setup (Phase 2, Part 0.1 / 0.2)

Manual AWS setup is done. This is what exists and how WS-D (`scripts/cache_sync.py`) should talk to it.

## Bucket

- Name: `restaurant-menu-corpus`
- Region: `us-west-2` (same region as the `ec2-devbox` training instance, for free same-region egress)
- Private: Block Public Access is ON (all four settings)
- Default encryption: SSE-S3 (AES256)
- Object ownership: `BucketOwnerEnforced` — ACLs are disabled, access is IAM-policy-only

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