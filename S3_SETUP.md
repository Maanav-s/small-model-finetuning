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
