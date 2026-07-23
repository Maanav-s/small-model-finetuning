# S3 bucket setup

**This is a pointer.** The canonical, up-to-date S3 reference — bucket properties,
the v2 object layout, the sync tool, and IAM/credentials — lives in
[notes/S3_setup.md](notes/S3_setup.md). Edit that file, not this one.

TL;DR: bucket `restaurant-menu-corpus`, region `us-west-2`, prefix `v2/`; sync the
`data/` artifacts with `uv run python scripts/infra/corpus_sync.py push` / `pull`.
