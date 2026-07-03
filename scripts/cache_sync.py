"""Sync the Phase-2 `data/` artifacts with S3 (WS-D, notes/phase2_plan.md).

`data/` is git-ignored; its source of truth is `s3://$S3_BUCKET/$S3_PREFIX/`.
This script mirrors the artifact set defined by contract 1.1:

  cache.sqlite       restaurants.jsonl       splits.json
  labels.jsonl       traces/<restaurant_id>.json  (synced per-file)

  uv run python scripts/cache_sync.py push                 # local -> S3
  uv run python scripts/cache_sync.py pull                 # S3 -> local
  uv run python scripts/cache_sync.py push --only traces --only splits.json
  uv run python scripts/cache_sync.py push --dry-run       # plan only, no network

Configuration comes from the repo-root `.env` (see .env.example): `S3_BUCKET`
(required unless --dry-run), `S3_PREFIX` (optional key prefix). Credentials use
boto3's default chain — EC2 instance profile if present, else `AWS_ACCESS_KEY_ID`
/ `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` from the environment/.env.

WAL rule: `cache.sqlite` runs in WAL mode, so its live state spans the db plus
`-wal`/`-shm` sidecars — uploading the bare file mid-run risks a stale or torn
snapshot. `push` therefore snapshots it first (`VACUUM INTO` a temp file, which
folds in the WAL and verifies cleanly), uploads the snapshot under the canonical
`cache.sqlite` key, and deletes the temp. The snapshot (and its integrity check)
also runs under --dry-run so the step is exercisable without a bucket.

Skip-unchanged guard: every upload stores the file's md5 as object *metadata*
(`x-amz-meta-md5`), because the S3 ETag stops being a plain md5 for multipart
uploads. Comparison order: metadata md5 -> single-part ETag -> size (weak
fallback). Unchanged files are skipped and logged. `pull` never deletes local
files that are absent remotely; it logs them as kept.

--dry-run makes no network calls at all (so it runs before the bucket exists);
it prints the plan with local hashes, substituting a placeholder bucket if
S3_BUCKET is unset.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load S3_BUCKET / S3_PREFIX / AWS_* from the repo-root .env regardless of cwd
# (same pattern as src/gemma/run_agent.py).
load_dotenv(REPO_ROOT / ".env")

DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Contract 1.1 artifact set, as paths relative to data/. "traces" is a
# directory, synced per-file.
FLAT_ARTIFACTS = ("cache.sqlite", "restaurants.jsonl", "splits.json", "labels.jsonl")
TRACES_DIR = "traces"
ARTIFACTS = FLAT_ARTIFACTS + (TRACES_DIR,)

PLACEHOLDER_BUCKET = "DRY-RUN-PLACEHOLDER-BUCKET"


# --------------------------------------------------------------------------
# Hashing + sqlite snapshot helpers
# --------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    """Hex md5 of a file, streamed (cache.sqlite can be large)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_sqlite(src: Path, dst: Path) -> None:
    """Snapshot a (possibly WAL-mode, possibly live) sqlite db to `dst`.

    `VACUUM INTO` writes a compact, self-contained copy that folds in any
    pending -wal content, without touching the source db's journal mode or
    blocking concurrent readers. `dst` must not already exist.
    """
    if dst.exists():
        dst.unlink()
    # Deliberately NOT mode=ro: opening read-write lets sqlite recover the WAL
    # if the last writer exited uncleanly (read-only connections can't, and
    # fail with SQLITE_READONLY_RECOVERY). VACUUM INTO itself never mutates
    # the source db's content.
    con = sqlite3.connect(str(src))
    try:
        con.execute("VACUUM INTO ?", (str(dst),))
    finally:
        con.close()


def verify_sqlite(path: Path) -> str:
    """Open a sqlite file read-only and run an integrity check ('ok' if sound)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()


# --------------------------------------------------------------------------
# Skip-unchanged guard (pure — unit-testable without boto3)
# --------------------------------------------------------------------------

def is_unchanged(local_md5: str, local_size: int, head: dict | None) -> tuple[bool, str]:
    """Compare a local file against an S3 head_object response.

    Returns (unchanged, reason). Comparison order:
      1. our `md5` object metadata (reliable regardless of multipart ETags),
      2. the ETag itself when it is a plain md5 (single-part upload, no '-'),
      3. size only (weak fallback for multipart objects lacking our metadata).
    """
    if head is None:
        return False, "remote object missing"
    meta_md5 = (head.get("Metadata") or {}).get("md5")
    if meta_md5:
        if meta_md5 == local_md5:
            return True, "md5 metadata matches"
        return False, "md5 metadata differs"
    etag = (head.get("ETag") or "").strip('"')
    if etag and "-" not in etag:
        if etag == local_md5:
            return True, "ETag (single-part md5) matches"
        return False, "ETag (single-part md5) differs"
    if head.get("ContentLength") == local_size:
        return True, "size matches (multipart ETag, no md5 metadata — weak check)"
    return False, "size differs"


# --------------------------------------------------------------------------
# S3 wrapper (client injectable for tests; boto3 imported lazily so --dry-run
# and unit tests never touch it)
# --------------------------------------------------------------------------

class S3Remote:
    def __init__(self, bucket: str, prefix: str, client=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3  # default credential chain: instance profile -> env keys

            self._client = boto3.client("s3")
        return self._client

    def key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def url(self, rel: str) -> str:
        return f"s3://{self.bucket}/{self.key(rel)}"

    def head(self, rel: str) -> dict | None:
        """head_object, or None if the key does not exist."""
        from botocore.exceptions import ClientError

        try:
            return self.client.head_object(Bucket=self.bucket, Key=self.key(rel))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def upload(self, path: Path, rel: str, md5: str) -> None:
        # md5 stored as object metadata: the guard's reliable comparison key
        # (ETags are not md5s for multipart uploads).
        self.client.upload_file(
            str(path),
            self.bucket,
            self.key(rel),
            ExtraArgs={"Metadata": {"md5": md5, "mtime": str(path.stat().st_mtime)}},
        )

    def download(self, rel: str, path: Path) -> None:
        """Download atomically: to a temp file beside the target, then rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
        os.close(fd)
        try:
            self.client.download_file(self.bucket, self.key(rel), tmp)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def list_keys(self, rel_prefix: str) -> list[str]:
        """Relative paths (under our prefix) of all objects below rel_prefix/."""
        full = self.key(rel_prefix).rstrip("/") + "/"
        # self.key("") is "<prefix>/" when a prefix is set, "" otherwise.
        strip = len(self.key(""))
        rels = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):  # skip directory markers
                    rels.append(key[strip:])
        return rels


# --------------------------------------------------------------------------
# push / pull
# --------------------------------------------------------------------------

def _log(action: str, msg: str) -> None:
    print(f"[{action}] {msg}")


def _local_files_for(name: str, data_dir: Path) -> list[tuple[Path, str]]:
    """(local_path, rel_key) pairs for one artifact name. Missing -> []."""
    if name == TRACES_DIR:
        root = data_dir / TRACES_DIR
        if not root.is_dir():
            return []
        return sorted(
            (p, p.relative_to(data_dir).as_posix())
            for p in root.rglob("*")
            if p.is_file()
        )
    p = data_dir / name
    return [(p, name)] if p.is_file() else []


def do_push(remote: S3Remote, data_dir: Path, names: list[str], dry_run: bool) -> dict:
    counts = {"pushed": 0, "skipped": 0, "missing": 0}
    tag = "push (dry-run)" if dry_run else "push"
    for name in names:
        pairs = _local_files_for(name, data_dir)
        if not pairs:
            _log(tag, f"missing local: {data_dir / name} — nothing to push")
            counts["missing"] += 1
            continue
        for path, rel in pairs:
            src = path
            snapshot = None
            try:
                if rel == "cache.sqlite":
                    # WAL rule: never upload the bare db — snapshot via VACUUM
                    # INTO so pending -wal content is folded in and the upload
                    # is a self-contained, torn-write-free copy. Runs in
                    # --dry-run too, so the step is verifiable without a bucket.
                    fd, tmp = tempfile.mkstemp(prefix="cache-snapshot-", suffix=".sqlite")
                    os.close(fd)
                    snapshot = Path(tmp)
                    snapshot_sqlite(path, snapshot)
                    integrity = verify_sqlite(snapshot)
                    _log(tag, f"snapshotted {path} -> {snapshot} "
                              f"({snapshot.stat().st_size} bytes, integrity_check={integrity})")
                    if integrity != "ok":
                        raise RuntimeError(f"snapshot of {path} failed integrity_check: {integrity}")
                    src = snapshot
                local_md5 = md5_file(src)
                local_size = src.stat().st_size
                if dry_run:
                    _log(tag, f"would push {rel} -> {remote.url(rel)} "
                              f"(md5={local_md5}, {local_size} bytes; remote not checked in dry-run)")
                    counts["pushed"] += 1
                    continue
                unchanged, reason = is_unchanged(local_md5, local_size, remote.head(rel))
                if unchanged:
                    _log(tag, f"skipped {rel} (unchanged: {reason})")
                    counts["skipped"] += 1
                else:
                    remote.upload(src, rel, local_md5)
                    _log(tag, f"pushed {rel} -> {remote.url(rel)} ({reason}; md5={local_md5})")
                    counts["pushed"] += 1
            finally:
                if snapshot is not None and snapshot.exists():
                    snapshot.unlink()
    return counts


def do_pull(remote: S3Remote, data_dir: Path, names: list[str], dry_run: bool) -> dict:
    counts = {"pulled": 0, "skipped": 0, "missing": 0, "kept_local_only": 0}
    tag = "pull (dry-run)" if dry_run else "pull"

    def pull_one(rel: str, head: dict | None) -> None:
        local = data_dir / rel
        if head is None:
            if local.exists():
                _log(tag, f"missing remote: {remote.url(rel)} (local copy kept, never deleted)")
                counts["kept_local_only"] += 1
            else:
                _log(tag, f"missing remote: {remote.url(rel)} — nothing to pull")
                counts["missing"] += 1
            return
        if local.is_file():
            unchanged, reason = is_unchanged(md5_file(local), local.stat().st_size, head)
            if unchanged:
                _log(tag, f"skipped {rel} (unchanged: {reason})")
                counts["skipped"] += 1
                return
            why = reason
        else:
            why = "no local copy"
        remote.download(rel, local)
        if rel == "cache.sqlite":
            # The pulled file is a clean snapshot; stale WAL sidecars from a
            # previous local db would corrupt it on next open.
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(local) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
                    _log(tag, f"removed stale sidecar {sidecar.name} (pulled db is a clean snapshot)")
        _log(tag, f"pulled {remote.url(rel)} -> {local} ({why})")
        counts["pulled"] += 1

    for name in names:
        if dry_run:
            if name == TRACES_DIR:
                _log(tag, f"would list {remote.url(TRACES_DIR)}/ and pull each object "
                          f"into {data_dir / TRACES_DIR}/ (remote not listed in dry-run)")
            else:
                _log(tag, f"would pull {remote.url(name)} -> {data_dir / name} "
                          f"if present and changed (remote not checked in dry-run)")
            counts["pulled"] += 1
            continue
        if name == TRACES_DIR:
            remote_rels = set(remote.list_keys(TRACES_DIR))
            for rel in sorted(remote_rels):
                pull_one(rel, remote.head(rel))
            # Never delete local files absent remotely — log them instead.
            for path, rel in _local_files_for(TRACES_DIR, data_dir):
                if rel not in remote_rels:
                    _log(tag, f"local-only: {path} (absent remotely; kept, never deleted)")
                    counts["kept_local_only"] += 1
            if not remote_rels:
                _log(tag, f"missing remote: {remote.url(TRACES_DIR)}/ has no objects")
                counts["missing"] += 1
        else:
            pull_one(name, remote.head(name))
    return counts


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _normalize_only(values: list[str] | None) -> list[str]:
    if not values:
        return list(ARTIFACTS)
    out = []
    for v in values:
        v = v.strip().strip("/")
        if v not in ARTIFACTS:
            sys.exit(f"error: --only {v!r} is not a Phase-2 artifact; "
                     f"choose from: {', '.join(ARTIFACTS)}")
        if v not in out:
            out.append(v)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd, help_ in (("push", "upload local data/ artifacts to S3"),
                       ("pull", "download S3 artifacts into data/")):
        p = sub.add_parser(cmd, help=help_)
        p.add_argument(
            "--only", action="append", metavar="NAME",
            help=f"sync only this artifact (repeatable); one of: {', '.join(ARTIFACTS)}",
        )
        p.add_argument(
            "--dry-run", action="store_true",
            help="print what WOULD transfer (and why) without any S3 network calls; "
            "runs even without S3_BUCKET (placeholder bucket). The cache.sqlite "
            "VACUUM INTO snapshot still executes so it stays verifiable.",
        )
        p.add_argument(
            "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
            help=f"local artifact root (default: {DEFAULT_DATA_DIR})",
        )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    bucket = os.environ.get("S3_BUCKET", "").strip()
    prefix = os.environ.get("S3_PREFIX", "").strip()
    if not bucket:
        if not args.dry_run:
            sys.exit(
                "error: S3_BUCKET is not set. Add S3_BUCKET (and optionally S3_PREFIX) "
                "to the repo-root .env — see .env.example — or export it in the "
                "environment. To preview the sync plan without a bucket, re-run "
                "with --dry-run."
            )
        bucket = PLACEHOLDER_BUCKET
        print(f"note: S3_BUCKET unset; dry-run using placeholder bucket {bucket!r}")

    remote = S3Remote(bucket, prefix)
    names = _normalize_only(args.only)
    print(f"{args.command}: {'DRY RUN, ' if args.dry_run else ''}"
          f"data dir {args.data_dir} <-> s3://{bucket}/{prefix + '/' if prefix else ''} "
          f"(artifacts: {', '.join(names)})")

    if args.command == "push":
        counts = do_push(remote, args.data_dir, names, args.dry_run)
    else:
        counts = do_pull(remote, args.data_dir, names, args.dry_run)
    print(f"{args.command} done: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
