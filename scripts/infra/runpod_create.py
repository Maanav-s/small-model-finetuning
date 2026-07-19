"""Create a RunPod GPU pod that can actually serve Gemma-4 on vLLM.

WHY THIS EXISTS: `runpodctl create pod` has **no CUDA/driver filter** (runpod/runpodctl#253),
so it lands on whatever host the scheduler picks -- in practice CUDA 12.8 / driver 570.
vLLM >=0.20 is the first version with Gemma-4 support and it ships torch 2.11+**cu130**,
which requires **driver >=580 (CUDA 13)**. There is no vLLM pin that escapes this (0.12/0.18
are cu128 but have no gemma4; 0.20+ have gemma4 but are cu130), so a 12.8 host simply cannot
serve Gemma-4 at any version. Symptom:

    RuntimeError: The NVIDIA driver on your system is too old (found version 12080)

The **REST API** exposes `allowedCudaVersions`, which the CLI does not -- that is the whole
point of this script. Note the container **image is irrelevant** to this problem: the driver
is a property of the HOST, so picking a "CUDA 13 image" does NOT help. You must filter the host.

Auth: reads the api key from ~/.runpod/config.toml (same file runpodctl uses), or $RUNPOD_API_KEY.

    uv run python scripts/runpod_create.py --name gemma-vllm --gpu "NVIDIA A100 80GB PCIe"
    uv run python scripts/runpod_create.py --dry-run          # print the request body only
"""

import argparse
import json
import os
import pathlib
import sys
import tomllib
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1/pods"

# The driver requirement, as an allowedCudaVersions enum value. The API accepts
# '13.0','12.9',...,'11.8'; anything below 13.0 cannot run vLLM's cu130 torch.
CUDA_13 = "13.0"

# Same image the 12.8 pods used -- deliberately unchanged, to make the point that the
# image is not the fix. Only allowedCudaVersions moves the needle.
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    cfg = pathlib.Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        got = tomllib.loads(cfg.read_text()).get("apikey")
        if got:
            return got
    sys.exit("no RunPod API key: set $RUNPOD_API_KEY or run `runpodctl config`")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="gemma-vllm")
    p.add_argument("--gpu", default="NVIDIA A100 80GB PCIe",
                   help="gpuTypeId, e.g. 'NVIDIA A100 80GB PCIe' / 'NVIDIA H100 PCIe'")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--cuda", default=CUDA_13,
                   help=f"allowedCudaVersions entry (default {CUDA_13}; <13.0 cannot serve Gemma-4 on vLLM)")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--disk", type=int, default=180,
                   help="container disk GB; needs ~16 (merged) + ~15 (text-only) + vllm wheels")
    p.add_argument("--dry-run", action="store_true", help="print the request body, do not create")
    args = p.parse_args()

    body = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": args.gpu_count,
        # THE FIX: pin the host driver. Without this you get CUDA 12.8 and vLLM dies.
        "allowedCudaVersions": [args.cuda],
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "ports": ["8000/http", "22/tcp"],
        "cloudType": "SECURE",
        "supportPublicIp": True,
    }

    if args.dry_run:
        print(json.dumps(body, indent=2))
        return

    req = urllib.request.Request(
        REST, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key()}",
                 # RunPod sits behind Cloudflare, which 403s urllib's default UA.
                 "User-Agent": "curl/8.4.0"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as f:
            d = json.load(f)
    except urllib.error.HTTPError as e:
        sys.exit(f"create failed: HTTP {e.code}\n{e.read().decode()[:600]}")

    print(json.dumps({k: d.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "machineId")}, indent=2))
    print(f"\nssh: check `nvidia-smi` reports driver >= 580 / CUDA {args.cuda} before installing vLLM.")


if __name__ == "__main__":
    main()
