"""Entry point: run the Phase 1 agent loop (src/run_agent.py) with 4-bit
quantization on.

run_agent.py does its argparse + model load at module top level, so we invoke
it as a subprocess with --quantize rather than importing it. Any extra args are
forwarded through (e.g. `uv run python main.py` just runs it quantized).
"""

import subprocess
import sys
from pathlib import Path

RUN_AGENT = Path(__file__).parent / "src" / "run_agent.py"


def main():
    cmd = [sys.executable, str(RUN_AGENT), "--quantize", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
