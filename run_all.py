"""
run_all.py
----------
Unified entrypoint for classical and neural tracks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print(f"[run_all] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run classical/neural experiment tracks")
    parser.add_argument("--track", choices=["classical", "neural", "all"], default="all")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--glove-path", type=str, default="")
    args = parser.parse_args()

    py = sys.executable
    if args.track in ("classical", "all"):
        cmd = [py, "run_classical.py"]
        if args.experiment is not None:
            cmd += ["--experiment", str(args.experiment)]
        _run(cmd)

    if args.track in ("neural", "all"):
        cmd = [py, "run_neural.py"]
        if args.experiment is not None:
            cmd += ["--experiment", str(args.experiment)]
        if args.skip_bert:
            cmd += ["--skip-bert"]
        if args.glove_path:
            cmd += ["--glove-path", args.glove_path]
        _run(cmd)


if __name__ == "__main__":
    main()
