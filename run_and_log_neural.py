"""
run_and_log_neural.py
---------------------
Run neural experiments and auto-log:
1) parameters before run
2) summary metrics after run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import uuid


LOG_PATH = "neural_training_log.txt"


def _read_summary_rows(experiment: int) -> list[dict]:
    path = os.path.join("outputs", "neural", f"exp{experiment}", "summary_mean_std.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_log(block: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(block + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run run_neural.py and log params/results.")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--glove-path", type=str, default="")
    parser.add_argument("--skip-bert", action="store_true")
    args = parser.parse_args()

    run_id = f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    start_time = dt.datetime.now().isoformat(timespec="seconds")
    cmd = ["python", "run_neural.py", "--experiment", str(args.experiment), "--seeds"] + [str(s) for s in args.seeds]
    if args.glove_path:
        cmd += ["--glove-path", args.glove_path]
    if args.skip_bert:
        cmd += ["--skip-bert"]

    pre_block = "\n".join(
        [
            "=" * 80,
            f"RUN_ID: {run_id}",
            f"START_TIME: {start_time}",
            f"COMMAND: {' '.join(cmd)}",
            f"PARAMS: {json.dumps({'experiment': args.experiment, 'seeds': args.seeds, 'glove_path': args.glove_path, 'skip_bert': args.skip_bert}, ensure_ascii=False)}",
            "STATUS: RUNNING",
            "RESULTS: PENDING",
            "=" * 80,
        ]
    )
    _append_log(pre_block)

    proc = subprocess.run(cmd, check=False)
    end_time = dt.datetime.now().isoformat(timespec="seconds")
    summary_rows = _read_summary_rows(args.experiment)
    result_payload = {
        "return_code": proc.returncode,
        "end_time": end_time,
        "summary_rows": summary_rows,
    }
    post_block = "\n".join(
        [
            f"RUN_ID: {run_id}",
            f"STATUS: {'DONE' if proc.returncode == 0 else 'FAILED'}",
            f"RESULTS: {json.dumps(result_payload, ensure_ascii=False)}",
            "-" * 80,
        ]
    )
    _append_log(post_block)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
