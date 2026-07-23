"""Run the H2/H3 experiment matrix: budgets x strategies x seeds.

Each run is an independent process; --parallel P runs P at a time, each pinned
to total_threads/P torch threads so processes don't fight over cores.

Usage:
    python scripts/run_matrix.py --data-dir data/kowiki --steps 10000 \
        --budgets 4 4.5 --strategies connectivity random kquant uniform \
        --seeds 1337 1338 1339 --parallel 4 --total-threads 8 --out runs/matrix

Completed runs (summary.json exists) are skipped, so the matrix is resumable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument(
        "--steps", type=int, default=None,
        help="omit to use the --config / train.py default",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--budgets", type=float, nargs="+", default=[4.0])
    ap.add_argument(
        "--strategies", type=str, nargs="+",
        default=[
            "connectivity", "fisher", "random", "random_churn",
            "kquant", "uniform",
        ],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 1338, 1339])
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--total-threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--out", type=str, default="runs/matrix")
    ap.add_argument(
        "--config", type=str, default="",
        help="base JSON config passed to every run (CLI flags override it)",
    )
    args = ap.parse_args()

    threads = max(1, args.total_threads // args.parallel)
    jobs = []
    for budget, strategy, seed in itertools.product(
        args.budgets, args.strategies, args.seeds
    ):
        name = f"b{budget}_{strategy}_s{seed}"
        out_dir = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out_dir, "summary.json")):
            print(f"skip (done): {name}")
            continue
        cmd = [
            sys.executable, "-m", "myelin.train",
            *(["--config", args.config] if args.config else []),
            "--strategy", strategy,
            "--seed", str(seed),
            *(["--steps", str(args.steps)] if args.steps is not None else []),
            *(
                ["--batch-size", str(args.batch_size)]
                if args.batch_size is not None
                else []
            ),
            "--budget", str(budget),
            "--out", out_dir,
            "--threads", str(threads),
        ]
        if args.synthetic:
            cmd.append("--synthetic")
        else:
            cmd += ["--data-dir", args.data_dir]
        jobs.append((name, cmd))

    print(f"{len(jobs)} jobs, {args.parallel} parallel x {threads} threads each")
    os.makedirs(args.out, exist_ok=True)
    running: list[tuple[str, subprocess.Popen]] = []
    failed = []
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < args.parallel:
            name, cmd = queue.pop(0)
            log = open(os.path.join(args.out, f"{name}.log"), "w")
            print(f"start: {name}")
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            running.append(
                (
                    name,
                    subprocess.Popen(
                        cmd, stdout=log, stderr=subprocess.STDOUT, cwd=repo_root
                    ),
                )
            )
        done_idx = None
        for i, (name, proc) in enumerate(running):
            rc = proc.poll()
            if rc is not None:
                done_idx = i
                print(f"{'done' if rc == 0 else 'FAILED'}: {name}")
                if rc != 0:
                    failed.append(name)
                break
        if done_idx is not None:
            running.pop(done_idx)
        else:
            import time

            time.sleep(2)

    print(json.dumps({"failed": failed}, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
