#!/usr/bin/env python3
"""Build and run selected OpenACC/NVHPC benchmarks without parsing timings.

Examples
--------
    cd /hs/work0/home/users/u0001967/HeCBench/src3
    python3 scripts/run_acc_nvd_benchmarks.py idivide jacobi
    python3 scripts/run_acc_nvd_benchmarks.py --benchmark-list acc-benchmarks.txt

Each requested benchmark is resolved to ``<name>-<variant>`` (or can be passed
with its suffix), then the script runs exactly::

    make -f Makefile.NVD
    make -f Makefile.NVD run

The complete build and run output is stored in one log per benchmark.  This
script deliberately does not extract timings and never creates or edits a CSV.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def load_list(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names


def resolve_benchmark(src_dir: Path, name: str, variant: str) -> Path | None:
    suffix = f"-{variant}"
    candidates = [name] if name.endswith(suffix) else [f"{name}{suffix}", name]
    for candidate in candidates:
        path = src_dir / candidate
        if path.is_dir():
            return path
    return None


def run_step(command: list[str], directory: Path, log) -> int:
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    return subprocess.run(
        command,
        cwd=directory,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        check=False,
    ).returncode


def append_program_logs(directory: Path, log) -> None:
    """Copy the files produced by the NVIDIA Makefile command into *log*."""
    for filename in ("log.std", "log.err", "log.time"):
        source = directory / filename
        if not source.is_file():
            continue
        log.write(f"\n--- {filename} ---\n")
        with source.open("r", errors="replace") as input_log:
            log.write(input_log.read())
        if source.stat().st_size and not source.read_bytes().endswith(b"\n"):
            log.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and run OpenACC/NVHPC benchmarks, retaining raw logs only."
    )
    parser.add_argument("benchmarks", nargs="*", help="Benchmark names, e.g. idivide or idivide-acc.")
    parser.add_argument("--benchmark-list", type=Path, help="Text file containing one benchmark name per line.")
    parser.add_argument(
        "--variant", choices=("acc", "acc_gv", "omp_nvc"), default="acc",
        help="Implementation suffix to run (default: acc).",
    )
    parser.add_argument(
        "--src-dir", type=Path, default=Path(__file__).resolve().parent.parent,
        help="Directory that contains <benchmark>-acc directories (default: src3).",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        help="Output directory for raw logs (default: <src-dir>/results-acc-nvd).",
    )
    args = parser.parse_args()

    requested = list(args.benchmarks)
    if args.benchmark_list:
        requested.extend(load_list(args.benchmark_list))
    if not requested:
        parser.error("pass at least one benchmark name or --benchmark-list")

    src_dir = args.src_dir.resolve()
    log_dir = (args.log_dir or src_dir / "results-acc-nvd").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str, int]] = []
    successes: list[str] = []
    for requested_name in requested:
        bench_dir = resolve_benchmark(src_dir, requested_name, args.variant)
        if bench_dir is None:
            failures.append((requested_name, "resolve", 1))
            print(f"[ FAIL ] {requested_name} (directory not found)")
            continue
        makefile = bench_dir / "Makefile.NVD"
        if not makefile.is_file():
            failures.append((bench_dir.name, "build", 1))
            print(f"[ FAIL ] {bench_dir.name} (Makefile.NVD not found)")
            continue

        log_path = log_dir / f"{bench_dir.name}.log"
        started = time.monotonic()
        with log_path.open("w") as log:
            log.write(f"=== {bench_dir.name} ===\nbenchmark dir : {bench_dir}\n")
            build_rc = run_step(["make", "-f", "Makefile.NVD"], bench_dir, log)
            if build_rc == 0:
                run_rc = run_step(["make", "-f", "Makefile.NVD", "run"], bench_dir, log)
                append_program_logs(bench_dir, log)
            else:
                run_rc = None
            log.write(f"\nwall time : {time.monotonic() - started:.3f} s\n")

        if build_rc != 0:
            failures.append((bench_dir.name, "build", build_rc))
            print(f"[ FAIL ] {bench_dir.name} (build rc={build_rc})")
        elif run_rc != 0:
            failures.append((bench_dir.name, "run", run_rc))
            print(f"[ FAIL ] {bench_dir.name} (run rc={run_rc})")
        else:
            successes.append(bench_dir.name)
            print(f"[  OK  ] {bench_dir.name}")

    print(f"\nLogs : {log_dir}")
    print(f"Passed: {len(successes)}")
    if failures:
        print(f"Failed: {len(failures)}")
        for name, stage, rc in failures:
            print(f"  - {name}  (stage={stage}, rc={rc})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
