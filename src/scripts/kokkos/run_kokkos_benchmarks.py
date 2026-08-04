#!/usr/bin/env python3
"""
run_kokkos_benchmarks.py

Build and run every Kokkos benchmark found in the current directory.

Usage
-----
    cd /path/to/HeCBench/src
    python3 ../scripts/run_kokkos_benchmarks.py
    python3 ../scripts/run_kokkos_benchmarks.py --benchmark-list ../scripts/kokkos-benchmark-list.txt

The script discovers every sub-directory matching ``*-kokkos`` that contains
a ``CMakeLists.txt`` and performs the equivalent of::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \\
          -DKokkos_ROOT=$Kokkos_ROOT \\
          -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER
    cmake --build build -j
    cmake --build build --target run

A benchmark is reported as *success* only when all three steps return zero.
Successful benchmarks are recorded in ``kokkos-passed.txt`` next to the logs,
so later runs can skip them instead of rebuilding everything.
Use ``--benchmark-list`` to run only a selected batch of benchmarks.
At the end of the run a summary is printed and the script exits with a
non-zero status if at least one benchmark failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KOKKOS_ROOT = os.environ.get(
    "Kokkos_ROOT", str(Path.home() / "kokkos-install")
)
DEFAULT_CXX = os.environ.get(
    "CMAKE_CXX_COMPILER", str(Path.home() / "kokkos" / "bin" / "nvcc_wrapper")
)


@dataclass
class BenchmarkResult:
    name: str
    stage: str = "configure"  # configure / build / run / done
    returncode: int = 0
    duration: float = 0.0
    log_path: Path | None = None
    error_excerpt: str = ""
    times: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_kokkos_benchmarks(root: Path) -> list[Path]:
    """Return every ``*-kokkos`` directory under *root* that has a CMakeLists.txt."""
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("-kokkos"))
    return [p for p in candidates if (p / "CMakeLists.txt").is_file()]


def run_step(
    cmd: list[str],
    cwd: Path,
    log_file,
    env: dict[str, str],
    timeout: int | None,
) -> int:
    """Run *cmd* in *cwd*, append output to *log_file*, return the exit code."""
    log_file.write(f"\n$ {' '.join(cmd)}\n")
    log_file.flush()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        log_file.write(f"\n[run_kokkos_benchmarks] timeout after {timeout}s\n")
        return 124  # conventional timeout exit code


def last_lines(path: Path, n: int = 20) -> str:
    try:
        with path.open("r", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:]).rstrip()
    except OSError:
        return ""


def load_passed(path: Path) -> set[str]:
    try:
        with path.open("r") as f:
            return {
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
    except FileNotFoundError:
        return set()


def load_benchmark_list(path: Path) -> list[str]:
    """Read benchmark names from *path*, ignoring blank lines and comments."""
    names: list[str] = []
    with path.open("r") as f:
        for line in f:
            name = line.split("#", 1)[0].strip()
            if not name:
                continue
            if not name.endswith("-kokkos"):
                name = f"{name}-kokkos"
            names.append(name)
    return names


def append_passed(path: Path, name: str, passed: set[str]) -> None:
    if name in passed:
        return
    with path.open("a") as f:
        f.write(f"{name}\n")
    passed.add(name)


def benchmark_base_name(name: str) -> str:
    return name.removesuffix("-kokkos")


def default_bench_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "scripts" / "benchmarks" / "subset.json"


def load_benchmark_regexes(path: Path) -> dict[str, str]:
    try:
        with path.open() as f:
            data = json.load(f)
    except OSError:
        return {}
    regexes: dict[str, str] = {}
    for name, entry in data.items():
        if isinstance(entry, list) and entry and isinstance(entry[0], str):
            regexes[name] = entry[0]
    return regexes


def _match_to_float(match) -> float | None:
    if isinstance(match, tuple):
        for item in match:
            if item:
                try:
                    return float(item)
                except ValueError:
                    continue
        return None
    try:
        return float(match)
    except ValueError:
        return None


def _unit_to_seconds(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "ns":
        return value * 1e-9
    if unit in {"us", "µs"}:
        return value * 1e-6
    if unit == "ms":
        return value * 1e-3
    return value


def extract_timing_values(log_path: Path, bench_name: str, regexes: dict[str, str]) -> list[float]:
    """Extract benchmark timing values from a Kokkos log.

    When the benchmark exists in autohecbench's benchmark data, use the same
    regex and keep the same unit as autohecbench. If not, fall back to a generic
    time-line parser and normalize those generic values to seconds.
    """
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return []

    regex = regexes.get(benchmark_base_name(bench_name))
    if regex:
        try:
            matches = re.findall(regex, text)
        except re.error:
            matches = []
        values = [value for match in matches if (value := _match_to_float(match)) is not None]
        if values:
            return [sum(values)]

    values: list[float] = []
    number_pattern = r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?"
    unit_pattern = r"ns|us|µs|ms|s|sec|secs|seconds"
    time_line = re.compile(
        rf"({number_pattern})"
        rf"\s*(?:\(({unit_pattern})\)|\b({unit_pattern})\b)"
    )
    unit_anywhere = re.compile(rf"\(({unit_pattern})\)|\b({unit_pattern})\b")
    number_anywhere = re.compile(number_pattern)
    positive_words = ("average", "total", "kernel", "execution", "elapsed", "runtime", "compute", "device", "offload", "inference")
    negative_words = ("cmake", "build", "built", "compile", "generating", "written to", "timeout after")
    for line in text.splitlines():
        lower = line.lower()
        if "time" not in lower and "runtime" not in lower:
            continue
        if not any(word in lower for word in positive_words):
            continue
        if any(word in lower for word in negative_words):
            continue
        matches = time_line.findall(line)
        for number, unit_a, unit_b in matches:
            unit = unit_a or unit_b
            values.append(_unit_to_seconds(float(number), unit))
        if not matches:
            unit_match = unit_anywhere.search(line)
            numbers = number_anywhere.findall(line)
            if unit_match and numbers:
                unit = unit_match.group(1) or unit_match.group(2)
                values.append(_unit_to_seconds(float(numbers[-1]), unit))

    return [sum(values)] if values else []


def load_csv_rows(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    if not path.is_file():
        return rows
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if row:
                rows[row[0]] = row[1:]
    return rows


def write_csv_rows(path: Path, rows: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for name in sorted(rows):
            writer.writerow([name, *rows[name]])


def update_csv_row(rows: dict[str, list[str]], name: str, values: list[float]) -> bool:
    if not values:
        return False
    rows[name] = [f"{value:.12g}" for value in values]
    return True


def build_and_run(
    bench_dir: Path,
    kokkos_root: str,
    cxx: str,
    log_dir: Path,
    timeout: int | None,
    keep_build: bool,
) -> BenchmarkResult:
    name = bench_dir.name
    res = BenchmarkResult(name=name)
    log_path = log_dir / f"{name}.log"
    res.log_path = log_path

    build_dir = bench_dir / "build"

    env = os.environ.copy()
    # Useful defaults that some Kokkos installs may need.
    env.setdefault("NVCC_WRAPPER_DEFAULT_COMPILER", "g++")

    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"=== {name} ===\n")
        log.write(f"benchmark dir : {bench_dir}\n")
        log.write(f"Kokkos_ROOT   : {kokkos_root}\n")
        log.write(f"CXX compiler  : {cxx}\n\n")

        # Always start from a clean build directory to avoid stale cache issues
        # (e.g. switching Kokkos_ROOT between L40S and Hopper installs).
        if build_dir.exists():
            shutil.rmtree(build_dir)

        # 1. configure
        rc = run_step(
            [
                "cmake",
                "-S", ".",
                "-B", "build",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DKokkos_ROOT={kokkos_root}",
                f"-DCMAKE_CXX_COMPILER={cxx}",
            ],
            cwd=bench_dir,
            log_file=log,
            env=env,
            timeout=timeout,
        )
        if rc != 0:
            res.stage = "configure"
            res.returncode = rc
            res.duration = time.monotonic() - start
            res.error_excerpt = last_lines(log_path)
            return res

        # 2. build
        rc = run_step(
            ["cmake", "--build", "build", "-j"],
            cwd=bench_dir,
            log_file=log,
            env=env,
            timeout=timeout,
        )
        if rc != 0:
            res.stage = "build"
            res.returncode = rc
            res.duration = time.monotonic() - start
            res.error_excerpt = last_lines(log_path)
            return res

        # 3. run
        rc = run_step(
            ["cmake", "--build", "build", "--target", "run"],
            cwd=bench_dir,
            log_file=log,
            env=env,
            timeout=timeout,
        )
        res.duration = time.monotonic() - start
        if rc != 0:
            res.stage = "run"
            res.returncode = rc
            res.error_excerpt = last_lines(log_path)
            return res

    res.stage = "done"
    res.returncode = 0
    if not keep_build:
        shutil.rmtree(build_dir, ignore_errors=True)
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing *-kokkos sub-directories (default: cwd).",
    )
    parser.add_argument(
        "--kokkos-root",
        default=DEFAULT_KOKKOS_ROOT,
        help=f"Path to Kokkos install (default: {DEFAULT_KOKKOS_ROOT}).",
    )
    parser.add_argument(
        "--cxx",
        default=DEFAULT_CXX,
        help=f"C++ compiler for CMake (default: {DEFAULT_CXX}).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory where per-benchmark logs are written (default: <src>/kokkos-logs).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-step timeout in seconds (default: no limit).",
    )
    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="Keep each benchmark's build/ directory after a successful run.",
    )
    parser.add_argument(
        "--benchmark-list",
        type=Path,
        default=None,
        help="Run only benchmarks listed in this file, one name per line.",
    )
    parser.add_argument(
        "--passed-file",
        type=Path,
        default=None,
        help="File listing benchmarks to skip after they passed (default: <log-dir>/kokkos-passed.txt).",
    )
    parser.add_argument(
        "--rerun-passed",
        action="store_true",
        help="Ignore the passed file and run benchmarks even if they already passed.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV file where extracted Kokkos timings are written (default: <src>/kokkos.csv).",
    )
    parser.add_argument(
        "--bench-data",
        type=Path,
        default=default_bench_data_path(),
        help="autohecbench benchmark data JSON used for timing regexes.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable kokkos.csv generation.",
    )
    args = parser.parse_args()

    src_dir: Path = args.src_dir.resolve()
    if not src_dir.is_dir():
        print(f"error: {src_dir} is not a directory", file=sys.stderr)
        return 2

    log_dir: Path = (args.log_dir or (src_dir / "kokkos-logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    passed_file: Path = (args.passed_file or (log_dir / "kokkos-passed.txt")).resolve()
    passed_file.parent.mkdir(parents=True, exist_ok=True)
    passed = set() if args.rerun_passed else load_passed(passed_file)
    csv_output: Path = (args.csv_output or (src_dir / "kokkos.csv")).resolve()
    timing_regexes = load_benchmark_regexes(args.bench_data.resolve())
    csv_rows = {} if args.no_csv else load_csv_rows(csv_output)

    benchmarks = find_kokkos_benchmarks(src_dir)
    if not benchmarks:
        print(f"No *-kokkos benchmark with CMakeLists.txt found in {src_dir}", file=sys.stderr)
        return 1
    selected_file: Path | None = args.benchmark_list.resolve() if args.benchmark_list else None
    if selected_file:
        try:
            selected_names = load_benchmark_list(selected_file)
        except OSError as exc:
            print(f"error: cannot read benchmark list {selected_file}: {exc}", file=sys.stderr)
            return 2

        by_name = {bench.name: bench for bench in benchmarks}
        missing = [name for name in selected_names if name not in by_name]
        benchmarks = [by_name[name] for name in selected_names if name in by_name]

        if missing:
            print(f"warning: {len(missing)} benchmark(s) from {selected_file} were not found:", file=sys.stderr)
            for name in missing:
                print(f"  - {name}", file=sys.stderr)
        if not benchmarks:
            print(f"No listed benchmark with CMakeLists.txt found in {src_dir}", file=sys.stderr)
            return 1

    print(f"Found {len(benchmarks)} Kokkos benchmark(s) in {src_dir}")
    print(f"Kokkos_ROOT = {args.kokkos_root}")
    print(f"CXX         = {args.cxx}")
    print(f"Logs        = {log_dir}")
    if not args.no_csv:
        print(f"CSV output  = {csv_output}")
        print(f"Bench data  = {args.bench_data.resolve()}")
    if selected_file:
        print(f"Benchmark list = {selected_file}")
    print(f"Passed file = {passed_file}")
    if args.rerun_passed:
        print("Passed file is ignored because --rerun-passed is set.")
    elif passed:
        print(f"Skipping {len(passed)} benchmark(s) already listed as passed.")
    print()

    results: list[BenchmarkResult] = []
    skipped: list[str] = []
    for bench in benchmarks:
        if bench.name in passed:
            skipped.append(bench.name)
            print(f"[ SKIP ] {bench.name}  (already passed)")
            if not args.no_csv and bench.name not in csv_rows:
                log_path = log_dir / f"{bench.name}.log"
                values = extract_timing_values(log_path, bench.name, timing_regexes)
                if update_csv_row(csv_rows, bench.name, values):
                    print(f"[  CSV ] {bench.name}  ({', '.join(csv_rows[bench.name])})")
            continue

        print(f"[ .. ] {bench.name}", flush=True)
        res = build_and_run(
            bench_dir=bench,
            kokkos_root=args.kokkos_root,
            cxx=args.cxx,
            log_dir=log_dir,
            timeout=args.timeout,
            keep_build=args.keep_build,
        )
        results.append(res)
        ok = res.returncode == 0
        tag = "PASS" if ok else f"FAIL@{res.stage}"
        print(f"[{tag:>6}] {bench.name}  ({res.duration:.1f}s)  log: {res.log_path}")
        if ok and not args.rerun_passed:
            append_passed(passed_file, res.name, passed)
        if ok and not args.no_csv and res.log_path:
            res.times = extract_timing_values(res.log_path, res.name, timing_regexes)
            if update_csv_row(csv_rows, res.name, res.times):
                print(f"[  CSV ] {res.name}  ({', '.join(csv_rows[res.name])})")
            else:
                print(f"[  CSV ] {res.name}  (no timing value found)")

    if not args.no_csv:
        write_csv_rows(csv_output, csv_rows)
        print(f"\nWrote Kokkos timing CSV to {csv_output}")

    # ----------------------------------------------------------------- summary
    successes = [r for r in results if r.returncode == 0]
    failures = [r for r in results if r.returncode != 0]

    print("\n" + "=" * 70)
    print(
        f"Summary: {len(successes)}/{len(results)} run passed, "
        f"{len(failures)} failed, {len(skipped)} skipped"
    )
    print("=" * 70)

    if skipped:
        print("\nskipped:")
        for name in skipped:
            print(f"  - {name}")

    if successes:
        print("\nsuccess:")
        for r in successes:
            print(f"  - {r.name}")

    if failures:
        print("\nfailed:")
        for r in failures:
            print(f"  - {r.name}  (stage={r.stage}, rc={r.returncode})")
            if r.error_excerpt:
                for line in r.error_excerpt.splitlines()[-10:]:
                    print(f"      {line}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
