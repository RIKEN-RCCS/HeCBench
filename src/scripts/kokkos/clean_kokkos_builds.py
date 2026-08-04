#!/usr/bin/env python3
"""
Remove CMake build directories from every src/*-kokkos benchmark.

Usage:
    python3 scripts/clean_kokkos_builds.py
    python3 scripts/clean_kokkos_builds.py --dry-run
    python3 scripts/clean_kokkos_builds.py --src-root /path/to/HeCBench/src
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def default_src_root() -> Path:
    return Path(__file__).resolve().parents[3] / "src"


def find_kokkos_build_dirs(src_root: Path) -> list[Path]:
    return sorted(
        bench / "build"
        for bench in src_root.iterdir()
        if bench.is_dir()
        and bench.name.endswith("-kokkos")
        and (bench / "build").is_dir()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove build/ directories under src/*-kokkos benchmarks."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=default_src_root(),
        help="Path to the HeCBench src directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print directories that would be removed.",
    )
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    if not src_root.is_dir():
        parser.error(f"src root does not exist: {src_root}")

    build_dirs = find_kokkos_build_dirs(src_root)
    if not build_dirs:
        print(f"No Kokkos build directories found under {src_root}")
        return 0

    action = "Would remove" if args.dry_run else "Removing"
    for build_dir in build_dirs:
        print(f"{action}: {build_dir}")
        if not args.dry_run:
            shutil.rmtree(build_dir)

    suffix = "would be removed" if args.dry_run else "removed"
    print(f"\n{len(build_dirs)} build directories {suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
