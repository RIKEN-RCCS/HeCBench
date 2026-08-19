#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
src3_dir="$(cd -- "$script_dir/../.." && pwd)"
list_file="$script_dir/bench_names.txt"
log_dir="$src3_dir/fortran-logs"
dry_run=false
timeout_seconds=600

usage() {
  echo "Usage: $0 [--list FILE] [--timeout SECONDS] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      list_file="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --timeout)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
      timeout_seconds="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

[[ -f "$list_file" ]] || { echo "Benchmark list not found: $list_file" >&2; exit 2; }
mkdir -p "$log_dir"

mapfile -t benchmarks < <(awk 'NF && $1 !~ /^#/' "$list_file")
total=${#benchmarks[@]}
passed=0
failed=0

for benchmark in "${benchmarks[@]}"; do
  benchmark_dir="$src3_dir/${benchmark}-fortran"
  printf '[%d/%d] %s\n' "$((passed + failed + 1))" "$total" "$benchmark"

  if [[ ! -d "$benchmark_dir" || ! -f "$benchmark_dir/Makefile" ]]; then
    echo "Missing benchmark directory or Makefile: $benchmark_dir" >&2
    ((failed += 1))
    continue
  fi

  if [[ "$dry_run" == true ]]; then
    echo "make -C $benchmark_dir run"
    ((passed += 1))
    continue
  fi

  log_file="$log_dir/${benchmark}.log"
  if (
    printf 'Benchmark: %s\n' "$benchmark"
    printf 'Directory: %s\n' "$benchmark_dir"
    printf 'Started: %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    timeout --foreground "${timeout_seconds}s" make -C "$benchmark_dir" run
  ) 2>&1 | tee "$log_file"; then
    if grep -Eq '^[[:space:]]*(FAIL|Fail)([[:space:]]|$)' "$log_file"; then
      echo "Failed verification: $benchmark" >&2
      printf 'Result: FAIL\n' | tee -a "$log_file" >&2
      ((failed += 1))
    else
      printf 'Result: PASS\n' | tee -a "$log_file"
      ((passed += 1))
    fi
  else
    run_status=${PIPESTATUS[0]}
    if [[ $run_status -eq 124 ]]; then
      printf 'Timed out after %s seconds: %s\n' "$timeout_seconds" "$benchmark" >&2
      printf 'Result: TIMEOUT\n' | tee -a "$log_file" >&2
    fi
    echo "Failed: $benchmark" >&2
    printf 'Result: FAIL\n' | tee -a "$log_file" >&2
    ((failed += 1))
  fi
done

printf 'Completed: %d passed, %d failed, %d total.\n' "$passed" "$failed" "$total"
((failed == 0))
