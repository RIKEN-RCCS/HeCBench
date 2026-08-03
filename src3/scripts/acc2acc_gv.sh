#!/usr/bin/env bash

set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: $0 BENCHMARK|BENCHMARK_LIST_FILE [BENCHMARK|BENCHMARK_LIST_FILE ...]" >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_root=$(cd -- "$script_dir/.." && pwd)
had_error=false
benchmarks=()

for input in "$@"; do
    if [[ -f $input ]]; then
        while IFS= read -r benchmark || [[ -n $benchmark ]]; do
            benchmark=${benchmark%$'\r'}
            [[ -z $benchmark || $benchmark == \#* ]] && continue
            benchmarks+=("$benchmark")
        done < "$input"
    else
        benchmarks+=("$input")
    fi
done

total=${#benchmarks[@]}
generated=0

transform_file() {
    local file=$1
    local tmp

    tmp=$(mktemp "${file}.tmp.XXXXXX")
    awk '
        function write_directive(directive, has_gang, has_vector, additions) {
            if (directive ~ /^[[:blank:]]*#[[:blank:]]*pragma[[:blank:]]+acc[[:blank:]]+parallel[[:blank:]]+loop/) {
                has_gang = directive ~ /(^|[^[:alnum:]_])gang([^[:alnum:]_]|$)/
                has_vector = directive ~ /(^|[^[:alnum:]_])vector([^[:alnum:]_]|$)/

                additions = ""
                if (!has_gang)
                    additions = additions " gang"
                if (!has_vector)
                    additions = additions " vector"
                sub(/loop/, "loop" additions, directive)
            }
            printf "%s", directive
        }

        {
            directive = $0
            while (directive ~ /\\[[:blank:]]*$/) {
                if (getline next_line <= 0) {
                    break
                }
                directive = directive "\n" next_line
            }
            write_directive(directive)
            printf "\n"
        }
    ' "$file" > "$tmp"
    chmod --reference="$file" "$tmp"
    mv -- "$tmp" "$file"
}

for requested_name in "${benchmarks[@]}"; do
    if [[ $requested_name == */* ]]; then
        echo "Error: use only the benchmark name, without a path: $requested_name" >&2
        had_error=true
        continue
    fi

    benchmark=${requested_name%-acc}
    source_dir="$bench_root/$benchmark-acc"
    target_dir="$bench_root/$benchmark-acc_gv"

    if [[ ! -d $source_dir ]]; then
        echo "Error: ACC version not found: $source_dir" >&2
        had_error=true
        continue
    fi

    echo "Generating ${benchmark}-acc_gv from ${benchmark}-acc"
    rm -rf -- "$target_dir"
    cp -a -- "$source_dir" "$target_dir"

    while IFS= read -r -d '' file; do
        transform_file "$file"
    done < <(find "$target_dir" -type f \( \
        -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
        -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' \
    \) -print0)

    ((generated += 1))
done

echo "Generated: $generated/$total ACC_GV benchmark(s)."

if [[ $had_error == true ]]; then
    exit 1
fi
