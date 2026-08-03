#!/usr/bin/env python3
"""Generate a self-contained SVG comparison of ACC, ACC_GV, and OMP_NVC CSV files."""

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


REPORT_BENCHMARKS = (
    "inversek2j epistasis scel testSNAP bilateral libor contract nqueen "
    "degrid tissue lulesh miniWeather cfd winograd overlay jenkins-hash "
    "aligned-types distort keogh scatterAdd fluidSim s8n doh hellinger "
    "affine sw4ck clenergy nbody"
).split()


def load_times(path, suffix):
    with path.open(newline="") as stream:
        return {
            row[0].removesuffix(suffix): float(row[1])
            for row in csv.reader(stream)
            if len(row) >= 2
        }


def bar(value, left, width):
    """Map a positive speedup to a log2 horizontal coordinate."""
    minimum, maximum = 0.25, 128.0
    value = max(minimum, min(maximum, value))
    return left + width * (math.log2(value) - math.log2(minimum)) / (
        math.log2(maximum) - math.log2(minimum)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("acc", type=Path)
    parser.add_argument("acc_gv", type=Path)
    parser.add_argument("omp_nvc", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="*", default=REPORT_BENCHMARKS)
    args = parser.parse_args()

    acc = load_times(args.acc, "-acc")
    acc_gv = load_times(args.acc_gv, "-acc_gv")
    omp = load_times(args.omp_nvc, "-omp_nvc")
    rows = [
        (name, acc[name], acc_gv[name], omp[name])
        for name in args.benchmarks
        if name in acc and name in acc_gv and name in omp
    ]
    rows.sort(key=lambda row: row[1] / row[2], reverse=True)

    left, plot_width, top, row_height = 210, 850, 86, 23
    height = top + len(rows) * row_height + 72
    axis_y = top - 22
    ticks = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + plot_width + 40}" height="{height}" viewBox="0 0 {left + plot_width + 40} {height}">',
        '<style>text { font-family: sans-serif; fill: #202124; } .small { font-size: 11px; } .label { font-size: 12px; } .title { font-size: 18px; font-weight: bold; } .grid { stroke: #d9d9d9; stroke-width: 1; } .base { stroke: #555; stroke-width: 1.5; }</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" class="title">ACC baseline speedup — lower execution time is better</text>',
        '<text x="20" y="48" class="small">Bars show ACC time / variant time; values above 1 mean the variant is faster. Log₂ scale. All times are ms.</text>',
        '<rect x="20" y="60" width="14" height="8" fill="#f28e2b"/><text x="40" y="69" class="small">ACC_GV</text>',
        '<rect x="110" y="60" width="14" height="8" fill="#4e79a7"/><text x="130" y="69" class="small">OMP_NVC</text>',
    ]
    for tick in ticks:
        x = bar(tick, left, plot_width)
        css = "base" if tick == 1 else "grid"
        svg += [
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{height - 42}" class="{css}"/>',
            f'<text x="{x:.1f}" y="{axis_y - 5}" text-anchor="middle" class="small">{tick:g}×</text>',
        ]
    for index, (name, acc_time, gv_time, omp_time) in enumerate(rows):
        y = top + index * row_height
        gv_speedup, omp_speedup = acc_time / gv_time, acc_time / omp_time
        gv_x, omp_x = bar(gv_speedup, left, plot_width), bar(omp_speedup, left, plot_width)
        svg += [
            f'<text x="{left - 10}" y="{y + 13}" text-anchor="end" class="label">{escape(name)}</text>',
            f'<rect x="{min(left, gv_x):.1f}" y="{y + 2}" width="{abs(gv_x - left):.1f}" height="8" fill="#f28e2b"/>',
            f'<rect x="{min(left, omp_x):.1f}" y="{y + 12}" width="{abs(omp_x - left):.1f}" height="8" fill="#4e79a7"/>',
        ]
    svg += [
        f'<text x="{left}" y="{height - 15}" class="small">Selected report benchmarks with all three measurements: {len(rows)}.</text>',
        "</svg>",
    ]
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
