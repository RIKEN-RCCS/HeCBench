#!/usr/bin/env python3
"""
Generate performance figures comparing CUDA native vs Kokkos/CUDA
for 5 HeCBench benchmarks on NVIDIA GB10 (sm_121, CUDA 13.0).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'results', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ─── Measured data ─────────────────────────────────────────────────────────────

# bilateral: execution time (ms), lower is better
# kernel radius 3x3 / 6x6 / 9x9 on 1024x1024 image, 100 repeats
bilateral = {
    'labels': ['R=3', 'R=6', 'R=9'],
    'cuda':   [0.234, 0.794, 1.682],
    'kokkos': [0.635, 2.362, 4.387],
    'unit':   'ms',
}

# attention: execution time (ms)
# 4096 rows, 256 columns, 100 repeats
attention = {
    'labels': ['n=4096, d=256'],
    'cuda':   [0.143],
    'kokkos': [0.248],
    'unit':   'ms',
}

# babelstream: memory bandwidth (MB/s), higher is better
# float precision, 2^25 elements
babelstream = {
    'labels': ['Copy', 'Mul', 'Add', 'Triad', 'Dot', 'NStream'],
    'cuda':   [225356, 238081, 221327, 236167, 219455, 236991],
    'kokkos': [215818, 225595, 217772, 219254, 180796, 233583],
    'unit':   'MB/s',
}

# bitonic-sort: total time (ms), lower is better
# 2^23 = 8M elements
bitonic = {
    'labels': ['n=2^23'],
    'cuda':   [57.3],
    'kokkos': [93.3],
    'unit':   'ms',
}

# atan2: execution time per kernel call (µs)
atan2 = {
    'labels': ['f32', 'i32', 'i16'],
    'cuda':   [26.24, 24.28, 22.06],
    'kokkos': [52.73, 62.31, 61.00],
    'unit':   'µs',
}

# ─── Colour palette ────────────────────────────────────────────────────────────
C_CUDA   = '#1f77b4'   # blue
C_KOKKOS = '#ff7f0e'   # orange

# ─── Helper: grouped bar chart ────────────────────────────────────────────────

def grouped_bar(ax, labels, cuda_vals, kokkos_vals, unit,
                title, higher_is_better=False, ylog=False):
    x = np.arange(len(labels))
    w = 0.35
    bars_c = ax.bar(x - w/2, cuda_vals,   w, color=C_CUDA,   label='CUDA (native)')
    bars_k = ax.bar(x + w/2, kokkos_vals, w, color=C_KOKKOS, label='Kokkos/CUDA')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    if ylog:
        ax.set_yscale('log')

    # annotate speedup / slowdown
    for xi, (cv, kv) in enumerate(zip(cuda_vals, kokkos_vals)):
        ratio = cv / kv if higher_is_better else kv / cv
        colour = 'green' if ratio <= 1.1 else 'red'
        label  = f'{ratio:.1f}×'
        ax.text(xi, max(cv, kv) * 1.03, label,
                ha='center', va='bottom', fontsize=7, color=colour)

    return bars_c, bars_k


# ─── Figure 1 : bilateral filter ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
grouped_bar(ax, bilateral['labels'], bilateral['cuda'], bilateral['kokkos'],
            'Execution time (ms)',
            'Bilateral Filter  —  1024×1024 image, 100 runs\n(lower is better)')
ax.set_ylim(0, 5.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'bilateral.png'), dpi=150)
plt.close(fig)
print("Saved bilateral.png")

# ─── Figure 2 : attention ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4, 4))
grouped_bar(ax, attention['labels'], attention['cuda'], attention['kokkos'],
            'Execution time (ms)',
            'Attention  —  n=4096, d=256, 100 runs\n(lower is better)')
ax.set_ylim(0, 0.35)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'attention.png'), dpi=150)
plt.close(fig)
print("Saved attention.png")

# ─── Figure 3 : BabelStream ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
grouped_bar(ax, babelstream['labels'], babelstream['cuda'], babelstream['kokkos'],
            'Bandwidth (MB/s)',
            'BabelStream  —  float, 2²⁵ elements, 20 runs\n(higher is better)',
            higher_is_better=True)
ax.set_ylim(0, 280_000)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'babelstream.png'), dpi=150)
plt.close(fig)
print("Saved babelstream.png")

# ─── Figure 4 : bitonic-sort ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4, 4))
grouped_bar(ax, bitonic['labels'], bitonic['cuda'], bitonic['kokkos'],
            'Total time (ms)',
            'Bitonic Sort  —  2²³ elements\n(lower is better)')
ax.set_ylim(0, 115)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'bitonic_sort.png'), dpi=150)
plt.close(fig)
print("Saved bitonic_sort.png")

# ─── Figure 5 : atan2 ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 4))
grouped_bar(ax, atan2['labels'], atan2['cuda'], atan2['kokkos'],
            'Kernel time (µs)',
            'atan2  —  1M elements, 100 runs\n(lower is better)')
ax.set_ylim(0, 80)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'atan2.png'), dpi=150)
plt.close(fig)
print("Saved atan2.png")

# ─── Figure 6 : summary — normalized performance (CUDA=1) ────────────────────

benchmarks = {
    'BabelStream\nCopy':    babelstream['cuda'][0] / babelstream['kokkos'][0],
    'BabelStream\nTriad':   babelstream['cuda'][3] / babelstream['kokkos'][3],
    'Bilateral\nR=3':       bilateral['cuda'][0]   / bilateral['kokkos'][0],
    'Bilateral\nR=9':       bilateral['cuda'][2]   / bilateral['kokkos'][2],
    'Attention':             attention['cuda'][0]   / attention['kokkos'][0],
    'Bitonic\nSort':        bitonic['cuda'][0]     / bitonic['kokkos'][0],
    'atan2\nf32':           atan2['cuda'][0]       / atan2['kokkos'][0],
    'atan2\ni32':           atan2['cuda'][1]       / atan2['kokkos'][1],
}

labels = list(benchmarks.keys())
ratios = list(benchmarks.values())   # CUDA / Kokkos = Kokkos efficiency vs CUDA

colours = ['green' if r >= 0.9 else ('orange' if r >= 0.75 else 'red')
           for r in ratios]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(labels)), ratios, color=colours, edgecolor='black', linewidth=0.5)
ax.axhline(1.0, color='navy', linewidth=1.5, linestyle='--', label='CUDA native (baseline)')
ax.axhline(0.9, color='gray', linewidth=0.8, linestyle=':',  label='90% efficiency')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Kokkos performance relative to CUDA native', fontsize=10)
ax.set_ylim(0, 1.25)
ax.set_title('Kokkos/CUDA vs CUDA native — NVIDIA GB10 (sm_121, CUDA 13.0)\n'
             'Higher = Kokkos closer to native CUDA performance',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

for i, (r, b) in enumerate(zip(ratios, bars)):
    ax.text(b.get_x() + b.get_width()/2, r + 0.02, f'{r:.2f}',
            ha='center', va='bottom', fontsize=8)

green_patch  = mpatches.Patch(color='green',  label='≥ 90% efficiency')
orange_patch = mpatches.Patch(color='orange', label='75–90% efficiency')
red_patch    = mpatches.Patch(color='red',    label='< 75% efficiency')
ax.legend(handles=[green_patch, orange_patch, red_patch,
                   plt.Line2D([0], [0], color='navy', linewidth=1.5, linestyle='--')],
          labels=['≥ 90% efficiency', '75–90% efficiency', '< 75% efficiency',
                  'CUDA native (1.0)'],
          fontsize=9, loc='upper right')

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'summary_normalized.png'), dpi=150)
plt.close(fig)
print("Saved summary_normalized.png")

print("\nAll figures saved to:", OUTDIR)
