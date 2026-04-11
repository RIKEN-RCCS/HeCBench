"""
HeCBench Kokkos vs OpenMP CPU Performance Results
Execution times collected on a 4-core CPU with Kokkos 3.7.01 (OpenMP backend)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Collected Benchmark Data ────────────────────────────────────────────────
# Format: (benchmark_name, kokkos_time_us, omp_time_us, units, note)
# Times in microseconds (us)
# OMP benchmarks compiled with g++ -fopenmp -O3 (CPU target fallback)
# Kokkos benchmarks compiled with g++ -fopenmp (OpenMP backend)

benchmarks = {
    # Benchmark: (kokkos_us, omp_us, problem_description)
    "norm2\n(512M elems)": (130521, 192556, "Vector 2-norm\n(512M float elements)"),
    "softmax\n(1K×512)":   (1517, 2138, "Row-wise softmax\n(1024 slices × 512 elements)"),
    "stencil1d\n(1M elems)": (520, 9544, "1D 7-pt stencil\n(1M elements)"),
    "michalewicz\n(1M pts, d=10)": (108774, 107254, "Michalewicz func\n(1M points, dim=10)"),
    "projectile\n(10M pts)": (49285, 48955, "Projectile sim\n(10M points)"),
    "complex\n(1M, float)": (31187, 36012, "Complex arithmetic\n(1M elements, float)"),
    "wordcount\n(1G chars)": (787065, 813174, "Word count\n(1G characters)"),
    "haversine\n(10K pts)": (256, None, "Haversine distance\n(10K points)"),
    "damage\n(100K nodes)": (35, None, "Bond damage\n(100K nodes, TeamPolicy)"),
    "reverse\n(256×100)":   (31139, None, "Array reverse\n(256 elements, TeamPolicy)"),
}

# ─── Figure 1: Execution Time Comparison Bar Chart ───────────────────────────
# Only benchmarks where both Kokkos and OMP results are available
comparable = {k: v for k, v in benchmarks.items() if v[1] is not None}

bench_names = list(comparable.keys())
kokkos_times = [v[0] / 1000 for v in comparable.values()]  # Convert to ms
omp_times = [v[1] / 1000 for v in comparable.values()]

x = np.arange(len(bench_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width/2, kokkos_times, width, label='Kokkos (OpenMP)', color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, omp_times, width, label='OpenMP (CPU fallback)', color='coral', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Benchmark', fontsize=13)
ax.set_ylabel('Execution Time (ms)', fontsize=13)
ax.set_title('Kokkos (OpenMP) vs OpenMP CPU Execution Time\n(Lower is better)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(bench_names, fontsize=9, ha='center')
ax.legend(fontsize=12)
ax.set_yscale('log')
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h*1.05,
            f'{h:.1f}' if h < 1000 else f'{h:.0f}',
            ha='center', va='bottom', fontsize=7)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h*1.05,
            f'{h:.1f}' if h < 1000 else f'{h:.0f}',
            ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('/home/runner/work/HeCBench/HeCBench/results/fig1_execution_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_execution_time.png")

# ─── Figure 2: Speedup of Kokkos over OMP ────────────────────────────────────
speedups = [v[1] / v[0] for v in comparable.values()]

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#2ecc71' if s >= 1.0 else '#e74c3c' for s in speedups]
bars = ax.bar(bench_names, speedups, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Parity (1×)')
ax.set_xlabel('Benchmark', fontsize=13)
ax.set_ylabel('Speedup (OMP time / Kokkos time)', fontsize=13)
ax.set_title('Kokkos Speedup over OpenMP CPU\n(Green = Kokkos faster, Red = OMP faster)', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(bench_names)))
ax.set_xticklabels(bench_names, fontsize=9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.legend(fontsize=11)

for bar, s in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{s:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/runner/work/HeCBench/HeCBench/results/fig2_speedup.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_speedup.png")

# ─── Figure 3: norm2 Scaling Across Problem Sizes ────────────────────────────
# Data from norm2 benchmark (512K to 512M elements)
n_elements_m = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
kokkos_perf = [7.947679, 8.290758, 8.207490, 8.018694, 8.017035, 8.166561, 8.123482, 8.227807, 8.214876, 8.247868, 8.254402]
omp_perf    = [0.140001, 0.358831, 0.615721, 1.179482, 1.271860, 2.010012, 2.385952, 2.938461, 3.274626, 3.512366, 3.552208]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.semilogx(n_elements_m, kokkos_perf, 'bo-', label='Kokkos (OpenMP)', linewidth=2, markersize=7)
ax1.semilogx(n_elements_m, omp_perf, 'rs-', label='OpenMP (CPU fallback)', linewidth=2, markersize=7)
ax1.set_xlabel('Number of Elements (Millions)', fontsize=12)
ax1.set_ylabel('Performance (Gop/s)', fontsize=12)
ax1.set_title('norm2: Performance vs Problem Size', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Time comparison
kokkos_time = [131.935013, 252.950714, 511.033844, 1046.131592, 2092.696045,
               4108.759277, 8261.095703, 16312.698242, 32676.750000, 65092.085938, 130521.421875]
omp_time    = [7489.795898, 5844.405273, 6812.020508, 7112.113281, 13191.085938,
               16693.652344, 28126.666016, 45676.195312, 81974.382812, 152851.656250, 192556.140625]

ax2.loglog(n_elements_m, kokkos_time, 'bo-', label='Kokkos (OpenMP)', linewidth=2, markersize=7)
ax2.loglog(n_elements_m, omp_time, 'rs-', label='OpenMP (CPU fallback)', linewidth=2, markersize=7)
ax2.set_xlabel('Number of Elements (Millions)', fontsize=12)
ax2.set_ylabel('Execution Time (µs)', fontsize=12)
ax2.set_title('norm2: Execution Time vs Problem Size', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig('/home/runner/work/HeCBench/HeCBench/results/fig3_norm2_scaling.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_norm2_scaling.png")

# ─── Figure 4: Bandwidth Estimate for norm2 ──────────────────────────────────
# norm2 reads N floats (4 bytes each). Bandwidth = N*4 / time
n_elements = [m * 1e6 for m in n_elements_m]
kokkos_bw = [n * 4 / (t * 1e-6) / 1e9 for n, t in zip(n_elements, kokkos_time)]  # GB/s
omp_bw    = [n * 4 / (t * 1e-6) / 1e9 for n, t in zip(n_elements, omp_time)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(n_elements_m, kokkos_bw, 'bo-', label='Kokkos (OpenMP)', linewidth=2, markersize=7)
ax.semilogx(n_elements_m, omp_bw, 'rs-', label='OpenMP (CPU fallback)', linewidth=2, markersize=7)
ax.set_xlabel('Number of Elements (Millions)', fontsize=12)
ax.set_ylabel('Memory Bandwidth (GB/s)', fontsize=12)
ax.set_title('norm2: Effective Memory Bandwidth\n(Kokkos achieves higher bandwidth due to better parallelism)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)
plt.tight_layout()
plt.savefig('/home/runner/work/HeCBench/HeCBench/results/fig4_bandwidth.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_bandwidth.png")

# ─── Figure 5: All Kokkos benchmark execution times ──────────────────────────
all_names = list(benchmarks.keys())
all_kokkos_ms = [v[0] / 1000 for v in benchmarks.values()]
has_omp = [v[1] is not None for v in benchmarks.values()]

fig, ax = plt.subplots(figsize=(15, 7))
colors = ['steelblue' if h else 'lightsteelblue' for h in has_omp]
bars = ax.bar(all_names, all_kokkos_ms, color=colors, edgecolor='black', linewidth=0.5)

# Add hatching for omp-compared ones
for bar, h in zip(bars, has_omp):
    if not h:
        bar.set_hatch('//')

ax.set_xlabel('Benchmark', fontsize=12)
ax.set_ylabel('Execution Time (ms)', fontsize=12)
ax.set_title('Kokkos (OpenMP) Benchmark Execution Times\n(All 10 New Kokkos Ports)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

solid_patch = mpatches.Patch(color='steelblue', label='OMP comparison available')
hatch_patch = mpatches.Patch(facecolor='lightsteelblue', hatch='//', label='Kokkos-only (OMP not directly comparable)')
ax.legend(handles=[solid_patch, hatch_patch], fontsize=10)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h*1.1,
            f'{h:.2f}' if h < 1 else f'{h:.0f}',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/home/runner/work/HeCBench/HeCBench/results/fig5_all_kokkos.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_all_kokkos.png")

# ─── Print Summary Table ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("PERFORMANCE SUMMARY TABLE")
print("="*80)
print(f"{'Benchmark':<25} {'Kokkos (ms)':>12} {'OMP (ms)':>12} {'Speedup':>10}")
print("-"*80)
for name, (k, o, desc) in benchmarks.items():
    k_ms = k / 1000
    if o is not None:
        o_ms = o / 1000
        speedup = o / k
        print(f"{name.replace(chr(10),' '):<25} {k_ms:>12.3f} {o_ms:>12.3f} {speedup:>10.2f}×")
    else:
        print(f"{name.replace(chr(10),' '):<25} {k_ms:>12.3f} {'N/A':>12} {'N/A':>10}")
print("="*80)
print(f"\nEnvironment: 4-core CPU, Kokkos 3.7.01, OpenMP backend, g++ -O3 -fopenmp")
print(f"OMP benchmarks compiled with: g++ -O3 -fopenmp (host-side target fallback)")
