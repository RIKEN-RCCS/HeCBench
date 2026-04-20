/*
 * ECL-GC Graph Coloring - Kokkos port
 *
 * Ported from gc-omp (ECL-GC algorithm).
 * Original copyright 2020, Texas State University.
 * Authors: Guadalupe Rodriguez, Ghadeer Alabandi, Evan Powers, Martin Burtscher
 *
 * graph.h inlined from mis-cuda/graph.h.
 * Copyright 2016-2020, Texas State University.
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>

// ============================================================
// graph.h (ECL binary CSR format)
// ============================================================
#include <cstdlib>
#include <cstdio>

struct ECLgraph {
    int  nodes;
    int  edges;
    int *nindex;
    int *nlist;
    int *eweight;
};

static ECLgraph readECLgraph(const char *const fname)
{
    ECLgraph g;
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "ERROR: could not open file %s\n\n", fname); exit(-1); }
    if (fread(&g.nodes,  sizeof(g.nodes),  1, f) != 1) { fprintf(stderr, "ERROR: failed to read nodes\n\n");  exit(-1); }
    if (fread(&g.edges,  sizeof(g.edges),  1, f) != 1) { fprintf(stderr, "ERROR: failed to read edges\n\n");  exit(-1); }
    if (g.nodes < 1 || g.edges < 0) { fprintf(stderr, "ERROR: node or edge count too low\n\n"); exit(-1); }
    g.nindex  = (int *)malloc((g.nodes + 1) * sizeof(int));
    g.nlist   = (int *)malloc(g.edges       * sizeof(int));
    g.eweight = (int *)malloc(g.edges       * sizeof(int));
    if (!g.nindex || !g.nlist || !g.eweight) { fprintf(stderr, "ERROR: malloc failed\n\n"); exit(-1); }
    if (fread(g.nindex,  sizeof(int), g.nodes + 1, f) != (size_t)(g.nodes + 1))
        fprintf(stderr, "ERROR: failed to read neighbor index list\n\n");
    if (fread(g.nlist,   sizeof(int), g.edges,      f) != (size_t)g.edges)
        fprintf(stderr, "ERROR: failed to read neighbor list\n\n");
    int cnt = (int)fread(g.eweight, sizeof(int), g.edges, f);
    if (cnt == 0) { free(g.eweight); g.eweight = nullptr; }
    fclose(f);
    return g;
}

static void freeECLgraph(ECLgraph &g)
{
    free(g.nindex);  g.nindex  = nullptr;
    free(g.nlist);   g.nlist   = nullptr;
    if (g.eweight) { free(g.eweight); g.eweight = nullptr; }
}

// ============================================================
// ECL-GC constants
// ============================================================
static constexpr int BPI = 32;
static constexpr int MSB = 1 << (BPI - 1);
static constexpr int Mask = (1 << (BPI / 2)) - 1;

KOKKOS_INLINE_FUNCTION
static unsigned int hash_fn(unsigned int val)
{
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    return (val >> 16) ^ val;
}

// ============================================================
// Kokkos view aliases
// ============================================================
using IntView  = Kokkos::View<int*>;
using HostIntView = IntView::HostMirror;

// ============================================================
// init kernel
// ============================================================
static int run_init(
    int nodes, int edges,
    IntView d_nidx, IntView d_nlist,
    IntView d_nlist2, IntView d_posscol, IntView d_posscol2,
    IntView d_color,  IntView d_wl,
    IntView d_wlsize)
{
    // Reset wlsize and posscol2
    Kokkos::deep_copy(d_wlsize, 0);
    Kokkos::deep_copy(d_posscol2, -1);

    int maxrange = -1;
    Kokkos::parallel_reduce(
        "gc_init",
        Kokkos::RangePolicy<>(0, nodes),
        KOKKOS_LAMBDA(int v, int &lmax) {
            const int beg   = d_nidx(v);
            const int end   = d_nidx(v + 1);
            const int degv  = end - beg;
            const bool cond = (degv >= BPI);
            int pos         = beg;

            if (cond) {
                int tmp = Kokkos::atomic_fetch_add(&d_wlsize(0), 1);
                d_wl(tmp) = v;
                for (int i = beg; i < end; i++) {
                    const int nei  = d_nlist(i);
                    const int degn = d_nidx(nei + 1) - d_nidx(nei);
                    if ((degv < degn) ||
                        ((degv == degn) && (hash_fn(v) < hash_fn(nei))) ||
                        ((degv == degn) && (hash_fn(v) == hash_fn(nei)) && (v < nei))) {
                        d_nlist2(pos) = nei;
                        pos++;
                    }
                }
            } else {
                int active = 0;
                for (int i = beg; i < end; i++) {
                    const int nei  = d_nlist(i);
                    const int degn = d_nidx(nei + 1) - d_nidx(nei);
                    if ((degv < degn) ||
                        ((degv == degn) && (hash_fn(v) < hash_fn(nei))) ||
                        ((degv == degn) && (hash_fn(v) == hash_fn(nei)) && (v < nei))) {
                        active |= (unsigned int)MSB >> (i - beg);
                        pos++;
                    }
                }
                const int range = pos - beg;
                lmax = std::max(lmax, range);
                d_color(v) = (cond || (range == 0)) ? (range << (BPI / 2)) : active;
                d_posscol(v) = (range >= BPI) ? -1 : ((unsigned int)MSB >> range);
                return;  // Skip the common update below for small-degree path
            }
            // Large-degree path update
            const int range = pos - beg;
            lmax = std::max(lmax, range);
            d_color(v)   = (range << (BPI / 2));
            d_posscol(v) = (range >= BPI) ? -1 : ((unsigned int)MSB >> range);
        },
        Kokkos::Max<int>(maxrange));
    Kokkos::fence();

    if (maxrange >= Mask) { printf("too many active neighbors\n"); exit(-1); }

    // Retrieve wlsize
    auto h_wlsize = Kokkos::create_mirror_view(d_wlsize);
    Kokkos::deep_copy(h_wlsize, d_wlsize);
    return h_wlsize(0);
}

// ============================================================
// runLarge kernel
// ============================================================
static void run_large(
    IntView d_nidx, IntView d_nlist2,
    IntView d_posscol, IntView d_posscol2,
    IntView d_color,
    IntView d_wl, int wlsize)
{
    if (wlsize == 0) return;

    IntView d_again("again", 1);
    auto    h_again = Kokkos::create_mirror_view(d_again);

    bool again = true;
    while (again) {
        Kokkos::deep_copy(d_again, 0);

        Kokkos::parallel_for(
            "gc_runLarge",
            Kokkos::RangePolicy<>(0, wlsize),
            KOKKOS_LAMBDA(int w) {
                const int v     = d_wl(w);
                int data        = Kokkos::atomic_fetch_or(&d_color(v), 0);  // atomic read
                const int range = data >> (BPI / 2);
                if (range <= 0) return;

                const int beg     = d_nidx(v);
                int       pcol    = d_posscol(v);
                const int mincol  = data & Mask;
                const int maxcol  = mincol + range;
                const int end     = beg + maxcol;
                const int offs    = beg / BPI;

                bool shortcut = true;
                bool done     = true;

                for (int i = beg; i < end; i++) {
                    const int nei     = d_nlist2(i);
                    int       neidata = Kokkos::atomic_fetch_or(&d_color(nei), 0);
                    const int neirange = neidata >> (BPI / 2);
                    if (neirange == 0) {
                        const int neicol = neidata;
                        if (neicol < BPI) {
                            pcol &= ~((unsigned int)MSB >> neicol);
                        } else {
                            if (mincol <= neicol && neicol < maxcol) {
                                int pc = Kokkos::atomic_fetch_or(&d_posscol2(offs + neicol / BPI), 0);
                                if ((pc << (neicol % BPI)) < 0)
                                    Kokkos::atomic_and(&d_posscol2(offs + neicol / BPI),
                                                       ~((unsigned int)MSB >> (neicol % BPI)));
                            }
                        }
                    } else {
                        done = false;
                        const int neimincol = neidata & Mask;
                        const int neimaxcol = neimincol + neirange;
                        if (neimincol <= mincol && neimaxcol >= mincol) shortcut = false;
                    }
                }

                int val = pcol;
                int mc  = 0;
                if (pcol == 0) {
                    mc = std::max(1, mincol / BPI) - 1;
                    do {
                        mc++;
                        val = Kokkos::atomic_fetch_or(&d_posscol2(offs + mc), 0);
                    } while (val == 0);
                }
                int newmincol = mc * BPI + __builtin_clz(val);
                if (mincol != newmincol) shortcut = false;

                int newcolor;
                if (shortcut || done) {
                    pcol     = (newmincol < BPI) ? ((unsigned int)MSB >> newmincol) : 0;
                    newcolor = newmincol;
                } else {
                    pcol     = d_posscol(v);  // restore (unused path below)
                    int rng  = maxcol - newmincol;
                    newcolor = (rng << (BPI / 2)) | newmincol;
                    Kokkos::atomic_store(&d_again(0), 1);
                }
                d_posscol(v) = pcol;
                Kokkos::atomic_exchange(&d_color(v), newcolor);
            });
        Kokkos::fence();

        Kokkos::deep_copy(h_again, d_again);
        again = (h_again(0) != 0);
    }
}

// ============================================================
// runSmall kernel
// ============================================================
static void run_small(
    int nodes,
    IntView d_nidx, IntView d_nlist,
    IntView d_posscol, IntView d_color)
{
    IntView d_again("again", 1);
    auto    h_again = Kokkos::create_mirror_view(d_again);

    bool again = true;
    while (again) {
        Kokkos::deep_copy(d_again, 0);

        Kokkos::parallel_for(
            "gc_runSmall",
            Kokkos::RangePolicy<>(0, nodes),
            KOKKOS_LAMBDA(int v) {
                int pcol = Kokkos::atomic_fetch_or(&d_posscol(v), 0);
                if (__builtin_popcount(pcol) <= 1) return;

                const int beg    = d_nidx(v);
                int       active = d_color(v);
                int       allnei = 0;
                int       keep   = active;

                while (active != 0) {
                    const int old  = active;
                    active &= active - 1;
                    const int curr = old ^ active;
                    const int i    = beg + __builtin_clz(curr);
                    const int nei  = d_nlist(i);
                    int neipcol    = Kokkos::atomic_fetch_or(&d_posscol(nei), 0);
                    allnei |= neipcol;
                    if ((pcol & neipcol) == 0) {
                        pcol &= pcol - 1;
                        keep ^= curr;
                    } else if (__builtin_popcount(neipcol) == 1) {
                        pcol ^= neipcol;
                        keep ^= curr;
                    }
                }

                if (keep != 0) {
                    const int best = (unsigned int)MSB >> __builtin_clz(pcol);
                    if ((best & ~allnei) != 0) {
                        pcol = best;
                        keep = 0;
                    }
                }

                if (keep != 0) Kokkos::atomic_store(&d_again(0), 1);
                int final_color = (keep == 0) ? __builtin_clz(pcol) : keep;
                d_color(v) = final_color;
                Kokkos::atomic_exchange(&d_posscol(v), pcol);
            });
        Kokkos::fence();

        Kokkos::deep_copy(h_again, d_again);
        again = (h_again(0) != 0);
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[])
{
    printf("ECL-GC Kokkos v1.0 (%s)\n", __FILE__);
    printf("Copyright 2020 Texas State University\n\n");

    if (argc != 4) {
        printf("USAGE: %s <input_file_name> <thread_count> <repeat>\n\n", argv[0]);
        exit(-1);
    }
    if (BPI != (int)(sizeof(int) * 8)) {
        printf("ERROR: bits per int size must be %zu\n\n", sizeof(int) * 8);
        exit(-1);
    }

    // thread_count is used by OMP; for Kokkos it is informational only.
    const int repeat = atoi(argv[3]);

    ECLgraph g = readECLgraph(argv[1]);
    printf("input: %s\n",  argv[1]);
    printf("nodes: %d\n",  g.nodes);
    printf("edges: %d\n",  g.edges);
    printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

    Kokkos::initialize(argc, argv);
    {
        // Copy graph to device
        IntView d_nidx ("d_nidx",  g.nodes + 1);
        IntView d_nlist("d_nlist", g.edges);
        {
            auto h_nidx  = Kokkos::create_mirror_view(d_nidx);
            auto h_nlist = Kokkos::create_mirror_view(d_nlist);
            for (int i = 0; i <= g.nodes; i++) h_nidx(i)  = g.nindex[i];
            for (int i = 0;  i < g.edges; i++) h_nlist(i) = g.nlist[i];
            Kokkos::deep_copy(d_nidx,  h_nidx);
            Kokkos::deep_copy(d_nlist, h_nlist);
        }

        IntView d_color   ("d_color",    g.nodes);
        IntView d_nlist2  ("d_nlist2",   g.edges);
        IntView d_posscol ("d_posscol",  g.nodes);
        IntView d_posscol2("d_posscol2", g.edges / BPI + 1);
        IntView d_wl      ("d_wl",       g.nodes);
        IntView d_wlsize  ("d_wlsize",   1);

        double runtime = 0.0;

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int n = 0; n < repeat; n++) {
            int wlsize = run_init(g.nodes, g.edges,
                                  d_nidx, d_nlist,
                                  d_nlist2, d_posscol, d_posscol2,
                                  d_color, d_wl, d_wlsize);
            run_large(d_nidx, d_nlist2, d_posscol, d_posscol2, d_color, d_wl, wlsize);
            run_small(g.nodes, d_nidx, d_nlist, d_posscol, d_color);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t_end - t_start;
        runtime = elapsed.count() / repeat;

        printf("average runtime (%d runs):    %.6f s\n", repeat, runtime);
        printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
        printf("throughput: %.6f Medges/s\n", g.edges * 0.000001 / runtime);

        // Copy result back for verification
        auto h_color = Kokkos::create_mirror_view(d_color);
        Kokkos::deep_copy(h_color, d_color);

        bool ok = true;
        for (int v = 0; v < g.nodes && ok; v++) {
            if (h_color(v) < 0) {
                printf("ERROR: unprocessed node %d (deg %d)\n\n",
                       v, g.nindex[v + 1] - g.nindex[v]);
                ok = false;
                break;
            }
            for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
                if (h_color(g.nlist[i]) == h_color(v)) {
                    printf("ERROR: adjacent nodes with same color %d (%d %d)\n\n",
                           h_color(v), v, g.nlist[i]);
                    ok = false;
                    break;
                }
            }
        }
        printf("%s\n", ok ? "PASS" : "FAIL");

        const int vals = 16;
        int c[vals] = {};
        int cols = -1;
        for (int v = 0; v < g.nodes; v++) {
            cols = std::max(cols, h_color(v));
            if (h_color(v) < vals) c[h_color(v)]++;
        }
        cols++;
        printf("Number of distinct colors used: %d\n", cols);

        int sum = 0;
        for (int i = 0; i < std::min(vals, cols); i++) {
            sum += c[i];
            printf("color %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / g.nodes);
        }
    }
    Kokkos::finalize();

    delete [] (int *)nullptr;  // suppress unused-variable warnings
    freeECLgraph(g);
    return 0;
}
