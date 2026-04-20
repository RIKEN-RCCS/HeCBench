/*
Friends-of-Friends (FoF) Halo Finder – simplified cosmology benchmark.
Kokkos port (OpenMP backend).

Algorithm:
  1. Generate nParticles random 3D positions in [0,1]^3
  2. Link particles within linking length b into groups (halos)
     b = 0.2 * mean interparticle spacing = 0.2 / cbrt(nParticles)
  3. Use union-find for group assignment
  4. Count halos and their sizes

Kokkos is used for:
  - Generating random particle positions (parallel_for)
  - Pair distance computation and union-find (parallel_for with atomics)
  - Halo counting and statistics (parallel_reduce)

Usage: ./main [nParticles [repeat]]
  nParticles = number of particles (default 10000)
  repeat     = benchmark repetitions (default 1)
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <climits>

// ---------------------------------------------------------------------------
// Simple LCG random number generator (device-compatible)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
float lcg_rand(unsigned long long& state)
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

// ---------------------------------------------------------------------------
// Union-Find: find root with path compression (device)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int find_root(Kokkos::View<int*>& label, int v)
{
    int root = v;
    while (label(root) != root) root = label(root);
    // Path compression
    while (label(v) != root) {
        int next = label(v);
        label(v) = root;
        v = next;
    }
    return root;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int nPart  = 10000;
    int repeat = 1;
    if (argc > 1) nPart  = atoi(argv[1]);
    if (argc > 2) repeat = atoi(argv[2]);

    printf("nParticles=%d\n", nPart);

    // Mean interparticle spacing (for unit cube)
    const float mean_spacing = (float)cbrt(1.0 / nPart);
    const float b            = 0.2f * mean_spacing;
    const float b2           = b * b;
    printf("Linking length: %f\n", (double)b);

    Kokkos::initialize(argc, argv);
    {
        using ViewF = Kokkos::View<float*>;
        using ViewI = Kokkos::View<int*>;

        ViewF d_x("x", nPart);
        ViewF d_y("y", nPart);
        ViewF d_z("z", nPart);
        ViewI d_label("label", nPart);
        ViewI d_haloSize("haloSize", nPart);

        // Generate random positions
        Kokkos::parallel_for("genParts", nPart, KOKKOS_LAMBDA(int i) {
            unsigned long long state = (unsigned long long)i * 6364136223846793005ULL + 1;
            d_x(i) = lcg_rand(state);
            d_y(i) = lcg_rand(state);
            d_z(i) = lcg_rand(state);
        });

        double total_ms = 0.0;
        int    last_nHalos  = 0;
        int    last_largest = 0;

        for (int rep = 0; rep < repeat; rep++) {

            // Initialize union-find
            Kokkos::parallel_for("initUF", nPart, KOKKOS_LAMBDA(int i) {
                d_label(i) = i;
            });

            auto t1 = std::chrono::steady_clock::now();

            // Pair-wise FoF linking
            // For each particle i, scan particles j > i
            Kokkos::parallel_for("fof", nPart, KOKKOS_LAMBDA(int i) {
                float xi = d_x(i), yi = d_y(i), zi = d_z(i);
                for (int j = i + 1; j < nPart; j++) {
                    float dx = xi - d_x(j);
                    float dy = yi - d_y(j);
                    float dz = zi - d_z(j);
                    if (dx*dx + dy*dy + dz*dz < b2) {
                        // Union i and j using pointer jumping
                        int ri = i, rj = j;
                        // Find roots
                        while (d_label(ri) != ri) ri = d_label(ri);
                        while (d_label(rj) != rj) rj = d_label(rj);
                        // Union: smaller root wins
                        while (ri != rj) {
                            if (ri < rj) {
                                int old = Kokkos::atomic_compare_exchange(&d_label(rj), rj, ri);
                                if (old == rj) break;
                                rj = old;
                                while (d_label(rj) != rj) rj = d_label(rj);
                            } else {
                                int old = Kokkos::atomic_compare_exchange(&d_label(ri), ri, rj);
                                if (old == ri) break;
                                ri = old;
                                while (d_label(ri) != ri) ri = d_label(ri);
                            }
                        }
                    }
                }
            });

            // Path compression pass
            Kokkos::parallel_for("compress", nPart, KOKKOS_LAMBDA(int i) {
                int root = d_label(i);
                while (d_label(root) != root) root = d_label(root);
                d_label(i) = root;
            });

            Kokkos::fence();
            auto t2 = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Count halo sizes
            Kokkos::deep_copy(d_haloSize, 0);
            Kokkos::parallel_for("countSizes", nPart, KOKKOS_LAMBDA(int i) {
                Kokkos::atomic_fetch_add(&d_haloSize(d_label(i)), 1);
            });

            // Count number of halos (roots with size > 0)
            int nHalos = 0;
            Kokkos::parallel_reduce("countHalos", nPart,
                KOKKOS_LAMBDA(int i, int& cnt) {
                    if (d_label(i) == i && d_haloSize(i) > 0) cnt++;
                }, nHalos);

            // Find largest halo
            int largest = 0;
            Kokkos::parallel_reduce("largestHalo", nPart,
                KOKKOS_LAMBDA(int i, int& mx) {
                    if (d_haloSize(i) > mx) mx = d_haloSize(i);
                },
                Kokkos::Max<int>(largest));

            last_nHalos  = nHalos;
            last_largest = largest;
        }

        printf("Found %d halos\n",        last_nHalos);
        printf("Largest halo: %d particles\n", last_largest);
        printf("Kernel time: %.3f (ms)\n", total_ms / repeat);
    }
    Kokkos::finalize();
    return 0;
}
