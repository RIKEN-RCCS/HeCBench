/*
graphB+ balancing algorithm for signed social network graphs
Kokkos port (OpenMP backend)

Algorithm:
  1. Read signed graph (edges with weights -1, 0, +1)
  2. For each iteration: build BFS spanning tree from a high-degree root
  3. 2-color the tree based on edge signs (negative edge flips color)
  4. Find balanced connected components via union-find on consistent edges
  5. Compute hop distances from largest CC
  6. Accumulate inCC: nodes at even hop distance from largest CC get +1
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <climits>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <tuple>

// ---------------------------------------------------------------------------
// Graph reading
// ---------------------------------------------------------------------------
static void readGraph(const char* filename,
                      int& N, int& E,
                      std::vector<int>& nindex,
                      std::vector<int>& nlist,
                      std::vector<int>& eweight,
                      std::vector<int>& origID)
{
    FILE* fin = fopen(filename, "rt");
    if (!fin) { fprintf(stderr, "ERROR: cannot open %s\n", filename); exit(1); }

    char buf[256];
    fgets(buf, sizeof(buf), fin); // skip header

    std::map<int,int> nodeMap;
    std::set<std::pair<int,int>> seen;
    std::vector<std::tuple<int,int,int>> edgeList;
    int cnt = 0;

    int src, dst, wei;
    while (fscanf(fin, "%d,%d,%d", &src, &dst, &wei) == 3) {
        if (src == dst) continue;
        if (wei < -1 || wei > 1) continue;
        int a = std::min(src, dst), b = std::max(src, dst);
        if (seen.count({a, b})) continue;
        seen.insert({a, b});
        if (!nodeMap.count(src)) nodeMap[src] = cnt++;
        if (!nodeMap.count(dst)) nodeMap[dst] = cnt++;
        edgeList.push_back({nodeMap[src], nodeMap[dst], wei});
    }
    fclose(fin);

    N = cnt;
    E = (int)edgeList.size() * 2; // undirected: 2 directed entries per edge

    std::vector<int> deg(N, 0);
    for (auto& [u, v, w] : edgeList) { deg[u]++; deg[v]++; }

    nindex.resize(N + 1);
    nindex[0] = 0;
    for (int i = 0; i < N; i++) nindex[i+1] = nindex[i] + deg[i];

    nlist.resize(E);
    eweight.resize(E);
    std::vector<int> pos(N, 0);
    for (auto& [u, v, w] : edgeList) {
        nlist  [nindex[u] + pos[u]] = v;
        eweight[nindex[u] + pos[u]] = w;
        pos[u]++;
        nlist  [nindex[v] + pos[v]] = u;
        eweight[nindex[v] + pos[v]] = w;
        pos[v]++;
    }

    origID.resize(N);
    for (auto& [orig, mapped] : nodeMap) origID[mapped] = orig;
}

// ---------------------------------------------------------------------------
// Union-find representative (host helper, used in serial output)
// ---------------------------------------------------------------------------
static int rep(int v, std::vector<int>& lbl) {
    while (lbl[v] != v) { lbl[v] = lbl[lbl[v]]; v = lbl[v]; }
    return v;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    printf("graphB+ balancing code for signed social network graphs\n");

    if (argc != 4) {
        printf("USAGE: %s input_file iterations output_file\n", argv[0]);
        return 1;
    }

    printf("verification is off\n");

    int N = 0, E = 0;
    std::vector<int> h_nindex, h_nlist, h_eweight, h_origID;
    readGraph(argv[1], N, E, h_nindex, h_nlist, h_eweight, h_origID);

    printf("nodes: %d\n", N);
    printf("edges: %d\n", E);

    const int iterations = atoi(argv[2]);

    // Choose roots: highest-degree nodes first
    std::vector<int> root(N);
    for (int i = 0; i < N; i++) root[i] = i;
    std::partial_sort(root.begin(),
                      root.begin() + std::min(iterations, N),
                      root.end(),
                      [&](int a, int b){
                          return (h_nindex[a+1]-h_nindex[a]) > (h_nindex[b+1]-h_nindex[b]);
                      });

    Kokkos::initialize(argc, argv);
    {
        using ViewI  = Kokkos::View<int*>;
        using ViewI2 = Kokkos::View<int*>; // same type, named for clarity

        ViewI d_nindex ("nindex",  N + 1);
        ViewI d_nlist  ("nlist",   E);
        ViewI d_eweight("eweight", E);

        // Per-iteration working arrays
        ViewI d_parent ("parent",  N);
        ViewI d_color  ("color",   N);  // 2-coloring: 0 or 1
        ViewI d_frontier("frontier", N);
        ViewI d_nextFr ("nextFr",  N);
        ViewI d_fsize  ("fsize",   1);
        ViewI d_nsize  ("nsize",   1);

        // Label / union-find
        ViewI d_label  ("label",   N);
        ViewI d_ccCnt  ("ccCnt",   N);  // CC sizes
        ViewI d_hop    ("hop",     N);  // hop count per CC label

        // Accumulators (persist across iterations)
        ViewI d_inCC   ("inCC",    N);
        ViewI d_inTree ("inTree",  E);
        ViewI d_negCnt ("negCnt",  E);

        // Copy graph data
        {
            auto hn = Kokkos::create_mirror_view(d_nindex);
            auto hl = Kokkos::create_mirror_view(d_nlist);
            auto he = Kokkos::create_mirror_view(d_eweight);
            for (int i = 0; i <= N; i++) hn(i) = h_nindex[i];
            for (int i = 0; i < E;  i++) hl(i) = h_nlist[i];
            for (int i = 0; i < E;  i++) he(i) = h_eweight[i];
            Kokkos::deep_copy(d_nindex,  hn);
            Kokkos::deep_copy(d_nlist,   hl);
            Kokkos::deep_copy(d_eweight, he);
        }

        // Zero accumulators
        Kokkos::deep_copy(d_inCC,   0);
        Kokkos::deep_copy(d_inTree, 0);
        Kokkos::deep_copy(d_negCnt, 0);

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; iter++) {
            const int r = root[iter % N];

            // ----------------------------------------------------------
            // Step 1: BFS spanning tree + 2-coloring
            // ----------------------------------------------------------
            Kokkos::parallel_for("initBFS", N, KOKKOS_LAMBDA(int i) {
                d_parent(i) = -1;
                d_color (i) = -1;
            });
            Kokkos::parallel_for("initRoot", 1, KOKKOS_LAMBDA(int) {
                d_parent(r)    = r;
                d_color (r)    = 0;
                d_frontier(0)  = r;
                d_fsize(0)     = 1;
                d_nsize(0)     = 0;
            });

            int fsize = 1;
            int level = 0;
            while (fsize > 0) {
                Kokkos::parallel_for("clearNSize", 1, KOKKOS_LAMBDA(int) {
                    d_nsize(0) = 0;
                });

                Kokkos::parallel_for("BFSexpand", fsize, KOKKOS_LAMBDA(int fi) {
                    const int node      = d_frontier(fi);
                    const int nodeColor = d_color(node);
                    for (int j = d_nindex(node); j < d_nindex(node+1); j++) {
                        const int nbr = d_nlist(j);
                        // Claim unvisited neighbor
                        if (Kokkos::atomic_compare_exchange(&d_parent(nbr), -1, node) == -1) {
                            const int idx = Kokkos::atomic_fetch_add(&d_nsize(0), 1);
                            d_nextFr(idx) = nbr;
                            // 2-color: negative edge flips color
                            d_color(nbr) = (d_eweight(j) < 0) ? (1 - nodeColor) : nodeColor;
                        }
                    }
                });

                // Swap frontiers
                Kokkos::parallel_for("swapFr", N, KOKKOS_LAMBDA(int i) {
                    if (i < d_nsize(0)) d_frontier(i) = d_nextFr(i);
                });

                auto h_ns = Kokkos::create_mirror_view(d_nsize);
                Kokkos::deep_copy(h_ns, d_nsize);
                fsize = h_ns(0);
                level++;
            }

            // ----------------------------------------------------------
            // Step 2: Update inTree and negCnt for tree edges
            // ----------------------------------------------------------
            Kokkos::parallel_for("markTree", N, KOKKOS_LAMBDA(int v) {
                const int par = d_parent(v);
                if (par < 0 || par == v) return;
                // Find the edge from v to par
                for (int j = d_nindex(v); j < d_nindex(v+1); j++) {
                    if (d_nlist(j) == par) {
                        Kokkos::atomic_fetch_add(&d_inTree(j), 1);
                        // Frustrated if sign is inconsistent with 2-coloring
                        const int w  = d_eweight(j);
                        const int cv = d_color(v), cp = d_color(par);
                        bool ok = (w >= 0) ? (cv == cp) : (cv != cp);
                        if (!ok) Kokkos::atomic_fetch_add(&d_negCnt(j), 1);
                        break;
                    }
                }
            });

            // ----------------------------------------------------------
            // Step 3: Union-find on balanced (non-frustrated) edges
            // ----------------------------------------------------------
            Kokkos::parallel_for("initLabel", N, KOKKOS_LAMBDA(int v) {
                d_label(v) = v;
            });

            // Hook: multiple rounds to propagate components
            for (int round = 0; round < 20; round++) {
                Kokkos::parallel_for("hook", N, KOKKOS_LAMBDA(int v) {
                    for (int j = d_nindex(v); j < d_nindex(v+1); j++) {
                        const int u = d_nlist(j);
                        const int w = d_eweight(j);
                        const int cv = d_color(v), cu = d_color(u);
                        // Balanced edge: positive & same color, or negative & different color
                        bool balanced = (w >= 0) ? (cv == cu) : (cv != cu);
                        if (!balanced) continue;
                        // Union v and u: point larger label root to smaller
                        int lv = d_label(v), lu = d_label(u);
                        while (lv != lu) {
                            if (lv > lu) {
                                int old = Kokkos::atomic_compare_exchange(&d_label(lv), lv, lu);
                                if (old == lv) break;
                                lv = old;
                            } else {
                                int old = Kokkos::atomic_compare_exchange(&d_label(lu), lu, lv);
                                if (old == lu) break;
                                lu = old;
                            }
                        }
                    }
                });

                // Path compression
                Kokkos::parallel_for("flatten", N, KOKKOS_LAMBDA(int v) {
                    int root_v = d_label(v);
                    while (root_v != d_label(root_v)) root_v = d_label(root_v);
                    d_label(v) = root_v;
                });
            }

            // ----------------------------------------------------------
            // Step 4: Count CC sizes
            // ----------------------------------------------------------
            Kokkos::deep_copy(d_ccCnt, 0);
            Kokkos::parallel_for("ccSize", N, KOKKOS_LAMBDA(int v) {
                Kokkos::atomic_fetch_add(&d_ccCnt(d_label(v)), 1);
            });

            // Find largest CC: two passes – first find max size, then find its label
            int max_cc_size = 0;
            Kokkos::parallel_reduce("findMaxCC", N,
                KOKKOS_LAMBDA(int v, int& mx) {
                    if (d_ccCnt(v) > mx) mx = d_ccCnt(v);
                },
                Kokkos::Max<int>(max_cc_size));

            const int mcs = max_cc_size;
            int largest_lbl = N - 1; // will be overwritten
            Kokkos::parallel_reduce("findLargestLabel", N,
                KOKKOS_LAMBDA(int v, int& best) {
                    if (d_ccCnt(v) == mcs && v < best) best = v;
                },
                Kokkos::Min<int>(largest_lbl));

            // ----------------------------------------------------------
            // Step 5: BFS across CC-graph to compute hop distances
            //         (treating edges between CCs as the BFS edges)
            // ----------------------------------------------------------
            const int large = largest_lbl; // capture for lambda

            Kokkos::deep_copy(d_hop, INT_MAX - 1);
            Kokkos::parallel_for("initHop", 1, KOKKOS_LAMBDA(int) {
                d_hop(large) = 0;
            });

            // Build per-CC hop counts via Bellman-Ford on CC adjacency
            // (simple iterative relaxation)
            for (int bfIter = 0; bfIter < 30; bfIter++) {
                Kokkos::parallel_for("bellmanFord", N, KOKKOS_LAMBDA(int v) {
                    const int lv = d_label(v);
                    for (int j = d_nindex(v); j < d_nindex(v+1); j++) {
                        const int u  = d_nlist(j);
                        const int lu = d_label(u);
                        if (lv == lu) continue;
                        // Relax: hop[lu] = min(hop[lu], hop[lv]+1)
                        if (d_hop(lv) != INT_MAX - 1) {
                            int newDist = d_hop(lv) + 1;
                            Kokkos::atomic_fetch_min(&d_hop(lu), newDist);
                        }
                        if (d_hop(lu) != INT_MAX - 1) {
                            int newDist = d_hop(lu) + 1;
                            Kokkos::atomic_fetch_min(&d_hop(lv), newDist);
                        }
                    }
                });
            }

            // ----------------------------------------------------------
            // Step 6: Increment inCC for nodes at even hop distance
            // ----------------------------------------------------------
            Kokkos::parallel_for("incCC", N, KOKKOS_LAMBDA(int v) {
                const int h = d_hop(d_label(v));
                if (h != INT_MAX - 1 && (h % 2) == 0) {
                    Kokkos::atomic_fetch_add(&d_inCC(v), 1);
                }
            });
        } // end iterations

        Kokkos::fence();
        auto t_end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(t_end - t_start).count();
        printf("Runtime: %.3f s\n", runtime);

        // ----------------------------------------------------------
        // Output results
        // ----------------------------------------------------------
        auto h_inCC   = Kokkos::create_mirror_view(d_inCC);
        auto h_inTree = Kokkos::create_mirror_view(d_inTree);
        auto h_negCnt = Kokkos::create_mirror_view(d_negCnt);
        Kokkos::deep_copy(h_inCC,   d_inCC);
        Kokkos::deep_copy(h_inTree, d_inTree);
        Kokkos::deep_copy(h_negCnt, d_negCnt);

        FILE* fout = fopen(argv[3], "wt");
        if (!fout) { fprintf(stderr, "ERROR: cannot open output file %s\n", argv[3]); }
        else {
            fprintf(fout, "original node ID, percentage node was in agreeable majority\n");
            for (int i = 0; i < N; i++)
                fprintf(fout, "%d,%.1f\n", h_origID[i], 100.0 * h_inCC(i) / iterations);

            fprintf(fout, "source node ID, destination node ID, percentage edge was in tree, percentage edge was negative\n");
            for (int v = 0; v < N; v++) {
                for (int j = h_nindex[v]; j < h_nindex[v+1]; j++) {
                    const int u = h_nlist[j];
                    if (v < u) {
                        fprintf(fout, "%d,%d,%.1f,%.1f\n",
                            h_origID[v], h_origID[u],
                            100.0 * h_inTree(j) / iterations,
                            100.0 * h_negCnt(j) / iterations);
                    }
                }
            }
            fclose(fout);
        }
    }
    Kokkos::finalize();
    return 0;
}
