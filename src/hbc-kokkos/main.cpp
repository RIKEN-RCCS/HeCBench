/*
Betweenness Centrality (BC) computation on graphs.
Kokkos port (OpenMP backend).

Algorithm (Brandes):
  For each source vertex s:
    1. BFS: compute d[v] = distance, sigma[v] = number of shortest paths
    2. Backward: delta[v] = dependency score
    3. BC[v] += delta[v] / 2

Level boundaries are tracked host-side (std::vector) so we avoid
device-side level-array bookkeeping.

If no input file given, generates a random graph (1000 nodes, ~5000 edges).
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <climits>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>

struct Graph {
    int n, m;
    int *R, *C;
};

static Graph makeRandomGraph(int n, int target_edges, unsigned seed)
{
    srand(seed);
    std::set<std::pair<int,int>> edgeSet;
    while ((int)edgeSet.size() < target_edges) {
        int u = rand() % n, v = rand() % n;
        if (u == v) continue;
        edgeSet.insert({u, v});
        edgeSet.insert({v, u});
    }
    std::vector<std::vector<int>> adj(n);
    for (auto& [u, v] : edgeSet) adj[u].push_back(v);

    Graph g;
    g.n = n;
    g.R = new int[n + 1];
    g.R[0] = 0;
    for (int i = 0; i < n; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        g.R[i+1] = g.R[i] + (int)adj[i].size();
    }
    g.m = g.R[n];
    g.C = new int[g.m];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (int)adj[i].size(); j++)
            g.C[g.R[i] + j] = adj[i][j];
    return g;
}

static Graph parseGraph(const char* fname)
{
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    std::vector<std::pair<int,int>> edges;
    int maxNode = -1, u, v;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%d %d", &u, &v) == 2 ||
            sscanf(line, "%d,%d", &u, &v) == 2) {
            if (u == v) continue;
            edges.push_back({u, v});
            edges.push_back({v, u});
            maxNode = std::max(maxNode, std::max(u, v));
        }
    }
    fclose(f);
    int n = maxNode + 1;
    std::vector<std::vector<int>> adj(n);
    for (auto& [a, b] : edges) adj[a].push_back(b);

    Graph g;
    g.n = n;
    g.R = new int[n + 1];
    g.R[0] = 0;
    for (int i = 0; i < n; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        g.R[i+1] = g.R[i] + (int)adj[i].size();
    }
    g.m = g.R[n];
    g.C = new int[g.m];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (int)adj[i].size(); j++)
            g.C[g.R[i] + j] = adj[i][j];
    return g;
}

int main(int argc, char* argv[])
{
    const char* infile = nullptr;
    int  k_src  = -1;
    bool approx = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f") && i+1 < argc)  infile = argv[++i];
        else if (!strcmp(argv[i], "-k") && i+1 < argc) k_src = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--approx"))  approx = true;
    }

    Graph g = infile ? parseGraph(infile) : makeRandomGraph(1000, 5000, 42);
    printf("Number of nodes: %d\n", g.n);
    printf("Number of edges: %d\n", g.m);

    const int n = g.n, m = g.m;

    std::vector<int> sources;
    if (approx && k_src > 0) {
        srand(123);
        std::set<int> chosen;
        while ((int)chosen.size() < std::min(k_src, n)) chosen.insert(rand() % n);
        sources.assign(chosen.begin(), chosen.end());
    } else {
        sources.resize(n);
        for (int i = 0; i < n; i++) sources[i] = i;
        if (k_src > 0 && k_src < n) sources.resize(k_src);
    }
    const int nSrc = (int)sources.size();

    Kokkos::initialize(argc, argv);
    {
        using ViewI  = Kokkos::View<int*>;
        using ViewLL = Kokkos::View<long long*>;
        using ViewD  = Kokkos::View<double*>;
        using ViewF  = Kokkos::View<float*>;

        ViewI  d_R    ("R",     n + 1);
        ViewI  d_C    ("C",     m);
        ViewI  d_dist ("dist",  n);
        ViewLL d_sigma("sigma", n);
        ViewD  d_delta("delta", n);
        ViewF  d_bc   ("bc",    n);
        ViewI  d_Q    ("Q",     n);   // current BFS frontier
        ViewI  d_nextQ("nextQ", n);   // next frontier
        ViewI  d_Qsz  ("Qsz",  1);
        ViewI  d_NQsz ("NQsz", 1);
        ViewI  d_stack("stack", n);   // nodes in BFS order (for backward pass)

        {
            auto hR = Kokkos::create_mirror_view(d_R);
            auto hC = Kokkos::create_mirror_view(d_C);
            for (int i = 0; i <= n; i++) hR(i) = g.R[i];
            for (int i = 0; i < m;  i++) hC(i) = g.C[i];
            Kokkos::deep_copy(d_R, hR);
            Kokkos::deep_copy(d_C, hC);
        }

        Kokkos::deep_copy(d_bc, 0.0f);

        auto t1 = std::chrono::high_resolution_clock::now();

        for (int si = 0; si < nSrc; si++) {
            const int s = sources[si];

            // ---- Initialize per-source arrays ----
            Kokkos::parallel_for("initBC", n, KOKKOS_LAMBDA(int v) {
                d_dist (v) = -1;
                d_sigma(v) = 0LL;
                d_delta(v) = 0.0;
            });
            Kokkos::parallel_for("initSrc", 1, KOKKOS_LAMBDA(int) {
                d_dist (s) = 0;
                d_sigma(s) = 1LL;
                d_Q(0)     = s;
                d_Qsz (0)  = 1;
            });

            // Host-side level boundary list: level L occupies stack[levelBeg[L]..levelBeg[L+1])
            std::vector<int> levelBeg;
            levelBeg.push_back(0);

            int stackPtr = 0;
            int qsize    = 1;

            // ---- BFS forward pass ----
            while (qsize > 0) {
                // Push current frontier onto stack
                const int sp = stackPtr;
                const int qs = qsize;
                Kokkos::parallel_for("pushStack", qs, KOKKOS_LAMBDA(int i) {
                    d_stack(sp + i) = d_Q(i);
                });
                stackPtr += qsize;
                levelBeg.push_back(stackPtr);

                // Expand frontier
                Kokkos::parallel_for("clearNQ", 1, KOKKOS_LAMBDA(int) {
                    d_NQsz(0) = 0;
                });

                Kokkos::parallel_for("BFSfwd", qs, KOKKOS_LAMBDA(int fi) {
                    const int u  = d_Q(fi);
                    const int du = d_dist(u);
                    for (int e = d_R(u); e < d_R(u+1); e++) {
                        const int v = d_C(e);
                        // First visit: claim the node
                        if (Kokkos::atomic_compare_exchange(&d_dist(v), -1, du + 1) == -1) {
                            const int idx = Kokkos::atomic_fetch_add(&d_NQsz(0), 1);
                            d_nextQ(idx) = v;
                        }
                        // Any path through u at the right distance contributes sigma
                        if (d_dist(v) == du + 1) {
                            Kokkos::atomic_fetch_add(&d_sigma(v), d_sigma(u));
                        }
                    }
                });

                // Swap frontiers
                Kokkos::parallel_for("swapQ", n, KOKKOS_LAMBDA(int i) {
                    if (i < d_NQsz(0)) d_Q(i) = d_nextQ(i);
                });

                auto hNQ = Kokkos::create_mirror_view(d_NQsz);
                Kokkos::deep_copy(hNQ, d_NQsz);
                qsize = hNQ(0);
            }

            const int numLevels = (int)levelBeg.size() - 1;

            // ---- BFS backward pass (reverse level order) ----
            for (int L = numLevels - 1; L >= 0; L--) {
                const int beg = levelBeg[L];
                const int end = levelBeg[L + 1];
                const int len = end - beg;
                if (len <= 0) continue;

                const int base = beg;
                Kokkos::parallel_for("BCback", len, KOKKOS_LAMBDA(int ii) {
                    const int u = d_stack(base + ii);
                    double dep = 0.0;
                    for (int e = d_R(u); e < d_R(u+1); e++) {
                        const int v = d_C(e);
                        if (d_dist(v) == d_dist(u) + 1 && d_sigma(v) > 0LL) {
                            dep += ((double)d_sigma(u) / (double)d_sigma(v))
                                   * (1.0 + d_delta(v));
                        }
                    }
                    d_delta(u) = dep;
                });
            }

            // Accumulate BC (exclude source)
            Kokkos::parallel_for("accumBC", n, KOKKOS_LAMBDA(int v) {
                if (v != s) d_bc(v) += (float)(d_delta(v) / 2.0);
            });
        }

        Kokkos::fence();
        auto t2 = std::chrono::high_resolution_clock::now();
        float GPU_time_us =
            (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        printf("Time for GPU execution: %.9f s\n", GPU_time_us / 1.0e6f);
    }
    Kokkos::finalize();
    delete[] g.R;
    delete[] g.C;
    return 0;
}
