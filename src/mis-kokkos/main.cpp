/*
 * Kokkos port of ECL-MIS maximal independent set algorithm.
 * Uses a synthetic k-regular ring graph instead of file I/O.
 * Args: <nodes> <edges_per_node> <repeat>
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

// Use int (32-bit) for nstat to ensure Kokkos atomics work portably.
static const int STATUS_IN  = 0xfe;
static const int STATUS_OUT = 0;

KOKKOS_INLINE_FUNCTION
unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <nodes> <edges_per_node> <repeat>\n", argv[0]);
    return 1;
  }

  const int nodes        = atoi(argv[1]);
  int       half_k       = atoi(argv[2]) / 2;
  if (half_k < 1) half_k = 1;
  const int edges_per_node = half_k * 2;
  const int edges          = nodes * edges_per_node;
  const int repeat         = atoi(argv[3]);

  printf("ECL-MIS Kokkos: synthetic ring graph %d nodes, %d edges (degree %d)\n",
         nodes, edges, edges_per_node);

  // Build k-regular ring graph on host
  std::vector<int> h_nidx(nodes + 1), h_nlist(edges);
  for (int i = 0; i <= nodes; i++) h_nidx[i] = i * edges_per_node;
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < half_k; j++) {
      h_nlist[i * edges_per_node + j]          = (i + j + 1) % nodes;
      h_nlist[i * edges_per_node + half_k + j] = (i - j - 1 + nodes) % nodes;
    }
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_nidx ("nidx",  nodes + 1);
    Kokkos::View<int*> d_nlist("nlist", edges);
    Kokkos::View<int*> d_nstat("nstat", nodes);
    // 0-D view used as an atomic flag (missing = any unresolved candidate)
    Kokkos::View<int>  d_missing("missing");

    // Transfer graph to device
    {
      auto hv = Kokkos::create_mirror_view(d_nidx);
      for (int i = 0; i <= nodes; i++) hv(i) = h_nidx[i];
      Kokkos::deep_copy(d_nidx, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_nlist);
      for (int i = 0; i < edges; i++) hv(i) = h_nlist[i];
      Kokkos::deep_copy(d_nlist, hv);
    }

    const float avg       = (float)edges / nodes;
    const float scaledavg = ((STATUS_IN / 2) - 1) * avg;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < repeat; n++) {

      // ---- Phase 1: assign priority scores ----
      Kokkos::parallel_for("mis_score", nodes, KOKKOS_LAMBDA(int i) {
        int val    = STATUS_IN;
        int degree = d_nidx(i + 1) - d_nidx(i);
        if (degree > 0) {
          float x   = degree - (hash(i) * 0.00000000023283064365386962890625f);
          int   res = (int)(scaledavg / (avg + x));
          val = (res + res) | 1;
        }
        d_nstat(i) = val;
      });
      Kokkos::fence();

      // ---- Phase 2: conflict resolution (iterate until stable) ----
      int missing;
      do {
        Kokkos::deep_copy(d_missing, 0);

        Kokkos::parallel_for("mis_resolve", nodes, KOKKOS_LAMBDA(int v) {
          const int nv = d_nstat(v);
          if (!(nv & 1)) return;  // already resolved (even status)

          int i = d_nidx(v);
          // Advance past all neighbors that v beats
          while (i < d_nidx(v + 1) &&
                 ((nv > d_nstat(d_nlist(i))) ||
                  ((nv == d_nstat(d_nlist(i))) && (v > d_nlist(i))))) {
            i++;
          }

          if (i < d_nidx(v + 1)) {
            // v lost to a neighbor – still unresolved
            Kokkos::atomic_store(&d_missing(), 1);
          } else {
            // v wins: join MIS and eliminate neighbors
            for (int j = d_nidx(v); j < d_nidx(v + 1); j++)
              Kokkos::atomic_store(&d_nstat(d_nlist(j)), STATUS_OUT);
            d_nstat(v) = STATUS_IN;
          }
        });

        Kokkos::fence();
        Kokkos::deep_copy(missing, d_missing);
      } while (missing);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float runtime = (float)std::chrono::duration<double>(t_end - t_start).count() / repeat;
    printf("compute time:  %.6f s\n",         runtime);
    printf("throughput:    %.6f Mnodes/s\n",   nodes * 1e-6f / runtime);
    printf("throughput:    %.6f Medges/s\n",   edges * 1e-6f / runtime);

    // Verify MIS
    auto h_nstat = Kokkos::create_mirror_view(d_nstat);
    Kokkos::deep_copy(h_nstat, d_nstat);

    bool ok = true;
    for (int v = 0; v < nodes && ok; v++) {
      int st = h_nstat(v);
      if (st != STATUS_IN && st != STATUS_OUT) {
        fprintf(stderr, "ERROR: unprocessed node %d (status=%d)\n", v, st);
        ok = false;
      } else if (st == STATUS_IN) {
        for (int i = h_nidx[v]; i < h_nidx[v + 1] && ok; i++)
          if (h_nstat(h_nlist[i]) == STATUS_IN) {
            fprintf(stderr, "ERROR: adjacent nodes %d and %d both in MIS\n", v, h_nlist[i]);
            ok = false;
          }
      } else {
        bool found_in_nbr = false;
        for (int i = h_nidx[v]; i < h_nidx[v + 1]; i++)
          if (h_nstat(h_nlist[i]) == STATUS_IN) { found_in_nbr = true; break; }
        if (!found_in_nbr) {
          fprintf(stderr, "ERROR: node %d excluded but has no MIS neighbor (set not maximal)\n", v);
          ok = false;
        }
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  Kokkos::finalize();
  return 0;
}
