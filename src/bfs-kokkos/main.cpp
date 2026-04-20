#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <Kokkos_Core.hpp>

struct Node {
  int starting;
  int no_of_edges;
};

void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost_ref)
{
  char stop;
  do {
    stop = 0;
    for (int tid = 0; tid < no_of_nodes; tid++) {
      if (h_graph_mask[tid] == 1) {
        h_graph_mask[tid] = 0;
        for (int i = h_graph_nodes[tid].starting;
             i < (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++) {
          int id = h_graph_edges[i];
          if (!h_graph_visited[id]) {
            h_cost_ref[id] = h_cost_ref[tid] + 1;
            h_updating_graph_mask[id] = 1;
          }
        }
      }
    }
    for (int tid = 0; tid < no_of_nodes; tid++) {
      if (h_updating_graph_mask[tid] == 1) {
        h_graph_mask[tid] = 1;
        h_graph_visited[tid] = 1;
        stop = 1;
        h_updating_graph_mask[tid] = 0;
      }
    }
  } while (stop);
}

template<typename compare_type>
void compare_results(compare_type *ref, compare_type *result, int len) {
  for (int i = 0; i < len; i++) {
    if (ref[i] != result[i]) {
      printf("Mismatch at index %d: ref=%d, got=%d\n", i, (int)ref[i], (int)result[i]);
      printf("FAIL\n");
      return;
    }
  }
  printf("PASS\n");
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
    exit(0);
  }

  FILE *fp = fopen(argv[1], "r");
  if (!fp) {
    printf("Error Reading graph file\n");
    return 1;
  }

  int no_of_nodes;
  fscanf(fp, "%d", &no_of_nodes);

  Node *h_graph_nodes = (Node*) malloc(sizeof(Node) * no_of_nodes);
  char *h_graph_mask          = (char*) malloc(sizeof(char) * no_of_nodes);
  char *h_updating_graph_mask = (char*) malloc(sizeof(char) * no_of_nodes);
  char *h_graph_visited       = (char*) malloc(sizeof(char) * no_of_nodes);

  int start, edgeno;
  for (int i = 0; i < no_of_nodes; i++) {
    fscanf(fp, "%d %d", &start, &edgeno);
    h_graph_nodes[i].starting    = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i] = 0;
    h_updating_graph_mask[i] = 0;
    h_graph_visited[i] = 0;
  }

  int source = 0;
  fscanf(fp, "%d", &source);
  source = 0;
  h_graph_mask[source]    = 1;
  h_graph_visited[source] = 1;

  int edge_list_size;
  fscanf(fp, "%d", &edge_list_size);
  int *h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);
  for (int i = 0; i < edge_list_size; i++) {
    int id, cost;
    fscanf(fp, "%d", &id);
    fscanf(fp, "%d", &cost);
    h_graph_edges[i] = id;
  }
  if (fp) fclose(fp);

  int *h_cost     = (int*) malloc(sizeof(int) * no_of_nodes);
  int *h_cost_ref = (int*) malloc(sizeof(int) * no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++) {
    h_cost[i]     = -1;
    h_cost_ref[i] = -1;
  }
  h_cost[source]     = 0;
  h_cost_ref[source] = 0;

  printf("run bfs (#nodes = %d) on device\n", no_of_nodes);

  Kokkos::initialize(argc, argv);
  {
    using ViewChar  = Kokkos::View<char*>;
    using ViewInt   = Kokkos::View<int*>;
    using ViewNode  = Kokkos::View<Node*>;

    ViewNode  d_nodes("nodes",    no_of_nodes);
    ViewChar  d_mask("mask",      no_of_nodes);
    ViewChar  d_updating("upd",   no_of_nodes);
    ViewChar  d_visited("vis",    no_of_nodes);
    ViewInt   d_edges("edges",    edge_list_size);
    ViewInt   d_cost("cost",      no_of_nodes);

    auto h_d_nodes    = Kokkos::create_mirror_view(d_nodes);
    auto h_d_mask     = Kokkos::create_mirror_view(d_mask);
    auto h_d_updating = Kokkos::create_mirror_view(d_updating);
    auto h_d_visited  = Kokkos::create_mirror_view(d_visited);
    auto h_d_edges    = Kokkos::create_mirror_view(d_edges);
    auto h_d_cost     = Kokkos::create_mirror_view(d_cost);

    for (int i = 0; i < no_of_nodes; i++) {
      h_d_nodes(i)    = h_graph_nodes[i];
      h_d_mask(i)     = h_graph_mask[i];
      h_d_updating(i) = h_updating_graph_mask[i];
      h_d_visited(i)  = h_graph_visited[i];
      h_d_cost(i)     = h_cost[i];
    }
    for (int i = 0; i < edge_list_size; i++)
      h_d_edges(i) = h_graph_edges[i];

    Kokkos::deep_copy(d_nodes,    h_d_nodes);
    Kokkos::deep_copy(d_mask,     h_d_mask);
    Kokkos::deep_copy(d_updating, h_d_updating);
    Kokkos::deep_copy(d_visited,  h_d_visited);
    Kokkos::deep_copy(d_edges,    h_d_edges);
    Kokkos::deep_copy(d_cost,     h_d_cost);

    // over flag stored as scalar view
    Kokkos::View<int[1]> d_over("over");

    long time = 0;
    bool over = false;

    do {
      Kokkos::deep_copy(d_over, 0);

      auto tstart = std::chrono::steady_clock::now();

      // Kernel 1: expand frontier
      Kokkos::parallel_for("bfs_k1", no_of_nodes, KOKKOS_LAMBDA(int tid) {
        if (d_mask(tid)) {
          d_mask(tid) = 0;
          const int s = d_nodes(tid).starting;
          const int e = s + d_nodes(tid).no_of_edges;
          for (int i = s; i < e; i++) {
            int id = d_edges(i);
            if (!d_visited(id)) {
              d_cost(id) = d_cost(tid) + 1;
              d_updating(id) = 1;
            }
          }
        }
      });

      // Kernel 2: update masks
      Kokkos::parallel_for("bfs_k2", no_of_nodes, KOKKOS_LAMBDA(int tid) {
        if (d_updating(tid)) {
          d_mask(tid)     = 1;
          d_visited(tid)  = 1;
          d_over(0)       = 1;
          d_updating(tid) = 0;
        }
      });

      Kokkos::fence();
      auto tend = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();

      auto h_over = Kokkos::create_mirror_view(d_over);
      Kokkos::deep_copy(h_over, d_over);
      over = (h_over(0) != 0);
    } while (over);

    printf("Total kernel execution time : %f (us)\n", time * 1e-3f);

    Kokkos::deep_copy(h_d_cost, d_cost);
    for (int i = 0; i < no_of_nodes; i++)
      h_cost[i] = h_d_cost(i);
  }
  Kokkos::finalize();

  printf("run bfs (#nodes = %d) on host (cpu)\n", no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++) {
    h_graph_mask[i] = 0;
    h_updating_graph_mask[i] = 0;
    h_graph_visited[i] = 0;
  }
  source = 0;
  h_graph_mask[source]    = 1;
  h_graph_visited[source] = 1;
  run_bfs_cpu(no_of_nodes, h_graph_nodes, edge_list_size, h_graph_edges,
      h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);

  compare_results<int>(h_cost_ref, h_cost, no_of_nodes);

  free(h_graph_nodes);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
  free(h_cost_ref);
  free(h_graph_edges);
  return 0;
}
