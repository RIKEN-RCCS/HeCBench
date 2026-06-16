#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

#define _QUEENS_BLOCK_SIZE_ 128
#define _EMPTY_             -1

typedef struct queen_root {
  unsigned int control;
  int8_t board[12];
} QueenRoot;

// ---------------------------------------------------------------------------
// CPU helpers: prefix generation
// ---------------------------------------------------------------------------

inline void prefixesHandleSol(QueenRoot* root_prefixes, unsigned int flag,
                               const char* board, int initialDepth, int num_sol)
{
  root_prefixes[num_sol].control = flag;
  for (int i = 0; i < initialDepth; ++i)
    root_prefixes[num_sol].board[i] = board[i];
}

inline bool MCstillLegal(const char* board, const int r)
{
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
  int ld = board[r];
  int rd = board[r];
  for (int i = r - 1; i >= 0; --i) {
    --ld; ++rd;
    if (board[i] == ld || board[i] == rd) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Device-callable legality check (same logic, no short-circuit for device)
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
bool queens_stillLegal(const char* board, const int r)
{
  bool safe = true;
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) safe = false;
  int ld = board[r];
  int rd = board[r];
  for (int i = r - 1; i >= 0; --i) {
    --ld; ++rd;
    if (board[i] == ld || board[i] == rd) safe = false;
  }
  return safe;
}

// ---------------------------------------------------------------------------
// CPU: generate DFS prefixes up to initialDepth
// ---------------------------------------------------------------------------

unsigned long long BP_queens_prefixes(int size, int initialDepth,
                                      unsigned long long* tree_size,
                                      QueenRoot* root_prefixes)
{
  unsigned int flag = 0;
  int bit_test = 0;
  char vertice[20];
  unsigned long long local_tree = 0ULL;
  unsigned long long num_sol = 0;

  for (int i = 0; i < size; ++i) vertice[i] = -1;
  int nivel = 0;

  do {
    vertice[nivel]++;
    bit_test = 0;
    bit_test |= (1 << vertice[nivel]);
    if (vertice[nivel] == size) {
      vertice[nivel] = _EMPTY_;
    } else if (MCstillLegal(vertice, nivel) && !(flag & bit_test)) {
      flag |= (1ULL << vertice[nivel]);
      nivel++;
      ++local_tree;
      if (nivel == initialDepth) {
        prefixesHandleSol(root_prefixes, flag, vertice, initialDepth, num_sol);
        num_sol++;
      } else continue;
    } else continue;
    nivel--;
    flag &= ~(1ULL << vertice[nivel]);
  } while (nivel >= 0);

  *tree_size = local_tree;
  return num_sol;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  // Accept either <size> <repeat>  or  <size> <initial_depth> <repeat>
  if (argc != 3 && argc != 4) {
    printf("Usage: %s <size> [<initial_depth>] <repeat>\n", argv[0]);
    return 1;
  }

  int size, initialDepth, repeat;
  if (argc == 3) {
    size   = atoi(argv[1]);
    repeat = atoi(argv[2]);
    // Auto-select a reasonable prefix depth
    initialDepth = size / 3;
    if (initialDepth < 1) initialDepth = 1;
  } else {
    size         = atoi(argv[1]);
    initialDepth = atoi(argv[2]);
    repeat       = atoi(argv[3]);
  }

  printf("\n### Initial depth: %d - Size: %d:\n", initialDepth, size);

  unsigned long long tree_size = 0ULL;
  unsigned long long qtd_sols_global = 0ULL;
  unsigned int nMaxPrefixos = 75580635;

  QueenRoot*          root_prefixes_h       = (QueenRoot*)malloc(sizeof(QueenRoot) * nMaxPrefixos);
  unsigned long long* vector_of_tree_size_h = (unsigned long long*)malloc(sizeof(unsigned long long) * nMaxPrefixos);
  unsigned long long* solutions_h           = (unsigned long long*)malloc(sizeof(unsigned long long) * nMaxPrefixos);

  if (!root_prefixes_h || !vector_of_tree_size_h || !solutions_h) {
    printf("Error: host out of memory\n");
    free(root_prefixes_h);
    free(vector_of_tree_size_h);
    free(solutions_h);
    return 1;
  }

  // Generate prefix nodes on CPU
  unsigned long long n_explorers =
    BP_queens_prefixes(size, initialDepth, &tree_size, root_prefixes_h);

  printf("\n### Regular BP-DFS search. ###\n");

  Kokkos::initialize(argc, argv);
  {
    // Build device views
    Kokkos::View<QueenRoot*>          d_prefixes("prefixes",  n_explorers);
    Kokkos::View<unsigned long long*> d_sols(    "sols",      n_explorers);
    Kokkos::View<unsigned long long*> d_tree(    "tree_size", n_explorers);

    // Upload prefix data
    auto h_prefixes = Kokkos::create_mirror_view(d_prefixes);
    for (unsigned long long i = 0; i < n_explorers; ++i)
      h_prefixes[i] = root_prefixes_h[i];
    Kokkos::deep_copy(d_prefixes, h_prefixes);

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::deep_copy(d_sols, (unsigned long long)0);
      Kokkos::deep_copy(d_tree, (unsigned long long)0);

      int N           = size;
      int depthGlobal = initialDepth;

      Kokkos::parallel_for("nqueen",
        Kokkos::RangePolicy<>(0, (int64_t)n_explorers),
        KOKKOS_LAMBDA(int64_t idx) {
          unsigned int flag = 0;
          unsigned int bit_test = 0;
          char vertice[20];
          unsigned long long qtd_solutions_thread = 0ULL;
          unsigned long long local_tree_size      = 0ULL;

          for (int i = 0; i < N; ++i) vertice[i] = _EMPTY_;

          flag = d_prefixes[idx].control;
          for (int i = 0; i < depthGlobal; ++i)
            vertice[i] = (char)d_prefixes[idx].board[i];

          int depth = depthGlobal;

          do {
            vertice[depth]++;
            bit_test = 0;
            bit_test |= (1 << vertice[depth]);
            if (vertice[depth] == N) {
              vertice[depth] = _EMPTY_;
            } else if (!(flag & bit_test) && queens_stillLegal(vertice, depth)) {
              ++local_tree_size;
              flag |= (1ULL << vertice[depth]);
              depth++;
              if (depth == N) {
                ++qtd_solutions_thread;
              } else continue;
            } else continue;
            depth--;
            flag &= ~(1ULL << vertice[depth]);
          } while (depth >= depthGlobal);

          d_sols[idx] = qtd_solutions_thread;
          d_tree[idx] = local_tree_size;
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9) / repeat);

    // Copy results back to host
    auto h_sols = Kokkos::create_mirror_view(d_sols);
    auto h_tree = Kokkos::create_mirror_view(d_tree);
    Kokkos::deep_copy(h_sols, d_sols);
    Kokkos::deep_copy(h_tree, d_tree);

    for (unsigned long long i = 0; i < n_explorers; ++i) {
      solutions_h[i]           = h_sols[i];
      vector_of_tree_size_h[i] = h_tree[i];
    }
  }
  Kokkos::finalize();

  printf("\nTree size: %llu", tree_size);

  for (unsigned long long i = 0; i < n_explorers; ++i) {
    if (solutions_h[i] > 0)           qtd_sols_global += solutions_h[i];
    if (vector_of_tree_size_h[i] > 0) tree_size       += vector_of_tree_size_h[i];
  }

  printf("\nNumber of solutions found: %llu \nTree size: %llu\n",
         qtd_sols_global, tree_size);

  // Reference check for size=15, initialDepth=7
  if (size == 15 && initialDepth == 7) {
    if (qtd_sols_global == 2279184 && tree_size == 171129071)
      printf("PASS\n");
    else
      printf("FAIL\n");
  }

  free(root_prefixes_h);
  free(vector_of_tree_size_h);
  free(solutions_h);
  return 0;
}
