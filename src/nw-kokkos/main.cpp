// Kokkos port of nw-omp
// Needleman-Wunsch sequence alignment using wavefront block decomposition.
// Args: <dimension> <penalty>  (dimension must be a multiple of 16)

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>

#define BLOCK_SIZE 16
#define LIMIT      -999

// Indices into team scratch arrays (macros reference local pointers in each lambda)
#define NW_SCORE(i, j) score_l[(j) + (i) * (BLOCK_SIZE + 1)]
#define NW_REF(i, j)   ref_l[(j) + (i) * BLOCK_SIZE]

KOKKOS_INLINE_FUNCTION int maximum(int a, int b, int c) {
  return a >= b ? (a >= c ? a : c) : (b >= c ? b : c);
}

static int blosum62[24][24] = {
  { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
  {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
  {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
  {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
  {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
  {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
  {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
  {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
  {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
  {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
  {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
  {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
  {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
  { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
  { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
  {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
  {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
  { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
  {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
  {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

// Simple O(N^2) sequential host reference (row-major fill, same recurrence as GPU kernel).
void nw_host(int* input_itemsets, int* reference, int max_cols, int penalty)
{
  for (int i = 1; i < max_cols; i++)
    for (int j = 1; j < max_cols; j++) {
      input_itemsets[i * max_cols + j] = maximum(
          input_itemsets[(i - 1) * max_cols + (j - 1)] + reference[i * max_cols + j],
          input_itemsets[ i      * max_cols + (j - 1)] - penalty,
          input_itemsets[(i - 1) * max_cols +  j     ] - penalty);
    }
}

using TeamPol    = Kokkos::TeamPolicy<>;
using MemberT    = TeamPol::member_type;
using ScratchSp  = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using ScratchInt = Kokkos::View<int*, ScratchSp, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Total scratch per team (ints):
//   score: (BLOCK_SIZE+1)*(BLOCK_SIZE+1)
//   ref  : BLOCK_SIZE*BLOCK_SIZE
static constexpr int SCORE_SIZE  = (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1);
static constexpr int REF_SIZE    = BLOCK_SIZE * BLOCK_SIZE;
static constexpr int SCRATCH_TOT = SCORE_SIZE + REF_SIZE;

int main(int argc, char* argv[])
{
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <dimension> <penalty>\n", argv[0]);
    return 1;
  }

  const int dim     = atoi(argv[1]);
  const int penalty = atoi(argv[2]);

  if (dim % 16 != 0) {
    fprintf(stderr, "Dimension must be a multiple of 16\n");
    return 1;
  }

  printf("WG size of kernel = %d\n", BLOCK_SIZE);

  const int max_cols = dim + 1;
  const int N        = max_cols * max_cols;

  int* reference        = (int*)malloc(N * sizeof(int));
  int* h_input_itemsets = (int*)malloc(N * sizeof(int));  // host reference copy
  int* input_itemsets   = (int*)malloc(N * sizeof(int));  // for GPU then verification

  srand(7);

  // Initialise to zero
  for (int i = 0; i < N; i++)
    input_itemsets[i] = h_input_itemsets[i] = 0;

  // Random boundary columns (i=1..max_cols-1, j=0)
  for (int i = 1; i < max_cols; i++)
    h_input_itemsets[i * max_cols] = input_itemsets[i * max_cols] = rand() % 10 + 1;
  // Random boundary rows (i=0, j=1..max_cols-1)
  for (int j = 1; j < max_cols; j++)
    h_input_itemsets[j] = input_itemsets[j] = rand() % 10 + 1;

  // Reference matrix from blosum62
  for (int i = 1; i < max_cols; i++)
    for (int j = 1; j < max_cols; j++)
      reference[i * max_cols + j] = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];

  // Apply penalties along boundaries (overrides random values)
  for (int i = 1; i < max_cols; i++)
    h_input_itemsets[i * max_cols] = input_itemsets[i * max_cols] = -i * penalty;
  for (int j = 1; j < max_cols; j++)
    h_input_itemsets[j] = input_itemsets[j] = -j * penalty;

  const int block_width = dim / BLOCK_SIZE;
  const int scratch_size = ScratchInt::shmem_size(SCRATCH_TOT);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> input_d("input_d", N);
    Kokkos::View<int*> reference_d("reference_d", N);

    auto input_h = Kokkos::create_mirror_view(input_d);
    auto ref_h   = Kokkos::create_mirror_view(reference_d);
    for (int i = 0; i < N; i++) { input_h(i) = input_itemsets[i]; ref_h(i) = reference[i]; }
    Kokkos::deep_copy(input_d, input_h);
    Kokkos::deep_copy(reference_d, ref_h);

    auto t_start = std::chrono::steady_clock::now();

    // ---- Upper-left wavefront pass ----
    for (int blk = 1; blk <= block_width; blk++) {
      TeamPol policy(blk, BLOCK_SIZE);
      policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

      Kokkos::parallel_for("nw_upper", policy,
        KOKKOS_LAMBDA(const MemberT& team) {
          ScratchInt scratch(team.team_scratch(0), SCRATCH_TOT);
          int* score_l = scratch.data();
          int* ref_l   = scratch.data() + SCORE_SIZE;

          const int bx = team.league_rank();
          const int tx = team.team_rank();

          const int b_index_x = bx;
          const int b_index_y = blk - 1 - bx;

          const int base_off  = max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
          const int index     = base_off + tx + max_cols + 1;
          const int index_n   = base_off + tx + 1;
          const int index_w   = base_off + max_cols;
          const int index_nw  = base_off;

          if (tx == 0) NW_SCORE(0, 0) = input_d(index_nw);

          for (int ty = 0; ty < BLOCK_SIZE; ty++)
            NW_REF(ty, tx) = reference_d(index + max_cols * ty);

          NW_SCORE(tx + 1, 0) = input_d(index_w + max_cols * tx);
          NW_SCORE(0, tx + 1) = input_d(index_n);

          team.team_barrier();

          for (int m = 0; m < BLOCK_SIZE; m++) {
            if (tx <= m) {
              int tiy = m - tx + 1, tix = tx + 1;
              NW_SCORE(tiy, tix) = maximum(
                  NW_SCORE(tiy - 1, tix - 1) + NW_REF(tiy - 1, tix - 1),
                  NW_SCORE(tiy,     tix - 1) - penalty,
                  NW_SCORE(tiy - 1, tix)     - penalty);
            }
            team.team_barrier();
          }
          for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
            if (tx <= m) {
              int tiy = BLOCK_SIZE - tx, tix = tx + BLOCK_SIZE - m;
              NW_SCORE(tiy, tix) = maximum(
                  NW_SCORE(tiy - 1, tix - 1) + NW_REF(tiy - 1, tix - 1),
                  NW_SCORE(tiy,     tix - 1) - penalty,
                  NW_SCORE(tiy - 1, tix)     - penalty);
            }
            team.team_barrier();
          }

          for (int ty = 0; ty < BLOCK_SIZE; ty++)
            input_d(index + max_cols * ty) = NW_SCORE(ty + 1, tx + 1);
        });
      Kokkos::fence();
    }

    // ---- Lower-right wavefront pass ----
    for (int blk = block_width - 1; blk >= 1; blk--) {
      TeamPol policy(blk, BLOCK_SIZE);
      policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

      Kokkos::parallel_for("nw_lower", policy,
        KOKKOS_LAMBDA(const MemberT& team) {
          ScratchInt scratch(team.team_scratch(0), SCRATCH_TOT);
          int* score_l = scratch.data();
          int* ref_l   = scratch.data() + SCORE_SIZE;

          const int bx = team.league_rank();
          const int tx = team.team_rank();

          const int b_index_x = bx + block_width - blk;
          const int b_index_y = block_width - bx - 1;

          const int base_off  = max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
          const int index     = base_off + tx + max_cols + 1;
          const int index_n   = base_off + tx + 1;
          const int index_w   = base_off + max_cols;
          const int index_nw  = base_off;

          if (tx == 0) NW_SCORE(0, 0) = input_d(index_nw);

          for (int ty = 0; ty < BLOCK_SIZE; ty++)
            NW_REF(ty, tx) = reference_d(index + max_cols * ty);

          NW_SCORE(tx + 1, 0) = input_d(index_w + max_cols * tx);
          NW_SCORE(0, tx + 1) = input_d(index_n);

          team.team_barrier();

          for (int m = 0; m < BLOCK_SIZE; m++) {
            if (tx <= m) {
              int tiy = m - tx + 1, tix = tx + 1;
              NW_SCORE(tiy, tix) = maximum(
                  NW_SCORE(tiy - 1, tix - 1) + NW_REF(tiy - 1, tix - 1),
                  NW_SCORE(tiy,     tix - 1) - penalty,
                  NW_SCORE(tiy - 1, tix)     - penalty);
            }
            team.team_barrier();
          }
          for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
            if (tx <= m) {
              int tiy = BLOCK_SIZE - tx, tix = tx + BLOCK_SIZE - m;
              NW_SCORE(tiy, tix) = maximum(
                  NW_SCORE(tiy - 1, tix - 1) + NW_REF(tiy - 1, tix - 1),
                  NW_SCORE(tiy,     tix - 1) - penalty,
                  NW_SCORE(tiy - 1, tix)     - penalty);
            }
            team.team_barrier();
          }

          for (int ty = 0; ty < BLOCK_SIZE; ty++)
            input_d(index + max_cols * ty) = NW_SCORE(ty + 1, tx + 1);
        });
      Kokkos::fence();
    }

    auto t_end = std::chrono::steady_clock::now();
    printf("Total kernel execution time: %f (s)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9f);

    // Copy result back
    Kokkos::deep_copy(input_h, input_d);
    for (int i = 0; i < N; i++) input_itemsets[i] = input_h(i);
  }
  Kokkos::finalize();

  // ---- Host reference ----
  nw_host(h_input_itemsets, reference, max_cols, penalty);

  int err = memcmp(input_itemsets, h_input_itemsets, N * sizeof(int));
  printf("%s\n", err ? "FAIL" : "PASS");

  free(reference);
  free(input_itemsets);
  free(h_input_itemsets);
  return 0;
}
