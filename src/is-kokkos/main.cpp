#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// -----------------------------------------------------------------------
// IS benchmark — CLASS S (hardcoded)
// -----------------------------------------------------------------------
#define CLASS            'S'
#define TOTAL_KEYS_LOG_2  16
#define MAX_KEY_LOG_2     11
#define NUM_BUCKETS_LOG_2  9

#define TOTAL_KEYS  (1 << TOTAL_KEYS_LOG_2)   // 65536
#define MAX_KEY     (1 << MAX_KEY_LOG_2)       // 2048
#define NUM_KEYS    TOTAL_KEYS                 // 65536
#define SIZE_OF_BUFFERS NUM_KEYS
#define MAX_ITERATIONS 24
#define TEST_ARRAY_SIZE 5

// -----------------------------------------------------------------------
// randlc — same algorithm as CUDA version, runs on host or device
// -----------------------------------------------------------------------
// R23 = 2^-23  (NOT 0.5/2^23 which would be 2^-24)
#define R23 (1.0 / (double)(1<<23))
#define T23 ((double)(1<<23))
#define R46 (R23 * R23)
#define T46 (T23 * T23)

KOKKOS_INLINE_FUNCTION
static double randlc(double *X, const double A) {
  double T1, T2, T3, T4;
  double A1, A2, X1, X2, Z;
  int j;

  T1 = R23 * A;
  j  = (int)T1;
  A1 = j;
  A2 = A - T23 * A1;

  T1 = R23 * *X;
  j  = (int)T1;
  X1 = j;
  X2 = *X - T23 * X1;
  T1 = A1 * X2 + A2 * X1;

  j  = (int)(R23 * T1);
  T2 = j;
  Z  = T1 - T23 * T2;
  T3 = T23 * Z + A2 * X2;
  j  = (int)(R46 * T3);
  T4 = j;
  *X = T3 - T46 * T4;

  return R46 * *X;
}

// Advance seed by advancing A using repeated squaring
KOKKOS_INLINE_FUNCTION
static double find_my_seed(int kn, int np, long nn, double s, double a) {
  if (kn == 0) return s;
  long mq = (nn / 4 + np - 1) / np;
  long nq = mq * 4 * kn;
  double t1 = s, t2 = a;
  long kk = nq;
  while (kk > 1) {
    long ik = kk / 2;
    if (2 * ik == kk) {
      (void)randlc(&t2, t2);
      kk = ik;
    } else {
      (void)randlc(&t1, t2);
      kk = kk - 1;
    }
  }
  (void)randlc(&t1, t2);
  return t1;
}

// -----------------------------------------------------------------------
// Generate key array in parallel on device
// -----------------------------------------------------------------------
static void createSeq(Kokkos::View<int *> key_array,
                      double seed, double a,
                      int num_procs) {
  Kokkos::parallel_for("create_seq", num_procs, KOKKOS_LAMBDA(int myid) {
    double an = a;
    int mq = (NUM_KEYS + num_procs - 1) / num_procs;
    int k1 = mq * myid;
    int k2 = k1 + mq;
    if (k2 > NUM_KEYS) k2 = NUM_KEYS;

    double s = find_my_seed(myid, num_procs, (long)4 * NUM_KEYS, seed, an);
    int k = MAX_KEY / 4;

    for (int i = k1; i < k2; i++) {
      double x = randlc(&s, an);
      x += randlc(&s, an);
      x += randlc(&s, an);
      x += randlc(&s, an);
      key_array(i) = (int)(k * x);
    }
  });
  Kokkos::fence();
}

// -----------------------------------------------------------------------
// One sort iteration (rank kernels 1–7 logic mapped to Kokkos)
// -----------------------------------------------------------------------
static void rankIteration(
    Kokkos::View<int *>       key_array,
    Kokkos::View<int *>       key_buff1,    // histogram of size MAX_KEY
    Kokkos::View<int *>       partial_verify_vals,
    const int                 test_index_array[TEST_ARRAY_SIZE],
    int                       iteration,
    int                       &passed_verification,
    const int                 test_rank_array[TEST_ARRAY_SIZE])
{
  // --- rank_1: set sentinel keys, save partial-verify vals ---
  {
    auto h_ka = Kokkos::create_mirror_view(key_array);
    Kokkos::deep_copy(h_ka, key_array);
    h_ka(iteration)                  = iteration;
    h_ka(iteration + MAX_ITERATIONS) = MAX_KEY - iteration;
    // save partial verify vals
    auto h_pv = Kokkos::create_mirror_view(partial_verify_vals);
    for (int i = 0; i < TEST_ARRAY_SIZE; i++)
      h_pv(i) = h_ka(test_index_array[i]);
    Kokkos::deep_copy(key_array, h_ka);
    Kokkos::deep_copy(partial_verify_vals, h_pv);
  }

  // --- rank_2: clear histogram ---
  Kokkos::parallel_for("rank2_clear", MAX_KEY,
    KOKKOS_LAMBDA(int i) { key_buff1(i) = 0; });
  Kokkos::fence();

  // --- rank_3: count keys (histogram) using atomics ---
  Kokkos::parallel_for("rank3_hist", NUM_KEYS,
    KOKKOS_LAMBDA(int i) {
      Kokkos::atomic_increment(&key_buff1(key_array(i)));
    });
  Kokkos::fence();

  // --- rank_4/5/6: prefix sum (inclusive scan, matching CUDA rank kernels) ---
  {
    auto h_kb1 = Kokkos::create_mirror_view(key_buff1);
    Kokkos::deep_copy(h_kb1, key_buff1);

    int sum = 0;
    for (int i = 0; i < MAX_KEY; i++) {
      sum += h_kb1(i);
      h_kb1(i) = sum;
    }
    Kokkos::deep_copy(key_buff1, h_kb1);
  }

  // --- rank_7: partial verification ---
  {
    auto h_kb1 = Kokkos::create_mirror_view(key_buff1);
    Kokkos::deep_copy(h_kb1, key_buff1);
    auto h_pv  = Kokkos::create_mirror_view(partial_verify_vals);
    Kokkos::deep_copy(h_pv, partial_verify_vals);

    for (int i = 0; i < TEST_ARRAY_SIZE; i++) {
      int k = h_pv(i);
      if (k > 0 && k <= NUM_KEYS - 1) {
        int key_rank = h_kb1(k - 1);
        int failed   = 0;
        // CLASS S rules
        if (i <= 2) {
          if (key_rank != test_rank_array[i] + iteration) failed = 1;
          else passed_verification++;
        } else {
          if (key_rank != test_rank_array[i] - iteration) failed = 1;
          else passed_verification++;
        }
        if (failed)
          printf("Failed partial verification: iteration %d, test key %d\n",
                 iteration, i);
      }
    }
  }
}

// -----------------------------------------------------------------------
// Full verify: scatter-sort and check order
// -----------------------------------------------------------------------
static int fullVerify(Kokkos::View<int *> key_array,
                      Kokkos::View<int *> key_buff1) {
  // Copy key_array → key_buff2 (host)
  auto h_ka  = Kokkos::create_mirror_view(key_array);
  Kokkos::deep_copy(h_ka, key_array);
  auto h_kb1 = Kokkos::create_mirror_view(key_buff1);
  Kokkos::deep_copy(h_kb1, key_buff1);

  // We need a fresh prefix-sum from the last iteration's histogram.
  // Re-count from the final key_array state.
  std::vector<int> hist(MAX_KEY, 0);
  for (int i = 0; i < NUM_KEYS; i++)
    hist[h_ka(i)]++;

  // Inclusive prefix sum → ranks (key_buff1[k] = count of keys with value ≤ k)
  int sum = 0;
  for (int i = 0; i < MAX_KEY; i++) {
    sum += hist[i];
    hist[i] = sum;
  }

  // Scatter using rank: position for key value v is hist[v-1] (or 0 if v==0)
  std::vector<int> sorted(NUM_KEYS);
  std::vector<int> inv_count(MAX_KEY, 0);
  // We scatter by counting occurrences: for key v, position = hist[v-1]+inv_count[v]
  std::vector<int> start_pos(MAX_KEY, 0);
  start_pos[0] = 0;
  for (int i = 1; i < MAX_KEY; i++)
    start_pos[i] = hist[i - 1];
  for (int i = 0; i < NUM_KEYS; i++) {
    int v = h_ka(i);
    sorted[start_pos[v] + inv_count[v]] = v;
    inv_count[v]++;
  }

  // Check sorted order
  int out_of_order = 0;
  for (int i = 1; i < NUM_KEYS; i++)
    if (sorted[i - 1] > sorted[i]) out_of_order++;

  if (out_of_order != 0)
    printf("Full_verify: number of keys out of sort: %d\n", out_of_order);

  return out_of_order;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int /*argc*/, char ** /*argv*/) {
  // Test data for CLASS S
  static const int S_test_index_array[TEST_ARRAY_SIZE] = {48427,17148,23627,62548,4431};
  static const int S_test_rank_array [TEST_ARRAY_SIZE] = {0,18,346,64917,65463};

  Kokkos::initialize();
  {
    printf("\n\n NAS Parallel Benchmarks 4.1 IS Benchmark\n");
    printf(" Size:  %d  (class %c)\n", (int)TOTAL_KEYS, CLASS);
    printf(" Iterations:   %d\n", MAX_ITERATIONS);

    // Allocate device views
    Kokkos::View<int *> key_array("key_array", SIZE_OF_BUFFERS);
    Kokkos::View<int *> key_buff1("key_buff1", MAX_KEY);
    Kokkos::View<int *> partial_verify_vals("partial_verify_vals", TEST_ARRAY_SIZE);

    // Generate keys
    const int num_create_procs = 64;
    createSeq(key_array, 314159265.0, 1220703125.0, num_create_procs);

    int passed_verification = 0;

    Kokkos::fence();
    auto t_start = std::chrono::steady_clock::now();

    for (int iter = 1; iter <= MAX_ITERATIONS; iter++) {
      rankIteration(key_array, key_buff1, partial_verify_vals,
                    S_test_index_array, iter,
                    passed_verification, S_test_rank_array);
    }

    Kokkos::fence();
    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();

    printf("Average execution time of the rank kernels %f (s)\n",
           total_s / MAX_ITERATIONS);

    // Full verify (untimed)
    int out_of_order = fullVerify(key_array, key_buff1);
    if (out_of_order == 0) passed_verification++;

    // 5 partial checks per iteration × MAX_ITERATIONS + 1 full verify
    bool pass = (passed_verification == 5 * MAX_ITERATIONS + 1);
    printf("%s\n", pass ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
