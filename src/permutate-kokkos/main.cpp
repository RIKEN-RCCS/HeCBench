/*
 * Kokkos port of the GPU-based parallel IID test (NIST SP 800-90B).
 * Original: gpu_permutation_testing.cpp / kernel_functions.hpp / device_functions.hpp
 */

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdio>

// ============================================================
// LCG random (device-safe)
// ============================================================
KOKKOS_INLINE_FUNCTION
uint32_t LCG_random(uint64_t *seed)
{
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (uint32_t)(*seed);
}

// ============================================================
// Device helper functions (all KOKKOS_INLINE_FUNCTION)
// ============================================================

KOKKOS_INLINE_FUNCTION
void dev_test1(double *out_max, const double mean, const uint8_t *data,
               const uint32_t len, const uint32_t N, uint32_t tid)
{
  double max = 0, temp = 0, sum = 0;
  for (uint64_t i = 0; i < len; i++) {
    sum += data[i * N + tid];
    temp = fabs(sum - ((i + 1) * mean));
    if (max < temp) max = temp;
  }
  *out_max = max;
}

KOKKOS_INLINE_FUNCTION
void dev_test2_6(double *out1, double *out2, double *out3, double *out4, double *out5,
                 const double median, const uint8_t *data,
                 const uint32_t len, const uint32_t N, uint32_t tid)
{
  *out1 = 1; *out2 = 0; *out3 = 0; *out4 = 1; *out5 = 0;
  uint32_t run1 = 1, run2 = 1, pos = 0;
  bool f1 = 0, f2 = 0, fm1 = 0, fm2 = 0;

  if (data[tid] <= data[N + tid])           f1 = 1;
  if (data[tid] >= median)                  fm1 = 1;

  for (uint64_t i = 1; i < (len - 1); i++) {
    pos += f1;
    f2 = 0; fm2 = 0;
    if (data[i * N + tid] <= data[(i + 1) * N + tid])  f2 = 1;
    if (data[i * N + tid] >= median)                    fm2 = 1;

    if (f1 == f2) { run1++; }
    else {
      *out1 += 1;
      if (run1 > *out2) *out2 = run1;
      run1 = 1;
    }
    if (fm1 == fm2) { run2++; }
    else {
      *out4 += 1;
      if (run2 > *out5) *out5 = run2;
      run2 = 1;
    }
    f1 = f2; fm1 = fm2;
  }
  pos += f1;

  fm2 = (data[(len - 1) * N + tid] >= median) ? 1 : 0;
  if (fm1 == fm2) { run2++; }
  else {
    *out4 += 1;
    if (run2 > *out5) *out5 = run2;
  }

  *out3 = (pos > (len - pos)) ? (double)pos : (double)(len - pos);
}

KOKKOS_INLINE_FUNCTION
void dev_test7_8(double *out_average, double *out_max, const uint8_t *data,
                 const uint32_t size, const uint32_t len, const uint32_t N, uint32_t tid)
{
  uint64_t i = 0, j = 0, k = 0;
  bool dups[256];
  uint32_t cnt = 0, max = 0;
  double avg = 0;

  while (i + j < len) {
    for (k = 0; k < (uint32_t)(1 << size); k++) dups[k] = false;
    while (i + j < len) {
      uint8_t v = data[(i + j) * N + tid];
      if (dups[v]) {
        avg += j;
        if (j > max) max = j;
        cnt++;
        i += j; j = 0;
        break;
      } else {
        dups[v] = true;
        ++j;
      }
    }
    ++i;
  }
  *out_average = (cnt > 0) ? avg / (double)cnt : 0.0;
  *out_max     = (double)max;
}

KOKKOS_INLINE_FUNCTION
void dev_test9_and_14(double *out_num, double *out_strength, const uint8_t *data,
                      const uint32_t len, const uint32_t N, uint32_t tid, const uint32_t lag)
{
  double temp1 = 0, temp2 = 0;
  for (uint64_t i = 0; i < len - lag; i++) {
    if (data[i * N + tid] == data[(i + lag) * N + tid]) temp1++;
    temp2 += (data[i * N + tid] * data[(i + lag) * N + tid]);
  }
  *out_num = temp1;
  *out_strength = temp2;
}

KOKKOS_INLINE_FUNCTION
uint8_t dev_hammingweight(uint8_t d)
{
  uint8_t tmp = 0;
  for (int b = 0; b < 8; b++) tmp += (d >> b) & 0x1;
  return tmp;
}

KOKKOS_INLINE_FUNCTION
void dev_binary_test2_4(double *out_num, double *out_len, double *out_max,
                        const uint8_t *data, const uint32_t len, const uint32_t N, uint32_t tid)
{
  uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0, pos = 0;
  bool bflag1 = 0, bflag2 = 0;

  if (dev_hammingweight(data[tid]) <= dev_hammingweight(data[N + tid])) bflag1 = 1;

  for (uint32_t i = 1; i < len - 1; i++) {
    pos += bflag1;
    bflag2 = (dev_hammingweight(data[i * N + tid]) <= dev_hammingweight(data[(i + 1) * N + tid])) ? 1 : 0;
    if (bflag1 == bflag2) { len_runs++; }
    else {
      num_runs++;
      if (len_runs > max_len_runs) max_len_runs = len_runs;
      len_runs = 1;
    }
    bflag1 = bflag2;
  }
  pos += bflag1;
  *out_num = (double)num_runs;
  *out_len = (double)max_len_runs;
  *out_max = (double)(pos > (uint32_t)(len - pos) ? pos : (uint32_t)(len - pos));
}

KOKKOS_INLINE_FUNCTION
void dev_binary_test9_and_14(double *out_num, double *out_strength, const uint8_t *data,
                              const uint32_t len, const uint32_t N, uint32_t tid, const uint32_t lag)
{
  double temp1 = 0, temp2 = 0;
  for (uint32_t i = 0; i < len - lag; i++) {
    uint8_t hw1 = dev_hammingweight(data[i * N + tid]);
    uint8_t hw2 = dev_hammingweight(data[(i + lag) * N + tid]);
    if (hw1 == hw2) temp1++;
    temp2 += hw1 * hw2;
  }
  *out_num = temp1;
  *out_strength = temp2;
}

KOKKOS_INLINE_FUNCTION
void dev_test5_6(double *out_num, double *out_len, const double median,
                 const uint8_t *data, const uint32_t len, const uint32_t N, uint32_t tid)
{
  uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0;
  bool bflag1 = (data[tid] >= median) ? 1 : 0, bflag2 = 0;

  for (uint32_t i = 1; i < len; i++) {
    bflag2 = (data[i * N + tid] >= median) ? 1 : 0;
    if (bflag1 == bflag2) { len_runs++; }
    else {
      num_runs++;
      if (len_runs > max_len_runs) max_len_runs = len_runs;
      len_runs = 1;
    }
    bflag1 = bflag2;
  }
  *out_num = (double)num_runs;
  *out_len = (double)max_len_runs;
}

// ============================================================
// Kokkos-based kernel wrappers
// ============================================================

using ViewU8    = Kokkos::View<uint8_t*>;
using ViewU32   = Kokkos::View<uint32_t*>;
using ViewD     = Kokkos::View<double*>;
using ViewU8_hm = Kokkos::View<uint8_t*>::HostMirror;

// Increment helper to avoid verbosity with atomic_fetch_add
#define ATOMIC_INC(arr, idx) Kokkos::atomic_fetch_add(&(arr)(idx), 1u)

// Compare double result against reference and update 3 consecutive counters
#define CMP_AND_COUNT(res, ref, base)                         \
  if ((res) > (ref))       { ATOMIC_INC(d_counts, (base));   } \
  else if ((res) == (ref)) { ATOMIC_INC(d_counts, (base)+1); } \
  else                     { ATOMIC_INC(d_counts, (base)+2); }

// Float-cast version used for test1
#define CMP_AND_COUNT_F(res, ref, base)                                          \
  if ((float)(res) > (float)(ref))       { ATOMIC_INC(d_counts, (base));   } \
  else if ((float)(res) == (float)(ref)) { ATOMIC_INC(d_counts, (base)+1); } \
  else                                   { ATOMIC_INC(d_counts, (base)+2); }

// ---- shuffling_kernel ----
void shuffling_kernel(ViewU8 d_Ndata, ViewU8 d_data,
                      uint32_t len, uint32_t N,
                      uint32_t num_block, uint32_t num_thread)
{
  int64_t size = (int64_t)num_block * num_thread;
  Kokkos::parallel_for("shuffle",
    Kokkos::RangePolicy<int64_t>(0, size),
    KOKKOS_LAMBDA(int64_t tid) {
      uint64_t seed = 0;
      for (uint64_t i = 0; i < len; i++) {
        uint64_t idx = i * N + tid;
        d_Ndata[idx] = d_data[i];
        seed += d_Ndata[idx];
      }
      seed = seed ^ (uint64_t)tid;

      uint64_t j = len - 1;
      while (j > 0) {
        uint64_t random = LCG_random(&seed) % j;
        uint64_t idx  = random * N + tid;
        uint64_t idx2 = j * N + tid;
        uint8_t tmp = d_Ndata[idx];
        d_Ndata[idx]  = d_Ndata[idx2];
        d_Ndata[idx2] = tmp;
        j--;
      }
    });
}

// ---- statistical_tests_kernel (non-binary) ----
// OMP used 2*num_block teams: group 0 → test7_8, group 1 → test1,test2_6,test9_14
void statistical_tests_kernel(
  ViewU32 d_counts, ViewD d_results,
  double mean, double median,
  ViewU8 d_Ndata,
  uint32_t size, uint32_t len, uint32_t N,
  uint32_t num_block, uint32_t num_thread)
{
  int64_t work_size = (int64_t)num_block * num_thread;

  // Group 0: test7_8
  Kokkos::parallel_for("stat_test78",
    Kokkos::RangePolicy<int64_t>(0, work_size),
    KOKKOS_LAMBDA(int64_t tid) {
      double r1 = 0, r2 = 0;
      dev_test7_8(&r1, &r2, d_Ndata.data(), size, len, N, (uint32_t)tid);
      CMP_AND_COUNT(r1, d_results[6],  18);
      CMP_AND_COUNT(r2, d_results[7],  21);
    });

  // Group 1: tests 1, 2-6, 9&14 (×5 lags)
  Kokkos::parallel_for("stat_rest",
    Kokkos::RangePolicy<int64_t>(0, work_size),
    KOKKOS_LAMBDA(int64_t tid) {
      double r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0;
      const uint8_t *Nd = d_Ndata.data();

      dev_test1(&r1, mean, Nd, len, N, (uint32_t)tid);
      CMP_AND_COUNT_F(r1, d_results[0], 0);

      dev_test2_6(&r1, &r2, &r3, &r4, &r5, median, Nd, len, N, (uint32_t)tid);
      CMP_AND_COUNT(r1, d_results[1],  3);
      CMP_AND_COUNT(r2, d_results[2],  6);
      CMP_AND_COUNT(r3, d_results[3],  9);
      CMP_AND_COUNT(r4, d_results[4], 12);
      CMP_AND_COUNT(r5, d_results[5], 15);

      const uint32_t lags[5]       = {1, 2, 8, 16, 32};
      const int      test9_base[5] = {24, 27, 30, 33, 36};
      const int      test14_base[5]= {39, 42, 45, 48, 51};
      for (int k = 0; k < 5; k++) {
        dev_test9_and_14(&r1, &r2, Nd, len, N, (uint32_t)tid, lags[k]);
        CMP_AND_COUNT(r1, d_results[8  + k], test9_base[k]);
        CMP_AND_COUNT(r2, d_results[13 + k], test14_base[k]);
      }
    });
}

// ---- binary_shuffling_kernel ----
void binary_shuffling_kernel(ViewU8 d_Ndata, ViewU8 d_bNdata, ViewU8 d_data,
                              uint32_t len, uint32_t blen, uint32_t N,
                              int num_block, int num_thread)
{
  int64_t size = (int64_t)num_block * num_thread;
  Kokkos::parallel_for("binary_shuffle",
    Kokkos::RangePolicy<int64_t>(0, size),
    KOKKOS_LAMBDA(int64_t tid) {
      uint64_t seed = 0;
      for (uint32_t i = 0; i < len; i++) {
        d_Ndata[i * N + tid] = d_data[i];
        seed += d_data[i];
      }
      seed = seed ^ (uint64_t)tid;

      uint32_t j = len - 1;
      while (j > 0) {
        uint32_t random = LCG_random(&seed) % j;
        uint8_t tmp            = d_Ndata[random * N + tid];
        d_Ndata[random * N + tid] = d_Ndata[j * N + tid];
        d_Ndata[j * N + tid]   = tmp;
        j--;
      }

      for (uint32_t i = 0; i < blen; i++) {
        uint8_t tmp = (d_Ndata[8 * i * N + tid] & 0x1) << 7;
        tmp ^= (d_Ndata[(8 * i + 1) * N + tid] & 0x1) << 6;
        tmp ^= (d_Ndata[(8 * i + 2) * N + tid] & 0x1) << 5;
        tmp ^= (d_Ndata[(8 * i + 3) * N + tid] & 0x1) << 4;
        tmp ^= (d_Ndata[(8 * i + 4) * N + tid] & 0x1) << 3;
        tmp ^= (d_Ndata[(8 * i + 5) * N + tid] & 0x1) << 2;
        tmp ^= (d_Ndata[(8 * i + 6) * N + tid] & 0x1) << 1;
        tmp ^= (d_Ndata[(8 * i + 7) * N + tid] & 0x1);
        d_bNdata[i * N + tid] = tmp;
      }
    });
}

// ---- binary_statistical_tests_kernel ----
// OMP used 4*num_block teams split into 4 groups → 4 separate kernels
void binary_statistical_tests_kernel(
  ViewU32 d_counts, ViewD d_results,
  double mean, double median,
  ViewU8 d_Ndata, ViewU8 d_bNdata,
  uint32_t size_bits, uint32_t len, uint32_t blen, uint32_t N,
  uint32_t num_block, uint32_t num_thread)
{
  int64_t work_size = (int64_t)num_block * num_thread;

  // Group 0: test1 on Ndata
  Kokkos::parallel_for("bin_test1",
    Kokkos::RangePolicy<int64_t>(0, work_size),
    KOKKOS_LAMBDA(int64_t tid) {
      double r1 = 0;
      dev_test1(&r1, mean, d_Ndata.data(), len, N, (uint32_t)tid);
      CMP_AND_COUNT_F(r1, d_results[0], 0);
    });

  // Group 1: test5_6 on Ndata; binary_test2_4 on bNdata
  Kokkos::parallel_for("bin_test5_6_2_4",
    Kokkos::RangePolicy<int64_t>(0, work_size),
    KOKKOS_LAMBDA(int64_t tid) {
      double r1 = 0, r2 = 0, r3 = 0;
      dev_test5_6(&r1, &r2, median, d_Ndata.data(), len, N, (uint32_t)tid);
      CMP_AND_COUNT(r1, d_results[4], 12);
      CMP_AND_COUNT(r2, d_results[5], 15);

      dev_binary_test2_4(&r1, &r2, &r3, d_bNdata.data(), blen, N, (uint32_t)tid);
      CMP_AND_COUNT(r1, d_results[1],  3);
      CMP_AND_COUNT(r2, d_results[2],  6);
      CMP_AND_COUNT(r3, d_results[3],  9);
    });

  // Group 2: test7_8 on bNdata (size=8 bits for binary)
  Kokkos::parallel_for("bin_test7_8",
    Kokkos::RangePolicy<int64_t>(0, work_size),
    KOKKOS_LAMBDA(int64_t tid) {
      double r1 = 0, r2 = 0;
      dev_test7_8(&r1, &r2, d_bNdata.data(), 8, blen, N, (uint32_t)tid);
      CMP_AND_COUNT(r1, d_results[6], 18);
      CMP_AND_COUNT(r2, d_results[7], 21);
    });

  // Group 3: binary_test9_and_14 (×5 lags)
  Kokkos::parallel_for("bin_test9_14",
    Kokkos::RangePolicy<int64_t>(0, work_size),
    KOKKOS_LAMBDA(int64_t tid) {
      double r1 = 0, r2 = 0;
      const uint32_t lags[5]    = {1, 2, 8, 16, 32};
      const int      t9_base[5] = {24, 27, 30, 33, 36};
      const int     t14_base[5] = {39, 42, 45, 48, 51};
      for (int k = 0; k < 5; k++) {
        dev_binary_test9_and_14(&r1, &r2, d_bNdata.data(), blen, N, (uint32_t)tid, lags[k]);
        CMP_AND_COUNT(r1, d_results[8  + k], t9_base[k]);
        CMP_AND_COUNT(r2, d_results[13 + k], t14_base[k]);
      }
    });
}

// ============================================================
// Top-level permutation testing function
// ============================================================
bool gpu_permutation_testing(double *gpu_runtime,
                              uint32_t *counts, double *results,
                              double mean, double median,
                              uint8_t *data,
                              uint32_t size, uint32_t len,
                              uint32_t N, uint32_t num_block, uint32_t num_thread)
{
  uint32_t loop = (10000 + N - 1) / N;
  uint32_t blen = 0;
  if (size == 1) {
    blen = (len + 7) / 8;
  }
  size_t Nlen  = (size_t)N * len;
  size_t Nblen = (size_t)N * blen;

  // Device views
  ViewU8  d_data   ("data",    len);
  ViewU32 d_counts ("counts",  54);
  ViewD   d_results("results", 18);
  ViewU8  d_Ndata  ("Ndata",   Nlen);
  ViewU8  d_bNdata ("bNdata",  Nblen > 0 ? Nblen : 1);

  // Host mirrors
  auto h_data    = Kokkos::create_mirror_view(d_data);
  auto h_counts  = Kokkos::create_mirror_view(d_counts);
  auto h_results = Kokkos::create_mirror_view(d_results);

  // Copy inputs to device
  for (uint32_t i = 0; i < len; i++)  h_data[i]    = data[i];
  for (int i = 0; i < 54; i++)        h_counts[i]  = counts[i];
  for (int i = 0; i < 18; i++)        h_results[i] = results[i];

  Kokkos::deep_copy(d_data,    h_data);
  Kokkos::deep_copy(d_counts,  h_counts);
  Kokkos::deep_copy(d_results, h_results);

  uint8_t num_runtest = 0;

  auto start = std::chrono::steady_clock::now();

  for (uint32_t i = 0; i < loop; i++) {
    if (size == 1) {
      binary_shuffling_kernel(d_Ndata, d_bNdata, d_data, len, blen, N, num_block, num_thread);
      binary_statistical_tests_kernel(d_counts, d_results, mean, median,
                                       d_Ndata, d_bNdata, size, len, blen, N, num_block, num_thread);
    } else {
      shuffling_kernel(d_Ndata, d_data, len, N, num_block, num_thread);
      statistical_tests_kernel(d_counts, d_results, mean, median,
                                d_Ndata, size, len, N, num_block, num_thread);
    }
    Kokkos::fence();

    Kokkos::deep_copy(h_counts, d_counts);

    num_runtest = 0;
    for (int t = 0; t < 18; t++) {
      if (((h_counts[3 * t] + h_counts[3 * t + 1]) > 5) &&
          ((h_counts[3 * t + 1] + h_counts[3 * t + 2]) > 5))
        num_runtest++;
    }
    if (num_runtest == 18) break;
  }

  auto end = std::chrono::steady_clock::now();
  *gpu_runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;

  // Copy final counts back to caller
  for (int i = 0; i < 54; i++) counts[i] = h_counts[i];

  return (num_runtest == 18);
}

// ============================================================
// main
// ============================================================
int main(int argc, char *argv[])
{
  uint32_t len        = 1000;
  uint32_t N          = 100;
  uint32_t num_block  = 4;
  uint32_t num_thread = 64;

  if (argc >= 2) len        = atoi(argv[1]);
  if (argc >= 3) N          = atoi(argv[2]);
  if (argc >= 4) num_block  = atoi(argv[3]);
  if (argc >= 5) num_thread = atoi(argv[4]);

  Kokkos::initialize(argc, argv);
  {
    uint8_t *data = (uint8_t *)malloc(len);
    srand(42);
    for (uint32_t i = 0; i < len; i++) data[i] = (uint8_t)(rand() % 256);

    // Compute mean and median
    double mean = 0.0;
    for (uint32_t i = 0; i < len; i++) mean += data[i];
    mean /= len;

    // Simple median approximation (sort a copy)
    uint8_t *tmp = (uint8_t *)malloc(len);
    memcpy(tmp, data, len);
    for (uint32_t i = 0; i < len - 1; i++)
      for (uint32_t j = 0; j < len - i - 1; j++)
        if (tmp[j] > tmp[j + 1]) { uint8_t t = tmp[j]; tmp[j] = tmp[j+1]; tmp[j+1] = t; }
    double median = tmp[len / 2];
    free(tmp);

    double results[18] = {0};
    uint32_t counts[54] = {0};

    printf("permutate: len=%u N=%u num_block=%u num_thread=%u\n",
           len, N, num_block, num_thread);

    double gpu_runtime = 0.0;
    bool iid = gpu_permutation_testing(&gpu_runtime, counts, results,
                                        mean, median, data,
                                        /*size=*/2, len, N, num_block, num_thread);

    printf("IID result: %s\n", iid ? "IID" : "Non-IID");
    printf("GPU runtime: %.6f s\n", gpu_runtime);

    free(data);
  }
  Kokkos::finalize();
  return 0;
}
