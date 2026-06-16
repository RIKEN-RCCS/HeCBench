#include <cfloat>
#include <iostream>
#include <sstream>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Kokkos_Core.hpp>

#ifdef SINGLE_PRECISION
#define T float
#define EPISON 1e-4f
#else
#define T double
#define EPISON 1e-6
#endif

// Complex type usable in Kokkos lambdas
struct T2 {
  T x, y;
  KOKKOS_INLINE_FUNCTION T2() : x(0), y(0) {}
  KOKKOS_INLINE_FUNCTION T2(T x_, T y_) : x(x_), y(y_) {}
};

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

KOKKOS_INLINE_FUNCTION T2 get_exp_1_8()  { return T2( 1, -1); }
KOKKOS_INLINE_FUNCTION T2 get_exp_1_4()  { return T2( 0, -1); }
KOKKOS_INLINE_FUNCTION T2 get_exp_3_8()  { return T2(-1, -1); }
KOKKOS_INLINE_FUNCTION T2 get_iexp_1_8() { return T2( 1,  1); }
KOKKOS_INLINE_FUNCTION T2 get_iexp_1_4() { return T2( 0,  1); }
KOKKOS_INLINE_FUNCTION T2 get_iexp_3_8() { return T2(-1,  1); }

#ifdef SINGLE_PRECISION
KOKKOS_INLINE_FUNCTION T2 exp_i(T phi) {
  return T2(cosf(phi), sinf(phi));
}
#else
KOKKOS_INLINE_FUNCTION T2 exp_i(T phi) {
  return T2(cos(phi), sin(phi));
}
#endif

KOKKOS_INLINE_FUNCTION T2 cmplx_mul(T2 a, T2 b) {
  return T2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
KOKKOS_INLINE_FUNCTION T2 cm_fl_mul(T2 a, T b) {
  return T2(b*a.x, b*a.y);
}
KOKKOS_INLINE_FUNCTION T2 cmplx_add(T2 a, T2 b) {
  return T2(a.x + b.x, a.y + b.y);
}
KOKKOS_INLINE_FUNCTION T2 cmplx_sub(T2 a, T2 b) {
  return T2(a.x - b.x, a.y - b.y);
}

// FFT butterfly macros
#define FFT2(a0, a1)                        \
{                                           \
  T2 c0 = *(a0);                           \
  *(a0) = cmplx_add(c0, *(a1));            \
  *(a1) = cmplx_sub(c0, *(a1));            \
}

#define FFT4(a0, a1, a2, a3)               \
{                                          \
  FFT2((a0), (a2));                        \
  FFT2((a1), (a3));                        \
  *(a3) = cmplx_mul(*(a3), get_exp_1_4()); \
  FFT2((a0), (a1));                        \
  FFT2((a2), (a3));                        \
}

#define FFT8(a)                                                          \
{                                                                        \
  FFT2(&(a)[0], &(a)[4]);                                               \
  FFT2(&(a)[1], &(a)[5]);                                               \
  FFT2(&(a)[2], &(a)[6]);                                               \
  FFT2(&(a)[3], &(a)[7]);                                               \
  (a)[5] = cm_fl_mul(cmplx_mul((a)[5], get_exp_1_8()), (T)M_SQRT1_2);  \
  (a)[6] = cmplx_mul((a)[6], get_exp_1_4());                            \
  (a)[7] = cm_fl_mul(cmplx_mul((a)[7], get_exp_3_8()), (T)M_SQRT1_2);  \
  FFT4(&(a)[0], &(a)[1], &(a)[2], &(a)[3]);                            \
  FFT4(&(a)[4], &(a)[5], &(a)[6], &(a)[7]);                            \
}

#define IFFT2 FFT2

#define IFFT4(a0, a1, a2, a3)               \
{                                           \
  IFFT2((a0), (a2));                        \
  IFFT2((a1), (a3));                        \
  *(a3) = cmplx_mul(*(a3), get_iexp_1_4()); \
  IFFT2((a0), (a1));                        \
  IFFT2((a2), (a3));                        \
}

#define IFFT8(a)                                                          \
{                                                                         \
  IFFT2(&(a)[0], &(a)[4]);                                               \
  IFFT2(&(a)[1], &(a)[5]);                                               \
  IFFT2(&(a)[2], &(a)[6]);                                               \
  IFFT2(&(a)[3], &(a)[7]);                                               \
  (a)[5] = cm_fl_mul(cmplx_mul((a)[5], get_iexp_1_8()), (T)M_SQRT1_2); \
  (a)[6] = cmplx_mul((a)[6], get_iexp_1_4());                           \
  (a)[7] = cm_fl_mul(cmplx_mul((a)[7], get_iexp_3_8()), (T)M_SQRT1_2); \
  IFFT4(&(a)[0], &(a)[1], &(a)[2], &(a)[3]);                           \
  IFFT4(&(a)[4], &(a)[5], &(a)[6], &(a)[7]);                           \
}

// Run one pass (forward or inverse) of the 512-point FFT on device
// sign: -1 for forward, +1 for inverse
template <class MemberType>
KOKKOS_INLINE_FUNCTION
void run_fft_pass(
    const MemberType &team,
    Kokkos::View<T2 *> source,
    T *smem,  // scratch: 8*8*9 elements
    int sign)
{
  int teamId = team.league_rank();
  int tid = team.team_rank();
  int blockIdx = teamId * 512 + tid;
  int hi = tid >> 3;
  int lo = tid & 7;
  T2 data[8];
  const int reversed[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  // Load global data
  for (int i = 0; i < 8; i++) data[i] = source(blockIdx + i * 64);

  if (sign < 0) {
    FFT8(data);
    // Twiddle (forward, size 512)
    for (int j = 1; j < 8; j++)
      data[j] = cmplx_mul(data[j],
                  exp_i(((T)-2 * (T)M_PI * reversed[j] / (T)512) * tid));
  } else {
    IFFT8(data);
    // Twiddle (inverse, size 512)
    for (int j = 1; j < 8; j++)
      data[j] = cmplx_mul(data[j],
                  exp_i(((T)2 * (T)M_PI * reversed[j] / (T)512) * tid));
  }

  // First transpose: stride 66
  for (int i = 0; i < 8; i++) smem[hi * 8 + lo + i * 66] = data[reversed[i]].x;
  team.team_barrier();
  for (int i = 0; i < 8; i++) data[i].x = smem[lo * 66 + hi + i * 8];
  team.team_barrier();
  for (int i = 0; i < 8; i++) smem[hi * 8 + lo + i * 66] = data[reversed[i]].y;
  team.team_barrier();
  for (int i = 0; i < 8; i++) data[i].y = smem[lo * 66 + hi + i * 8];
  team.team_barrier();

  if (sign < 0) {
    FFT8(data);
    // Twiddle (forward, size 64)
    for (int j = 1; j < 8; j++)
      data[j] = cmplx_mul(data[j],
                  exp_i(((T)-2 * (T)M_PI * reversed[j] / (T)64) * hi));
  } else {
    IFFT8(data);
    // Twiddle (inverse, size 64)
    for (int j = 1; j < 8; j++)
      data[j] = cmplx_mul(data[j],
                  exp_i(((T)2 * (T)M_PI * reversed[j] / (T)64) * hi));
  }

  // Second transpose: stride 72 (8*9)
  for (int i = 0; i < 8; i++) smem[hi * 8 + lo + i * 72] = data[reversed[i]].x;
  team.team_barrier();
  for (int i = 0; i < 8; i++) data[i].x = smem[hi * 72 + lo + i * 8];
  team.team_barrier();
  for (int i = 0; i < 8; i++) smem[hi * 8 + lo + i * 72] = data[reversed[i]].y;
  team.team_barrier();
  for (int i = 0; i < 8; i++) data[i].y = smem[hi * 72 + lo + i * 8];

  if (sign < 0) {
    FFT8(data);
  } else {
    IFFT8(data);
    // Normalize by 1/512
    for (int i = 0; i < 8; i++) {
      data[i].x = data[i].x / (T)512;
      data[i].y = data[i].y / (T)512;
    }
  }

  // Store global data (bit-reversed)
  for (int i = 0; i < 8; i++)
    source(blockIdx + i * 64) = data[reversed[i]];
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <problem size> <number of passes>\n", argv[0]);
    printf("Problem size [0-3]: 0=1M, 1=8M, 2=96M, 3=256M\n");
    return 1;
  }

  srand(2);

  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  int probSizes[4] = {1, 8, 96, 256};
  unsigned long bytes = probSizes[select];
  bytes *= 1024 * 1024;

  int half_n_ffts = bytes / (512 * sizeof(T2) * 2);
  const int n_ffts = half_n_ffts * 2;
  const int half_n_cmplx = half_n_ffts * 512;
  unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
  const int N = half_n_cmplx * 2;

  fprintf(stdout, "used_bytes=%lu, N=%d\n", used_bytes, N);

  T2 *source_h = (T2 *)malloc(used_bytes);
  T2 *reference = (T2 *)malloc(used_bytes);

  for (int i = 0; i < half_n_cmplx; i++) {
    source_h[i].x = (rand() / (float)RAND_MAX) * 2 - 1;
    source_h[i].y = (rand() / (float)RAND_MAX) * 2 - 1;
    source_h[i + half_n_cmplx].x = source_h[i].x;
    source_h[i + half_n_cmplx].y = source_h[i].y;
  }
  memcpy(reference, source_h, used_bytes);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<T2 *> d_source("source", N);
    auto h_source = Kokkos::create_mirror_view(d_source);
    for (int i = 0; i < N; i++) h_source(i) = source_h[i];
    Kokkos::deep_copy(d_source, h_source);

    using team_policy_t = Kokkos::TeamPolicy<>;
    using member_type   = team_policy_t::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchView   = Kokkos::View<T *, scratch_space, Kokkos::MemoryUnmanaged>;

    // smem per team = 8*8*9 T values (covers both stride-66 and stride-72 layouts)
    const int smem_size = 8 * 8 * 9 * sizeof(T);

    team_policy_t policy(n_ffts, 64);
    policy = policy.set_scratch_size(0, Kokkos::PerTeam(smem_size));

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < passes; k++) {
      // Forward FFT pass
      Kokkos::parallel_for(
          "fft_forward", policy,
          KOKKOS_LAMBDA(const member_type &team) {
            ScratchView smem(team.team_scratch(0), 8 * 8 * 9);
            run_fft_pass(team, d_source, static_cast<T *>(smem.data()), -1);
          });
      Kokkos::fence();

      // Inverse FFT pass
      Kokkos::parallel_for(
          "fft_inverse", policy,
          KOKKOS_LAMBDA(const member_type &team) {
            ScratchView smem(team.team_scratch(0), 8 * 8 * 9);
            run_fft_pass(team, d_source, static_cast<T *>(smem.data()), +1);
          });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / passes << " (s)\n";

    Kokkos::deep_copy(h_source, d_source);
    for (int i = 0; i < N; i++) source_h[i] = h_source(i);
  }
  Kokkos::finalize();

  // Verification
  bool error = false;
  for (int i = 0; i < N; i++) {
    if (fabs((T)source_h[i].x - (T)reference[i].x) > EPISON) { error = true; break; }
    if (fabs((T)source_h[i].y - (T)reference[i].y) > EPISON) { error = true; break; }
  }
  std::cout << (error ? "FAIL" : "PASS") << std::endl;

  free(reference);
  free(source_h);
  return 0;
}
