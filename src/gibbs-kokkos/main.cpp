/*---------------------------------------------------------------
  Original author: Zebulun Arendsee, March 26, 2013
  Ported to Kokkos
----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define PI 3.14159265359f

// ── Host-side random helpers ──────────────────────────────────

static float rnorm()
{
  float U1 = rand() / float(RAND_MAX);
  float U2 = rand() / float(RAND_MAX);
  return sqrtf(-2.f * logf(U1)) * cosf(2.f * PI * U2);
}

static float rgamma_host(float a, float b)
{
  if (a > 1.f) {
    float d = a - 1.f / 3.f;
    float c = 1.f / sqrtf(9.f * d);
    for (;;) {
      float Z = rnorm();
      if (Z > -1.f / c) {
        float V = powf(1.f + c * Z, 3.f);
        float U = rand() / (float)RAND_MAX;
        if (logf(U) <= 0.5f * Z * Z + d - d * V + d * logf(V))
          return d * V / b;
      }
    }
  } else {
    float x = rgamma_host(a + 1.f, b);
    return x * powf(rand() / (float)RAND_MAX, 1.f / a);
  }
}

static float sample_a(float a, float b, int N, float log_sum)
{
  static float sigma = 2.f;
  float proposal = rnorm() * sigma + a;
  if (proposal <= 0.f) return a;
  float log_ar = (proposal - a) * log_sum
               + N * (proposal - a) * logf(b)
               - N * (lgammaf(proposal) - lgammaf(a));
  float U = rand() / float(RAND_MAX);
  if (logf(U) < log_ar) { sigma *= 1.1f; return proposal; }
  else                   { sigma /= 1.1f; return a; }
}

static float sample_b(float a, int N, float flat_sum)
{
  return rgamma_host(N * a + 1.f, flat_sum);
}

// ── Device-side LCG helpers (KOKKOS_INLINE_FUNCTION) ─────────

KOKKOS_INLINE_FUNCTION float lcg_rand(unsigned int &s)
{
  s = 1664525u * s + 1013904223u;
  return (float)(s >> 8) * (1.0f / (float)(1u << 24));
}

KOKKOS_INLINE_FUNCTION float lcg_normal(unsigned int &s)
{
  float u1, u2;
  do { u1 = lcg_rand(s); } while (u1 == 0.f);
  u2 = lcg_rand(s);
  return sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265359f * u2);
}

// Marsaglia algorithm for gamma (device)
KOKKOS_INLINE_FUNCTION float rgamma_device(unsigned int &s, float a, float b)
{
  float d = a - 1.f / 3.f;
  float c = 1.f / sqrtf(9.f * d);
  for (;;) {
    float Z = lcg_normal(s);
    if (Z > -1.f / c) {
      float V = powf(1.f + c * Z, 3.f);
      float U = lcg_rand(s);
      if (logf(U) <= 0.5f * Z * Z + d - d * V + d * logf(V))
        return d * V / b;
    }
  }
}

// ─────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
  const int seed = 123;
  srand(seed);

  int trials = 1000;
  if (argc > 2) trials = atoi(argv[2]);

  if (argc < 2) { printf("Please provide input filename\n"); return 1; }
  FILE *fp = fopen(argv[1], "r");
  if (!fp) { printf("Cannot read file\n"); return 1; }

  int N = 0;
  char line[128];
  while (fgets(line, sizeof(line), fp)) N++;
  rewind(fp);

  int   *h_y = (int *)  malloc(sizeof(int)   * N);
  float *h_n = (float *)malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++) fscanf(fp, "%d %f", &h_y[i], &h_n[i]);
  fclose(fp);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int *>          d_y     ("y",      N);
    Kokkos::View<float *>        d_n     ("n",      N);
    Kokkos::View<float *>        d_theta ("theta",  N);
    Kokkos::View<unsigned int *> d_states("states", N);

    // Initialise device data
    {
      auto m_y = Kokkos::create_mirror_view(d_y);
      auto m_n = Kokkos::create_mirror_view(d_n);
      auto m_s = Kokkos::create_mirror_view(d_states);
      for (int i = 0; i < N; i++) {
        m_y(i) = h_y[i];
        m_n(i) = h_n[i];
        // Give each thread a distinct seed derived from the global seed
        m_s(i) = (unsigned int)(seed * 1664525u + (unsigned int)i * 22695477u + 1u);
      }
      Kokkos::deep_copy(d_y, m_y);
      Kokkos::deep_copy(d_n, m_n);
      Kokkos::deep_copy(d_states, m_s);
    }

    float  a = 20.f, b = 1.f;
    double mean_a = 0.0, mean_b = 0.0, total_time = 0.0;

    for (int iter = 0; iter < trials; iter++) {
      auto t0 = std::chrono::steady_clock::now();

      // ── sample_theta ──
      const float fa = a, fb = b;
      Kokkos::parallel_for("sample_theta", N, KOKKOS_LAMBDA(int id) {
        unsigned int s = d_states(id);
        const float hyperA = fa + (float)d_y(id);
        const float hyperB = fb + d_n(id);
        if (hyperA < 1.f)
          d_theta(id) = rgamma_device(s, hyperA + 1.f, hyperB)
                        * powf(lcg_rand(s), 1.f / hyperA);
        else
          d_theta(id) = rgamma_device(s, hyperA, hyperB);
        d_states(id) = s;
      });

      // ── reductions: flat sum and log sum of theta ──
      float flat_sum = 0.f, log_sum = 0.f;
      Kokkos::parallel_reduce("flat_sum", N,
        KOKKOS_LAMBDA(int id, float &acc) { acc += d_theta(id); },
        flat_sum);
      Kokkos::parallel_reduce("log_sum", N,
        KOKKOS_LAMBDA(int id, float &acc) { acc += logf(d_theta(id)); },
        log_sum);

      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();
      total_time += (double)std::chrono::duration_cast<
                      std::chrono::nanoseconds>(t1 - t0).count();

      a = sample_a(a, b, N, log_sum);
      mean_a += a;
      b = sample_b(a, N, flat_sum);
      mean_b += b;
    }

    printf("Average execution time of kernels: %f (us)\n",
           (total_time * 1e-3) / trials);
    printf("a = %lf (avg), b = %lf (avg)\n",
           mean_a / trials, mean_b / trials);
  }
  Kokkos::finalize();

  free(h_y);
  free(h_n);
  return 0;
}
