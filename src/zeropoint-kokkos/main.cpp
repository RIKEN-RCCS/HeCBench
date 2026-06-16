/*
 * Zero-point quantization kernel.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

void zero_point(
    Kokkos::View<const float*>   x_min,
    Kokkos::View<const float*>   x_max,
    int32_t qmin, int32_t qmax,
    int size,
    bool preserve_sparsity,
    Kokkos::View<float*>   scale,
    Kokkos::View<int32_t*> zero_pt)
{
  Kokkos::parallel_for("zero_point", size, KOKKOS_LAMBDA(int i) {
    float min_val = x_min(i);
    float max_val = x_max(i);

    if (min_val < 0.f && max_val > 0.f && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;
      double max_scale = Kokkos::max(
          Kokkos::fabs((double)(min_val / symmetric_qmin)),
          Kokkos::fabs((double)(max_val / symmetric_qmax)));
      min_val = (float)(max_scale * symmetric_qmin);
      max_val = (float)(max_scale * symmetric_qmax);
    }

    min_val = Kokkos::min(min_val, 0.f);
    max_val = Kokkos::max(max_val, 0.f);
    float s = (float)((static_cast<double>(max_val) - min_val) / (qmax - qmin));

    if (s == 0.f || Kokkos::isinf(1.f / s)) s = 0.1f;
    scale(i) = s;

    double zp_from_min = qmin - min_val / static_cast<double>(s);
    double zp_from_max = qmax - max_val / static_cast<double>(s);
    double err_min = Kokkos::fabs((double)qmin) + Kokkos::fabs(min_val / static_cast<double>(s));
    double err_max = Kokkos::fabs((double)qmax) + Kokkos::fabs(max_val / static_cast<double>(s));
    double initial_zp = (err_min < err_max) ? zp_from_min : zp_from_max;

    if (min_val < 0.f && max_val > 0.f && preserve_sparsity)
      initial_zp = static_cast<double>(qmin + qmax) / 2.0;

    int32_t nudged;
    if (initial_zp < qmin)       nudged = qmin;
    else if (initial_zp > qmax)  nudged = qmax;
    else                         nudged = (int32_t)Kokkos::round(initial_zp);
    zero_pt(i) = nudged;
  });
  Kokkos::fence();
}

// CPU reference
void reference(const float* x_min, const float* x_max,
               int32_t qmin, int32_t qmax, int size, bool preserve_sparsity,
               float* scale, int32_t* zp)
{
  for (int i = 0; i < size; i++) {
    float min_val = x_min[i];
    float max_val = x_max[i];

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;
      double max_scale = std::max(
          fabs(min_val / symmetric_qmin), fabs(max_val / symmetric_qmax));
      min_val = (float)(max_scale * symmetric_qmin);
      max_val = (float)(max_scale * symmetric_qmax);
    }
    min_val = std::min(min_val, 0.f);
    max_val = std::max(max_val, 0.f);
    scale[i] = (float)((static_cast<double>(max_val) - min_val) / (qmax - qmin));
    if (scale[i] == 0.f || std::isinf(1.f / scale[i])) scale[i] = 0.1f;

    double zp_min = qmin - min_val / static_cast<double>(scale[i]);
    double zp_max = qmax - max_val / static_cast<double>(scale[i]);
    double err_min = std::abs(qmin) + std::abs(min_val / static_cast<double>(scale[i]));
    double err_max = std::abs(qmax) + std::abs(max_val / static_cast<double>(scale[i]));
    double init_zp = (err_min < err_max) ? zp_min : zp_max;

    if (min_val < 0 && max_val > 0 && preserve_sparsity)
      init_zp = (double)(qmin + qmax) / 2.0;

    int32_t nudged;
    if (init_zp < qmin)       nudged = qmin;
    else if (init_zp > qmax)  nudged = qmax;
    else                      nudged = (int32_t)nearbyint(init_zp);
    zp[i] = nudged;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of min/max values> <repeat>\n", argv[0]);
    return 1;
  }
  const int size   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int32_t qmin = -127;
  const int32_t qmax =  127;
  const bool preserve_sparsity = true;

  float    *min_h = (float*)    malloc(size * sizeof(float));
  float    *max_h = (float*)    malloc(size * sizeof(float));
  float    *scale_ref = (float*)    malloc(size * sizeof(float));
  int32_t  *zp_ref    = (int32_t*)  malloc(size * sizeof(int32_t));
  float    *scale_out = (float*)    malloc(size * sizeof(float));
  int32_t  *zp_out    = (int32_t*)  malloc(size * sizeof(int32_t));

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-1.f, 1.f);
  for (int i = 0; i < size; i++) {
    min_h[i] = distr(g);
    max_h[i] = distr(g);
  }

  reference(min_h, max_h, qmin, qmax, size, preserve_sparsity, scale_ref, zp_ref);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*>   d_min("d_min", size);
    Kokkos::View<float*>   d_max("d_max", size);
    Kokkos::View<float*>   d_scale("d_scale", size);
    Kokkos::View<int32_t*> d_zp("d_zp", size);

    auto h_min = Kokkos::create_mirror_view(d_min);
    auto h_max = Kokkos::create_mirror_view(d_max);
    for (int i = 0; i < size; i++) { h_min(i) = min_h[i]; h_max(i) = max_h[i]; }
    Kokkos::deep_copy(d_min, h_min);
    Kokkos::deep_copy(d_max, h_max);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      zero_point(d_min, d_max, qmin, qmax, size, preserve_sparsity, d_scale, d_zp);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of zero-point kernel: %f (us)\n",
           (time * 1e-3f) / repeat);

    auto h_scale = Kokkos::create_mirror_view(d_scale);
    auto h_zp    = Kokkos::create_mirror_view(d_zp);
    Kokkos::deep_copy(h_scale, d_scale);
    Kokkos::deep_copy(h_zp,    d_zp);
    for (int i = 0; i < size; i++) {
      scale_out[i] = h_scale(i);
      zp_out[i]    = h_zp(i);
    }
  }
  Kokkos::finalize();

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (zp_out[i] != zp_ref[i] || fabsf(scale_out[i] - scale_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(min_h); free(max_h);
  free(scale_ref); free(zp_ref);
  free(scale_out); free(zp_out);
  return 0;
}
