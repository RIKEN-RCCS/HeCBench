/*
 * Lomb-Scargle periodogram.
 * Ported to Kokkos from the OMP target version.
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <Kokkos_Core.hpp>

void lombscargle_cpu(int x_shape, int freqs_shape,
                     const float *x, const float *y, const float *freqs,
                     float *pgram, float y_dot)
{
  for (int tid = 0; tid < freqs_shape; tid++) {
    float freq = freqs[tid];
    float xc = 0, xs = 0, cc = 0, ss = 0, cs = 0, c, s;
    for (int j = 0; j < x_shape; j++) {
      sincosf(freq * x[j], &s, &c);
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }
    float c_tau, s_tau;
    float tau = atan2f(2.f * cs, cc - ss) / (2.f * freq);
    sincosf(freq * tau, &s_tau, &c_tau);
    float c_tau2 = c_tau * c_tau;
    float s_tau2 = s_tau * s_tau;
    float cs_tau = 2.f * c_tau * s_tau;
    pgram[tid] = 0.5f * (
        (c_tau * xc + s_tau * xs) * (c_tau * xc + s_tau * xs) /
        (c_tau2 * cc + cs_tau * cs + s_tau2 * ss) +
        (c_tau * xs - s_tau * xc) * (c_tau * xs - s_tau * xc) /
        (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)) * y_dot;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int x_shape = 1000;
  const int freqs_shape = 100000;
  const float A = 2.f, w = 1.f, phi = 1.57f;
  const float y_dot = 2.f / 1.5f;

  float *x  = (float*) malloc(x_shape     * sizeof(float));
  float *y  = (float*) malloc(x_shape     * sizeof(float));
  float *f  = (float*) malloc(freqs_shape * sizeof(float));
  float *p  = (float*) malloc(freqs_shape * sizeof(float));
  float *p2 = (float*) malloc(freqs_shape * sizeof(float));

  for (int i = 0; i < x_shape; i++) {
    x[i] = 0.01f + i * (31.4f - 0.01f) / x_shape;
    y[i] = A * sinf(w * x[i] + phi);
  }
  for (int i = 0; i < freqs_shape; i++)
    f[i] = 0.01f + i * (10.f - 0.01f) / freqs_shape;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_x("d_x", x_shape);
    Kokkos::View<float*> d_y("d_y", x_shape);
    Kokkos::View<float*> d_f("d_f", freqs_shape);
    Kokkos::View<float*> d_p("d_p", freqs_shape);

    auto h_x = Kokkos::create_mirror_view(d_x);
    auto h_y = Kokkos::create_mirror_view(d_y);
    auto h_f = Kokkos::create_mirror_view(d_f);
    for (int i = 0; i < x_shape;     i++) { h_x(i) = x[i]; h_y(i) = y[i]; }
    for (int i = 0; i < freqs_shape; i++) h_f(i) = f[i];
    Kokkos::deep_copy(d_x, h_x);
    Kokkos::deep_copy(d_y, h_y);
    Kokkos::deep_copy(d_f, h_f);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("lombscargle", freqs_shape, KOKKOS_LAMBDA(int tid) {
        float freq = d_f(tid);
        float xc = 0, xs = 0, cc = 0, ss = 0, cs = 0;
        for (int j = 0; j < x_shape; j++) {
          float c = Kokkos::cos(freq * d_x(j));
          float s = Kokkos::sin(freq * d_x(j));
          xc += d_y(j) * c;
          xs += d_y(j) * s;
          cc += c * c;
          ss += s * s;
          cs += c * s;
        }
        float c_tau, s_tau;
        float tau = Kokkos::atan2(2.f * cs, cc - ss) / (2.f * freq);
        c_tau = Kokkos::cos(freq * tau);
        s_tau = Kokkos::sin(freq * tau);
        float c_tau2 = c_tau * c_tau;
        float s_tau2 = s_tau * s_tau;
        float cs_tau = 2.f * c_tau * s_tau;
        d_p(tid) = 0.5f * (
            (c_tau * xc + s_tau * xs) * (c_tau * xc + s_tau * xs) /
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss) +
            (c_tau * xs - s_tau * xc) * (c_tau * xs - s_tau * xc) /
            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)) * y_dot;
      });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3) / repeat);

    auto h_p = Kokkos::create_mirror_view(d_p);
    Kokkos::deep_copy(h_p, d_p);
    for (int i = 0; i < freqs_shape; i++) p[i] = h_p(i);
  }
  Kokkos::finalize();

  lombscargle_cpu(x_shape, freqs_shape, x, y, f, p2, y_dot);

  bool error = false;
  for (int i = 0; i < freqs_shape; i++) {
    if (fabsf(p[i] - p2[i]) > 1e-3f) {
      printf("%.3f %.3f\n", p[i], p2[i]);
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(x); free(y); free(f); free(p); free(p2);
  return 0;
}
