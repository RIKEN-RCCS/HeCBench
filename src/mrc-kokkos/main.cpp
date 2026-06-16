#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>

// Inlined from mrc-cuda/reference.h
void reference(const int N, const int *Y, const float *X1, const float *X2,
               const float *dOutput, const float margin,
               float *dX1, float *dX2) {
  for (int i = 0; i < N; i++) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) { dX1[i] = dX2[i] = 0.f; }
    else { dX1[i] = -Y[i] * dOutput[i]; dX2[i] = Y[i] * dOutput[i]; }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  float *h_X1  = (float*)malloc(length * sizeof(float));
  float *h_X2  = (float*)malloc(length * sizeof(float));
  float *h_O   = (float*)malloc(length * sizeof(float));
  int   *h_Y   = (int*)  malloc(length * sizeof(int));
  float *r_dX1 = (float*)malloc(length * sizeof(float));
  float *r_dX2 = (float*)malloc(length * sizeof(float));

  const float m = 0.01f;
  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-2.f, 2.f);
  for (int i = 0; i < length; i++) {
    h_X1[i] = distr(g); h_X2[i] = distr(g);
    h_O[i]  = distr(g); h_Y[i]  = (distr(g) < 0) ? -1 : 1;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_X1("d_X1", length);
    Kokkos::View<float*> d_X2("d_X2", length);
    Kokkos::View<float*> d_O("d_O",   length);
    Kokkos::View<int*>   d_Y("d_Y",   length);
    Kokkos::View<float*> d_dX1("d_dX1", length);
    Kokkos::View<float*> d_dX2("d_dX2", length);

    {
      auto hX1 = Kokkos::create_mirror_view(d_X1);
      auto hX2 = Kokkos::create_mirror_view(d_X2);
      auto hO  = Kokkos::create_mirror_view(d_O);
      auto hY  = Kokkos::create_mirror_view(d_Y);
      for (int i = 0; i < length; i++) {
        hX1(i) = h_X1[i]; hX2(i) = h_X2[i];
        hO(i)  = h_O[i];  hY(i)  = h_Y[i];
      }
      Kokkos::deep_copy(d_X1, hX1); Kokkos::deep_copy(d_X2, hX2);
      Kokkos::deep_copy(d_O, hO);   Kokkos::deep_copy(d_Y, hY);
    }

    // Warmup
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("mrc1_warm", length, KOKKOS_LAMBDA(int idx) {
        float dist = -(float)d_Y(idx) * (d_X1(idx) - d_X2(idx)) + m;
        d_dX1(idx) = dist < 0.f ? 0.f : -(float)d_Y(idx) * d_O(idx);
        d_dX2(idx) = dist < 0.f ? 0.f :  (float)d_Y(idx) * d_O(idx);
      });
      Kokkos::parallel_for("mrc2_warm", length, KOKKOS_LAMBDA(int idx) {
        float y = (float)d_Y(idx);
        float o = d_O(idx);
        float dist = -y * (d_X1(idx) - d_X2(idx)) + m;
        d_dX1(idx) = dist < 0.f ? 0.f : -y * o;
        d_dX2(idx) = dist < 0.f ? 0.f :  y * o;
      });
    }
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("mrc1", length, KOKKOS_LAMBDA(int idx) {
        float dist = -(float)d_Y(idx) * (d_X1(idx) - d_X2(idx)) + m;
        if (dist < 0.f) { d_dX1(idx) = d_dX2(idx) = 0.f; }
        else { d_dX1(idx) = -(float)d_Y(idx) * d_O(idx);
               d_dX2(idx) =  (float)d_Y(idx) * d_O(idx); }
      });
      Kokkos::fence();
    }
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of MRC kernel: %f (us)\n", (time * 1e-3f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("mrc2", length, KOKKOS_LAMBDA(int idx) {
        float y = (float)d_Y(idx);
        float o = d_O(idx);
        float dist = -y * (d_X1(idx) - d_X2(idx)) + m;
        d_dX1(idx) = dist < 0.f ? 0.f : -y * o;
        d_dX2(idx) = dist < 0.f ? 0.f :  y * o;
      });
      Kokkos::fence();
    }
    end  = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of MRC2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

    auto hdX1 = Kokkos::create_mirror_view(d_dX1);
    auto hdX2 = Kokkos::create_mirror_view(d_dX2);
    Kokkos::deep_copy(hdX1, d_dX1);
    Kokkos::deep_copy(hdX2, d_dX2);

    reference(length, h_Y, h_X1, h_X2, h_O, m, r_dX1, r_dX2);

    bool ok = true;
    for (int i = 0; i < length; i++) {
      if (fabs(hdX1(i) - r_dX1[i]) > 1e-3 || fabs(hdX2(i) - r_dX2[i]) > 1e-3) {
        ok = false; break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  free(h_X1); free(h_X2); free(h_O); free(h_Y);
  free(r_dX1); free(r_dX2);
  return 0;
}
