// Rotary embedding benchmark (Kokkos port)
// Computes: o1 = x1*cos - x2*sin, o2 = x1*sin + x2*cos
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // Match original block/thread/work sizes
  constexpr int num_threads_val   = 32 * 4;   // 128
  constexpr int thread_work_size  = 4;
  constexpr int block_work_size   = thread_work_size * num_threads_val; // 512
  const int64_t numel = (int64_t)block_work_size * 10000; // 5120000

  printf("Number of elements: %zu\n", (size_t)numel);

  const size_t sz = (size_t)numel * sizeof(float);

  // Host arrays
  float* h_x1  = (float*)malloc(sz);
  float* h_x2  = (float*)malloc(sz);
  float* h_cos = (float*)malloc(sz);
  float* h_sin = (float*)malloc(sz);
  float* h_o1  = (float*)malloc(sz);
  float* h_o2  = (float*)malloc(sz);

  for (int64_t i = 0; i < numel; i++) {
    h_x1[i]  = (float)(i + 1) / numel;
    h_x2[i]  = (float)(i + 1) / numel;
    h_cos[i] = cosf((float)i / powf(10000.f, (float)i / numel));
    h_sin[i] = sinf((float)i / powf(10000.f, (float)i / numel));
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_x1 ("x1",  numel);
    Kokkos::View<float*> d_x2 ("x2",  numel);
    Kokkos::View<float*> d_cos("cos", numel);
    Kokkos::View<float*> d_sin("sin", numel);
    Kokkos::View<float*> d_o1 ("o1",  numel);
    Kokkos::View<float*> d_o2 ("o2",  numel);

    {
      auto hv_x1  = Kokkos::create_mirror_view(d_x1);
      auto hv_x2  = Kokkos::create_mirror_view(d_x2);
      auto hv_cos = Kokkos::create_mirror_view(d_cos);
      auto hv_sin = Kokkos::create_mirror_view(d_sin);
      for (int64_t i = 0; i < numel; i++) {
        hv_x1(i)  = h_x1[i];
        hv_x2(i)  = h_x2[i];
        hv_cos(i) = h_cos[i];
        hv_sin(i) = h_sin[i];
      }
      Kokkos::deep_copy(d_x1,  hv_x1);
      Kokkos::deep_copy(d_x2,  hv_x2);
      Kokkos::deep_copy(d_cos, hv_cos);
      Kokkos::deep_copy(d_sin, hv_sin);
    }

    // Warmup
    Kokkos::parallel_for("rotary_warmup", numel, KOKKOS_LAMBDA(const int64_t i) {
      d_o1(i) = d_x1(i) * d_cos(i) - d_x2(i) * d_sin(i);
      d_o2(i) = d_x1(i) * d_sin(i) + d_x2(i) * d_cos(i);
    });
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("rotary", numel, KOKKOS_LAMBDA(const int64_t i) {
        d_o1(i) = d_x1(i) * d_cos(i) - d_x2(i) * d_sin(i);
        d_o2(i) = d_x1(i) * d_sin(i) + d_x2(i) * d_cos(i);
      });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", (float)time * 1e-3f / repeat);

    // Copy back and verify
    auto hv_o1 = Kokkos::create_mirror_view(d_o1);
    auto hv_o2 = Kokkos::create_mirror_view(d_o2);
    Kokkos::deep_copy(hv_o1, d_o1);
    Kokkos::deep_copy(hv_o2, d_o2);
    for (int64_t i = 0; i < numel; i++) {
      h_o1[i] = hv_o1(i);
      h_o2[i] = hv_o2(i);
    }

    bool ok = true;
    for (int64_t i = 0; i < numel; i++) {
      float r1 = h_x1[i] * h_cos[i] - h_x2[i] * h_sin[i];
      float r2 = h_x1[i] * h_sin[i] + h_x2[i] * h_cos[i];
      if (fabsf(r1 - h_o1[i]) > 1e-3f || fabsf(r2 - h_o2[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  free(h_x1); free(h_x2); free(h_cos); free(h_sin); free(h_o1); free(h_o2);
  return 0;
}
