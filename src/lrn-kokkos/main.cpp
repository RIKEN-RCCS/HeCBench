#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>

using namespace std::chrono;

static inline int64_t imax(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

void Forward(int repeat)
{
  const int64_t ndims  = 5;
  const int64_t size   = 5;
  const float   alpha  = 0.000122f;
  const float   beta   = 0.750000f;
  const float   k      = 1.000000f;
  const int64_t N      = 6;
  const int64_t C      = 150;
  const int64_t D      = 100;
  const int64_t H      = 160;
  const int64_t W      = 160;
  const int64_t stride_mb = C*D*H*W;
  const int64_t wk_size   = N*C*D*H*W;

  std::vector<float> src_h(wk_size, 0);
  std::vector<float> dst_h(wk_size, 0);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++)
    src_h[i] = rand() / (float)RAND_MAX;

  Kokkos::View<float*> src_d("src", wk_size);
  Kokkos::View<float*> dst_d("dst", wk_size);

  auto src_mirror = Kokkos::create_mirror_view(src_d);
  for (int64_t i = 0; i < wk_size; i++) src_mirror(i) = src_h[i];
  Kokkos::deep_copy(src_d, src_mirror);

  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size *= 2) {

    auto start = high_resolution_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("lrn_fwd", wk_size,
        KOKKOS_LAMBDA(const int64_t idx) {
          const float* src_ = src_d.data();
          float*       dst_ = dst_d.data();

          auto data_off = [=](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) -> int64_t {
            return mb * stride_mb + c * H * W + h * W + w;
          };

          auto ker = [=](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) -> float {
            float sum = 0.f;
            const int64_t half_size = (size - 1) / 2;
            const int64_t c_st = oc - half_size > 0 ? oc - half_size : 0;
            const int64_t c_en = oc + half_size + 1 < C ? oc + half_size + 1 : C;
            for (int64_t c = c_st; c < c_en; ++c) {
              const float s = src_[data_off(mb, c, od, oh, ow)];
              sum += s * s;
            }
            sum = k + alpha * sum / size;
            const float s = src_[data_off(mb, oc, od, oh, ow)];
            return s * sqrtf(1.0f / (sqrtf(sum) * sum));
          };

          const int64_t n = (idx / (C * D * H * W)) % N;
          const int64_t c = (idx / (D * H * W))     % C;
          const int64_t d = (idx / (H * W))          % D;
          const int64_t h = (idx / W)                % H;
          const int64_t w = idx                      % W;

          const int64_t off = data_off(n, c, d, h, w);
          dst_[off] = ker(n, c, d, h, w);
        });
      Kokkos::fence();
    }

    auto stop = high_resolution_clock::now();
    float time = duration_cast<microseconds>(stop - start).count() / 1e6f;
    printf("Average execution time of lrn_fwd_kernel: %.6f sec \n", time / repeat);

    float data_inGB = (2 * wk_size * sizeof(float)) / 1e9f;
    printf("Kernel bandwidth: %.6f GB/s \n", data_inGB * repeat / time);
  }

  auto dst_mirror = Kokkos::create_mirror_view(dst_d);
  Kokkos::deep_copy(dst_mirror, dst_d);
  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) checksum += dst_mirror(i);
  printf("Checksum: %lf\n", checksum / wk_size);
}

void Backward(int repeat)
{
  const int64_t size      = 5;
  const float   alpha     = 0.000122f;
  const float   beta      = 0.750000f;
  const float   k         = 1.000000f;
  const int64_t N         = 5;
  const int64_t C         = 150;
  const int64_t D         = 100;
  const int64_t H         = 160;
  const int64_t W         = 160;
  const int64_t stride_mb = C*D*H*W;
  const int64_t wk_size   = N*C*D*H*W;

  std::vector<float> src_h(wk_size), dst_h(wk_size), diff_src_h(wk_size);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++)
    dst_h[i] = diff_src_h[i] = src_h[i] = rand() / (float)RAND_MAX;

  Kokkos::View<float*> src_d("src",       wk_size);
  Kokkos::View<float*> dst_d("dst",       wk_size);
  Kokkos::View<float*> diff_d("diff_src", wk_size);

  {
    auto m = Kokkos::create_mirror_view(src_d);
    for (int64_t i = 0; i < wk_size; i++) m(i) = src_h[i];
    Kokkos::deep_copy(src_d, m);
  }
  {
    auto m = Kokkos::create_mirror_view(dst_d);
    for (int64_t i = 0; i < wk_size; i++) m(i) = dst_h[i];
    Kokkos::deep_copy(dst_d, m);
  }
  {
    auto m = Kokkos::create_mirror_view(diff_d);
    for (int64_t i = 0; i < wk_size; i++) m(i) = diff_src_h[i];
    Kokkos::deep_copy(diff_d, m);
  }

  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size *= 2) {

    auto start = high_resolution_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("lrn_bwd", wk_size,
        KOKKOS_LAMBDA(const int64_t idx) {
          const float* src_       = src_d.data();
          const float* dst_       = dst_d.data();
          float*       diff_src_  = diff_d.data();

          auto data_off = [=](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) -> int64_t {
            return mb * stride_mb + c * H * W + h * W + w;
          };

          auto get_omega = [=](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) -> float {
            float sum = 0.f;
            const int64_t half_size = (size - 1) / 2;
            const int64_t c_st = oc - half_size > 0 ? oc - half_size : 0;
            const int64_t c_en = oc + half_size + 1 < C ? oc + half_size + 1 : C;
            for (int64_t c = c_st; c < c_en; ++c) {
              const float s = src_[data_off(mb, c, od, oh, ow)];
              sum += s * s;
            }
            return k + alpha * sum / size;
          };

          auto ker = [=](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) -> float {
            float A = 0.f, B = 0.f;
            const int64_t half_size = (size - 1) / 2;
            const int64_t c_st = oc - half_size > 0 ? oc - half_size : 0;
            const int64_t c_en = oc + half_size + 1 < C ? oc + half_size + 1 : C;
            for (int64_t c = c_st; c < c_en; ++c) {
              const int64_t off    = data_off(mb, c, od, oh, ow);
              const float omega    = get_omega(mb, c, od, oh, ow);
              const float omega_ib = sqrtf(1.0f / (sqrtf(omega) * omega));
              const float tmp      = omega_ib * dst_[off];
              if (c == oc) A = tmp;
              B += src_[off] * tmp / omega;
            }
            const int64_t off = data_off(mb, oc, od, oh, ow);
            B *= 2.0f * alpha * beta * src_[off] / size;
            return A - B;
          };

          const int64_t n = (idx / (C * D * H * W)) % N;
          const int64_t c = (idx / (D * H * W))     % C;
          const int64_t d = (idx / (H * W))          % D;
          const int64_t h = (idx / W)                % H;
          const int64_t w = idx                      % W;

          diff_src_[data_off(n, c, d, h, w)] = ker(n, c, d, h, w);
        });
      Kokkos::fence();
    }

    auto stop = high_resolution_clock::now();
    float time = duration_cast<microseconds>(stop - start).count() / 1e6f;
    printf("Average execution time of lrn_bwd_kernel: %.6f sec \n", time / repeat);

    float data_inGB = (3 * wk_size * sizeof(float)) / 1e9f;
    printf("Kernel bandwidth: %.6f GB/s \n", data_inGB * repeat / time);
  }

  auto dst_mirror = Kokkos::create_mirror_view(dst_d);
  Kokkos::deep_copy(dst_mirror, dst_d);
  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) checksum += dst_mirror(i);
  printf("Checksum: %lf\n", checksum / wk_size);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    Forward(repeat);
    Backward(repeat);
  }
  Kokkos::finalize();
  return 0;
}
