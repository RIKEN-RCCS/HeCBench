// Port of relu-cuda to Kokkos.
//
// CUDA used cuda_fp16 (half precision).  Kokkos has no built-in half support
// across all backends, so we use float instead (as directed).
//
// ReluGrad: backprop[i] = (feature[i] > 0) ? gradient[i] : 0
// Relu:     each 32-bit word packs 4 signed bytes; clamp each byte to >= 0.
//
// Inline reference functions replace the separate reference.h header.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

// ---- Reference (host) implementations --------------------------------------

static void ReluGrad_reference(int count,
                                const float *gradient,
                                const float *feature,
                                float       *backprop)
{
  for (int i = 0; i < count; i++)
    backprop[i] = (feature[i] > 0.f) ? gradient[i] : 0.f;
}

static void Relu_reference(int count, const int *input, int *output)
{
  for (int i = 0; i < count; i++) {
    signed char c1 = (signed char)( input[i]        & 0xFF);
    signed char c2 = (signed char)((input[i] >>  8) & 0xFF);
    signed char c3 = (signed char)((input[i] >> 16) & 0xFF);
    signed char c4 = (signed char)((input[i] >> 24) & 0xFF);
    unsigned x = (unsigned)(c1 > 0 ? c1 : 0);
    unsigned y = (unsigned)(c2 > 0 ? c2 : 0);
    unsigned z = (unsigned)(c3 > 0 ? c3 : 0);
    unsigned w = (unsigned)(c4 > 0 ? c4 : 0);
    output[i] = (int)(w << 24 | z << 16 | y << 8 | x);
  }
}

// ---- Main ------------------------------------------------------------------

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <count> <repeat>\n", argv[0]);
    return 1;
  }

  const int count  = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    // =========================================================================
    // ReluGrad (float)
    // =========================================================================

    std::vector<float> h_gradient(count);
    std::vector<float> h_feature(count);
    std::vector<float> h_backprop(count);
    std::vector<float> r_backprop(count);

    std::mt19937 engine(19937);
    std::uniform_real_distribution<float> real_dist(-1.f, 1.f);

    for (int i = 0; i < count; i++) {
      h_feature[i]  = real_dist(engine);
      h_gradient[i] = 1.f;
    }

    ReluGrad_reference(count, h_gradient.data(), h_feature.data(),
                       r_backprop.data());

    Kokkos::View<float *> d_gradient("d_gradient", count);
    Kokkos::View<float *> d_feature ("d_feature",  count);
    Kokkos::View<float *> d_backprop("d_backprop", count);

    {
      auto hv_g = Kokkos::create_mirror_view(d_gradient);
      auto hv_f = Kokkos::create_mirror_view(d_feature);
      for (int i = 0; i < count; i++) {
        hv_g(i) = h_gradient[i];
        hv_f(i) = h_feature[i];
      }
      Kokkos::deep_copy(d_gradient, hv_g);
      Kokkos::deep_copy(d_feature,  hv_f);
    }

    // impl1 (every element individually)
    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for(
          "ReluGrad_impl1", count,
          KOKKOS_LAMBDA(int idx) {
            d_backprop(idx) =
                (d_feature(idx) > 0.f) ? d_gradient(idx) : 0.f;
          });
    }
    Kokkos::fence();

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count();
    printf("Average execution time of ReluGrad_impl1 Kernel: %f (us)\n",
           (time * 1e-3f) / repeat);

    {
      auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     d_backprop);
      int fail = 0;
      for (int i = 0; i < count; i++) {
        if (std::fabs(hv(i) - r_backprop[i]) > 1e-3f) {
          fail = 1;
          break;
        }
      }
      printf("%s\n", fail ? "FAIL" : "PASS");
    }

    // impl2 (vectorized 8-element groups – conceptually the same in Kokkos,
    // the compiler/backend will vectorize; we keep the same parallel_for)
    Kokkos::fence();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for(
          "ReluGrad_impl2", count,
          KOKKOS_LAMBDA(int idx) {
            d_backprop(idx) =
                (d_feature(idx) > 0.f) ? d_gradient(idx) : 0.f;
          });
    }
    Kokkos::fence();

    end  = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(
               end - start).count();
    printf("Average execution time of ReluGrad_impl2 Kernel: %f (us)\n",
           (time * 1e-3f) / repeat);

    {
      auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     d_backprop);
      int fail = 0;
      for (int i = 0; i < count; i++) {
        if (std::fabs(hv(i) - r_backprop[i]) > 1e-3f) {
          fail = 1;
          break;
        }
      }
      printf("%s\n", fail ? "FAIL" : "PASS");
    }

    // =========================================================================
    // Relu (packed int32 -> 4 x int8 clamped)
    // =========================================================================

    std::vector<int> h_in(count);
    std::vector<int> h_out(count);
    std::vector<int> r_out(count);

    std::uniform_int_distribution<unsigned char> int_dist(0, 255);
    for (int i = 0; i < count; i++) {
      h_in[i] = (unsigned)int_dist(engine)        |
                (unsigned)int_dist(engine) <<  8   |
                (unsigned)int_dist(engine) << 16   |
                (unsigned)int_dist(engine) << 24;
    }

    Relu_reference(count, h_in.data(), r_out.data());

    Kokkos::View<int *> d_in ("d_in",  count);
    Kokkos::View<int *> d_out("d_out", count);

    {
      auto hv = Kokkos::create_mirror_view(d_in);
      for (int i = 0; i < count; i++)
        hv(i) = h_in[i];
      Kokkos::deep_copy(d_in, hv);
    }

    // impl1
    Kokkos::fence();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for(
          "Relu_impl1", count,
          KOKKOS_LAMBDA(int idx) {
            int v = d_in(idx);
            // Extract 4 signed bytes and clamp to >= 0
            auto b0 = static_cast<signed char>( v        & 0xFF);
            auto b1 = static_cast<signed char>((v >>  8) & 0xFF);
            auto b2 = static_cast<signed char>((v >> 16) & 0xFF);
            auto b3 = static_cast<signed char>((v >> 24) & 0xFF);
            unsigned x = (unsigned)(b0 > 0 ? b0 : 0);
            unsigned y = (unsigned)(b1 > 0 ? b1 : 0);
            unsigned z = (unsigned)(b2 > 0 ? b2 : 0);
            unsigned w = (unsigned)(b3 > 0 ? b3 : 0);
            d_out(idx) = (int)(w << 24 | z << 16 | y << 8 | x);
          });
    }
    Kokkos::fence();

    end  = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(
               end - start).count();
    printf("Average execution time of Relu_impl1 Kernel : %f (us)\n",
           (time * 1e-3f) / repeat);

    {
      auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     d_out);
      for (int i = 0; i < count; i++)
        h_out[i] = hv(i);
      int fail = std::memcmp(h_out.data(), r_out.data(),
                             count * sizeof(int));
      printf("%s\n", fail ? "FAIL" : "PASS");
    }

    // impl2 (same logic; separate timing pass)
    Kokkos::fence();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for(
          "Relu_impl2", count,
          KOKKOS_LAMBDA(int idx) {
            int v = d_in(idx);
            auto b0 = static_cast<signed char>( v        & 0xFF);
            auto b1 = static_cast<signed char>((v >>  8) & 0xFF);
            auto b2 = static_cast<signed char>((v >> 16) & 0xFF);
            auto b3 = static_cast<signed char>((v >> 24) & 0xFF);
            unsigned x = (unsigned)(b0 > 0 ? b0 : 0);
            unsigned y = (unsigned)(b1 > 0 ? b1 : 0);
            unsigned z = (unsigned)(b2 > 0 ? b2 : 0);
            unsigned w = (unsigned)(b3 > 0 ? b3 : 0);
            d_out(idx) = (int)(w << 24 | z << 16 | y << 8 | x);
          });
    }
    Kokkos::fence();

    end  = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(
               end - start).count();
    printf("Average execution time of Relu_impl2 Kernel: %f (us)\n",
           (time * 1e-3f) / repeat);

    {
      auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     d_out);
      for (int i = 0; i < count; i++)
        h_out[i] = hv(i);
      int fail = std::memcmp(h_out.data(), r_out.data(),
                             count * sizeof(int));
      printf("%s\n", fail ? "FAIL" : "PASS");
    }
  }
  Kokkos::finalize();
  return 0;
}
