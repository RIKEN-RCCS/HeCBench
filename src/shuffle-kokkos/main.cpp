#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>

#define BUF_SIZE 256
#define PATTERN  0xDEADBEEF

// CPU reference for matrix transpose: reverse within each subgroup
static void matrixTransposeCPUReference(float *output, const float *input,
                                        unsigned numGroups, unsigned subGroupSize)
{
  for (unsigned i = 0; i < numGroups; ++i)
    for (unsigned j = 0; j < subGroupSize; j++)
      output[i * subGroupSize + j] = input[i * subGroupSize + subGroupSize - j - 1];
}

static void verifyBroadcast(const int *out, int subGroupSize, int pattern = 0)
{
  int expected = pattern;
  if (pattern == 0)
    for (int i = 0; i < subGroupSize; i++) expected += i;

  int errors = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != expected) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
      break;
    }
  }
  if (errors == 0) std::cout << "PASS\n";
  else             std::cout << "FAIL\n";
}

static void verifyTransposeMatrix(const float *TransposeMatrix,
                                   const float *cpuTransposeMatrix,
                                   int total, int subGroupSize)
{
  float eps = 1.0e-6f;
  int errors = 0;
  for (int i = 0; i < total; i++) {
    if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ITEM: " << i
                << " cpu: " << cpuTransposeMatrix[i]
                << " gpu: " << TransposeMatrix[i] << "\n";
      errors++;
      break;
    }
  }
  if (errors == 0) std::cout << "PASS\n";
  else             std::cout << "FAIL\n";
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <repeat for broadcast> <repeat for matrix transpose>\n";
    return 1;
  }
  const int repeat  = atoi(argv[1]);
  const int repeat2 = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    std::cout << "Broadcast using shuffle functions\n";

    Kokkos::View<int*> d_out("d_out", BUF_SIZE);
    auto h_out = Kokkos::create_mirror_view(d_out);

    // ----------------------------------------------------------------
    // XOR-reduction broadcast (emulated: all threads in a subgroup of
    // size S get sum(0..S-1))
    // ----------------------------------------------------------------
    std::cout << "Broadcast using the shuffle xor function (subgroup sizes 8, 16, and 32) \n";

    // Subgroup size 8 – warmup
    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("bcast_xor_sg8_warmup", BUF_SIZE,
          KOKKOS_LAMBDA(const int i) {
            int sum = 0;
            for (int j = 0; j < 8; j++) sum += j;
            d_out(i) = sum;
          });
      Kokkos::fence();
    }

    // Subgroup size 8 – timed
    {
      auto begin = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        Kokkos::parallel_for("bcast_xor_sg8", BUF_SIZE,
            KOKKOS_LAMBDA(const int i) {
              int sum = 0;
              for (int j = 0; j < 8; j++) sum += j;
              d_out(i) = sum;
            });
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  * 1e-3 / repeat;
      std::cout << "Average kernel time (subgroup size = 8): " << us << " (us)\n";
      Kokkos::deep_copy(h_out, d_out);
      verifyBroadcast(h_out.data(), 8);
    }

    // Subgroup size 16
    {
      auto begin = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        Kokkos::parallel_for("bcast_xor_sg16", BUF_SIZE,
            KOKKOS_LAMBDA(const int i) {
              int sum = 0;
              for (int j = 0; j < 16; j++) sum += j;
              d_out(i) = sum;
            });
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  * 1e-3 / repeat;
      std::cout << "Average kernel time (subgroup size = 16): " << us << " (us)\n";
      Kokkos::deep_copy(h_out, d_out);
      verifyBroadcast(h_out.data(), 16);
    }

    // Subgroup size 32
    {
      auto begin = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        Kokkos::parallel_for("bcast_xor_sg32", BUF_SIZE,
            KOKKOS_LAMBDA(const int i) {
              int sum = 0;
              for (int j = 0; j < 32; j++) sum += j;
              d_out(i) = sum;
            });
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  * 1e-3 / repeat;
      std::cout << "Average kernel time (subgroup size = 32): " << us << " (us)\n";
      Kokkos::deep_copy(h_out, d_out);
      verifyBroadcast(h_out.data(), 32);
    }

    // ----------------------------------------------------------------
    // Broadcast from lane 0 (all threads get PATTERN)
    // ----------------------------------------------------------------
    std::cout << "Broadcast using the shuffle function (subgroup sizes 8, 16, and 32) \n";

    // Subgroup size 8
    {
      auto begin = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        Kokkos::parallel_for("bcast_sg8", BUF_SIZE,
            KOKKOS_LAMBDA(const int i) { d_out(i) = (int)PATTERN; });
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  * 1e-3 / repeat;
      std::cout << "Average kernel time (subgroup size = 8): " << us << " (us)\n";
      Kokkos::deep_copy(h_out, d_out);
      verifyBroadcast(h_out.data(), 8, (int)PATTERN);
    }

    // Subgroup size 16
    {
      auto begin = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        Kokkos::parallel_for("bcast_sg16", BUF_SIZE,
            KOKKOS_LAMBDA(const int i) { d_out(i) = (int)PATTERN; });
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  * 1e-3 / repeat;
      std::cout << "Average kernel time (subgroup size = 16): " << us << " (us)\n";
      Kokkos::deep_copy(h_out, d_out);
      verifyBroadcast(h_out.data(), 16, (int)PATTERN);
    }

    // Subgroup size 32
    {
      auto begin = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        Kokkos::parallel_for("bcast_sg32", BUF_SIZE,
            KOKKOS_LAMBDA(const int i) { d_out(i) = (int)PATTERN; });
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  * 1e-3 / repeat;
      std::cout << "Average kernel time (subgroup size = 32): " << us << " (us)\n";
      Kokkos::deep_copy(h_out, d_out);
      verifyBroadcast(h_out.data(), 32, (int)PATTERN);
    }

    // ----------------------------------------------------------------
    // Matrix transpose using shuffle (reverse within subgroup)
    // ----------------------------------------------------------------
    std::cout << "matrix transpose using the shuffle function (subgroup sizes are 8, 16, and 32)\n";

    const int total = 1 << 27;

    Kokkos::View<float*> d_in ("d_Matrix",    total);
    Kokkos::View<float*> d_tpx("d_Transpose", total);

    {
      auto h_in = Kokkos::create_mirror_view(d_in);
      for (int i = 0; i < total; i++) h_in(i) = (float)i * 10.f;
      Kokkos::deep_copy(d_in, h_in);
    }

    std::vector<float> cpu_tp(total);

    auto run_transpose = [&](int sg) {
      {
        auto begin = std::chrono::steady_clock::now();
        for (int n = 0; n < repeat2; n++) {
          Kokkos::parallel_for("transpose", total,
              KOKKOS_LAMBDA(const int i) {
                int group = i / sg;
                int lane  = i % sg;
                d_tpx(group * sg + lane) = d_in(group * sg + sg - lane - 1);
              });
          Kokkos::fence();
        }
        auto end = std::chrono::steady_clock::now();
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                    * 1e-3 / repeat2;
        std::cout << "Average kernel time (subgroup size = " << sg << "): " << us << " (us)\n";
      }

      auto h_tp = Kokkos::create_mirror_view(d_tpx);
      Kokkos::deep_copy(h_tp, d_tpx);

      auto h_in = Kokkos::create_mirror_view(d_in);
      Kokkos::deep_copy(h_in, d_in);

      matrixTransposeCPUReference(cpu_tp.data(), h_in.data(), total / sg, sg);
      verifyTransposeMatrix(h_tp.data(), cpu_tp.data(), total, sg);
    };

    run_transpose(8);
    run_transpose(16);
    run_transpose(32);
  }
  Kokkos::finalize();
  return 0;
}
