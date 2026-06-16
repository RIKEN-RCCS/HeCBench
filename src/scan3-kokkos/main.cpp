#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>

bool isPowerOf2(int n) { return n > 0 && (n & (n - 1)) == 0; }
int roundToPowerOf2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

void scanCPUReference(float* output, const float* input, unsigned int length) {
  output[0] = 0.0f;
  for (unsigned int i = 1; i < length; i++)
    output[i] = input[i-1] + output[i-1];
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input_length>\n";
    return 1;
  }
  int iterations = atoi(argv[1]);
  int length = atoi(argv[2]);

  if (iterations < 1) {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if (!isPowerOf2(length))
    length = roundToPowerOf2(length);

  const unsigned int sizeBytes = length * sizeof(float);

  std::vector<float> input(length);
  std::vector<float> output(length, 0.0f);
  std::vector<float> refOutput(length, 0.0f);

  srand(42);
  for (int i = 0; i < length; i++) input[i] = (float)(rand() % 256);

  scanCPUReference(refOutput.data(), input.data(), length);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_input("d_input", length);
    Kokkos::View<float*> d_output("d_output", length);

    {
      auto h = Kokkos::create_mirror_view(d_input);
      for (int i = 0; i < length; i++) h(i) = input[i];
      Kokkos::deep_copy(d_input, h);
    }

    std::cout << "Executing kernel for " << iterations << " iterations\n";
    std::cout << "-------------------------------------------\n";

    // Warmup
    Kokkos::parallel_scan("ExclusiveScan_warmup", length,
      KOKKOS_LAMBDA(const int i, float& update, const bool final) {
        const float val = d_input(i);
        if (final) d_output(i) = update;
        update += val;
      });
    Kokkos::fence();

    auto t0 = std::chrono::steady_clock::now();
    for (int n = 0; n < iterations; n++) {
      Kokkos::parallel_scan("ExclusiveScan", length,
        KOKKOS_LAMBDA(const int i, float& update, const bool final) {
          const float val = d_input(i);
          if (final) d_output(i) = update;
          update += val;
        });
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::cout << "Average execution time of Kokkos exclusive scan: "
              << ns / iterations * 1e-3 << " (us)\n";

    auto h_out = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_out, d_output);

    bool pass = true;
    for (int i = 0; i < length; i++) {
      if (fabsf(h_out(i) - refOutput[i]) > 0.001f) { pass = false; break; }
    }
    std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
