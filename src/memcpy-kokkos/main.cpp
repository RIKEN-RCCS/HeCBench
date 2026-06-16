#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <Kokkos_Core.hpp>

#define NUM_SIZE 16

void setup(size_t* size) {
  for (int i = 0; i < NUM_SIZE; i++)
    size[i] = 1 << (i + 6);
}

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++)
    A[i] = val;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = atoi(argv[1]);

  size_t sizes[NUM_SIZE];
  setup(sizes);

  Kokkos::initialize(argc, argv);
  {
    for (int i = 0; i < NUM_SIZE; i++) {
      size_t nbytes = sizes[i];
      size_t len = nbytes / sizeof(int);

      int* A = (int*)malloc(nbytes);
      if (!A) {
        std::cerr << "Host memory allocation failed\n";
        return -1;
      }
      valSet(A, 1, nbytes);

      // Use a HostSpace view as the host buffer and a default space view as device buffer
      Kokkos::View<int*, Kokkos::HostSpace> h_A(A, len);
      Kokkos::View<int*> d_A("d_A", len);

      // Warmup H2D
      for (int j = 0; j < repeat; j++) {
        Kokkos::deep_copy(d_A, h_A);
      }
      Kokkos::fence();

      auto start = std::chrono::steady_clock::now();
      for (int j = 0; j < repeat; j++) {
        Kokkos::deep_copy(d_A, h_A);
      }
      Kokkos::fence();
      auto end = std::chrono::steady_clock::now();
      auto timeH2D = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      std::cout << "Copy " << nbytes << " bytes from host to device takes "
                << (timeH2D * 1e-3f) / repeat << " us" << std::endl;

      // Warmup D2H
      for (int j = 0; j < repeat; j++) {
        Kokkos::deep_copy(h_A, d_A);
      }
      Kokkos::fence();

      start = std::chrono::steady_clock::now();
      for (int j = 0; j < repeat; j++) {
        Kokkos::deep_copy(h_A, d_A);
      }
      Kokkos::fence();
      end = std::chrono::steady_clock::now();
      auto timeD2H = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      std::cout << "Copy " << nbytes << " bytes from device to host takes "
                << (timeD2H * 1e-3f) / repeat << " us" << std::endl;

      free(A);
      std::cout << "Timing gap in nanoseconds per byte: "
                << (float)std::abs(timeH2D - timeD2H) / (repeat * (long long)nbytes);
      std::cout << std::endl << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
