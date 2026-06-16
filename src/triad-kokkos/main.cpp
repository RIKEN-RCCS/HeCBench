// Stream Triad benchmark ported to Kokkos
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <Kokkos_Core.hpp>

using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space  = typename exec_space::memory_space;
using FView      = Kokkos::View<float*, mem_space>;

int main(int argc, char* argv[]) {
  int n_passes = 10;
  bool verbose = false;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-n" && i+1 < argc)
      n_passes = atoi(argv[++i]);
    else if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose")
      verbose = true;
  }

  Kokkos::initialize(argc, argv);
  {
    const int nSizes = 9;
    const int blockSizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const int memSize = 16384;
    const int numMaxFloats  = 1024 * memSize / sizeof(float);
    const int halfNumFloats = numMaxFloats / 2;

    srand48(8650341L);
    float* h_mem = (float*)malloc(sizeof(float) * numMaxFloats);

    FView A("A", numMaxFloats);
    FView B("B", numMaxFloats);
    FView C("C", numMaxFloats);

    const float scalar = 1.75f;

    for (int i = 0; i < nSizes; ++i) {
      int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);

      // Initialise host data
      auto hA = Kokkos::create_mirror_view(A);
      auto hB = Kokkos::create_mirror_view(B);
      auto hC = Kokkos::create_mirror_view(C);

      for (int j = 0; j < numMaxFloats; j++) hC(j) = 0.0f;
      for (int j = 0; j < halfNumFloats; j++) {
        float v = (float)(drand48() * 10.0);
        hA(j) = hA(halfNumFloats+j) = v;
        hB(j) = hB(halfNumFloats+j) = v;
      }

      Kokkos::deep_copy(A, hA);
      Kokkos::deep_copy(B, hB);
      Kokkos::deep_copy(C, hC);

      if (verbose) {
        std::cout << ">> Executing Triad with vectors of length "
                  << numMaxFloats << " and block size of "
                  << elemsInBlock << " elements.\n";
        std::cout << "Block: " << blockSizes[i] << "KB\n";
      }

      Kokkos::fence();
      auto t0 = std::chrono::steady_clock::now();

      for (int pass = 0; pass < n_passes; ++pass) {
        // Process in elemsInBlock chunks
        for (int crtIdx = 0; crtIdx < numMaxFloats; crtIdx += elemsInBlock) {
          int elems = elemsInBlock;
          if (crtIdx + elems > numMaxFloats) elems = numMaxFloats - crtIdx;

          Kokkos::parallel_for(
            "triad",
            Kokkos::RangePolicy<exec_space>(crtIdx, crtIdx + elems),
            KOKKOS_LAMBDA(int gid) {
              C(gid) = A(gid) + scalar * B(gid);
            });
        }
      }

      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-9;

      double triad = ((double)numMaxFloats * 2.0 * n_passes) / (time * 1e9);
      double bdwth  = ((double)numMaxFloats * sizeof(float) * 3.0 * n_passes) / (time * 1e9);
      if (verbose) {
        std::cout << "Average TriadFlops " << triad << " GFLOPS/s\n";
        std::cout << "Average TriadBdwth " << bdwth << " GB/s\n";
      }

      // Verify
      Kokkos::deep_copy(hC, C);
      for (int j = 0; j < numMaxFloats; j++) h_mem[j] = hC(j);

      bool ok = true;
      for (int j = 0; j < halfNumFloats; j++) {
        if (h_mem[j] != h_mem[j + halfNumFloats]) {
          std::cout << "hostMem[" << j << "]=" << h_mem[j]
                    << " differs from twin [" << (j+halfNumFloats) << "]: "
                    << h_mem[j+halfNumFloats] << "\n";
          ok = false;
          break;
        }
      }
      std::cout << (ok ? "PASS" : "FAIL") << "\n";
    }
    free(h_mem);
  }
  Kokkos::finalize();
  return 0;
}
