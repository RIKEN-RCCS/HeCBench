// Kokkos port of BabelStream memory-bandwidth benchmark.
// Reference: https://github.com/UoB-HPC/BabelStream
//
// Measures Copy / Mul / Add / Triad / Dot / NStream bandwidth (MB/s).
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <Kokkos_Core.hpp>

// Default parameters (may be overridden via --arraysize / --numtimes)
int          ARRAY_SIZE = 33554432;   // 2^25
unsigned int num_times  = 100;

#define SCALAR 0.4

template <typename T>
void run() {
  std::streamsize ss = std::cout.precision();

  std::cout << "Running kernels " << num_times << " times" << std::endl;

  if (sizeof(T) == sizeof(float))
    std::cout << "Precision: float" << std::endl;
  else
    std::cout << "Precision: double" << std::endl;

  const int N = ARRAY_SIZE;
  std::cout << std::setprecision(1) << std::fixed
    << "Array size: " << N * sizeof(T) * 1.0e-6 << " MB"
    << " (=" << N * sizeof(T) * 1.0e-9 << " GB)" << std::endl;
  std::cout << "Total size: " << 3.0 * N * sizeof(T) * 1.0e-6 << " MB"
    << " (=" << 3.0 * N * sizeof(T) * 1.0e-9 << " GB)" << std::endl;
  std::cout.precision(ss);

  // Allocate device arrays
  Kokkos::View<T*> da("a", N);
  Kokkos::View<T*> db("b", N);
  Kokkos::View<T*> dc("c", N);

  // Initialise
  Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(const int i) {
    da(i) = T(0.1);
    db(i) = T(0.2);
    dc(i) = T(0.0);
  });
  Kokkos::fence();

  const T scalar = T(SCALAR);

  // Timing storage: [copy, mul, add, triad, dot, nstream]
  std::vector<std::vector<double>> timings(6);

  std::chrono::high_resolution_clock::time_point t1, t2;

  for (unsigned int k = 0; k < num_times; k++) {
    // --- Copy: c = a ---
    t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("copy", N, KOKKOS_LAMBDA(const int i) {
      dc(i) = da(i);
    });
    Kokkos::fence();
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

    // --- Mul: b = scalar * c ---
    t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("mul", N, KOKKOS_LAMBDA(const int i) {
      db(i) = scalar * dc(i);
    });
    Kokkos::fence();
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

    // --- Add: c = a + b ---
    t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("add", N, KOKKOS_LAMBDA(const int i) {
      dc(i) = da(i) + db(i);
    });
    Kokkos::fence();
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

    // --- Triad: a = b + scalar * c ---
    t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("triad", N, KOKKOS_LAMBDA(const int i) {
      da(i) = db(i) + scalar * dc(i);
    });
    Kokkos::fence();
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

    // --- Dot: sum = a · b ---
    t1 = std::chrono::high_resolution_clock::now();
    T dot_result = T(0);
    Kokkos::parallel_reduce("dot", N, KOKKOS_LAMBDA(const int i, T& lsum) {
      lsum += da(i) * db(i);
    }, dot_result);
    Kokkos::fence();
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

    // --- NStream: a += b + scalar * c ---
    t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("nstream", N, KOKKOS_LAMBDA(const int i) {
      da(i) += db(i) + scalar * dc(i);
    });
    Kokkos::fence();
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());
  }

  // Display results
  std::cout
    << std::left << std::setw(12) << "Function"
    << std::left << std::setw(12) << "MBytes/sec"
    << std::left << std::setw(12) << "Min (sec)"
    << std::left << std::setw(12) << "Max"
    << std::left << std::setw(12) << "Average"
    << std::endl
    << std::fixed;

  std::vector<std::string> labels = {"Copy","Mul","Add","Triad","Dot","Nstream"};
  std::vector<size_t> sizes = {
    2 * sizeof(T) * N,
    2 * sizeof(T) * N,
    3 * sizeof(T) * N,
    3 * sizeof(T) * N,
    2 * sizeof(T) * N,
    4 * sizeof(T) * N
  };

  for (size_t i = 0; i < timings.size(); ++i) {
    auto minmax  = std::minmax_element(timings[i].begin() + 1, timings[i].end());
    double avg   = std::accumulate(timings[i].begin() + 1, timings[i].end(), 0.0)
                   / (double)(num_times - 1);
    double bw    = 1.0e-6 * sizes[i] / (*minmax.first);
    std::cout
      << std::left << std::setw(12) << labels[i]
      << std::left << std::setw(12) << std::setprecision(3) << bw
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
      << std::left << std::setw(12) << std::setprecision(5) << avg
      << std::endl;
  }
  std::cout << std::endl;
}

// ─── argument parsing ────────────────────────────────────────────────────────

int parseUInt(const char *str, unsigned int *output) {
  char *next; *output = strtoul(str, &next, 10);
  return !strlen(next);
}
int parseInt(const char *str, int *output) {
  char *next; *output = strtol(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[]) {
  for (int i = 1; i < argc; i++) {
    if (!std::string("--arraysize").compare(argv[i]) || !std::string("-s").compare(argv[i])) {
      if (++i >= argc || !parseInt(argv[i], &ARRAY_SIZE) || ARRAY_SIZE <= 0) {
        std::cerr << "Invalid array size." << std::endl; exit(EXIT_FAILURE);
      }
    } else if (!std::string("--numtimes").compare(argv[i]) || !std::string("-n").compare(argv[i])) {
      if (++i >= argc || !parseUInt(argv[i], &num_times) || num_times < 2) {
        std::cerr << "Invalid number of times (must be >=2)." << std::endl; exit(EXIT_FAILURE);
      }
    } else if (!std::string("--help").compare(argv[i]) || !std::string("-h").compare(argv[i])) {
      std::cout << "\nUsage: " << argv[0] << " [OPTIONS]\n"
                << "  -s  --arraysize  SIZE  Number of elements\n"
                << "  -n  --numtimes   NUM   Repeat count (>=2)\n\n";
      exit(EXIT_SUCCESS);
    }
    // Skip Kokkos-specific flags (e.g. --kokkos-device-id)
  }
}

int main(int argc, char *argv[]) {
  parseArguments(argc, argv);

  Kokkos::initialize(argc, argv);
  {
    run<float>();
    run<double>();
  }
  Kokkos::finalize();
  return 0;
}
