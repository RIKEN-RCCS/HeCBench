#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <Kokkos_Core.hpp>

#define NOW std::chrono::high_resolution_clock::now()

// End marker character
const char ETX = '\0';

// CPU reference implementation for verification
static std::string bwt_cpu(const std::string& sequence) {
  const int n = sequence.size();
  const char* seq = sequence.c_str();

  std::vector<int> table(n);
  for (int i = 0; i < n; i++) table[i] = i;

  std::list<int> sorted_table(table.begin(), table.end());
  sorted_table.sort([seq, n](const int& a, const int& b) -> bool {
    for (int i = 0; i < n; i++) {
      if (seq[(a + i) % n] != seq[(b + i) % n])
        return seq[(a + i) % n] < seq[(b + i) % n];
    }
    return false;
  });

  std::string transformed;
  for (auto r = sorted_table.begin(); r != sorted_table.end(); ++r)
    transformed += seq[(n + *r - 1) % n];

  return transformed;
}

KOKKOS_INLINE_FUNCTION
bool compare_rotations(int a, int b, const char* genome, int n) {
  if (a < 0) return false;
  if (b < 0) return true;
  for (int i = 0; i < n; i++) {
    char ca = genome[(a + i) % n];
    char cb = genome[(b + i) % n];
    if (ca != cb) return ca < cb;
  }
  return false;
}

static std::string bwt_device(const std::string& sequence) {
  const int n          = (int)sequence.size();
  const int block_size = 256;

  // Round table_size up to next power of 2 for bitonic sort
  int table_size = n;
  table_size--;
  table_size |= table_size >> 1;
  table_size |= table_size >> 2;
  table_size |= table_size >> 4;
  table_size |= table_size >> 8;
  table_size |= table_size >> 16;
  table_size++;

  Kokkos::View<int*>  d_table("d_table", table_size);
  Kokkos::View<char*> d_sequence("d_sequence", n);

  // Copy sequence to device
  {
    auto hv = Kokkos::create_mirror_view(d_sequence);
    for (int i = 0; i < n; i++) hv(i) = sequence[i];
    Kokkos::deep_copy(d_sequence, hv);
  }

  // generate_table: table[i] = i for i < n, -1 for padding
  Kokkos::parallel_for("generate_table", table_size,
    KOKKOS_LAMBDA(int i) {
      d_table(i) = (i < n) ? i : -1;
    });

  // Bitonic sort
  for (int k = 2; k <= table_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      Kokkos::parallel_for("bitonic_sort_step", table_size,
        KOKKOS_LAMBDA(int i) {
          int ixj = i ^ j;
          if (ixj > i && ixj < table_size) {
            bool f  = (i & k) == 0;
            int  t1 = d_table(i);
            int  t2 = d_table(ixj);
            // If out of order for current sort direction, swap
            if (compare_rotations(f ? t2 : t1, f ? t1 : t2,
                                  d_sequence.data(), n)) {
              d_table(i)   = t2;
              d_table(ixj) = t1;
            }
          }
        });
      Kokkos::fence();
    }
  }

  // Reconstruct BWT from sorted suffix array
  Kokkos::View<char*> d_transformed("d_transformed", n);
  Kokkos::parallel_for("reconstruct", n,
    KOKKOS_LAMBDA(int i) {
      d_transformed(i) = d_sequence((n + d_table(i) - 1) % n);
    });
  Kokkos::fence();

  // Copy result back to host
  auto hv = Kokkos::create_mirror_view(d_transformed);
  Kokkos::deep_copy(hv, d_transformed);

  std::string result(n, '\0');
  for (int i = 0; i < n; i++) result[i] = hv(i);
  return result;
}

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {
    const int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    std::cout << "running a sample sequence of length " << N << std::endl;

    const std::string alphabet("ATCG");
    srand(123);

    std::string sequence(N + 1, ETX);
    for (int i = 0; i < N; i++)
      sequence[i] = alphabet[rand() % alphabet.size()];
    sequence[N] = ETX;

    // CPU reference
    auto start   = NOW;
    auto cpu_seq = bwt_cpu(sequence);
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW - start);

    // Device run
    start        = NOW;
    auto gpu_seq = bwt_device(sequence);
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW - start);

    std::cout << "Host time:   " << cpu_time.count() << " ms" << std::endl;
    std::cout << "Device time: " << gpu_time.count() << " ms" << std::endl;

    if (cpu_seq == gpu_seq)
      std::cout << "PASS\n";
    else
      std::cout << "FAIL\n";
  }
  Kokkos::finalize();
  return 0;
}
