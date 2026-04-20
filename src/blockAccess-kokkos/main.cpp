#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <Kokkos_Core.hpp>

#define NUM 4

// Reference kernel: converts float->uchar by casting, vectorised 4-at-a-time
void reference_kernel(Kokkos::View<const float*> A,
                      Kokkos::View<unsigned char*> out,
                      const unsigned int n)
{
  Kokkos::parallel_for("reference",
    Kokkos::RangePolicy<>(0, n / 4),
    KOKKOS_LAMBDA(const int idx) {
      const int base = idx * 4;
      out(base + 0) = (unsigned char)(int)A(base + 0);
      out(base + 1) = (unsigned char)(int)A(base + 1);
      out(base + 2) = (unsigned char)(int)A(base + 2);
      out(base + 3) = (unsigned char)(int)A(base + 3);
    });
  Kokkos::fence();
}

// blockAccess kernel: same conversion, each work-item processes ITEMS_TO_LOAD
// elements in groups of NUM (mimics the blocked load/store pattern of the
// CUDA version where a whole CTA handles one tile at a time).
void blockAccess_kernel(Kokkos::View<const float*> A,
                        Kokkos::View<unsigned char*> out,
                        const unsigned int n)
{
  constexpr int ITEMS_TO_LOAD = 256 * NUM;  // block_size * NUM
  const int num_blocks = ((int)n + ITEMS_TO_LOAD - 1) / ITEMS_TO_LOAD;

  Kokkos::parallel_for("blockAccess",
    Kokkos::RangePolicy<>(0, num_blocks),
    KOKKOS_LAMBDA(const int bid) {
      const int base = bid * ITEMS_TO_LOAD;
      const int end  = (base + ITEMS_TO_LOAD < (int)n) ? base + ITEMS_TO_LOAD : (int)n;
      for (int i = base; i < end; i += NUM) {
        float vals[NUM];
        unsigned char qvals[NUM];
        for (int j = 0; j < NUM; j++)
          vals[j] = ((i + j) < end) ? A(i + j) : 0.0f;
        for (int j = 0; j < NUM; j++)
          qvals[j] = (unsigned char)(int)vals[j];
        for (int j = 0; j < NUM; j++)
          if ((i + j) < end) out(i + j) = qvals[j];
      }
    });
  Kokkos::fence();
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 4) {
      printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int nrows  = atoi(argv[1]);
    const int ncols  = atoi(argv[2]);
    const int repeat = atoi(argv[3]);

    const unsigned int n = (unsigned int)nrows * ncols;

    // Generate host data
    std::vector<float> h_A(n);
    std::mt19937 gen{19937};
    std::normal_distribution<float> dist{128.0f, 127.0f};
    for (unsigned int i = 0; i < n; i++)
      h_A[i] = dist(gen);

    // Allocate device views
    Kokkos::View<float*>         d_A("d_A", n);
    Kokkos::View<unsigned char*> d_out("d_out", n);

    // Copy input to device
    auto h_A_mirror = Kokkos::create_mirror_view(d_A);
    for (unsigned int i = 0; i < n; i++)
      h_A_mirror(i) = h_A[i];
    Kokkos::deep_copy(d_A, h_A_mirror);

    Kokkos::View<const float*> d_A_const = d_A;

    // --- Time reference kernel ---
    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++)
      reference_kernel(d_A_const, d_out, n);
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    long long time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the reference kernel: %f (us)\n",
           (time_ns * 1e-3) / repeat);

    // --- Time blockAccess kernel ---
    Kokkos::fence();
    start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++)
      blockAccess_kernel(d_A_const, d_out, n);
    Kokkos::fence();
    end = std::chrono::steady_clock::now();
    time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the blockAccess kernel: %f (us)\n",
           (time_ns * 1e-3) / repeat);

    // Copy result back and verify
    auto h_out_mirror = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_out_mirror, d_out);

    bool error = false;
    for (unsigned int i = 0; i < n; i++) {
      unsigned char expected = (unsigned char)(int)h_A[i];
      if (h_out_mirror(i) != expected) {
        printf("@%u: %u != %u\n", i, (unsigned)h_out_mirror(i), (unsigned)expected);
        error = true;
        break;
      }
    }
    printf("%s\n", error ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return 0;
}
