#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>

void eval_remap(const int N, const int repeat) {
  std::vector<int> h_input(N);
  std::vector<int> h_output(N);

  srand(123);
  for (int i = 0; i < N; i++)
    h_input[i] = rand() % N;

  long seq_time = 0, sort_time = 0, unique_time = 0,
       kernel_time = 0, copy_time = 0, alloc_time = 0, dealloc_time = 0;

  auto offload_start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {

    // --- Alloc ---
    auto t0 = std::chrono::steady_clock::now();
    Kokkos::View<int*> d_input("d_input", N);
    Kokkos::View<int*> d_output("d_output", N);
    Kokkos::View<int*> d_first_order("d_first_order", N);
    // d_second_order sized N; we fill only K entries
    Kokkos::View<int*> d_second_order("d_second_order", N);
    auto t1 = std::chrono::steady_clock::now();
    alloc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // --- Copy H→D (input) ---
    t0 = std::chrono::steady_clock::now();
    {
      auto hv = Kokkos::View<const int*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_input.data(), N);
      Kokkos::deep_copy(d_input, hv);
    }
    t1 = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // --- Sequence (iota on host, analogous to thrust::sequence) ---
    t0 = std::chrono::steady_clock::now();
    std::vector<int> h_order1(N);
    std::iota(h_order1.begin(), h_order1.end(), 0);
    t1 = std::chrono::steady_clock::now();
    seq_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // --- Sort by key: sort order1 indices so that h_input[order1[i]] is non-decreasing ---
    t0 = std::chrono::steady_clock::now();
    std::stable_sort(h_order1.begin(), h_order1.end(),
                     [&](int a, int b) { return h_input[a] < h_input[b]; });
    // Build sorted-input view used by unique step
    std::vector<int> h_sorted(N);
    for (int i = 0; i < N; i++) h_sorted[i] = h_input[h_order1[i]];
    t1 = std::chrono::steady_clock::now();
    sort_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // --- Unique by key: collect position of first occurrence of each unique value ---
    // Mirrors thrust::unique_by_key(sorted_input, order2) where order2 = iota(N)
    // Result: second_order[i] = position in sorted array of the first occurrence of unique element i
    t0 = std::chrono::steady_clock::now();
    std::vector<int> h_second_order;
    h_second_order.reserve(N);
    for (int i = 0; i < N; i++) {
      if (i == 0 || h_sorted[i] != h_sorted[i - 1])
        h_second_order.push_back(i);
    }
    int K = static_cast<int>(h_second_order.size());
    t1 = std::chrono::steady_clock::now();
    unique_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    if (rep == 0)
      printf("Percentage of unique elements: %.1f %%\n", (float)K * 100.f / N);

    // --- Copy order arrays D (needed by kernel) ---
    t0 = std::chrono::steady_clock::now();
    {
      auto hv = Kokkos::View<const int*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_order1.data(), N);
      Kokkos::deep_copy(d_first_order, hv);
    }
    {
      // Copy only K entries into d_second_order[0..K)
      auto hv = Kokkos::View<const int*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_second_order.data(), K);
      auto sub = Kokkos::subview(d_second_order, Kokkos::make_pair(0, K));
      Kokkos::deep_copy(sub, hv);
    }
    t1 = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // --- Remap kernel ---
    Kokkos::fence();
    t0 = std::chrono::steady_clock::now();

    Kokkos::parallel_for("remap", K, KOKKOS_LAMBDA(int i) {
      int idx = d_second_order[i];
      d_output[d_first_order[idx]] = i;
      // Fill all positions that belong to this unique element
      for (idx++; idx < N && (i == K - 1 || idx != d_second_order[i + 1]); idx++) {
        d_output[d_first_order[idx]] = i;
      }
    });

    Kokkos::fence();
    t1 = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // --- Copy D→H (output) ---
    t0 = std::chrono::steady_clock::now();
    {
      auto hv = Kokkos::View<int*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_output.data(), N);
      Kokkos::deep_copy(hv, d_output);
    }
    t1 = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Views destroyed at end of scope → implicit dealloc
  }

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      offload_end - offload_start).count();

  printf("Average offload time: %f (s)\n", offload_time * 1e-9f / repeat);
  printf("Average execution time of memory allocation : %f (us)\n",
         (alloc_time * 1e-3f) / repeat);
  printf("Average execution time of memory deallocation : %f (us)\n",
         (dealloc_time * 1e-3f) / repeat);
  printf("Average execution time of data copy : %f (us)\n",
         (copy_time * 1e-3f) / repeat);
  printf("Average execution time of sequence : %f (us)\n",
         (seq_time * 1e-3f) / repeat);
  printf("Average execution time of sort-by-key : %f (us)\n",
         (sort_time * 1e-3f) / repeat);
  printf("Average execution time of unique-by-key : %f (us)\n",
         (unique_time * 1e-3f) / repeat);
  printf("Average execution time of remap kernel: %f (us)\n",
         (kernel_time * 1e-3f) / repeat);

  // Verify / checksum
  int cs1 = 0, cs2 = 0;
  for (int i = 0; i < N - 1; i++) cs1 ^= h_output[i] - h_output[i + 1];
  for (int i = 0; i < N;     i++) cs2 ^= h_output[i];
  printf("Checksum: %d %d\n", cs1, cs2);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    for (int i = 0; i < 2; i++) {
      printf("\nRun %d\n", i);
      eval_remap(N, repeat);
    }
  }
  Kokkos::finalize();
  return 0;
}
