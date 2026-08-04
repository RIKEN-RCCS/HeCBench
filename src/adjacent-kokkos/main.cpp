#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

template <int BLOCK_THREADS>
void BlockAdjDiffKernel(const Kokkos::View<const int*>& d_in,
                        const Kokkos::View<int*>& d_out,
                        bool subtract_left,
                        int num_items)
{
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  const int grid_size = num_items / items_per_block;

  Kokkos::parallel_for("BlockAdjDiffKernel",
    Kokkos::RangePolicy<>(0, grid_size * items_per_block),
    KOKKOS_LAMBDA(const int gid) {
      const int local = gid % items_per_block;
      if (subtract_left) {
        d_out(gid) = (local == 0) ? d_in(gid) : d_in(gid) - d_in(gid - 1);
      } else {
        d_out(gid) = (local + 1 == items_per_block) ? d_in(gid) : d_in(gid) - d_in(gid + 1);
      }
    });
}

void Initialize(int* h_in, int num_items)
{
  for (int i = 0; i < num_items; ++i) {
    h_in[i] = i % 17;
  }
}

template <int BLOCK_THREADS>
void Test(int num_items, int repeat)
{
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  num_items = (num_items + items_per_block - 1) / items_per_block * items_per_block;
  const int grid_size = num_items / items_per_block;

  int* h_in = new int[num_items];
  int* h_out = new int[num_items];
  int* r_out = new int[num_items];

  Initialize(h_in, num_items);

  Kokkos::View<int*> d_in("d_in", num_items);
  Kokkos::View<int*> d_tmp("d_tmp", num_items);
  Kokkos::View<int*> d_out("d_out", num_items);

  auto h_in_view = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h_in, num_items);
  auto h_out_view = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h_out, num_items);
  Kokkos::deep_copy(d_in, h_in_view);

  for (int i = 0; i < repeat; i++) {
    BlockAdjDiffKernel<BLOCK_THREADS>(d_in, d_out, true, num_items);
  }
  Kokkos::deep_copy(h_out_view, d_out);

  for (int b = 0; b < grid_size; b++) {
    const int* in = h_in + b * items_per_block;
    int* out = r_out + b * items_per_block;
    for (int i = 0; i < items_per_block; i++) {
      out[i] = (i == 0) ? in[i] : in[i] - in[i - 1];
    }
  }

  int compare = std::memcmp(r_out, h_out, sizeof(int) * num_items);
  std::printf("%s\n", compare ? "FAIL" : "PASS");

  for (int i = 0; i < repeat; i++) {
    BlockAdjDiffKernel<BLOCK_THREADS>(d_in, d_out, false, num_items);
  }
  Kokkos::deep_copy(h_out_view, d_out);

  for (int b = 0; b < grid_size; b++) {
    const int* in = h_in + b * items_per_block;
    int* out = r_out + b * items_per_block;
    for (int i = 0; i < items_per_block; i++) {
      out[i] = (i + 1 == items_per_block) ? in[i] : in[i] - in[i + 1];
    }
  }

  compare = std::memcmp(r_out, h_out, sizeof(int) * num_items);
  std::printf("%s\n", compare ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    BlockAdjDiffKernel<BLOCK_THREADS>(d_in, d_tmp, true, num_items);
    BlockAdjDiffKernel<BLOCK_THREADS>(d_tmp, d_out, false, num_items);
  }
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the kernels (thread block size = %4d): %f (us)\n",
              BLOCK_THREADS, (time * 1e-3f) / repeat);

  delete[] h_in;
  delete[] h_out;
  delete[] r_out;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    Test<64>(nelems, repeat);
    Test<128>(nelems, repeat);
    Test<256>(nelems, repeat);
    Test<512>(nelems, repeat);
    Test<1024>(nelems, repeat);
  }
  Kokkos::finalize();

  return 0;
}
