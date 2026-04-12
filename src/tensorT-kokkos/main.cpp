#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// 1,2,3,4,5,6 -> 2,3,4,6,1,5
static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;

void verify(double *input, double *output) {
  int input_offset  = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  bool error = false;
  for (size_t i = 0; i < d5; i++) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] !=
        output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      printf("FAIL\n");
      error = true;
      break;
    }
  }
  if (!error) printf("PASS\n");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    double *input  = new double[data_size]();
    double *output = new double[data_size]();
    for (int i = 0; i < data_size; i++) input[i] = i;

    Kokkos::View<double*> d_input("d_input", data_size);
    Kokkos::View<double*> d_output("d_output", data_size);

    {
      auto h = Kokkos::create_mirror_view(d_input);
      for (int i = 0; i < data_size; i++) h(i) = input[i];
      Kokkos::deep_copy(d_input, h);
    }

    const int nblocks   = d4 * d5 * d6;
    const int tile_size = d1 * d2 * d3;

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      // Eliminate the shared tile: output[offset1+offset2] = input[local_offset + block_idx*tile_size]
      // since tile[k] = input[k + block_idx*tile_size] in the original.
      Kokkos::parallel_for("tensorT",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nblocks, tile_size}),
        KOKKOS_LAMBDA(int block_idx, int i) {
          // Constants embedded (all are compile-time constants)
          const int so[3]  = {d2, d3, d1};
          const int si[3]  = {d4, d5, d6};
          const float so_r[3] = {1.0f/d2, 1.0f/d3, 1.0f/d1};
          const float si_r[3] = {1.0f/d4, 1.0f/d5, 1.0f/d6};
          const int stol[3]   = {d1, d1*d2, 1};
          const int stog[3]   = {1, d2, d2*d3*d4*d6};
          const int sti[3]    = {d2*d3, d2*d3*d4*d6*d1, d2*d3*d4};

          // offset1 from block_idx
          int it = block_idx, im = 0, offset1 = 0;
          for (int k = 0; k < 3; k++) {
            im = (int)(it * si_r[k]);
            offset1 += sti[k] * (it - im * si[k]);
            it = im;
          }

          // offset2 and local_offset from i
          it = i;
          int offset2 = 0, local_offset = 0;
          for (int j = 0; j < 3; j++) {
            im = (int)(it * so_r[j]);
            int tmp = it - im * so[j];
            offset2      += stog[j] * tmp;
            local_offset += stol[j] * tmp;
            it = im;
          }

          d_output(offset1 + offset2) = d_input(local_offset + block_idx * tile_size);
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

    {
      auto h = Kokkos::create_mirror_view(d_output);
      Kokkos::deep_copy(h, d_output);
      for (int i = 0; i < data_size; i++) output[i] = h(i);
    }

    verify(input, output);
    delete[] input;
    delete[] output;
  }
  Kokkos::finalize();
  return 0;
}
