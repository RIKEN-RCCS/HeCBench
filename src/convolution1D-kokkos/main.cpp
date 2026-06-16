/*
 * 1D Convolution benchmark with multiple kernel variants.
 * Ported to Kokkos from the OMP target version.
 * Only the basic kernel is implemented (tiled versions omitted
 * as they require shared memory coordination).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256

template<typename T>
void conv1d(Kokkos::View<const T*> mask,
            Kokkos::View<const T*> in,
            Kokkos::View<T*>       out,
            int input_width, int mask_width)
{
  Kokkos::parallel_for("conv1d", input_width, KOKKOS_LAMBDA(int i) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      int idx = start + j;
      if (idx >= 0 && idx < input_width)
        s += in(idx) * mask(j);
    }
    out(i) = s;
  });
  Kokkos::fence();
}

template<typename T>
void reference(const T* h_in, T* h_out, const T* h_mask,
               int input_width, int mask_width)
{
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width)
        s += h_in[start + j] * h_mask[j];
    }
    h_out[i] = s;
  }
}

template<typename T>
void conv1D(int input_width, int mask_width, int repeat) {
  T *mask = (T*) malloc(mask_width * sizeof(T));
  T *a    = (T*) malloc(input_width * sizeof(T));
  T *b    = (T*) malloc(input_width * sizeof(T));

  srand(123);
  for (int i = 0; i < mask_width;  i++) mask[i] = (T)(rand() % 10 / 10.f);
  for (int i = 0; i < input_width; i++) a[i] = (T)(rand() % 10 / 10.f);

  Kokkos::View<T*> d_mask("d_mask", mask_width);
  Kokkos::View<T*> d_a("d_a", input_width);
  Kokkos::View<T*> d_b("d_b", input_width);

  auto h_mask = Kokkos::create_mirror_view(d_mask);
  auto h_a    = Kokkos::create_mirror_view(d_a);
  for (int i = 0; i < mask_width;  i++) h_mask(i) = mask[i];
  for (int i = 0; i < input_width; i++) h_a(i) = a[i];
  Kokkos::deep_copy(d_mask, h_mask);
  Kokkos::deep_copy(d_a, h_a);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    conv1d<T>(d_mask, d_a, d_b, input_width, mask_width);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d kernel: %f (us)\n", (time * 1e-3f) / repeat);

  auto h_b = Kokkos::create_mirror_view(d_b);
  Kokkos::deep_copy(h_b, d_b);
  for (int i = 0; i < input_width; i++) b[i] = h_b(i);

  // Verify
  T *ref = (T*) malloc(input_width * sizeof(T));
  reference(a, ref, mask, input_width, mask_width);
  bool ok = true;
  for (int i = 0; i < input_width; i++) {
    double diff = fabs((double)b[i] - (double)ref[i]);
    if (diff > 1e-3) { ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(mask); free(a); free(b); free(ref);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }
  int input_width = atoi(argv[1]);
  input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
      printf("\n---------------------\n");
      printf("Mask width: %d\n", mask_width);
      printf("1D convolution (FP64)\n");
      conv1D<double>(input_width, mask_width, repeat);
      printf("1D convolution (FP32)\n");
      conv1D<float>(input_width, mask_width, repeat);
    }
  }
  Kokkos::finalize();
  return 0;
}
