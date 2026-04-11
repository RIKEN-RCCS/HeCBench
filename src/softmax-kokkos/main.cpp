#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <Kokkos_Core.hpp>

void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest) {
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 1; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of slices> <slice size> <repeat>\n", argv[0]);
    return 1;
  }

  const int numSlice  = atoi(argv[1]);
  const int sliceSize = atoi(argv[2]);
  const int repeat    = atoi(argv[3]);
  const int numElem   = numSlice * sliceSize;

  float* input      = (float*) malloc(sizeof(float) * numElem);
  float* output_gpu = (float*) malloc(sizeof(float) * numElem);
  float* output_cpu = (float*) malloc(sizeof(float) * numElem);

  srand(2);
  for (int i = 0; i < numSlice; i++)
    for (int j = 0; j < sliceSize; j++)
      input[i * sliceSize + j] = (float)(rand() % 13);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_input("d_input", numElem);
    Kokkos::View<float*> d_output("d_output", numElem);

    auto h_input = Kokkos::create_mirror_view(d_input);
    for (int i = 0; i < numElem; i++) h_input(i) = input[i];
    Kokkos::deep_copy(d_input, h_input);
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("softmax", numSlice,
        KOKKOS_LAMBDA(const int i) {
          float max_ = d_input(i * sliceSize);
          for (int j = 1; j < sliceSize; j++) {
            float val = d_input(i * sliceSize + j);
            if (val > max_) max_ = val;
          }
          float sum = 0.f;
          for (int j = 0; j < sliceSize; j++) {
            sum += expf(d_input(i * sliceSize + j) - max_);
          }
          for (int j = 0; j < sliceSize; j++) {
            d_output(i * sliceSize + j) =
              expf(d_input(i * sliceSize + j) - max_) / sum;
          }
        });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    auto h_output = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_output, d_output);
    for (int i = 0; i < numElem; i++) output_gpu[i] = h_output(i);
  }
  Kokkos::finalize();

  bool ok = true;
  softMax_cpu(numSlice, sliceSize, input, output_cpu);
  for (int i = 0; i < numElem; i++) {
    if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-3f) {
      printf("@index %d host: %f device: %f\n", i, output_cpu[i], output_gpu[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output_cpu);
  free(output_gpu);
  return 0;
}
