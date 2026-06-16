#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
float michalewicz(const float* xValues, const int dim) {
  float result = 0;
  for (int i = 0; i < dim; ++i) {
    float a = sinf(xValues[i]);
    float b = sinf(((i + 1) * xValues[i] * xValues[i]) / (float)M_PI);
    float c = powf(b, 20); // m = 10
    result += a * c;
  }
  return -1.0f * result;
}

void Error(float value, int dim) {
  printf("Global minima = %f\n", value);
  float trueMin = 0.0f;
  if      (dim == 2)  trueMin = -1.8013f;
  else if (dim == 5)  trueMin = -4.687658f;
  else if (dim == 10) trueMin = -9.66015f;
  printf("Error = %f\n", fabsf(trueMin - value));
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n      = atol(argv[1]);
  const int    repeat = atoi(argv[2]);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0f, 4.0f);

  const int dims[] = {2, 5, 10};

  Kokkos::initialize(argc, argv);
  {
    for (int d = 0; d < 3; d++) {
      const int    dim  = dims[d];
      const size_t size = n * dim;

      // Generate random host data
      Kokkos::View<float*, Kokkos::HostSpace> h_values("h_values", size);
      for (size_t i = 0; i < size; i++) h_values(i) = dis(gen);

      Kokkos::View<float*> d_values("d_values", size);
      Kokkos::deep_copy(d_values, h_values);

      float minValue = 0.0f;

      Kokkos::fence();
      auto t_start = std::chrono::steady_clock::now();

      for (int r = 0; r < repeat; r++) {
        Kokkos::parallel_reduce(
          "michalewicz",
          n,
          KOKKOS_LAMBDA(const size_t j, float& local_min) {
            float val = michalewicz(d_values.data() + j * dim, dim);
            if (val < local_min) local_min = val;
          },
          Kokkos::Min<float>(minValue));
      }

      Kokkos::fence();
      auto t_end = std::chrono::steady_clock::now();
      auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
      printf("Average execution time of kernel (dim = %d): %f (us)\n",
             dim, (time * 1e-3f) / repeat);

      Error(minValue, dim);
    }
  }
  Kokkos::finalize();
  return 0;
}
