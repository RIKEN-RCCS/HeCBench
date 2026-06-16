#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>

// Inlined helpers from glu-cuda/reference.h
inline int64_t size_from_dim(int k, std::vector<int> &dims) {
  int64_t r = 1;
  for (size_t i = k; i < dims.size(); i++) r *= dims[i];
  return r;
}

inline int64_t size_to_dim(int k, std::vector<int> &dims) {
  int64_t r = 1;
  for (int i = 0; i < k; i++) r *= dims[i];
  return r;
}

static float sigmoid_ref(float x) {
  if (x >= 0) return 1.f / (1.f + expf(-x));
  const float e = expf(x);
  return e / (1.f + e);
}

void ComputeGlu(int M, int split_dim, int N, const float *Xdata, float *Ydata) {
  const int yStride = split_dim * N;
  const int xStride = 2 * yStride;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < split_dim; j++) {
      for (int k = 0; k < N; k++) {
        float x1 = Xdata[i * xStride + j * N + k];
        float x2 = Xdata[i * xStride + (j + split_dim) * N + k];
        Ydata[i * yStride + j * N + k] = x1 * sigmoid_ref(x2);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int ndims    = atoi(argv[1]);
  const int dim_size = atoi(argv[2]);
  const int repeat   = atoi(argv[3]);

  std::vector<int> Xshape(ndims, dim_size);
  std::vector<int> Yshape = Xshape;

  printf("Shape of input tensor: ( ");
  for (int i = 0; i < ndims; i++) printf("%d ", Xshape[i]);
  printf(")\n");

  uint64_t nelems = size_from_dim(0, Xshape);
  float *X     = (float*)malloc(nelems * sizeof(float));
  float *Y_ref = (float*)malloc(nelems * sizeof(float));

  std::default_random_engine generator(123);
  std::uniform_real_distribution<float> distribution(-6.f, 6.f);
  for (uint64_t i = 0; i < nelems; i++) X[i] = distribution(generator);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_X("d_X", nelems);
    Kokkos::View<float*> d_Y("d_Y", nelems);

    {
      auto hX = Kokkos::create_mirror_view(d_X);
      for (uint64_t i = 0; i < nelems; i++) hX(i) = X[i];
      Kokkos::deep_copy(d_X, hX);
    }

    for (int input_dim = -1; input_dim < 3 * (ndims - 1); input_dim++) {
      const int split_index = (input_dim == -1) ? ndims - 1 : (input_dim % ndims);

      if (Yshape[split_index] % 2 != 0) {
        printf("Split dimension %d should be divided by two. Skip\n", Yshape[split_index]);
        continue;
      }

      const int split_dim_size = Yshape[split_index] / 2;
      const int m = (int)size_to_dim(split_index, Xshape);
      const int n = (int)size_from_dim(split_index + 1, Xshape);

      ComputeGlu(m, split_dim_size, n, X, Y_ref);

      auto start = std::chrono::steady_clock::now();

      for (int iter = 0; iter < repeat; iter++) {
        const int total = m * split_dim_size * n;
        Kokkos::parallel_for("glu", total,
          KOKKOS_LAMBDA(int index) {
            const int xOffset = 2 * split_dim_size * n;
            const int yOffset =     split_dim_size * n;
            const int i = index / split_dim_size / n;
            const int j = index / n % split_dim_size;
            const int k = index % n;
            const float x1 = d_X(i * xOffset +                   j * n + k);
            const float x2 = d_X(i * xOffset + (j + split_dim_size) * n + k);
            d_Y(i * yOffset + j * n + k) = x1 * (1.f / (1.f + Kokkos::exp(-x2)));
          });
        Kokkos::fence();
      }

      auto end  = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of GLU kernel (split dimension = %d): %f (us)\n",
             split_index, (time * 1e-3f) / repeat);

      auto hY = Kokkos::create_mirror_view(d_Y);
      Kokkos::deep_copy(hY, d_Y);

      bool ok = true;
      for (uint64_t i = 0; i < nelems / 2; i++) {
        if (fabsf(hY(i) - Y_ref[i]) > 1e-3f) { ok = false; break; }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }
  }
  Kokkos::finalize();

  free(X);
  free(Y_ref);
  return 0;
}
