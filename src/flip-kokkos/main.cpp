/*
 * Tensor flip (dimension reversal).
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <Kokkos_Core.hpp>

void property(const char* name, std::vector<int64_t> p)
{
  printf("%s: ( ", name);
  for (size_t i = 0; i < p.size(); i++) printf("%ld ", p[i]);
  printf(")\n");
}

// CPU reference
template <typename scalar_t>
void flip_kernel_cpu(const scalar_t* in_tensor, scalar_t* out_tensor, int64_t n,
                     const int64_t* flip_dims, int64_t flip_dims_size,
                     const int64_t* strides, const int64_t* strides_cont,
                     const int64_t* shape, int64_t total_dims)
{
  for (int64_t lid = 0; lid < n; lid++) {
    int64_t cur_indices = lid, rem = 0, dst_offset = 0;
    for (int64_t i = 0; i < total_dims; i++) {
      int64_t temp = cur_indices;
      cur_indices = cur_indices / strides_cont[i];
      rem = temp - cur_indices * strides_cont[i];
      for (int64_t j = 0; j < flip_dims_size; j++)
        if (i == flip_dims[j]) cur_indices = shape[i] - 1 - cur_indices;
      dst_offset += cur_indices * strides[i];
      cur_indices = rem;
    }
    out_tensor[lid] = in_tensor[dst_offset];
  }
}

template <typename scalar_t>
void flip(int64_t num_dims, int64_t num_flip_dims, int32_t dim_size, int32_t repeat)
{
  std::vector<int64_t> flip_v, shape_v, stride_v;

  for (int64_t i = 0; i < num_dims; i++) shape_v.push_back(dim_size);

  int64_t n = 1;
  for (int64_t i = 0; i < num_dims; i++) n *= shape_v[i];

  for (int64_t i = 0; i < num_flip_dims; i++) flip_v.push_back(i);

  // Row-major strides (3D only for simplicity)
  stride_v.push_back(shape_v[1] * shape_v[2]);
  stride_v.push_back(shape_v[2]);
  stride_v.push_back(1);

  property("shape", shape_v);
  property("flip_dims", flip_v);
  property("stride", stride_v);

  scalar_t *input  = (scalar_t*) malloc(n * sizeof(scalar_t));
  scalar_t *output = (scalar_t*) malloc(n * sizeof(scalar_t));
  scalar_t *output_ref = (scalar_t*) malloc(n * sizeof(scalar_t));
  for (int64_t i = 0; i < n; i++) input[i] = (scalar_t)i;

  Kokkos::View<scalar_t*> d_input("d_input", n);
  Kokkos::View<scalar_t*> d_output("d_output", n);
  Kokkos::View<int64_t*>  d_shape("d_shape", num_dims);
  Kokkos::View<int64_t*>  d_flip("d_flip", num_flip_dims);
  Kokkos::View<int64_t*>  d_stride("d_stride", num_dims);
  Kokkos::View<int64_t*>  d_stride_cont("d_stride_cont", num_dims);

  auto h_input       = Kokkos::create_mirror_view(d_input);
  auto h_shape       = Kokkos::create_mirror_view(d_shape);
  auto h_flip        = Kokkos::create_mirror_view(d_flip);
  auto h_stride      = Kokkos::create_mirror_view(d_stride);
  auto h_stride_cont = Kokkos::create_mirror_view(d_stride_cont);

  for (int64_t i = 0; i < n;            i++) h_input(i) = input[i];
  for (int64_t i = 0; i < num_dims;     i++) { h_shape(i) = shape_v[i]; h_stride(i) = stride_v[i]; h_stride_cont(i) = stride_v[i]; }
  for (int64_t i = 0; i < num_flip_dims; i++) h_flip(i) = flip_v[i];

  Kokkos::deep_copy(d_input, h_input);
  Kokkos::deep_copy(d_shape, h_shape);
  Kokkos::deep_copy(d_flip, h_flip);
  Kokkos::deep_copy(d_stride, h_stride);
  Kokkos::deep_copy(d_stride_cont, h_stride_cont);

  // Warmup + verify
  Kokkos::parallel_for("flip", n, KOKKOS_LAMBDA(int64_t linear_index) {
    int64_t cur_indices = linear_index, rem = 0, dst_offset = 0;
    for (int64_t i = 0; i < num_dims; i++) {
      int64_t temp = cur_indices;
      cur_indices = cur_indices / d_stride_cont(i);
      rem = temp - cur_indices * d_stride_cont(i);
      for (int64_t j = 0; j < num_flip_dims; j++)
        if (i == d_flip(j)) cur_indices = d_shape(i) - 1 - cur_indices;
      dst_offset += cur_indices * d_stride(i);
      cur_indices = rem;
    }
    d_output(linear_index) = d_input(dst_offset);
  });
  Kokkos::fence();

  auto h_output = Kokkos::create_mirror_view(d_output);
  Kokkos::deep_copy(h_output, d_output);
  for (int64_t i = 0; i < n; i++) output[i] = h_output(i);

  flip_kernel_cpu(input, output_ref, n, flip_v.data(), num_flip_dims,
                  stride_v.data(), stride_v.data(), shape_v.data(), num_dims);

  int error = memcmp(output, output_ref, n * sizeof(scalar_t));
  printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();
  for (int32_t i = 0; i < repeat; i++) {
    Kokkos::parallel_for("flip_bench", n, KOKKOS_LAMBDA(int64_t linear_index) {
      int64_t cur_indices = linear_index, rem = 0, dst_offset = 0;
      for (int64_t dim = 0; dim < num_dims; dim++) {
        int64_t temp = cur_indices;
        cur_indices = cur_indices / d_stride_cont(dim);
        rem = temp - cur_indices * d_stride_cont(dim);
        for (int64_t j = 0; j < num_flip_dims; j++)
          if (dim == d_flip(j)) cur_indices = d_shape(dim) - 1 - cur_indices;
        dst_offset += cur_indices * d_stride(dim);
        cur_indices = rem;
      }
      d_output(linear_index) = d_input(dst_offset);
    });
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the flip kernel: %f (ms)\n", (time * 1e-6f) / repeat);

  free(input); free(output); free(output_ref);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t num_dims  = atoi(argv[1]);
  const int64_t dim_size  = atoi(argv[2]);
  const int32_t repeat    = atoi(argv[3]);
  const int64_t num_flip_dims = num_dims;

  if (num_dims != 3) {
    printf("This implementation supports only 3D tensors.\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    printf("=========== Data type is FP32 ==========\n");
    flip<float>(num_dims, num_flip_dims, dim_size, repeat);

    printf("=========== Data type is FP64 ==========\n");
    flip<double>(num_dims, num_flip_dims, dim_size, repeat);
  }
  Kokkos::finalize();
  return 0;
}
