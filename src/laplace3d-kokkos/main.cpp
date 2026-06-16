//
// Program to solve Laplace equation on a regular 3D grid
// Ported to Kokkos from the OMP target version.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>

// 6-point stencil Laplace kernel
void laplace3d(int NX, int NY, int NZ,
               Kokkos::View<const float*> u1,
               Kokkos::View<float*> u2)
{
  Kokkos::parallel_for("laplace3d",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NZ, NY, NX}),
      KOKKOS_LAMBDA(int k, int j, int i) {
        int ind = i + j * NX + k * NX * NY;
        if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) {
          u2(ind) = u1(ind);  // Dirichlet b.c.'s
        } else {
          u2(ind) = (u1(ind - 1)     + u1(ind + 1)
                   + u1(ind - NX)    + u1(ind + NX)
                   + u1(ind - NX*NY) + u1(ind + NX*NY)) * (1.0f / 6.0f);
        }
      });
  Kokkos::fence();
}

void reference(int NX, int NY, int NZ, float* h_u1, float* h_u2) {
  const float sixth = 1.0f / 6.0f;
  for (int k = 0; k < NZ; k++) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {
        int ind = i + j * NX + k * NX * NY;
        if (i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1)
          h_u2[ind] = h_u1[ind];
        else
          h_u2[ind] = (h_u1[ind-1]     + h_u1[ind+1]
                     + h_u1[ind-NX]    + h_u1[ind+NX]
                     + h_u1[ind-NX*NY] + h_u1[ind+NX*NY]) * sixth;
      }
    }
  }
}

void printHelp(void) {
  printf("Usage: laplace3d NX NY NZ REPEAT VERIFY\n");
  printf("6-point stencil 3D Laplace test\n");
  printf("Example: ./main 256 128 128 100 1\n");
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printHelp();
    return 1;
  }

  const int NX     = atoi(argv[1]);
  const int NY     = atoi(argv[2]);
  const int NZ     = atoi(argv[3]);
  const int REPEAT = atoi(argv[4]);
  const int verify = atoi(argv[5]);

  if (NX <= 0 || NX % 32 != 0 || NY <= 0 || NZ <= 0 || REPEAT <= 0) return 1;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);
  printf("Result verification %s\n", verify ? "enabled" : "disabled");

  const size_t grid3D_size  = (size_t)NX * NY * NZ;
  const size_t grid3D_bytes = grid3D_size * sizeof(float);

  float *h_u1 = (float*) malloc(grid3D_bytes);
  float *h_u2 = (float*) malloc(grid3D_bytes);
  float *h_u3 = (float*) malloc(grid3D_bytes);

  for (int k = 0; k < NZ; k++) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {
        int ind = i + j * NX + k * NX * NY;
        h_u2[ind] = h_u1[ind] =
            (i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) ? 1.0f : 0.0f;
      }
    }
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_u1("d_u1", grid3D_size);
    Kokkos::View<float*> d_u2("d_u2", grid3D_size);

    auto hv_u1 = Kokkos::create_mirror_view(d_u1);
    for (size_t i = 0; i < grid3D_size; i++) hv_u1(i) = h_u1[i];
    Kokkos::deep_copy(d_u1, hv_u1);

    // Warmup
    laplace3d(NX, NY, NZ, d_u1, d_u2);

    auto start = std::chrono::steady_clock::now();
    for (int i = 1; i <= REPEAT; ++i) {
      laplace3d(NX, NY, NZ, d_u1, d_u2);
      Kokkos::deep_copy(d_u1, d_u2);  // swap: copy d_u2 -> d_u1
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / REPEAT);

    auto hv_u1_out = Kokkos::create_mirror_view(d_u1);
    Kokkos::deep_copy(hv_u1_out, d_u1);
    for (size_t i = 0; i < grid3D_size; i++) h_u1[i] = hv_u1_out(i);
  }
  Kokkos::finalize();

  if (verify) {
    for (int i = 1; i <= REPEAT; ++i) {
      reference(NX, NY, NZ, h_u2, h_u3);
      std::swap(h_u2, h_u3);
    }

    float err = 0.f;
    for (int k = 0; k < NZ; k++)
      for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++) {
          int ind = i + j * NX + k * NX * NY;
          err += (h_u1[ind] - h_u2[ind]) * (h_u1[ind] - h_u2[ind]);
        }
    printf("\nrms error = %f\n", sqrtf(err / (NX * NY * NZ)));
  }

  free(h_u1);
  free(h_u2);
  free(h_u3);
  return 0;
}
