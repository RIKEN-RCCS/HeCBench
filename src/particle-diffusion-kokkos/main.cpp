/***************************************************************************
 * Kokkos port of particle-diffusion-sycl benchmark.
 * Original copyright 2020 Intel Corporation (MIT).
 **************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 3) {
      std::cout << " Incorrect number of parameters\n";
      std::cout << " Usage: " << argv[0]
                << " <Number of iterations within the kernel>"
                << " <Kernel execution count>\n\n";
      Kokkos::finalize();
      return 1;
    }

    int nIterations = std::stoi(argv[1]);
    int nRepeat     = std::stoi(argv[2]);

    const size_t grid_size   = 21;
    const size_t n_particles = 147456;
    const float  radius      = 0.5f;

    // Host grid (2-D)
    int** grid = new int*[grid_size];
    for (size_t i = 0; i < grid_size; i++) {
      grid[i] = new int[grid_size]();
    }

    // Host random arrays (allocated on host, copied to device)
    size_t nrand = n_particles * (size_t)nIterations;

    Kokkos::View<float*>  d_randomX("d_randomX",  nrand);
    Kokkos::View<float*>  d_randomY("d_randomY",  nrand);
    Kokkos::View<float*>  d_particleX("d_particleX", n_particles);
    Kokkos::View<float*>  d_particleY("d_particleY", n_particles);
    Kokkos::View<size_t*> d_map("d_map", n_particles * grid_size * grid_size);

    auto h_randomX   = Kokkos::create_mirror_view(d_randomX);
    auto h_randomY   = Kokkos::create_mirror_view(d_randomY);
    auto h_particleX = Kokkos::create_mirror_view(d_particleX);
    auto h_particleY = Kokkos::create_mirror_view(d_particleY);
    auto h_map       = Kokkos::create_mirror_view(d_map);

    const size_t scale = 100;
    srand(17);
    for (size_t i = 0; i < nrand; i++) {
      h_randomX(i) = (float)(rand() % scale);
      h_randomY(i) = (float)(rand() % scale);
    }
    for (size_t i = 0; i < n_particles; i++) {
      h_particleX(i) = 10.0f;
      h_particleY(i) = 10.0f;
    }
    for (size_t i = 0; i < n_particles * grid_size * grid_size; i++) h_map(i) = 0;

    Kokkos::deep_copy(d_randomX, h_randomX);
    Kokkos::deep_copy(d_randomY, h_randomY);

    std::cout << " The number of iterations is " << nIterations << std::endl;
    std::cout << " The number of kernel execution is " << nRepeat << std::endl;
    std::cout << " The number of particles is " << n_particles << std::endl;

    double time_total = 0.0;
    const int nIter_i    = nIterations;
    const size_t gs      = grid_size;
    const size_t np      = n_particles;

    for (int rep = 0; rep < nRepeat; rep++) {
      Kokkos::deep_copy(d_particleX, h_particleX);
      Kokkos::deep_copy(d_particleY, h_particleY);
      Kokkos::deep_copy(d_map, h_map);

      Kokkos::fence();
      auto t0 = std::chrono::steady_clock::now();

      Kokkos::parallel_for("motionsim", np, KOKKOS_LAMBDA(size_t ii) {
        float pX = d_particleX(ii);
        float pY = d_particleY(ii);
        size_t map_base = ii * gs * gs;

        for (int iter = 0; iter < nIter_i; iter++) {
          float randnumX = d_randomX((size_t)iter * np + ii);
          float randnumY = d_randomY((size_t)iter * np + ii);

          float displacementX = randnumX / 1000.0f - 0.0495f;
          float displacementY = randnumY / 1000.0f - 0.0495f;

          pX += displacementX;
          pY += displacementY;

          float dX = pX - Kokkos::trunc(pX);
          float dY = pY - Kokkos::trunc(pY);

          int iX = (int)Kokkos::floor(pX);
          int iY = (int)Kokkos::floor(pY);

          if ((pX < (float)gs) && (pY < (float)gs) && (pX >= 0.f) && (pY >= 0.f)) {
            if (dX * dX + dY * dY <= radius * radius)
              d_map(map_base + (size_t)iY * gs + (size_t)iX)++;
          }
        }

        d_particleX(ii) = pX;
        d_particleY(ii) = pY;
      });

      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();
      time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }

    std::cout << "\nAverage kernel execution time: "
              << (time_total * 1e-9) / nRepeat << " (s)" << std::endl;

    // Copy map back and accumulate into grid
    Kokkos::deep_copy(h_map, d_map);

    for (size_t i = 0; i < n_particles; i++) {
      for (size_t y = 0; y < grid_size; y++) {
        for (size_t x = 0; x < grid_size; x++) {
          size_t val = h_map(i * grid_size * grid_size + y * grid_size + x);
          if (val > 0) grid[y][x] += (int)val;
        }
      }
    }

    if (grid_size <= 64) {
      std::cout << "\n ********************** OUTPUT GRID: " << std::endl;
      for (size_t i = 0; i < grid_size; i++) {
        for (size_t j = 0; j < grid_size; j++)
          std::cout << std::setw(3) << grid[i][j] << " ";
        std::cout << std::endl;
      }
    }

    for (size_t i = 0; i < grid_size; i++) delete[] grid[i];
    delete[] grid;
  }
  Kokkos::finalize();
  return 0;
}
