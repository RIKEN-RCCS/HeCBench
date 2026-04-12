/*
** PROGRAM: heat equation solve
**
** PURPOSE: This program will explore use of an explicit
**          finite difference method to solve the heat
**          equation under a method of manufactured solution (MMS)
**          scheme. The solution has been set to be a simple
**          function based on exponentials and trig functions.
**
**          A finite difference scheme is used on a 1000x1000 cube.
**          A total of 0.5 units of time are simulated.
**
**          The MMS solution has been adapted from
**          G.W. Recktenwald (2011). Finite difference approximations
**          to the Heat Equation. Portland State University.
**
**
** USAGE:   Run with two arguments:
**          First is the number of cells.
**          Second is the number of timesteps.
**
**          For example, with 100x100 cells and 10 steps:
**
**          ./main 100 10
**
**
** HISTORY: Written by Tom Deakin, Oct 2018
**          Ported to SYCL by Tom Deakin, Nov 2019
**          Ported to OpenCL by Tom Deakin, Jan 2020
**          Ported to Kokkos, 2024
**
*/

#include <iostream>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

#define PI acos(-1.0)
#define LINE "--------------------"

double solution(const double t, const double x, const double y,
                const double alpha, const double length) {
  return exp(-2.0 * alpha * PI * PI * t / (length * length)) *
         sin(PI * x / length) * sin(PI * y / length);
}

double l2norm(const int n, const double *u, const int nsteps, const double dt,
              const double alpha, const double dx, const double length) {
  double time = dt * (double)nsteps;
  double l2 = 0.0;
  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, x, y, alpha, length);
      l2 += (u[i + j * n] - answer) * (u[i + j * n] - answer);
      x += dx;
    }
    y += dx;
  }
  return sqrt(l2);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto start = std::chrono::high_resolution_clock::now();

    int n = 1000;
    int nsteps = 10;

    if (argc == 3) {
      n = atoi(argv[1]);
      if (n < 0) {
        std::cerr << "Error: n must be positive" << std::endl;
        Kokkos::finalize();
        return EXIT_FAILURE;
      }
      nsteps = atoi(argv[2]);
      if (nsteps < 0) {
        std::cerr << "Error: nsteps must be positive" << std::endl;
        Kokkos::finalize();
        return EXIT_FAILURE;
      }
    }

    double alpha  = 0.1;
    double length = 1000.0;
    double dx     = length / (n + 1);
    double dt     = 0.5 / nsteps;
    double r      = alpha * dt / (dx * dx);

    std::cout
        << std::endl
        << " MMS heat equation" << std::endl << std::endl
        << LINE << std::endl
        << "Problem input" << std::endl << std::endl
        << " Grid size: " << n << " x " << n << std::endl
        << " Cell width: " << dx << std::endl
        << " Grid length: " << length << "x" << length << std::endl
        << std::endl
        << " Alpha: " << alpha << std::endl
        << std::endl
        << " Steps: " << nsteps << std::endl
        << " Total time: " << dt * (double)nsteps << std::endl
        << " Time step: " << dt << std::endl
        << LINE << std::endl;

    std::cout << "Stability" << std::endl << std::endl;
    std::cout << " r value: " << r << std::endl;
    if (r > 0.5)
      std::cout << " Warning: unstable" << std::endl;
    std::cout << LINE << std::endl;

    Kokkos::View<double *> u("u", n * n);
    Kokkos::View<double *> u_tmp("u_tmp", n * n);

    // Initial value kernel
    Kokkos::parallel_for(
        "initial_value", n * n, KOKKOS_LAMBDA(int idx) {
          int i = idx % n;
          int j = idx / n;
          double x = dx * (i + 1);
          double y = dx * (j + 1);
          u(idx) = sin(PI * x / length) * sin(PI * y / length);
        });

    // Zero u_tmp
    Kokkos::parallel_for(
        "zero", n * n, KOKKOS_LAMBDA(int idx) { u_tmp(idx) = 0.0; });

    Kokkos::fence();

    const double r2 = 1.0 - 4.0 * r;

    auto tic = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < nsteps; ++t) {
      Kokkos::parallel_for(
          "solve", n * n, KOKKOS_LAMBDA(int idx) {
            int i = idx % n;
            int j = idx / n;
            u_tmp(idx) =
                r2 * u(idx) +
                r * ((i < n - 1) ? u(idx + 1) : 0.0) +
                r * ((i > 0)     ? u(idx - 1) : 0.0) +
                r * ((j < n - 1) ? u(idx + n) : 0.0) +
                r * ((j > 0)     ? u(idx - n) : 0.0);
          });
      Kokkos::fence();

      // Pointer swap via deep_copy is expensive; swap the view labels instead
      auto tmp = u;
      u     = u_tmp;
      u_tmp = tmp;
    }

    Kokkos::fence();
    auto toc = std::chrono::high_resolution_clock::now();

    // Copy result to host
    auto u_host = Kokkos::create_mirror_view(u);
    Kokkos::deep_copy(u_host, u);

    double norm = l2norm(n, u_host.data(), nsteps, dt, alpha, dx, length);

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout
        << "Results" << std::endl << std::endl
        << "Error (L2norm): " << norm << std::endl
        << "Solve time (s): "
        << std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count()
        << std::endl
        << "Total time (s): "
        << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count()
        << std::endl
        << "Bandwidth (GB/s): "
        << 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) /
               std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count()
        << std::endl
        << LINE << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
