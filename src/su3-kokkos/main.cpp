// SU3 matrix-matrix multiplication benchmark – Kokkos port
// Ported from su3-omp by replacing OpenMP target offload with Kokkos.

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <sys/resource.h>

typedef std::chrono::system_clock Clock;

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef LDIM
#  define LDIM 32
#endif
#ifndef PRECISION
#  define PRECISION 2
#endif

// ---- Type definitions using Kokkos::complex for device compatibility --------

#if (PRECISION == 1)
  using Real   = float;
  using Complx = Kokkos::complex<float>;
  struct fsu3_matrix { Kokkos::complex<float> e[3][3]; };
  using su3_matrix = fsu3_matrix;
#else
  using Real   = double;
  using Complx = Kokkos::complex<double>;
  struct dsu3_matrix { Kokkos::complex<double> e[3][3]; };
  using su3_matrix = dsu3_matrix;
#endif

#define EVEN 0x02
#define ODD  0x01

struct site {
  su3_matrix link[4];
  int x, y, z, t;
  int index;
  char parity;
#if (PRECISION == 1)
  int pad[2];
#else
  int pad[10];
#endif
};

// ---- Global benchmark settings (mirroring original) -------------------------

unsigned int verbose = 1;
size_t       warmups = 1;

// ---- Helper: almost_equal for verification ----------------------------------

template <class T>
bool almost_equal(T x, T y, double tol)
{
  if (std::isnan(x) || std::isnan(y)) return false;
  return std::abs(x - y) < tol;
}

template <class T>
bool almost_equal(Kokkos::complex<T> x, Kokkos::complex<T> y, double tol)
{
  if (std::isnan(x.real()) || std::isnan(x.imag()) ||
      std::isnan(y.real()) || std::isnan(y.imag()))
    return false;
  T dr = x.real() - y.real();
  T di = x.imag() - y.imag();
  return std::sqrt(dr * dr + di * di) < tol;
}

// ---- Lattice initialisation (host only) -------------------------------------

void init_link(su3_matrix *s, Complx val) {
  for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 3; ++k)
      for (int l = 0; l < 3; ++l)
        s[j].e[k][l] = val;
}

void make_lattice(site *s, size_t n, Complx val) {
  int nx = n, ny = n, nz = n, nt = n;
  for (int t = 0; t < nt; t++) {
    int i = t * nz * ny * nx;
    for (int z = 0; z < nz; z++)
      for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++, i++) {
          s[i].x = x; s[i].y = y; s[i].z = z; s[i].t = t;
          s[i].index = x + nx * (y + ny * (z + nz * t));
          s[i].parity = ((x + y + z + t) % 2 == 0) ? EVEN : ODD;
          init_link(&s[i].link[0], val);
        }
  }
}

// ---- Kokkos SU3 kernel + timing ---------------------------------------------

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b,
                  std::vector<site> &c,
                  size_t total_sites, size_t iterations,
                  size_t threads_per_team, int /*use_device*/)
{
  if (threads_per_team == 0)
    threads_per_team = 36;

  const size_t num_work_items = total_sites * threads_per_team;

  if (verbose >= 1) {
    std::cout << "Number of teams = " << total_sites << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
    std::cout << "Number of work items = " << num_work_items << std::endl;
  }

  // Allocate device Views
  using DevView_site = Kokkos::View<site *, Kokkos::DefaultExecutionSpace>;
  using DevView_mat  = Kokkos::View<su3_matrix *, Kokkos::DefaultExecutionSpace>;

  DevView_site d_a("d_a", total_sites);
  DevView_site d_c("d_c", total_sites);
  DevView_mat  d_b("d_b", 4);

  // Create host mirrors and populate them
  auto h_a = Kokkos::create_mirror_view(d_a);
  auto h_b = Kokkos::create_mirror_view(d_b);

  for (size_t i = 0; i < total_sites; i++) h_a(i) = a[i];
  for (int  i = 0; i < 4;            i++) h_b(i) = b[i];

  Kokkos::deep_copy(d_a, h_a);
  Kokkos::deep_copy(d_b, h_b);

  // Raw device pointers – valid inside KOKKOS_LAMBDA when using CUDA/HIP etc.
  site       *ra = d_a.data();
  site       *rc = d_c.data();
  su3_matrix *rb = d_b.data();

  // Benchmark loop
  auto tstart = Clock::now();
  for (size_t iters = 0; iters < iterations + warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();

    Kokkos::parallel_for(
      "su3_mat_nn",
      Kokkos::RangePolicy<>(0, (int)num_work_items),
      KOKKOS_LAMBDA(int id) {
        int i = id / 36;
        if (i < (int)total_sites) {
          int j = (id % 36) / 9;
          int k = (id % 9)  / 3;
          int l =  id % 3;

          Complx cc(0.0, 0.0);
          for (int m = 0; m < 3; m++)
            cc += ra[i].link[j].e[k][m] * rb[j].e[m][l];
          rc[i].link[j].e[k][l] = cc;
        }
      });
    Kokkos::fence();
  }

  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - tstart).count();

  // Copy result back to host
  auto h_c = Kokkos::create_mirror_view(d_c);
  Kokkos::deep_copy(h_c, d_c);
  for (size_t i = 0; i < total_sites; i++) c[i] = h_c(i);

  // Checksum (identical logic to original)
  double sum = 0.0;
  for (int i = 0; i < (int)total_sites; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          sum += c[i].link[j].e[k][l].real();
  sum /= (double)total_sites;

  if (almost_equal(sum, 4.0 * sizeof(su3_matrix) / sizeof(Complx), 1E-6))
    printf("Checksum SUCCESS... checksum=%.0lf\n", sum);
  else
    printf("Checksum FAILURE\n");

  return ttotal / 1.0e6;
}

// ---- Main -------------------------------------------------------------------

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    size_t iterations      = ITERATIONS;
    size_t ldim            = LDIM;
    size_t threads_per_group = 128;
    int    device          = -1;

    int opt;
    while ((opt = getopt(argc, argv, ":hi:l:t:v:d:w:n:")) != -1) {
      switch (opt) {
      case 'i': iterations       = atoi(optarg); break;
      case 'l': ldim             = atoi(optarg); break;
      case 't': threads_per_group = atoi(optarg); break;
      case 'v': verbose          = atoi(optarg); break;
      case 'd': device           = atoi(optarg); break;
      case 'w': warmups          = atoi(optarg); break;
      case 'h':
        fprintf(stderr,
          "Usage: %s [-i iterations] [-l lattice dim] "
          "[-t threads/group] [-d device] [-v verbosity] [-w warmups]\n",
          argv[0]);
        Kokkos::finalize();
        return 1;
      }
    }

    size_t total_sites = ldim * ldim * ldim * ldim;
    std::vector<site>       a(total_sites), c(total_sites);
    std::vector<su3_matrix> b(4);

    make_lattice(a.data(), ldim, Complx{1.0, 0.0});
    init_link(b.data(), Complx{1.0 / 3.0, 0.0});

    if (verbose >= 1) {
      printf("Number of sites = %zu^4\n", ldim);
      printf("Executing %zu iterations with %zu warmups\n", iterations, warmups);
      if (threads_per_group != 0)
        printf("Threads per group = %zu\n", threads_per_group);
    }

    const double ttotal =
      su3_mat_nn(a, b, c, total_sites, iterations, threads_per_group, device);

    if (verbose >= 1)
      printf("Total kernel execution time = %f (s)\n", ttotal);

    const double tflop = (double)iterations * total_sites * 864.0;
    printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);

    const double memory_usage =
      (double)sizeof(site) * (a.capacity() + c.capacity()) +
      (double)sizeof(su3_matrix) * b.capacity();
    printf("Total GByte/s (GPU memory)  = %.3f\n",
           iterations * memory_usage / ttotal / 1.0e9);
    fflush(stdout);

    // Verification
    for (int i = 0; i < (int)total_sites; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 3; ++k)
          for (int l = 0; l < 3; ++l) {
            Complx cc(0.0, 0.0);
            for (int m = 0; m < 3; m++)
              cc += a[i].link[j].e[k][m] * b[j].e[m][l];
            assert(almost_equal(c[i].link[j].e[k][l], cc, 1E-6));
          }

    if (verbose >= 2) {
      printf("Total allocation for matrices = %.3f MiB\n",
             memory_usage / 1048576.0);
      struct rusage usage;
      if (getrusage(RUSAGE_SELF, &usage) == 0)
        printf("Approximate memory usage = %.3f MiB\n",
               (float)usage.ru_maxrss / 1024.0);
    }
  }
  Kokkos::finalize();
  return 0;
}
