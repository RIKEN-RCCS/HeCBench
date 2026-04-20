// Lindblad master equation / quantum dynamics – Kokkos port
// Consolidates hexciton-omp/{main,kernels,utils,reference}.cpp + utils.hpp
// Copyright (c) 2015 Matthias Noack (ma.noack.pr@gmail.com)
// Distributed under the Boost Software License, Version 1.0.

#include <Kokkos_Core.hpp>
#include <complex>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ─── Types & compile-time constants ──────────────────────────────────────────
#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
  using real_t = float;
  struct real_2_t { float x, y; };
  #define hbar  (1.f / acosf(-1.f))
  #define dt    1e-3f
  #define hdt   (dt / hbar)
#else
  using real_t = double;
  struct real_2_t { double x, y; };
  #define hbar  (1.0 / acos(-1.0))
  #define dt    1e-3
  #define hdt   (dt / hbar)
#endif

using complex_t = std::complex<real_t>;

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 1001
#endif
#ifndef NUM_WARMUP
#define NUM_WARMUP 1
#endif
#ifndef DIM
#define DIM 7
#endif
#ifndef NUM
#define NUM 2048
#endif
#ifndef DEFAULT_ALIGNMENT
#define DEFAULT_ALIGNMENT 64
#endif
#ifndef VEC_LENGTH_AUTO
#define VEC_LENGTH_AUTO 4
#endif
#ifndef PACKAGES_PER_WG
#define PACKAGES_PER_WG 64
#endif

template<typename T>
T* allocate_aligned(size_t size, size_t alignment = DEFAULT_ALIGNMENT)
{
  T* ptr = nullptr;
  if (posix_memalign((void**)&ptr, alignment, size * sizeof(T)))
    std::cerr << "posix_memalign failed\n";
  return ptr;
}

// ─── Host utilities ───────────────────────────────────────────────────────────

void initialise_sigma(complex_t* sigma_in, complex_t* sigma_out,
                      size_t dim, size_t num)
{
  size_t size_sigma = dim * dim;
  for (size_t sigma_id = 0; sigma_id < num; ++sigma_id)
    for (size_t i = 0; i < size_sigma; ++i) {
      real_t x = (real_t)sigma_id / num;
      real_t y = (real_t)i / size_sigma;
      sigma_in [sigma_id * size_sigma + i] = complex_t(x - y, y - x);
      sigma_out[sigma_id * size_sigma + i] = complex_t(0.0, 0.0);
    }
}

void initialise_hamiltonian(complex_t* hamiltonian, size_t dim)
{
  size_t size = dim * dim;
  for (size_t i = 0; i < size; ++i)
    hamiltonian[i] = 1.0 - (real_t)i / size;
}

void transform_matrix_scale_aos(complex_t* matrix, size_t dim)
{
  for (size_t i = 0; i < dim * dim; ++i)
    matrix[i] *= hdt;
}

void transform_matrix_aos_to_soa(complex_t* matrix, size_t dim)
{
  size_t size = dim * dim;
  complex_t* tmp = new complex_t[size];
  std::memcpy(tmp, matrix, sizeof(complex_t) * size);
  real_t* mr = reinterpret_cast<real_t*>(matrix);
  for (size_t i = 0; i < dim; ++i)
    for (size_t j = 0; j < dim; ++j) {
      mr[i * dim + j]        = tmp[i * dim + j].real();
      mr[size + i * dim + j] = tmp[i * dim + j].imag();
    }
  delete[] tmp;
}

// ─── Reference CPU implementation (correctness check) ────────────────────────

void commutator_reference(complex_t* sigma_in, complex_t* sigma_out,
                          complex_t* hamiltonian, size_t dim, size_t num_sigma)
{
  const size_t size_sigma = dim * dim;
  for (size_t n = 0; n < num_sigma; ++n) {
    size_t sigma_id = n * size_sigma;
    for (size_t i = 0; i < dim; ++i)
      for (size_t j = 0; j < dim; ++j) {
        complex_t tmp(0.0, 0.0);
        for (size_t k = 0; k < dim; ++k)
          tmp += hamiltonian[i*dim+k] * sigma_in[sigma_id + k*dim+j]
               - sigma_in[sigma_id + i*dim+k] * hamiltonian[k*dim+j];
        sigma_out[sigma_id + i*dim+j] -= complex_t(0.0,1.0) * (real_t)hdt * tmp;
      }
  }
}

real_t compare_matrices(complex_t* a, complex_t* b, size_t dim, size_t num)
{
  real_t dev = 0.0;
  size_t size = num * dim * dim;
  for (size_t i = 0; i < size; ++i)
    dev += std::abs(a[i] - b[i]);
  return dev;
}

// ─── Kokkos benchmark kernel (kernels 0–3) ────────────────────────────────────

long run_benchmark(
    complex_t* h_sigma_in_cmplx,
    complex_t* h_sigma_out_cmplx,
    complex_t* h_hamiltonian_cmplx,
    size_t size_sigma,
    size_t size_hamiltonian,
    int dim, int num,
    int kernel_id,
    bool scale_hamiltonian,
    bool hamiltonian_soa)
{
  initialise_hamiltonian(h_hamiltonian_cmplx, dim);
  if (scale_hamiltonian)
    transform_matrix_scale_aos(h_hamiltonian_cmplx, dim);
  if (hamiltonian_soa)
    transform_matrix_aos_to_soa(h_hamiltonian_cmplx, dim);

  initialise_sigma(h_sigma_in_cmplx, h_sigma_out_cmplx, dim, num);

  // Extract real_2_t arrays
  real_2_t* h_ham  = allocate_aligned<real_2_t>(size_hamiltonian);
  real_2_t* h_sin  = allocate_aligned<real_2_t>(size_sigma);
  real_2_t* h_sout = allocate_aligned<real_2_t>(size_sigma);

  for (size_t i = 0; i < size_hamiltonian; i++) {
    h_ham[i].x  = h_hamiltonian_cmplx[i].real();
    h_ham[i].y  = h_hamiltonian_cmplx[i].imag();
  }
  for (size_t i = 0; i < size_sigma; i++) {
    h_sin[i].x  = h_sigma_in_cmplx[i].real();
    h_sin[i].y  = h_sigma_in_cmplx[i].imag();
    h_sout[i].x = h_sigma_out_cmplx[i].real();
    h_sout[i].y = h_sigma_out_cmplx[i].imag();
  }

  // Device Views
  Kokkos::View<real_2_t*> d_ham ("ham",  size_hamiltonian);
  Kokkos::View<real_2_t*> d_sin ("sin",  size_sigma);
  Kokkos::View<real_2_t*> d_sout("sout", size_sigma);

  {
    auto hv_ham  = Kokkos::create_mirror_view(d_ham);
    auto hv_sin  = Kokkos::create_mirror_view(d_sin);
    auto hv_sout = Kokkos::create_mirror_view(d_sout);
    std::memcpy(hv_ham.data(),  h_ham,  size_hamiltonian*sizeof(real_2_t));
    std::memcpy(hv_sin.data(),  h_sin,  size_sigma*sizeof(real_2_t));
    std::memcpy(hv_sout.data(), h_sout, size_sigma*sizeof(real_2_t));
    Kokkos::deep_copy(d_ham,  hv_ham);
    Kokkos::deep_copy(d_sin,  hv_sin);
    Kokkos::deep_copy(d_sout, hv_sout);
  }

  // Zero-init output on device (shared initial sout for each iteration)
  Kokkos::View<real_2_t*> d_sout_init("sout_init", size_sigma);
  {
    auto hv = Kokkos::create_mirror_view(d_sout_init);
    std::memcpy(hv.data(), h_sout, size_sigma*sizeof(real_2_t));
    Kokkos::deep_copy(d_sout_init, hv);
  }

  long total_time = 0;
  const real_t hdtv = hdt;

  for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
    // Reset output each iteration
    Kokkos::deep_copy(d_sout, d_sout_init);

    auto t_start = std::chrono::steady_clock::now();

    switch (kernel_id) {
      case 0: {
        // empty kernel
        Kokkos::parallel_for("comm_empty",
          Kokkos::RangePolicy<>(0, num),
          KOKKOS_LAMBDA(int gid) {});
        break;
      }

      case 1: {
        // initial kernel: -i*dt/hbar * [H, sigma]
        Kokkos::parallel_for("comm_init",
          Kokkos::RangePolicy<>(0, num),
          KOKKOS_LAMBDA(int gid) {
            int sigma_id = gid * dim * dim;
            for (int i = 0; i < dim; ++i) {
              for (int j = 0; j < dim; ++j) {
                real_2_t tmp; tmp.x = 0.0f; tmp.y = 0.0f;
                for (int k = 0; k < dim; ++k) {
                  // commutator contribution (real part of -i*(H*s - s*H))
                  tmp.x += d_ham(i*dim+k).x * d_sin(sigma_id+k*dim+j).x
                         - d_sin(sigma_id+i*dim+k).x * d_ham(k*dim+j).x;
                  tmp.x -= d_ham(i*dim+k).y * d_sin(sigma_id+k*dim+j).y
                         - d_sin(sigma_id+i*dim+k).y * d_ham(k*dim+j).y;
                  tmp.y += d_ham(i*dim+k).x * d_sin(sigma_id+k*dim+j).y
                         - d_sin(sigma_id+i*dim+k).x * d_ham(k*dim+j).y;
                  tmp.y += d_ham(i*dim+k).y * d_sin(sigma_id+k*dim+j).x
                         - d_sin(sigma_id+i*dim+k).y * d_ham(k*dim+j).x;
                }
                // multiply by -i*dt/hbar: real += hdt*tmp.y, imag -= hdt*tmp.x
                d_sout(sigma_id+i*dim+j).x += hdtv * tmp.y;
                d_sout(sigma_id+i*dim+j).y -= hdtv * tmp.x;
              }
            }
          });
        break;
      }

      case 2: {
        // refactored kernel treating real/imag as interleaved (AoS -> r_2_t pairs)
        Kokkos::parallel_for("comm_refactor",
          Kokkos::RangePolicy<>(0, num),
          KOKKOS_LAMBDA(int gid) {
            int sid = gid * dim * dim;
            for (int i = 0; i < dim; ++i) {
              for (int j = 0; j < dim; ++j) {
                real_t tr = 0.0f, ti = 0.0f;
                for (int k = 0; k < dim; ++k) {
                  // H[i,k] * s[k,j] - s[i,k] * H[k,j]  (complex arithmetic)
                  tr += d_ham(i*dim+k).x * d_sin(sid+k*dim+j).x
                      - d_ham(i*dim+k).y * d_sin(sid+k*dim+j).y
                      - d_sin(sid+i*dim+k).x * d_ham(k*dim+j).x
                      + d_sin(sid+i*dim+k).y * d_ham(k*dim+j).y;
                  ti += d_ham(i*dim+k).x * d_sin(sid+k*dim+j).y
                      + d_ham(i*dim+k).y * d_sin(sid+k*dim+j).x
                      - d_sin(sid+i*dim+k).x * d_ham(k*dim+j).y
                      - d_sin(sid+i*dim+k).y * d_ham(k*dim+j).x;
                }
                d_sout(sid+i*dim+j).x += hdtv * ti;
                d_sout(sid+i*dim+j).y -= hdtv * tr;
              }
            }
          });
        break;
      }

      case 3: {
        // refactored kernel with direct store (same math, fused k-loop)
        Kokkos::parallel_for("comm_refactor_direct",
          Kokkos::RangePolicy<>(0, num),
          KOKKOS_LAMBDA(int gid) {
            int sid = gid * dim * dim;
            for (int i = 0; i < dim; ++i) {
              for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < dim; ++k) {
                  d_sout(sid+i*dim+j).x +=
                      hdtv * ( d_ham(i*dim+k).x * d_sin(sid+k*dim+j).y
                             + d_ham(i*dim+k).y * d_sin(sid+k*dim+j).x
                             - d_sin(sid+i*dim+k).x * d_ham(k*dim+j).y
                             - d_sin(sid+i*dim+k).y * d_ham(k*dim+j).x);
                  d_sout(sid+i*dim+j).y -=
                      hdtv * ( d_ham(i*dim+k).x * d_sin(sid+k*dim+j).x
                             - d_ham(i*dim+k).y * d_sin(sid+k*dim+j).y
                             - d_sin(sid+i*dim+k).x * d_ham(k*dim+j).x
                             + d_sin(sid+i*dim+k).y * d_ham(k*dim+j).y);
                }
              }
            }
          });
        break;
      }

      default:
        // For unimplemented kernels (4–24): fall back to kernel 1
        Kokkos::parallel_for("comm_fallback",
          Kokkos::RangePolicy<>(0, num),
          KOKKOS_LAMBDA(int gid) {
            int sigma_id = gid * dim * dim;
            for (int i = 0; i < dim; ++i)
              for (int j = 0; j < dim; ++j) {
                real_2_t tmp; tmp.x = 0.f; tmp.y = 0.f;
                for (int k = 0; k < dim; ++k) {
                  tmp.x += d_ham(i*dim+k).x * d_sin(sigma_id+k*dim+j).x
                         - d_sin(sigma_id+i*dim+k).x * d_ham(k*dim+j).x
                         - d_ham(i*dim+k).y * d_sin(sigma_id+k*dim+j).y
                         + d_sin(sigma_id+i*dim+k).y * d_ham(k*dim+j).y;
                  tmp.y += d_ham(i*dim+k).x * d_sin(sigma_id+k*dim+j).y
                         - d_sin(sigma_id+i*dim+k).x * d_ham(k*dim+j).y
                         + d_ham(i*dim+k).y * d_sin(sigma_id+k*dim+j).x
                         - d_sin(sigma_id+i*dim+k).y * d_ham(k*dim+j).x;
                }
                d_sout(sigma_id+i*dim+j).x += hdtv * tmp.y;
                d_sout(sigma_id+i*dim+j).y -= hdtv * tmp.x;
              }
          });
        break;
    }

    Kokkos::fence();
    auto t_end = std::chrono::steady_clock::now();

    if (iter >= NUM_WARMUP)
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                      t_end - t_start).count();
  }

  // Copy result back
  {
    auto hv_sout = Kokkos::create_mirror_view(d_sout);
    Kokkos::deep_copy(hv_sout, d_sout);
    for (size_t i = 0; i < size_sigma; i++) {
      h_sigma_out_cmplx[i] = complex_t(hv_sout(i).x, hv_sout(i).y);
    }
  }

  free(h_ham);
  free(h_sin);
  free(h_sout);
  return total_time;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    const size_t dim = DIM;
    const size_t num = NUM;
    size_t size_hamiltonian = dim * dim;
    size_t size_sigma       = size_hamiltonian * num;

    complex_t* hamiltonian = allocate_aligned<complex_t>(size_hamiltonian);
    complex_t* sigma_in    = allocate_aligned<complex_t>(size_sigma);
    complex_t* sigma_out   = allocate_aligned<complex_t>(size_sigma);
    complex_t* sigma_ref   = allocate_aligned<complex_t>(size_sigma);

    // Reference result
    initialise_hamiltonian(hamiltonian, dim);
    initialise_sigma(sigma_in, sigma_out, dim, num);
    commutator_reference(sigma_in, sigma_out, hamiltonian, dim, num);
    std::memcpy(sigma_ref, sigma_out, size_sigma * sizeof(complex_t));

    long ktime = 0;

    // Run kernels 0–3 (core AoS kernels)
    struct BenchSpec {
      int  kid;
      bool scale_h;
      bool soa_h;
      const char* name;
    };
    BenchSpec specs[] = {
      {0, false, false, "comm_empty"},
      {1, false, false, "comm_init"},
      {2, false, false, "comm_refactor"},
      {3, true,  false, "comm_refactor_direct_store"},
    };

    for (auto& s : specs) {
      long t = run_benchmark(sigma_in, sigma_out, hamiltonian,
                             size_sigma, size_hamiltonian,
                             (int)dim, (int)num,
                             s.kid, s.scale_h, s.soa_h);
      // Compare against reference for correctness (kernel 1+)
      if (s.kid > 0) {
        real_t dev = compare_matrices(sigma_out, sigma_ref, dim, num);
        printf("%-38s  deviation = %e  time = %ld ns\n", s.name, (double)dev, t);
      } else {
        printf("%-38s  time = %ld ns\n", s.name, t);
      }
      ktime += t;
    }

    printf("Total kernel time for selected benchmarks %lf (s)\n",
           ktime * 1e-9);

    free(hamiltonian);
    free(sigma_in);
    free(sigma_out);
    free(sigma_ref);
  }
  Kokkos::finalize();
  return 0;
}
