/*
 * Kokkos port of bicgstab.
 * Original used cuBLAS + cuSPARSE for a BiCGSTAB sparse linear solver.
 * This port implements the same algorithm using dense Kokkos parallel_for/reduce
 * for a simple tridiagonal test matrix, preserving the convergence and timing structure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

// Dense matrix-vector product: y = A*x  (A is m x m, stored row-major)
KOKKOS_INLINE_FUNCTION
void matvec(const Kokkos::View<double*>& A, const Kokkos::View<double*>& x,
            Kokkos::View<double*>& y, int m,
            const Kokkos::TeamPolicy<>::member_type& team) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m), [&](int i) {
    double s = 0.0;
    for (int j = 0; j < m; j++) s += A(i * m + j) * x(j);
    y(i) = s;
  });
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // Simple tridiagonal system: same as cuSPARSE example from CUDA version
    const int m = 1000;      // matrix size
    const int maxIter = 300;
    const double tol = 1e-10;

    // Build a tridiagonal matrix A: -1, 4, -1 (SPD, well-conditioned)
    std::vector<double> h_A(m * m, 0.0);
    std::vector<double> h_b(m, 1.0);
    for (int i = 0; i < m; i++) {
      h_A[i * m + i] = 4.0;
      if (i > 0)     h_A[i * m + (i-1)] = -1.0;
      if (i < m - 1) h_A[i * m + (i+1)] = -1.0;
    }

    Kokkos::View<double*> A("A", m * m);
    Kokkos::View<double*> b("b", m);
    Kokkos::View<double*> x("x", m);    // solution (init to 0)
    Kokkos::View<double*> r("r", m);    // residual
    Kokkos::View<double*> r0("r0", m);
    Kokkos::View<double*> p("p", m);
    Kokkos::View<double*> v("v", m);
    Kokkos::View<double*> s("s", m);
    Kokkos::View<double*> t("t", m);

    auto h_A_v = Kokkos::create_mirror_view(A);
    auto h_b_v = Kokkos::create_mirror_view(b);
    for (int i = 0; i < m * m; i++) h_A_v(i) = h_A[i];
    for (int i = 0; i < m; i++) h_b_v(i) = h_b[i];
    Kokkos::deep_copy(A, h_A_v);
    Kokkos::deep_copy(b, h_b_v);
    Kokkos::deep_copy(x, 0.0);

    // Helper lambdas
    auto dot = [&](Kokkos::View<double*> u, Kokkos::View<double*> w) -> double {
      double res = 0.0;
      Kokkos::parallel_reduce("dot", m, KOKKOS_LAMBDA(int i, double& s) {
        s += u(i) * w(i);
      }, res);
      return res;
    };

    auto axpy = [&](double alpha, Kokkos::View<double*> src, Kokkos::View<double*> dst) {
      Kokkos::parallel_for("axpy", m, KOKKOS_LAMBDA(int i) {
        dst(i) += alpha * src(i);
      });
    };

    auto copy_vec = [&](Kokkos::View<double*> src, Kokkos::View<double*> dst) {
      Kokkos::deep_copy(dst, src);
    };

    auto mv = [&](Kokkos::View<double*> in, Kokkos::View<double*> out) {
      Kokkos::parallel_for("mv", m, KOKKOS_LAMBDA(int i) {
        double s = 0.0;
        for (int j = 0; j < m; j++) s += A(i * m + j) * in(j);
        out(i) = s;
      });
    };

    auto nrm2 = [&](Kokkos::View<double*> u) -> double {
      double res = 0.0;
      Kokkos::parallel_reduce("nrm2", m, KOKKOS_LAMBDA(int i, double& s) {
        s += u(i) * u(i);
      }, res);
      return sqrt(res);
    };

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    // r = b - A*x (x=0 so r = b)
    copy_vec(b, r);
    copy_vec(r, r0);
    copy_vec(r, p);

    double rho = dot(r0, r);
    double nrm_r0 = nrm2(r);
    double threshold = tol * nrm_r0;
    double nrm_r = nrm_r0;

    int iter = 0;
    double alpha = 1.0, omega = 1.0;
    for (iter = 1; iter <= maxIter && nrm_r > threshold; iter++) {
      mv(p, v);
      double rv = dot(r0, v);
      alpha = rho / rv;

      // s = r - alpha*v
      copy_vec(r, s);
      axpy(-alpha, v, s);

      nrm_r = nrm2(s);
      if (nrm_r <= threshold) {
        axpy(alpha, p, x);
        break;
      }

      mv(s, t);
      omega = dot(t, s) / dot(t, t);

      // x += alpha*p + omega*s
      axpy(alpha, p, x);
      axpy(omega, s, x);

      // r = s - omega*t
      copy_vec(s, r);
      axpy(-omega, t, r);

      nrm_r = nrm2(r);

      double rho_new = dot(r0, r);
      double beta = (rho_new / rho) * (alpha / omega);
      rho = rho_new;

      // p = r + beta*(p - omega*v)
      axpy(-omega, v, p);
      Kokkos::parallel_for("scale_p", m, KOKKOS_LAMBDA(int i) {
        p(i) = r(i) + beta * p(i);
      });
    }
    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;

    printf("BiCGStab converged in %d iterations, residual = %e\n", iter-1, nrm_r);
    printf("Elapsed time: %.3f ms\n", time_ms);
  }
  Kokkos::finalize();
  return 0;
}
