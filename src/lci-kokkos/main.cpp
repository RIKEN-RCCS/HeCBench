#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>
#include "../lci-cuda/tables.h"

#define L_max 64
#define one_over_theta0 (5.0/(2.0*pow(M_PI,4.0)))
#define kappa (pow(M_PI,2.0)/3.0 - 2 * 1.2020569031595942)

KOKKOS_INLINE_FUNCTION double B(int l)
{
  return 2.0*(14.0*l*l + 7.0*l - 2.0)/(4.0*l-1.0)/(4.0*l+3.0);
}

KOKKOS_INLINE_FUNCTION double U(int l)
{
  return -(2.0*l-1)*(2.0*l+1)*(2.0*l+2)/(4.0*l+3)/(4.0*l+5);
}

KOKKOS_INLINE_FUNCTION double C(int l)
{
  return (2.0*l-1)*2.0*l*(2.0*l+2)/(4.0*l-3.0)/(4.0*l-1.0);
}

KOKKOS_INLINE_FUNCTION double alpha(int l, const double* dftab, const double* ftab)
{
  if (l < 0) return 0.0;
  if (l == 0) return 1.0;
  return dftab[2*l-1] / ftab[l];
}

KOKKOS_INLINE_FUNCTION double Omega(int l, int m, int n, const double* dftab, const double* ftab)
{
  return alpha(m-n+l, dftab, ftab) * alpha(m+n-l, dftab, ftab) *
         alpha(n-m+l, dftab, ftab) / alpha(m+n+l, dftab, ftab) *
         (4*l+1) / (2.0*(n+m+l)+1);
}

KOKKOS_INLINE_FUNCTION double Sum_Omega(int l, const double* c, const double* dftab, const double* ftab)
{
  double sum = 0.0;
  for (int m = 1; m < L_max; m++)
    for (int n = 1; n < L_max; n++)
      if (abs(m-n) < l+1) sum += Omega(l, m, n, dftab, ftab) * c[m] * c[n];
  return sum;
}

KOKKOS_INLINE_FUNCTION double Sum_NL(int l, const double* c)
{
  double sum = 0.0;
  for (int n = 1; n < L_max; n++)
    sum += pow(c[n], 2.0) / (4.0*n+1);
  return sum * (2*l-1)*(l+1)*c[l]/3.0;
}

void initial(double c[], int seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dis(-6.0, 6.0);
  std::uniform_real_distribution<double> temp(0.1, 0.9);
  for (int l = 1; l < L_max; l++) c[l] = dis(gen);
  c[0] = temp(gen);   // random temperature
  c[L_max] = 0.0;     // truncation; do not modify
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    const double t_init  = 0.1;
    const double t_final = 200.0;
    const double delta_t = 0.1;
    int dimension = L_max;
    int seed = dimension;

    int dftab_size = sizeof(double_fact_table) / sizeof(double_fact_table[0]);
    int ftab_size  = sizeof(fact_table)         / sizeof(fact_table[0]);

    Kokkos::View<double*> d_c("c",     dimension + 1);
    Kokkos::View<double*> d_n("n",     dimension + 1);
    Kokkos::View<double*> d_dftab("dftab", dftab_size);
    Kokkos::View<double*> d_ftab("ftab",   ftab_size);

    auto h_c     = Kokkos::create_mirror_view(d_c);
    auto h_n     = Kokkos::create_mirror_view(d_n);
    auto h_dftab = Kokkos::create_mirror_view(d_dftab);
    auto h_ftab  = Kokkos::create_mirror_view(d_ftab);

    for (int i = 0; i < dftab_size; i++) h_dftab(i) = double_fact_table[i];
    for (int i = 0; i < ftab_size;  i++) h_ftab(i)  = fact_table[i];
    Kokkos::deep_copy(d_dftab, h_dftab);
    Kokkos::deep_copy(d_ftab,  h_ftab);

    double* c_host = new double[dimension + 1];
    initial(c_host, seed);
    for (int i = 0; i <= dimension; i++) h_c(i) = c_host[i];
    Kokkos::deep_copy(d_c, h_c);

    float total_time = 0.f;

    for (double t_next = t_init + delta_t; t_next <= t_final; t_next += delta_t) {
      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for("RHS_f", L_max, KOKKOS_LAMBDA(int l) {
        double T = d_c(0);

        if (l == 0) {
          d_n(0) = -T/3.0/t_next * (1.0 + 0.1*d_c(1));
        } else {
          double B_bar = B(l) - 4.0/3.0;
          double LHS_119;
          if (l > 1)
            LHS_119 = 1.0/t_next * (U(l)*d_c(l+1) + (B_bar - 2.0/15.0*d_c(1)) + C(l)*d_c(l-1));
          else
            LHS_119 = 1.0/t_next * (U(1)*d_c(2)   + (B_bar - 2.0/15.0*d_c(1)) + C(1));

          double Sum1 = Sum_Omega(l, d_c.data(), d_dftab.data(), d_ftab.data());
          double Sum2 = Sum_NL(l, d_c.data());

          double RHS_119 = -T*one_over_theta0 * (
              (kappa + M_PI*M_PI*l*(2*l+1)/3.0)*d_c(l) +
              kappa*Sum1 + kappa*Sum2);

          d_n(l) = -LHS_119 + RHS_119;
        }
      });
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      initial(c_host, ++seed);
      for (int i = 0; i <= dimension; i++) h_c(i) = c_host[i];
      Kokkos::deep_copy(d_c, h_c);
    }

    printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);

    delete[] c_host;
  }
  Kokkos::finalize();
  return 0;
}
