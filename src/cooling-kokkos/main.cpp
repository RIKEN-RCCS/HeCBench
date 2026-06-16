/*
 * Primordial hydrogen/helium cooling curve.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

typedef double Real;

KOKKOS_INLINE_FUNCTION
Real primordial_cool(Real n, Real T, int heat_flag)
{
  Real n_h, Y, y, g_ff, cool;
  Real n_h0, n_hp, n_he0, n_hep, n_hepp, n_e, n_e_old;
  Real alpha_hp, alpha_hep, alpha_d, alpha_hepp, gamma_eh0, gamma_ehe0, gamma_ehep;
  Real le_h0, le_hep, li_h0, li_he0, li_hep, lr_hp, lr_hep, lr_hepp, ld_hep, l_ff;
  Real gamma_lh0, gamma_lhe0, gamma_lhep, e_h0, e_he0, e_hep, H;
  int n_iter;
  Real diff, tol;

  Y = 0.24;
  y = Y / (4 - 4 * Y);
  n_h = n;

  alpha_hp   = (8.4e-11)  * (1.0/Kokkos::sqrt(T)) * Kokkos::pow(T/1e3,-0.2)  * (1.0 / (1.0 + Kokkos::pow(T/1e6, 0.7)));
  alpha_hep  = (1.5e-10)  * Kokkos::pow(T,-0.6353);
  alpha_d    = (1.9e-3)   * Kokkos::pow(T,-1.5)   * Kokkos::exp(-470000.0/T)  * (1.0 + 0.3 * Kokkos::exp(-94000.0/T));
  alpha_hepp = (3.36e-10) * (1.0/Kokkos::sqrt(T)) * Kokkos::pow(T/1e3,-0.2)  * (1.0 / (1.0 + Kokkos::pow(T/1e6, 0.7)));
  gamma_eh0  = (5.85e-11) * Kokkos::sqrt(T) * Kokkos::exp(-157809.1/T)  * (1.0 / (1.0 + Kokkos::sqrt(T/1e5)));
  gamma_ehe0 = (2.38e-11) * Kokkos::sqrt(T) * Kokkos::exp(-285335.4/T)  * (1.0 / (1.0 + Kokkos::sqrt(T/1e5)));
  gamma_ehep = (5.68e-12) * Kokkos::sqrt(T) * Kokkos::exp(-631515.0/T)  * (1.0 / (1.0 + Kokkos::sqrt(T/1e5)));
  gamma_lh0 = 3.19851e-13;
  gamma_lhe0 = 3.13029e-13;
  gamma_lhep = 2.00541e-14;
  e_h0 = 2.4796e-24;
  e_he0 = 6.86167e-24;
  e_hep = 6.21868e-25;

  n_e = n_h;
  n_iter = 20;
  diff = 1.0;
  tol = 1.0e-6;
  if (heat_flag) {
    for (int i = 0; i < n_iter; i++) {
      n_e_old = n_e;
      n_h0   = n_h * alpha_hp / (alpha_hp + gamma_eh0 + gamma_lh0 / n_e);
      n_hp   = n_h - n_h0;
      n_hep  = y * n_h / (1.0 + (alpha_hep + alpha_d) / (gamma_ehe0 + gamma_lhe0 / n_e) + (gamma_ehep + gamma_lhep / n_e) / alpha_hepp);
      n_he0  = n_hep * (alpha_hep + alpha_d) / (gamma_ehe0 + gamma_lhe0 / n_e);
      n_hepp = n_hep * (gamma_ehep + gamma_lhep / n_e) / alpha_hepp;
      n_e    = n_hp + n_hep + 2 * n_hepp;
      diff = Kokkos::fabs(n_e_old - n_e);
      if (diff < tol) break;
    }
  } else {
    n_h0   = n_h * alpha_hp / (alpha_hp + gamma_eh0);
    n_hp   = n_h - n_h0;
    n_hep  = y * n_h / (1.0 + (alpha_hep + alpha_d) / gamma_ehe0 + gamma_ehep / alpha_hepp);
    n_he0  = n_hep * (alpha_hep + alpha_d) / gamma_ehe0;
    n_hepp = n_hep * gamma_ehep / alpha_hepp;
    n_e    = n_hp + n_hep + 2 * n_hepp;
  }

  le_h0  = (7.50e-19) * Kokkos::exp(-118348.0/T)  * (1.0/(1.0+Kokkos::sqrt(T/1e5))) * n_e * n_h0;
  le_hep = (5.54e-17) * Kokkos::pow(T,-0.397)      * Kokkos::exp(-473638.0/T) * (1.0/(1.0+Kokkos::sqrt(T/1e5))) * n_e * n_hep;
  li_h0  = (1.27e-21) * Kokkos::sqrt(T) * Kokkos::exp(-157809.1/T) * (1.0/(1.0+Kokkos::sqrt(T/1e5))) * n_e * n_h0;
  li_he0 = (9.38e-22) * Kokkos::sqrt(T) * Kokkos::exp(-285335.4/T) * (1.0/(1.0+Kokkos::sqrt(T/1e5))) * n_e * n_he0;
  li_hep = (4.95e-22) * Kokkos::sqrt(T) * Kokkos::exp(-631515.0/T) * (1.0/(1.0+Kokkos::sqrt(T/1e5))) * n_e * n_hep;
  lr_hp  = (8.70e-27) * Kokkos::sqrt(T) * Kokkos::pow(T/1e3,-0.2)  * (1.0/(1.0+Kokkos::pow(T/1e6,0.7))) * n_e * n_hp;
  lr_hep = (1.55e-26) * Kokkos::pow(T, 0.3647) * n_e * n_hep;
  lr_hepp= (3.48e-26) * Kokkos::sqrt(T) * Kokkos::pow(T/1e3,-0.2)  * (1.0/(1.0+Kokkos::pow(T/1e6,0.7))) * n_e * n_hepp;
  ld_hep = (1.24e-13) * Kokkos::pow(T,-1.5) * Kokkos::exp(-470000.0/T) * (1.0+0.3*Kokkos::exp(-94000.0/T)) * n_e * n_hep;
  g_ff   = 1.1 + 0.34 * Kokkos::exp(-(5.5 - Kokkos::log(T)) * (5.5 - Kokkos::log(T)) / 3.0);
  l_ff   = (1.42e-27) * g_ff * Kokkos::sqrt(T) * (n_hp + n_hep + 4 * n_hepp) * n_e;

  cool = le_h0 + le_hep + li_h0 + li_he0 + li_hep + lr_hp + lr_hep + lr_hepp + ld_hep + l_ff;

  H = 0.0;
  if (heat_flag) {
    H = n_h0 * e_h0 + n_he0 * e_he0 + n_hep * e_hep;
  }
  cool -= H;
  return cool;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int num    = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const Real n = 0.0899;

  Real *T   = (Real*) malloc(num * sizeof(Real));
  Real *h_r = (Real*) malloc(num * sizeof(Real));
  Real *d_r = (Real*) malloc(num * sizeof(Real));

  for (int i = 0; i < num; i++)
    T[i] = -275.0 + i * 275 * 2.0 / num;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<Real*> d_T("d_T", num);
    Kokkos::View<Real*> d_r_view("d_r", num);

    auto h_T = Kokkos::create_mirror_view(d_T);
    for (int i = 0; i < num; i++) h_T(i) = T[i];
    Kokkos::deep_copy(d_T, h_T);

    // Warmup
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("cool_warmup", num, KOKKOS_LAMBDA(int i) {
        d_r_view(i) = primordial_cool(n, d_T(i), 0);
      });
      Kokkos::fence();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("cool_kernel", num, KOKKOS_LAMBDA(int j) {
        d_r_view(j) = primordial_cool(n, d_T(j), 1);
      });
      Kokkos::fence();
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / repeat);

    auto h_r_view = Kokkos::create_mirror_view(d_r_view);
    Kokkos::deep_copy(h_r_view, d_r_view);
    for (int i = 0; i < num; i++) d_r[i] = h_r_view(i);
  }
  Kokkos::finalize();

  // Verify
  for (int i = 0; i < num; i++)
    h_r[i] = primordial_cool(n, T[i], 1);

  bool error = false;
  for (int i = 0; i < num; i++) {
    if (fabs(d_r[i] - h_r[i]) > 1e-3) { error = true; break; }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(T); free(h_r); free(d_r);
  return 0;
}
