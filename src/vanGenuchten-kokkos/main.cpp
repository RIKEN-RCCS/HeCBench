#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// Constants and reference implementation inlined from vanGenuchten-cuda/reference.h
static const double alpha_c   = 0.02;
static const double theta_S_c = 0.45;
static const double theta_R_c = 0.1;
static const double n_c       = 1.8;

void reference(const double *Ksat, const double *psi,
               double *C, double *theta, double *K, int size) {
  for (int i = 0; i < size; i++) {
    double lambda = n_c - 1.0;
    double m      = lambda / n_c;
    double _psi   = psi[i] * 100.0;
    double _theta = (_psi < 0.0)
      ? (theta_S_c - theta_R_c) / pow(1.0 + pow(alpha_c * (-_psi), n_c), m) + theta_R_c
      : theta_S_c;
    theta[i] = _theta;
    double Se = (_theta - theta_R_c) / (theta_S_c - theta_R_c);
    K[i] = Ksat[i] * sqrt(Se)
           * (1.0 - pow(1.0 - pow(Se, 1.0/m), m))
           * (1.0 - pow(1.0 - pow(Se, 1.0/m), m));
    C[i] = (_psi < 0.0)
      ? 100.0 * alpha_c * n_c * (1.0/n_c - 1.0)
        * pow(alpha_c * fabs(_psi), n_c - 1.0)
        * (theta_R_c - theta_S_c)
        * pow(pow(alpha_c * fabs(_psi), n_c) + 1.0, 1.0/n_c - 2.0)
      : 0.0;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: ./%s <dimX> <dimY> <dimZ> <repeat>\n", argv[0]);
    return 1;
  }
  const int dimX   = atoi(argv[1]);
  const int dimY   = atoi(argv[2]);
  const int dimZ   = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const int size   = dimX * dimY * dimZ;

  double *Ksat      = new double[size];
  double *psi       = new double[size];
  double *C_ref     = new double[size];
  double *theta_ref = new double[size];
  double *K_ref     = new double[size];

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 + (1.0 - 1e-6) * i / size;
    psi[i]  = -100.0 + 101.0 * i / size;
  }
  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_Ksat("d_Ksat", size);
    Kokkos::View<double*> d_psi("d_psi",   size);
    Kokkos::View<double*> d_C("d_C",       size);
    Kokkos::View<double*> d_theta("d_theta", size);
    Kokkos::View<double*> d_K("d_K",       size);

    {
      auto hK = Kokkos::create_mirror_view(d_Ksat);
      auto hP = Kokkos::create_mirror_view(d_psi);
      for (int i = 0; i < size; i++) { hK(i) = Ksat[i]; hP(i) = psi[i]; }
      Kokkos::deep_copy(d_Ksat, hK);
      Kokkos::deep_copy(d_psi,  hP);
    }

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("vanGenuchten", size,
        KOKKOS_LAMBDA(int i) {
          const double alpha_k   = 0.02;
          const double theta_S_k = 0.45;
          const double theta_R_k = 0.1;
          const double n_k       = 1.8;
          const double lambda_k  = n_k - 1.0;
          const double m_k       = lambda_k / n_k;

          double _psi = d_psi(i) * 100.0;
          double _theta = (_psi < 0.0)
            ? (theta_S_k - theta_R_k) / Kokkos::pow(1.0 + Kokkos::pow(alpha_k * (-_psi), n_k), m_k)
              + theta_R_k
            : theta_S_k;
          d_theta(i) = _theta;

          double Se = (_theta - theta_R_k) / (theta_S_k - theta_R_k);
          double t  = 1.0 - Kokkos::pow(1.0 - Kokkos::pow(Se, 1.0/m_k), m_k);
          d_K(i) = d_Ksat(i) * Kokkos::sqrt(Se) * t * t;

          d_C(i) = (_psi < 0.0)
            ? 100.0 * alpha_k * n_k * (1.0/n_k - 1.0)
              * Kokkos::pow(alpha_k * Kokkos::fabs(_psi), n_k - 1.0)
              * (theta_R_k - theta_S_k)
              * Kokkos::pow(Kokkos::pow(alpha_k * Kokkos::fabs(_psi), n_k) + 1.0, 1.0/n_k - 2.0)
            : 0.0;
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    auto hC     = Kokkos::create_mirror_view(d_C);
    auto hTheta = Kokkos::create_mirror_view(d_theta);
    auto hK     = Kokkos::create_mirror_view(d_K);
    Kokkos::deep_copy(hC, d_C);
    Kokkos::deep_copy(hTheta, d_theta);
    Kokkos::deep_copy(hK, d_K);

    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (fabs(hC(i) - C_ref[i]) > 1e-3 ||
          fabs(hTheta(i) - theta_ref[i]) > 1e-3 ||
          fabs(hK(i) - K_ref[i]) > 1e-3) {
        ok = false; break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  delete[] Ksat; delete[] psi;
  delete[] C_ref; delete[] theta_ref; delete[] K_ref;
  return 0;
}
