// Kokkos port of mriQ-omp
// Computes MRI-Q signal quality metric.
// Uses synthetic data instead of file I/O.
// Args: <numX> <numK> <repeat>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

#define PI    3.1415926535897932384626433832795029f
#define PIx2  6.2831853071795864769252867665590058f

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <numX> <numK> <repeat>\n", argv[0]);
    return 1;
  }
  const int numX   = atoi(argv[1]);
  const int numK   = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("%d pixels in output; %d samples in trajectory\n", numX, numK);

  // ---- Generate synthetic data on host ----
  std::vector<float> h_kx(numK), h_ky(numK), h_kz(numK);
  std::vector<float> h_phiR(numK), h_phiI(numK);
  std::vector<float> h_x(numX), h_y(numX), h_z(numX);

  for (int k = 0; k < numK; k++) {
    h_kx[k]   =  sinf(k * 0.1f) * 0.5f;
    h_ky[k]   =  cosf(k * 0.1f) * 0.5f;
    h_kz[k]   =  sinf(k * 0.07f) * 0.5f;
    h_phiR[k] =  sinf(k * 0.2f);
    h_phiI[k] =  cosf(k * 0.2f);
  }
  for (int i = 0; i < numX; i++) {
    h_x[i] = sinf(i * 0.13f) * 0.5f;
    h_y[i] = cosf(i * 0.13f) * 0.5f;
    h_z[i] = sinf(i * 0.09f) * 0.5f;
  }

  Kokkos::initialize(argc, argv);
  {
    // ---- Allocate device views ----
    Kokkos::View<float*> phiR("phiR", numK);
    Kokkos::View<float*> phiI("phiI", numK);
    Kokkos::View<float*> phiMag("phiMag", numK);
    Kokkos::View<float*> x("x", numX);
    Kokkos::View<float*> y("y", numX);
    Kokkos::View<float*> z("z", numX);
    Kokkos::View<float*> Qr("Qr", numX);
    Kokkos::View<float*> Qi("Qi", numX);
    Kokkos::View<kValues*> kVals("kVals", numK);

    // Copy to device
    auto h_phiR_v = Kokkos::create_mirror_view(phiR);
    auto h_phiI_v = Kokkos::create_mirror_view(phiI);
    auto h_x_v    = Kokkos::create_mirror_view(x);
    auto h_y_v    = Kokkos::create_mirror_view(y);
    auto h_z_v    = Kokkos::create_mirror_view(z);
    for (int k = 0; k < numK; k++) { h_phiR_v(k) = h_phiR[k]; h_phiI_v(k) = h_phiI[k]; }
    for (int i = 0; i < numX; i++) { h_x_v(i) = h_x[i]; h_y_v(i) = h_y[i]; h_z_v(i) = h_z[i]; }
    Kokkos::deep_copy(phiR, h_phiR_v);
    Kokkos::deep_copy(phiI, h_phiI_v);
    Kokkos::deep_copy(x, h_x_v);
    Kokkos::deep_copy(y, h_y_v);
    Kokkos::deep_copy(z, h_z_v);

    // ---- Phase 1: compute phiMag ----
    auto t0 = std::chrono::steady_clock::now();

    Kokkos::parallel_for("phiMag", numK, KOKKOS_LAMBDA(int k) {
      float r = phiR(k), im = phiI(k);
      phiMag(k) = r * r + im * im;
    });
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    printf("computePhiMag time: %f s\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9f);

    // Build kVals on host then copy
    auto h_kVals = Kokkos::create_mirror_view(kVals);
    auto h_phiMag = Kokkos::create_mirror_view(phiMag);
    Kokkos::deep_copy(h_phiMag, phiMag);
    for (int k = 0; k < numK; k++) {
      h_kVals(k).Kx      = h_kx[k];
      h_kVals(k).Ky      = h_ky[k];
      h_kVals(k).Kz      = h_kz[k];
      h_kVals(k).PhiMag  = h_phiMag(k);
    }
    Kokkos::deep_copy(kVals, h_kVals);

    // ---- Phase 2: compute Q (repeated for timing) ----
    double computeQ_time = 0.0;
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("initQ", numX, KOKKOS_LAMBDA(int i) {
        Qr(i) = 0.f;
        Qi(i) = 0.f;
      });
      Kokkos::fence();

      auto tq0 = std::chrono::steady_clock::now();

      Kokkos::parallel_for("computeQ", numX, KOKKOS_LAMBDA(int xIndex) {
        const float sX = x(xIndex);
        const float sY = y(xIndex);
        const float sZ = z(xIndex);
        float sQr = 0.f, sQi = 0.f;
        for (int k = 0; k < numK; k++) {
          float expArg = PIx2 * (kVals(k).Kx * sX +
                                  kVals(k).Ky * sY +
                                  kVals(k).Kz * sZ);
          sQr += kVals(k).PhiMag * cosf(expArg);
          sQi += kVals(k).PhiMag * sinf(expArg);
        }
        Qr(xIndex) = sQr;
        Qi(xIndex) = sQi;
      });
      Kokkos::fence();

      auto tq1 = std::chrono::steady_clock::now();
      computeQ_time +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(tq1 - tq0).count();
    }
    printf("Average computeQ time: %f s\n", computeQ_time * 1e-9 / repeat);

    // ---- Verify against host reference ----
    auto h_Qr = Kokkos::create_mirror_view(Qr);
    auto h_Qi = Kokkos::create_mirror_view(Qi);
    Kokkos::deep_copy(h_Qr, Qr);
    Kokkos::deep_copy(h_Qi, Qi);

    std::vector<float> ref_Qr(numX, 0.f), ref_Qi(numX, 0.f);
    for (int xIndex = 0; xIndex < numX; xIndex++) {
      float sX = h_x[xIndex], sY = h_y[xIndex], sZ = h_z[xIndex];
      float sQr = 0.f, sQi = 0.f;
      for (int k = 0; k < numK; k++) {
        float expArg = PIx2 * (h_kx[k] * sX + h_ky[k] * sY + h_kz[k] * sZ);
        sQr += h_phiMag(k) * cosf(expArg);
        sQi += h_phiMag(k) * sinf(expArg);
      }
      ref_Qr[xIndex] = sQr;
      ref_Qi[xIndex] = sQi;
    }

    double maxErrR = 0.0, maxErrI = 0.0;
    for (int i = 0; i < numX; i++) {
      maxErrR = fmax(maxErrR, fabs(h_Qr(i) - ref_Qr[i]));
      maxErrI = fmax(maxErrI, fabs(h_Qi(i) - ref_Qi[i]));
    }
    printf("Max Qr error: %e  Max Qi error: %e\n", maxErrR, maxErrI);
    printf("%s\n", (maxErrR < 1e-3f && maxErrI < 1e-3f) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
