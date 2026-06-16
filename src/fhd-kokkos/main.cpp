#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <#samples> <#voxels> <verify>\n", argv[0]);
    return 1;
  }
  const int samples = atoi(argv[1]);
  const int voxels  = atoi(argv[2]);
  const int verify  = atoi(argv[3]);

  float *h_rmu  = (float*)malloc(voxels * sizeof(float));
  float *h_imu  = (float*)malloc(voxels * sizeof(float));
  float *h_kx   = (float*)malloc(voxels * sizeof(float));
  float *h_ky   = (float*)malloc(voxels * sizeof(float));
  float *h_kz   = (float*)malloc(voxels * sizeof(float));
  float *h_rfhd = (float*)malloc(samples * sizeof(float));
  float *h_ifhd = (float*)malloc(samples * sizeof(float));
  float *h_x    = (float*)malloc(samples * sizeof(float));
  float *h_y    = (float*)malloc(samples * sizeof(float));
  float *h_z    = (float*)malloc(samples * sizeof(float));
  float *rfhd   = (float*)malloc(samples * sizeof(float));
  float *ifhd   = (float*)malloc(samples * sizeof(float));

  srand(2);
  for (int i = 0; i < samples; i++) {
    rfhd[i] = h_rfhd[i] = (float)i / samples;
    ifhd[i] = h_ifhd[i] = (float)i / samples;
    h_x[i] = 0.3f + (rand() % 2 ? 0.1f : -0.1f);
    h_y[i] = 0.2f + (rand() % 2 ? 0.1f : -0.1f);
    h_z[i] = 0.1f + (rand() % 2 ? 0.1f : -0.1f);
  }
  for (int i = 0; i < voxels; i++) {
    h_rmu[i] = (float)i / voxels;
    h_imu[i] = (float)i / voxels;
    h_kx[i] = 0.1f + (rand() % 2 ? 0.1f : -0.1f);
    h_ky[i] = 0.2f + (rand() % 2 ? 0.1f : -0.1f);
    h_kz[i] = 0.3f + (rand() % 2 ? 0.1f : -0.1f);
  }

  printf("Run FHd on a device\n");

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_rmu("rmu", voxels),  d_imu("imu", voxels);
    Kokkos::View<float*> d_kx("kx", voxels),    d_ky("ky", voxels),  d_kz("kz", voxels);
    Kokkos::View<float*> d_rfhd("rfhd", samples), d_ifhd("ifhd", samples);
    Kokkos::View<float*> d_x("x", samples), d_y("y", samples), d_z("z", samples);

    {
      auto mrmu  = Kokkos::create_mirror_view(d_rmu);
      auto mimu  = Kokkos::create_mirror_view(d_imu);
      auto mkx   = Kokkos::create_mirror_view(d_kx);
      auto mky   = Kokkos::create_mirror_view(d_ky);
      auto mkz   = Kokkos::create_mirror_view(d_kz);
      auto mrfhd = Kokkos::create_mirror_view(d_rfhd);
      auto mifhd = Kokkos::create_mirror_view(d_ifhd);
      auto mx    = Kokkos::create_mirror_view(d_x);
      auto my2   = Kokkos::create_mirror_view(d_y);
      auto mz    = Kokkos::create_mirror_view(d_z);

      for (int i = 0; i < voxels;  i++) { mrmu[i]=h_rmu[i]; mimu[i]=h_imu[i]; mkx[i]=h_kx[i]; mky[i]=h_ky[i]; mkz[i]=h_kz[i]; }
      for (int i = 0; i < samples; i++) { mrfhd[i]=rfhd[i]; mifhd[i]=ifhd[i]; mx[i]=h_x[i]; my2[i]=h_y[i]; mz[i]=h_z[i]; }

      Kokkos::deep_copy(d_rmu, mrmu);  Kokkos::deep_copy(d_imu, mimu);
      Kokkos::deep_copy(d_kx, mkx);   Kokkos::deep_copy(d_ky, mky);   Kokkos::deep_copy(d_kz, mkz);
      Kokkos::deep_copy(d_rfhd, mrfhd); Kokkos::deep_copy(d_ifhd, mifhd);
      Kokkos::deep_copy(d_x, mx); Kokkos::deep_copy(d_y, my2); Kokkos::deep_copy(d_z, mz);
    }

    auto start = std::chrono::steady_clock::now();

    Kokkos::parallel_for("fhd", samples,
      KOKKOS_LAMBDA(int n) {
        float r  = d_rfhd[n];
        float im = d_ifhd[n];
        const float xn = d_x[n], yn = d_y[n], zn = d_z[n];
        for (int m = 0; m < voxels; m++) {
          float e = 2.f * (float)M_PI * (d_kx[m]*xn + d_ky[m]*yn + d_kz[m]*zn);
          float c = cosf(e), s = sinf(e);
          r  += d_rmu[m]*c - d_imu[m]*s;
          im += d_imu[m]*c + d_rmu[m]*s;
        }
        d_rfhd[n] = r;
        d_ifhd[n] = im;
      });
    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Device execution time %f (s)\n", time * 1e-9f);

    {
      auto mrfhd = Kokkos::create_mirror_view(d_rfhd);
      auto mifhd = Kokkos::create_mirror_view(d_ifhd);
      Kokkos::deep_copy(mrfhd, d_rfhd);
      Kokkos::deep_copy(mifhd, d_ifhd);
      for (int i = 0; i < samples; i++) { rfhd[i] = mrfhd[i]; ifhd[i] = mifhd[i]; }
    }

    if (verify) {
      printf("Computing root mean square error between host and device results.\n");
      printf("This will take a while..\n");
      for (int n = 0; n < samples; n++) {
        float r = h_rfhd[n], im = h_ifhd[n];
        for (int m = 0; m < voxels; m++) {
          float e = 2.f*(float)M_PI*(h_kx[m]*h_x[n]+h_ky[m]*h_y[n]+h_kz[m]*h_z[n]);
          float c = cosf(e), s = sinf(e);
          r  += h_rmu[m]*c - h_imu[m]*s;
          im += h_imu[m]*c + h_rmu[m]*s;
        }
        h_rfhd[n] = r; h_ifhd[n] = im;
      }
      float err = 0.f;
      for (int i = 0; i < samples; i++)
        err += (h_rfhd[i]-rfhd[i])*(h_rfhd[i]-rfhd[i])
             + (h_ifhd[i]-ifhd[i])*(h_ifhd[i]-ifhd[i]);
      printf("RMSE = %f\n", sqrtf(err / (2*samples)));
    }
  }
  Kokkos::finalize();

  free(h_rmu); free(h_imu); free(h_kx); free(h_ky); free(h_kz);
  free(h_rfhd); free(h_ifhd); free(rfhd); free(ifhd);
  free(h_x); free(h_y); free(h_z);
  return 0;
}
