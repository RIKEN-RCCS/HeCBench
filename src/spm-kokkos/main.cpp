#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <chrono>

#define NUM_THREADS 128
#define NUM_BLOCKS  256

struct int3 { int x, y, z; };

KOKKOS_INLINE_FUNCTION
float interp(const int3 d, const unsigned char f[], float x, float y, float z)
{
  int ix = (int)floorf(x); float dx1 = x - ix; float dx2 = 1.f - dx1;
  int iy = (int)floorf(y); float dy1 = y - iy; float dy2 = 1.f - dy1;
  int iz = (int)floorf(z); float dz1 = z - iz; float dz2 = 1.f - dz1;

  const unsigned char *ff = f + ix - 1 + d.x * (iy - 1 + d.y * (iz - 1));
  int k222 = ff[0],      k122 = ff[1];
  int k212 = ff[d.x],    k112 = ff[d.x + 1];
  ff += d.x * d.y;
  int k221 = ff[0],      k121 = ff[1];
  int k211 = ff[d.x],    k111 = ff[d.x + 1];

  return (((k222*dx2 + k122*dx1)*dy2 + (k212*dx2 + k112*dx1)*dy1))*dz2 +
         (((k221*dx2 + k121*dx1)*dy2 + (k211*dx2 + k111*dx1)*dy1))*dz1;
}

void spm_reference(
  const float *M, const int data_size,
  const unsigned char *g_d, const unsigned char *f_d,
  const int3 dg, const int3 df,
  unsigned char *ivf_d, unsigned char *ivg_d, bool *data_threshold_d)
{
  const float ran[] = {
    0.656619f,0.891183f,0.488144f,0.992646f,0.373326f,0.531378f,0.181316f,0.501944f,0.422195f,
    0.660427f,0.673653f,0.95733f,0.191866f,0.111216f,0.565054f,0.969166f,0.0237439f,0.870216f,
    0.0268766f,0.519529f,0.192291f,0.715689f,0.250673f,0.933865f,0.137189f,0.521622f,0.895202f,
    0.942387f,0.335083f,0.437364f,0.471156f,0.14931f,0.135864f,0.532498f,0.725789f,0.398703f,
    0.358419f,0.285279f,0.868635f,0.626413f,0.241172f,0.978082f,0.640501f,0.229849f,0.681335f,
    0.665823f,0.134718f,0.0224933f,0.262199f,0.116515f,0.0693182f,0.85293f,0.180331f,0.0324186f,
    0.733926f,0.536517f,0.27603f,0.368458f,0.0128863f,0.889206f,0.866021f,0.254247f,0.569481f,
    0.159265f,0.594364f,0.3311f,0.658613f,0.863634f,0.567623f,0.980481f,0.791832f,0.152594f,
    0.833027f,0.191863f,0.638987f,0.669f,0.772088f,0.379818f,0.441585f,0.48306f,0.608106f,
    0.175996f,0.00202556f,0.790224f,0.513609f,0.213229f,0.10345f,0.157337f,0.407515f,0.407757f,
    0.0526927f,0.941815f,0.149972f,0.384374f,0.311059f,0.168534f,0.896648f};

  int x_datasize = dg.x - 2;
  int y_datasize = dg.y - 2;

  for (int i = 0; i < data_size; i++) {
    float xx_temp = (i % x_datasize) + 1.f;
    float yy_temp = ((int)floorf((float)i / x_datasize) % y_datasize) + 1.f;
    float zz_temp = (floorf((float)i / x_datasize)) / y_datasize + 1.f;

    float rx = xx_temp + ran[i % 97];
    float ry = yy_temp + ran[i % 97];
    float rz = zz_temp + ran[i % 97];

    float xp = M[0]*rx + M[4]*ry + M[ 8]*rz + M[12];
    float yp = M[1]*rx + M[5]*ry + M[ 9]*rz + M[13];
    float zp = M[2]*rx + M[6]*ry + M[10]*rz + M[14];

    if (zp >= 1.f && zp < df.z && yp >= 1.f && yp < df.y && xp >= 1.f && xp < df.x) {
      ivf_d[i] = (unsigned char)floorf(interp(df, f_d, xp, yp, zp) + 0.5f);
      ivg_d[i] = (unsigned char)floorf(interp(dg, g_d, rx, ry, rz) + 0.5f);
      data_threshold_d[i] = true;
    } else {
      ivf_d[i] = 0;
      ivg_d[i] = 0;
      data_threshold_d[i] = false;
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <dimension> <repeat>\n", argv[0]);
    return 1;
  }
  int v      = atoi(argv[1]);
  int repeat = atoi(argv[2]);

  int3 g_vol = {v, v, v};
  int3 f_vol = {v, v, v};

  const int data_size = (g_vol.x + 1) * (g_vol.y + 1) * (g_vol.z + 5);
  const int vol_size  = g_vol.x * g_vol.y * g_vol.z;

  int *hist_d = (int*)malloc(65536 * sizeof(int));
  int *hist_h = (int*)malloc(65536 * sizeof(int));
  memset(hist_d, 0, 65536 * sizeof(int));
  memset(hist_h, 0, 65536 * sizeof(int));

  srand(123);
  float M_h[16];
  for (int i = 0; i < 16; i++) M_h[i] = (float)rand() / (float)RAND_MAX;

  unsigned char *f_h = (unsigned char*)malloc(data_size);
  unsigned char *g_h = (unsigned char*)malloc(data_size);
  for (int i = 0; i < data_size; i++) {
    f_h[i] = rand() % 256;
    g_h[i] = rand() % 256;
  }

  unsigned char *ivf_h        = (unsigned char*)malloc(vol_size);
  unsigned char *ivg_h        = (unsigned char*)malloc(vol_size);
  bool          *data_thresh_h = (bool*)malloc(vol_size * sizeof(bool));

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<float*>         M_d("M", 16);
    Kokkos::View<unsigned char*> f_d("f", data_size);
    Kokkos::View<unsigned char*> g_d("g", data_size);
    Kokkos::View<unsigned char*> ivf_d("ivf", vol_size);
    Kokkos::View<unsigned char*> ivg_d("ivg", vol_size);
    Kokkos::View<bool*>          thresh_d("thresh", vol_size);

    // Host→device copies
    {
      auto M_hv   = Kokkos::View<float*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(M_h, 16);
      auto f_hv   = Kokkos::View<unsigned char*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(f_h, data_size);
      auto g_hv   = Kokkos::View<unsigned char*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(g_h, data_size);
      Kokkos::deep_copy(M_d, M_hv);
      Kokkos::deep_copy(f_d, f_hv);
      Kokkos::deep_copy(g_d, g_hv);
    }

    // Capture by value for the lambda
    const int3 dg = g_vol;
    const int3 df = f_vol;

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("spm",
        Kokkos::RangePolicy<>(0, vol_size),
        KOKKOS_LAMBDA(int i) {
          const float ran[] = {
            0.656619f,0.891183f,0.488144f,0.992646f,0.373326f,0.531378f,0.181316f,0.501944f,0.422195f,
            0.660427f,0.673653f,0.95733f,0.191866f,0.111216f,0.565054f,0.969166f,0.0237439f,0.870216f,
            0.0268766f,0.519529f,0.192291f,0.715689f,0.250673f,0.933865f,0.137189f,0.521622f,0.895202f,
            0.942387f,0.335083f,0.437364f,0.471156f,0.14931f,0.135864f,0.532498f,0.725789f,0.398703f,
            0.358419f,0.285279f,0.868635f,0.626413f,0.241172f,0.978082f,0.640501f,0.229849f,0.681335f,
            0.665823f,0.134718f,0.0224933f,0.262199f,0.116515f,0.0693182f,0.85293f,0.180331f,0.0324186f,
            0.733926f,0.536517f,0.27603f,0.368458f,0.0128863f,0.889206f,0.866021f,0.254247f,0.569481f,
            0.159265f,0.594364f,0.3311f,0.658613f,0.863634f,0.567623f,0.980481f,0.791832f,0.152594f,
            0.833027f,0.191863f,0.638987f,0.669f,0.772088f,0.379818f,0.441585f,0.48306f,0.608106f,
            0.175996f,0.00202556f,0.790224f,0.513609f,0.213229f,0.10345f,0.157337f,0.407515f,0.407757f,
            0.0526927f,0.941815f,0.149972f,0.384374f,0.311059f,0.168534f,0.896648f};

          int x_datasize = dg.x - 2;
          int y_datasize = dg.y - 2;

          float xx_temp = (i % x_datasize) + 1.f;
          float yy_temp = ((int)floorf((float)i / x_datasize) % y_datasize) + 1.f;
          float zz_temp = (floorf((float)i / x_datasize)) / y_datasize + 1.f;

          float rx = xx_temp + ran[i % 97];
          float ry = yy_temp + ran[i % 97];
          float rz = zz_temp + ran[i % 97];

          float xp = M_d(0)*rx + M_d(4)*ry + M_d( 8)*rz + M_d(12);
          float yp = M_d(1)*rx + M_d(5)*ry + M_d( 9)*rz + M_d(13);
          float zp = M_d(2)*rx + M_d(6)*ry + M_d(10)*rz + M_d(14);

          if (zp >= 1.f && zp < df.z && yp >= 1.f && yp < df.y && xp >= 1.f && xp < df.x) {
            ivf_d(i) = (unsigned char)floorf(interp(df, f_d.data(), xp, yp, zp) + 0.5f);
            ivg_d(i) = (unsigned char)floorf(interp(dg, g_d.data(), rx, ry, rz) + 0.5f);
            thresh_d(i) = true;
          } else {
            ivf_d(i) = 0;
            ivg_d(i) = 0;
            thresh_d(i) = false;
          }
        });
    }
    Kokkos::fence();

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

    // Copy results back
    {
      auto ivf_hv = Kokkos::View<unsigned char*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(ivf_h, vol_size);
      auto ivg_hv = Kokkos::View<unsigned char*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(ivg_h, vol_size);
      auto th_hv  = Kokkos::View<bool*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(data_thresh_h, vol_size);
      Kokkos::deep_copy(ivf_hv, ivf_d);
      Kokkos::deep_copy(ivg_hv, ivg_d);
      Kokkos::deep_copy(th_hv,  thresh_d);
    }
  }
  Kokkos::finalize();

  // Histogram on device results
  int count = 0;
  for (int i = 0; i < vol_size; i++) {
    if (data_thresh_h[i]) {
      hist_d[ivf_h[i] + ivg_h[i] * 256]++;
      count++;
    }
  }
  printf("Device count: %d\n", count);

  // Reference
  count = 0;
  spm_reference(M_h, vol_size, g_h, f_h, g_vol, f_vol, ivf_h, ivg_h, data_thresh_h);
  for (int i = 0; i < vol_size; i++) {
    if (data_thresh_h[i]) {
      hist_h[ivf_h[i] + ivg_h[i] * 256]++;
      count++;
    }
  }
  printf("Host count: %d\n", count);

  int max_diff = 0;
  for (int i = 0; i < 65536; i++)
    max_diff = std::max(max_diff, abs(hist_h[i] - hist_d[i]));
  printf("Maximum difference %d\n", max_diff);

  free(hist_h);
  free(hist_d);
  free(ivf_h);
  free(ivg_h);
  free(g_h);
  free(f_h);
  free(data_thresh_h);
  return 0;
}
