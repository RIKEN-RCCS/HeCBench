#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const float wx = -0.3f, wy = -0.6f, wz = 0.15f;
  const float norm_inv = 1.f / sqrtf(wx*wx + wy*wy + wz*wz);
  const float wax = wx*norm_inv, way = wy*norm_inv, waz = wz*norm_inv;
  const float angle = 0.5f;

  float *h_px = (float*)malloc(n*sizeof(float));
  float *h_py = (float*)malloc(n*sizeof(float));
  float *h_pz = (float*)malloc(n*sizeof(float));

  srand(123);
  for (int i = 0; i < n; i++) {
    float a = (float)rand(), b = (float)rand(), c = (float)rand();
    float d = sqrtf(a*a + b*b + c*c);
    h_px[i] = a/d; h_py[i] = b/d; h_pz[i] = c/d;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_px("px",n), d_py("py",n), d_pz("pz",n);
    Kokkos::View<float*> d_qx("qx",n), d_qy("qy",n), d_qz("qz",n);

    {
      auto mx = Kokkos::create_mirror_view(d_px);
      auto my = Kokkos::create_mirror_view(d_py);
      auto mz = Kokkos::create_mirror_view(d_pz);
      auto mx2 = Kokkos::create_mirror_view(d_qx);
      auto my2 = Kokkos::create_mirror_view(d_qy);
      auto mz2 = Kokkos::create_mirror_view(d_qz);
      for (int i=0;i<n;i++){mx[i]=h_px[i];my[i]=h_py[i];mz[i]=h_pz[i];}
      for (int i=0;i<n;i++){mx2[i]=h_px[i];my2[i]=h_py[i];mz2[i]=h_pz[i];}
      Kokkos::deep_copy(d_px,mx); Kokkos::deep_copy(d_py,my); Kokkos::deep_copy(d_pz,mz);
      Kokkos::deep_copy(d_qx,mx2); Kokkos::deep_copy(d_qy,my2); Kokkos::deep_copy(d_qz,mz2);
    }

    // float3 kernel
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("rotate3", n, KOKKOS_LAMBDA(int i) {
        float s, c;
        sincosf(angle, &s, &c);
        float px=d_px[i], py=d_py[i], pz=d_pz[i];
        float mc = 1.f - c;
        float m1=c+wax*wax*mc,  m2=waz*s+wax*way*mc,  m3=-way*s+wax*waz*mc;
        float m4=-waz*s+wax*way*mc, m5=c+way*way*mc,   m6=wax*s+way*waz*mc;
        float m7=way*s+wax*waz*mc,  m8=-wax*s+way*waz*mc, m9=c+waz*waz*mc;
        d_px[i] = px*m1+py*m2+pz*m3;
        d_py[i] = px*m4+py*m5+pz*m6;
        d_pz[i] = px*m7+py*m8+pz*m9;
      });
      Kokkos::fence();
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (float3): %f (us)\n", (time * 1e-3f) / repeat);

    // float4 kernel (same math, uses qx/qy/qz)
    start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("rotate4", n, KOKKOS_LAMBDA(int i) {
        float s, c;
        sincosf(angle, &s, &c);
        float px=d_qx[i], py=d_qy[i], pz=d_qz[i];
        float mc = 1.f - c;
        float m1=c+wax*wax*mc,  m2=waz*s+wax*way*mc,  m3=-way*s+wax*waz*mc;
        float m4=-waz*s+wax*way*mc, m5=c+way*way*mc,   m6=wax*s+way*waz*mc;
        float m7=way*s+wax*waz*mc,  m8=-wax*s+way*waz*mc, m9=c+waz*waz*mc;
        d_qx[i] = px*m1+py*m2+pz*m3;
        d_qy[i] = px*m4+py*m5+pz*m6;
        d_qz[i] = px*m7+py*m8+pz*m9;
      });
      Kokkos::fence();
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (float4): %f (us)\n", (time * 1e-3f) / repeat);
  }
  Kokkos::finalize();

  free(h_px); free(h_py); free(h_pz);
  return 0;
}
