#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

void haccmk_gold(
    int count1, float xxi, float yyi, float zzi,
    float fsrrmax2, float mp_rsm2,
    float* xx1, float* yy1, float* zz1, float* mass1,
    float* dxi, float* dyi, float* dzi)
{
  const float ma0=0.269327f, ma1=-0.0750978f, ma2=0.0114808f,
              ma3=-0.00109313f, ma4=0.0000605491f, ma5=-0.00000147177f;
  float xi=0,yi=0,zi=0;
  for (int j = 0; j < count1; j++) {
    float dxc = xx1[j]-xxi, dyc=yy1[j]-yyi, dzc=zz1[j]-zzi;
    float r2 = dxc*dxc + dyc*dyc + dzc*dzc;
    float m = (r2 < fsrrmax2) ? mass1[j] : 0.f;
    float f = r2 + mp_rsm2;
    f = m*(1.f/(f*sqrtf(f)) - (ma0+r2*(ma1+r2*(ma2+r2*(ma3+r2*(ma4+r2*ma5))))));
    xi += f*dxc; yi += f*dyc; zi += f*dzc;
  }
  *dxi=xi; *dyi=yi; *dzi=zi;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n1 = 784;   // outer (global size)
  const int n2 = 15000; // inner loop count (ilp)
  printf("Outer loop count is set %d\n", n1);
  printf("Inner loop count is set %d\n", n2);

  float* xx   = (float*)malloc(sizeof(float)*n2);
  float* yy   = (float*)malloc(sizeof(float)*n2);
  float* zz   = (float*)malloc(sizeof(float)*n2);
  float* mass = (float*)malloc(sizeof(float)*n2);
  float* vx2  = (float*)malloc(sizeof(float)*n2);
  float* vy2  = (float*)malloc(sizeof(float)*n2);
  float* vz2  = (float*)malloc(sizeof(float)*n2);
  float* vx2_hw = (float*)malloc(sizeof(float)*n2);
  float* vy2_hw = (float*)malloc(sizeof(float)*n2);
  float* vz2_hw = (float*)malloc(sizeof(float)*n2);

  const float fcoeff   = 0.23f;
  const float fsrrmax2 = 0.5f;
  const float mp_rsm2  = 0.03f;
  const float dx1      = 1.0f/n2, dy1=2.0f/n2, dz1=3.0f/n2;

  xx[0]=0; yy[0]=0; zz[0]=0; mass[0]=2.f;
  for (int i=1; i<n2; i++) {
    xx[i]   = xx[i-1]   + dx1;
    yy[i]   = yy[i-1]   + dy1;
    zz[i]   = zz[i-1]   + dz1;
    mass[i] = (float)i*0.01f + xx[i];
  }
  for (int i=0; i<n2; i++) vx2[i]=vy2[i]=vz2[i]=vx2_hw[i]=vy2_hw[i]=vz2_hw[i]=0.f;

  // Reference on CPU
  for (int i=0; i<n1; i++) {
    float dx2,dy2,dz2;
    haccmk_gold(n2, xx[i],yy[i],zz[i], fsrrmax2,mp_rsm2, xx,yy,zz,mass, &dx2,&dy2,&dz2);
    vx2[i] += dx2*fcoeff;
    vy2[i] += dy2*fcoeff;
    vz2[i] += dz2*fcoeff;
  }

  // Kokkos kernel
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> xx_d  ("xx",   n2);
    Kokkos::View<float*> yy_d  ("yy",   n2);
    Kokkos::View<float*> zz_d  ("zz",   n2);
    Kokkos::View<float*> mass_d("mass", n2);
    Kokkos::View<float*> vx2_d ("vx2",  n1);
    Kokkos::View<float*> vy2_d ("vy2",  n1);
    Kokkos::View<float*> vz2_d ("vz2",  n1);

    {
      auto m = Kokkos::create_mirror_view(xx_d);
      for(int i=0;i<n2;i++) m(i)=xx[i]; Kokkos::deep_copy(xx_d,m);
    }
    {
      auto m = Kokkos::create_mirror_view(yy_d);
      for(int i=0;i<n2;i++) m(i)=yy[i]; Kokkos::deep_copy(yy_d,m);
    }
    {
      auto m = Kokkos::create_mirror_view(zz_d);
      for(int i=0;i<n2;i++) m(i)=zz[i]; Kokkos::deep_copy(zz_d,m);
    }
    {
      auto m = Kokkos::create_mirror_view(mass_d);
      for(int i=0;i<n2;i++) m(i)=mass[i]; Kokkos::deep_copy(mass_d,m);
    }
    // zero out velocity
    {
      auto m = Kokkos::create_mirror_view(vx2_d);
      for(int i=0;i<n1;i++) m(i)=0.f; Kokkos::deep_copy(vx2_d,m);
    }
    {
      auto m = Kokkos::create_mirror_view(vy2_d);
      for(int i=0;i<n1;i++) m(i)=0.f; Kokkos::deep_copy(vy2_d,m);
    }
    {
      auto m = Kokkos::create_mirror_view(vz2_d);
      for(int i=0;i<n1;i++) m(i)=0.f; Kokkos::deep_copy(vz2_d,m);
    }

    const int ilp = n2;

    float total_time = 0.f;
    for (int r = 0; r < repeat; r++) {
      // Re-zero velocities each repeat (matching OMP #pragma omp target update)
      Kokkos::deep_copy(vx2_d, 0.f);
      Kokkos::deep_copy(vy2_d, 0.f);
      Kokkos::deep_copy(vz2_d, 0.f);

      auto t0 = std::chrono::steady_clock::now();

      Kokkos::parallel_for("haccmk", n1,
        KOKKOS_LAMBDA(const int i) {
          const float ma0=0.269327f, ma1=-0.0750978f, ma2=0.0114808f,
                      ma3=-0.00109313f, ma4=0.0000605491f, ma5=-0.00000147177f;
          float xi=0,yi=0,zi=0;
          float xxi = xx_d(i), yyi = yy_d(i), zzi = zz_d(i);
          for (int j=0; j<ilp; j++) {
            float dxc = xx_d(j)-xxi, dyc=yy_d(j)-yyi, dzc=zz_d(j)-zzi;
            float r2 = dxc*dxc + dyc*dyc + dzc*dzc;
            float m = (r2 < fsrrmax2) ? mass_d(j) : 0.f;
            float f = r2 + mp_rsm2;
            f = m*(1.f/(f*sqrtf(f)) - (ma0+r2*(ma1+r2*(ma2+r2*(ma3+r2*(ma4+r2*ma5))))));
            xi += f*dxc; yi += f*dyc; zi += f*dzc;
          }
          vx2_d(i) += xi * fcoeff;
          vy2_d(i) += yi * fcoeff;
          vz2_d(i) += zi * fcoeff;
        });
      Kokkos::fence();

      auto t1 = std::chrono::steady_clock::now();
      total_time += (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    }
    printf("Average kernel execution time %f (s)\n", (total_time*1e-9f)/repeat);

    // Copy back last iteration results
    {
      auto m = Kokkos::create_mirror_view(vx2_d);
      Kokkos::deep_copy(m, vx2_d);
      for(int i=0;i<n1;i++) vx2_hw[i]=m(i);
    }
    {
      auto m = Kokkos::create_mirror_view(vy2_d);
      Kokkos::deep_copy(m, vy2_d);
      for(int i=0;i<n1;i++) vy2_hw[i]=m(i);
    }
    {
      auto m = Kokkos::create_mirror_view(vz2_d);
      Kokkos::deep_copy(m, vz2_d);
      for(int i=0;i<n1;i++) vz2_hw[i]=m(i);
    }
  }
  Kokkos::finalize();

  // Verify
  int error = 0;
  const float eps = 1e-1f;
  for (int i=0; i<n1; i++) {
    if (fabsf(vx2[i]-vx2_hw[i])>eps) { printf("error at vx2[%d] %f %f\n",i,vx2[i],vx2_hw[i]); error=1; break; }
    if (fabsf(vy2[i]-vy2_hw[i])>eps) { printf("error at vy2[%d] %f %f\n",i,vy2[i],vy2_hw[i]); error=1; break; }
    if (fabsf(vz2[i]-vz2_hw[i])>eps) { printf("error at vz2[%d] %f %f\n",i,vz2[i],vz2_hw[i]); error=1; break; }
  }

  free(xx); free(yy); free(zz); free(mass);
  free(vx2); free(vy2); free(vz2);
  free(vx2_hw); free(vy2_hw); free(vz2_hw);

  printf("%s\n", error ? "FAIL" : "PASS");
  return 0;
}
