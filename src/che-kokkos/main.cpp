#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <random>
#include <fstream>
#include <Kokkos_Core.hpp>

#define DATAXSIZE 256
#define DATAYSIZE 256
#define DATAZSIZE 256

KOKKOS_INLINE_FUNCTION
double Laplacian(const double *c, double dx, double dy, double dz, int x, int y, int z)
{
  const int nx = DATAXSIZE-1, ny = DATAYSIZE-1, nz = DATAZSIZE-1;
  int xp = x+1, xn = x-1, yp = y+1, yn = y-1, zp = z+1, zn = z-1;
  if (xp>nx) xp=0; if (yp>ny) yp=0; if (zp>nz) zp=0;
  if (xn<0)  xn=nx; if (yn<0) yn=ny; if (zn<0) zn=nz;
#define IDX(iz,iy,ix) ((iz)*DATAYSIZE*DATAXSIZE + (iy)*DATAXSIZE + (ix))
  double cxx = (c[IDX(z,y,xp)] + c[IDX(z,y,xn)] - 2.0*c[IDX(z,y,x)]) / (dx*dx);
  double cyy = (c[IDX(z,yp,x)] + c[IDX(z,yn,x)] - 2.0*c[IDX(z,y,x)]) / (dy*dy);
  double czz = (c[IDX(zp,y,x)] + c[IDX(zn,y,x)] - 2.0*c[IDX(z,y,x)]) / (dz*dz);
#undef IDX
  return cxx + cyy + czz;
}

KOKKOS_INLINE_FUNCTION double GradX(const double *p,double dx,double dy,double dz,int x,int y,int z){
  int xp=x+1,xn=x-1;
  if(xp>DATAXSIZE-1)xp=0; if(xn<0)xn=DATAXSIZE-1;
  return (p[z*DATAYSIZE*DATAXSIZE+y*DATAXSIZE+xp]-p[z*DATAYSIZE*DATAXSIZE+y*DATAXSIZE+xn])/(2.0*dx);
}
KOKKOS_INLINE_FUNCTION double GradY(const double *p,double dx,double dy,double dz,int x,int y,int z){
  int yp=y+1,yn=y-1;
  if(yp>DATAYSIZE-1)yp=0; if(yn<0)yn=DATAYSIZE-1;
  return (p[z*DATAYSIZE*DATAXSIZE+yp*DATAXSIZE+x]-p[z*DATAYSIZE*DATAXSIZE+yn*DATAXSIZE+x])/(2.0*dy);
}
KOKKOS_INLINE_FUNCTION double GradZ(const double *p,double dx,double dy,double dz,int x,int y,int z){
  int zp=z+1,zn=z-1;
  if(zp>DATAZSIZE-1)zp=0; if(zn<0)zn=DATAZSIZE-1;
  return (p[(zp)*DATAYSIZE*DATAXSIZE+y*DATAXSIZE+x]-p[(zn)*DATAYSIZE*DATAXSIZE+y*DATAXSIZE+x])/(2.0*dz);
}
KOKKOS_INLINE_FUNCTION double freeEnergy(double c,double e_AA,double e_BB,double e_AB){
  return (((9.0/4.0)*((c*c+2.0*c+1.0)*e_AA+(c*c-2.0*c+1.0)*e_BB+2.0*(1.0-c*c)*e_AB))+
          ((3.0/2.0)*c*c)+((3.0/12.0)*c*c*c*c));
}

int main(int argc, char *argv[])
{
  const double dx=1.0,dy=1.0,dz=1.0,dt=0.01;
  const double e_AA=-(2.0/9.0),e_BB=-(2.0/9.0),e_AB=(2.0/9.0);
  const int t_f = atoi(argv[1]);
#ifndef DEBUG
  const int t_freq = t_f;
#else
  const int t_freq = 10;
#endif
  const double gamma=0.5, D=1.0;

  const int vol = DATAXSIZE*DATAYSIZE*DATAZSIZE;

  double *cold  = (double*) malloc(vol*sizeof(double));
  double *cnew  = (double*) malloc(vol*sizeof(double));
  double *muold = (double*) malloc(vol*sizeof(double));
  double *fold  = (double*) malloc(vol*sizeof(double));

  // initialize
  srand(2);
  for (int k=0;k<DATAZSIZE;k++)
    for (int j=0;j<DATAYSIZE;j++)
      for (int i=0;i<DATAXSIZE;i++) {
        double f = (double)rand()/RAND_MAX;
        cold[k*DATAYSIZE*DATAXSIZE+j*DATAXSIZE+i] = -1.0 + 2.0*f;
      }

  std::string name_c="./out/integral_c.txt", name_mu="./out/integral_mu.txt",
              name_f="./out/integral_f.txt";
  std::ofstream ofile_c(name_c), ofile_mu(name_mu), ofile_f(name_f);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_co("co", vol);
    Kokkos::View<double*> d_cn("cn", vol);
    Kokkos::View<double*> d_mu("mu", vol);
    Kokkos::View<double*> d_f ("f",  vol);

    auto h_co = Kokkos::create_mirror_view(d_co);
    for (int i=0;i<vol;i++) h_co(i)=cold[i];
    Kokkos::deep_copy(d_co, h_co);

    auto start = std::chrono::steady_clock::now();

    using MDpolicy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    MDpolicy policy({0,0,0},{DATAZSIZE,DATAYSIZE,DATAXSIZE});

    for (int t = 0; t < t_f; t++) {
      // chemicalPotential
      Kokkos::parallel_for("chePot", policy,
        KOKKOS_LAMBDA(int idz, int idy, int idx) {
          int id = idz*DATAYSIZE*DATAXSIZE + idy*DATAXSIZE + idx;
          d_mu(id) = 4.5*((d_co(id)+1.0)*e_AA+(d_co(id)-1)*e_BB-2.0*d_co(id)*e_AB)
            + 3.0*d_co(id) + d_co(id)*d_co(id)*d_co(id)
            - gamma*Laplacian(d_co.data(),dx,dy,dz,idx,idy,idz);
        });

      // localFreeEnergy
      Kokkos::parallel_for("cheF", policy,
        KOKKOS_LAMBDA(int idz, int idy, int idx) {
          int id = idz*DATAYSIZE*DATAXSIZE + idy*DATAXSIZE + idx;
          double gx=GradX(d_co.data(),dx,dy,dz,idx,idy,idz);
          double gy=GradY(d_co.data(),dx,dy,dz,idx,idy,idz);
          double gz=GradZ(d_co.data(),dx,dy,dz,idx,idy,idz);
          d_f(id) = freeEnergy(d_co(id),e_AA,e_BB,e_AB)
            + (gamma/2.0)*(gx*gx+gy*gy+gz*gz);
        });

      // cahnHilliard
      Kokkos::parallel_for("cheNew", policy,
        KOKKOS_LAMBDA(int idz, int idy, int idx) {
          int id = idz*DATAYSIZE*DATAXSIZE + idy*DATAXSIZE + idx;
          d_cn(id) = d_co(id) + dt*D*Laplacian(d_mu.data(),dx,dy,dz,idx,idy,idz);
        });

      if (t > 0 && t % (t_freq-1) == 0) {
        auto hcn = Kokkos::create_mirror_view(d_cn);
        auto hmu = Kokkos::create_mirror_view(d_mu);
        auto hf  = Kokkos::create_mirror_view(d_f);
        Kokkos::deep_copy(hcn, d_cn);
        Kokkos::deep_copy(hmu, d_mu);
        Kokkos::deep_copy(hf,  d_f);
        double ic=0,im=0,ifc=0;
        for (int i=0;i<vol;i++){ic+=hcn(i);im+=hmu(i);ifc+=hf(i);}
        ofile_c<<t<<","<<ic<<std::endl;
        ofile_mu<<t<<","<<im<<std::endl;
        ofile_f<<t<<","<<ifc<<std::endl;
      }

      // Swap: cnew -> cold
      Kokkos::parallel_for("cheSwap", vol, KOKKOS_LAMBDA(int id) {
        double tmp = d_cn(id);
        d_cn(id) = d_co(id);
        d_co(id) = tmp;
      });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Kernel execution time on the GPU (%d iterations) = %.3f (s)\n", t_f, time * 1e-9f);
  }
  Kokkos::finalize();

  free(cold); free(cnew); free(muold); free(fold);
  return 0;
}
