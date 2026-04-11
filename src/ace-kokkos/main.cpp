#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

//define the data set size (cubic volume)
#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400

#define SQ(x) ((x)*(x))

typedef double nRarray[DATAYSIZE][DATAXSIZE];

// Flat index macro for 3D arrays stored as [ix][iy][iz]
#define IDX(ix,iy,iz) ((ix)*DATAYSIZE*DATAXSIZE + (iy)*DATAXSIZE + (iz))

#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

KOKKOS_INLINE_FUNCTION
double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

KOKKOS_INLINE_FUNCTION
double GradientX(const double* phi,
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[IDX(x+1,y,z)] - phi[IDX(x-1,y,z)]) / (2.0*dx);
}

KOKKOS_INLINE_FUNCTION
double GradientY(const double* phi,
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[IDX(x,y+1,z)] - phi[IDX(x,y-1,z)]) / (2.0*dy);
}

KOKKOS_INLINE_FUNCTION
double GradientZ(const double* phi,
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[IDX(x,y,z+1)] - phi[IDX(x,y,z-1)]) / (2.0*dz);
}

KOKKOS_INLINE_FUNCTION
double Divergence(const double* phix,
                  const double* phiy,
                  const double* phiz,
                  double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) +
         GradientY(phiy,dx,dy,dz,x,y,z) +
         GradientZ(phiz,dx,dy,dz,x,y,z);
}

KOKKOS_INLINE_FUNCTION
double Laplacian(const double* phi,
                 double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[IDX(x+1,y,z)] + phi[IDX(x-1,y,z)] - 2.0 * phi[IDX(x,y,z)]) / SQ(dx);
  double phiyy = (phi[IDX(x,y+1,z)] + phi[IDX(x,y-1,z)] - 2.0 * phi[IDX(x,y,z)]) / SQ(dy);
  double phizz = (phi[IDX(x,y,z+1)] + phi[IDX(x,y,z-1)] - 2.0 * phi[IDX(x,y,z)]) / SQ(dz);
  return phixx + phiyy + phizz;
}

KOKKOS_INLINE_FUNCTION
double An(double phix, double phiy, double phiz, double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
    return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
           ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
           ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
  }
  else
  {
    return (1.0-((5.0/3.0)*epsilon));
  }
}

KOKKOS_INLINE_FUNCTION
double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An(phix,phiy,phiz,epsilon));
}

KOKKOS_INLINE_FUNCTION
double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An(phix,phiy,phiz,epsilon));
}

KOKKOS_INLINE_FUNCTION
double dFunc(double l, double m, double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0){
    return (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
  }
  else
  {
    return 0.0;
  }
}

void initializationPhi(double* phi, double r0)
{
  for (int ix = 0; ix < DATAXSIZE; ix++)
    for (int iy = 0; iy < DATAYSIZE; iy++)
      for (int iz = 0; iz < DATAZSIZE; iz++) {
        double r = sqrt(SQ(ix-0.5*DATAXSIZE) + SQ(iy-0.5*DATAYSIZE) + SQ(iz-0.5*DATAZSIZE));
        if (r < r0)
          phi[IDX(ix,iy,iz)] = 1.0;
        else
          phi[IDX(ix,iy,iz)] = -1.0;
      }
}

void initializationU(double* u, double r0, double delta)
{
  for (int ix = 0; ix < DATAXSIZE; ix++)
    for (int iy = 0; iy < DATAYSIZE; iy++)
      for (int iz = 0; iz < DATAZSIZE; iz++) {
        double r = sqrt(SQ(ix-0.5*DATAXSIZE) + SQ(iy-0.5*DATAYSIZE) + SQ(iz-0.5*DATAZSIZE));
        if (r < r0)
          u[IDX(ix,iy,iz)] = 0.0;
        else
          u[IDX(ix,iy,iz)] = -delta * (1.0 - exp(-(r-r0)));
      }
}

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {
    const int num_steps = atoi(argv[1]);  //6000;
    const double dx = 0.4;
    const double dy = 0.4;
    const double dz = 0.4;
    const double dt = 0.01;
    const double delta = 0.8;
    const double r0 = 5.0;
    const double epsilon = 0.07;
    const double W0 = 1.0;
    const double beta0 = 0.0;
    const double D = 2.0;
    const double d0 = 0.5;
    const double a1 = 1.25 / sqrt(2.0);
    const double a2 = 0.64;
    const double lambda = (W0*a1)/(d0);
    const double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));

    // overall data set sizes
    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;
    const int vol = nx * ny * nz;
    const size_t vol_in_bytes = sizeof(double) * vol;

    // storage for result stored on host
    double *phi_host = (double *)malloc(vol_in_bytes);
    double *u_host = (double *)malloc(vol_in_bytes);
    initializationPhi(phi_host, r0);
    initializationU(u_host, r0, delta);

#ifdef VERIFY
    nRarray *phi_ref = (nRarray *)malloc(vol_in_bytes);
    nRarray *u_ref = (nRarray *)malloc(vol_in_bytes);
    memcpy(phi_ref, phi_host, vol_in_bytes);
    memcpy(u_ref, u_host, vol_in_bytes);
    reference(phi_ref, u_ref, vol, num_steps);
#endif

    auto offload_start = std::chrono::steady_clock::now();

    // Kokkos device views
    Kokkos::View<double*> d_phiold("phiold", vol);
    Kokkos::View<double*> d_phinew("phinew", vol);
    Kokkos::View<double*> d_uold("uold", vol);
    Kokkos::View<double*> d_unew("unew", vol);
    Kokkos::View<double*> d_Fx("Fx", vol);
    Kokkos::View<double*> d_Fy("Fy", vol);
    Kokkos::View<double*> d_Fz("Fz", vol);

    // Create host mirrors and copy initial data
    auto h_phiold = Kokkos::create_mirror_view(d_phiold);
    auto h_uold = Kokkos::create_mirror_view(d_uold);

    for (int i = 0; i < vol; i++) {
      h_phiold(i) = phi_host[i];
      h_uold(i) = u_host[i];
    }

    Kokkos::deep_copy(d_phiold, h_phiold);
    Kokkos::deep_copy(d_uold, h_uold);

    int t = 0;

    auto start = std::chrono::steady_clock::now();

    using policy3d = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    while (t <= num_steps) {

      // calculateForce
      Kokkos::parallel_for("calculateForce",
        policy3d({0, 0, 0}, {DATAXSIZE, DATAYSIZE, DATAZSIZE}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) &&
              (iz < (DATAZSIZE-1)) && (ix > 0) &&
              (iy > 0) && (iz > 0)) {

            double phix = GradientX(d_phiold.data(),dx,dy,dz,ix,iy,iz);
            double phiy = GradientY(d_phiold.data(),dx,dy,dz,ix,iy,iz);
            double phiz = GradientZ(d_phiold.data(),dx,dy,dz,ix,iy,iz);
            double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
            double c = 16.0 * W0 * epsilon;
            double w = Wn(phix,phiy,phiz,epsilon,W0);
            double w2 = SQ(w);

            d_Fx(IDX(ix,iy,iz)) = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
            d_Fy(IDX(ix,iy,iz)) = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
            d_Fz(IDX(ix,iy,iz)) = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
          }
          else
          {
            d_Fx(IDX(ix,iy,iz)) = 0.0;
            d_Fy(IDX(ix,iy,iz)) = 0.0;
            d_Fz(IDX(ix,iy,iz)) = 0.0;
          }
        });

      // allenCahn
      Kokkos::parallel_for("allenCahn",
        policy3d({1, 1, 1}, {DATAXSIZE-1, DATAYSIZE-1, DATAZSIZE-1}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          double phix = GradientX(d_phiold.data(),dx,dy,dz,ix,iy,iz);
          double phiy = GradientY(d_phiold.data(),dx,dy,dz,ix,iy,iz);
          double phiz = GradientZ(d_phiold.data(),dx,dy,dz,ix,iy,iz);

          d_phinew(IDX(ix,iy,iz)) = d_phiold(IDX(ix,iy,iz)) +
           (dt / taun(phix,phiy,phiz,epsilon,tau0)) *
           (Divergence(d_Fx.data(),d_Fy.data(),d_Fz.data(),dx,dy,dz,ix,iy,iz) -
            dFphi(d_phiold(IDX(ix,iy,iz)), d_uold(IDX(ix,iy,iz)),lambda));
        });

      // boundaryConditionsPhi
      Kokkos::parallel_for("boundaryConditionsPhi",
        policy3d({0, 0, 0}, {DATAXSIZE, DATAYSIZE, DATAZSIZE}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          if (ix == 0 || ix == DATAXSIZE-1 ||
              iy == 0 || iy == DATAYSIZE-1 ||
              iz == 0 || iz == DATAZSIZE-1) {
            d_phinew(IDX(ix,iy,iz)) = -1.0;
          }
        });

      // thermalEquation
      Kokkos::parallel_for("thermalEquation",
        policy3d({1, 1, 1}, {DATAXSIZE-1, DATAYSIZE-1, DATAZSIZE-1}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          d_unew(IDX(ix,iy,iz)) = d_uold(IDX(ix,iy,iz)) +
            0.5*(d_phinew(IDX(ix,iy,iz))-
                 d_phiold(IDX(ix,iy,iz))) +
            dt * D * Laplacian(d_uold.data(),dx,dy,dz,ix,iy,iz);
        });

      // boundaryConditionsU
      Kokkos::parallel_for("boundaryConditionsU",
        policy3d({0, 0, 0}, {DATAXSIZE, DATAYSIZE, DATAZSIZE}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          if (ix == 0 || ix == DATAXSIZE-1 ||
              iy == 0 || iy == DATAYSIZE-1 ||
              iz == 0 || iz == DATAZSIZE-1) {
            d_unew(IDX(ix,iy,iz)) = -delta;
          }
        });

      // swapGrid for phi
      Kokkos::parallel_for("swapGridPhi",
        policy3d({0, 0, 0}, {DATAXSIZE, DATAYSIZE, DATAZSIZE}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          double tmp = d_phinew(IDX(ix,iy,iz));
          d_phinew(IDX(ix,iy,iz)) = d_phiold(IDX(ix,iy,iz));
          d_phiold(IDX(ix,iy,iz)) = tmp;
        });

      // swapGrid for u
      Kokkos::parallel_for("swapGridU",
        policy3d({0, 0, 0}, {DATAXSIZE, DATAYSIZE, DATAZSIZE}),
        KOKKOS_LAMBDA(const int ix, const int iy, const int iz) {
          double tmp = d_unew(IDX(ix,iy,iz));
          d_unew(IDX(ix,iy,iz)) = d_uold(IDX(ix,iy,iz));
          d_uold(IDX(ix,iy,iz)) = tmp;
        });

      t++;
    }

    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);

    // Copy results back
    Kokkos::deep_copy(h_phiold, d_phiold);
    Kokkos::deep_copy(h_uold, d_uold);

    for (int i = 0; i < vol; i++) {
      phi_host[i] = h_phiold(i);
      u_host[i] = h_uold(i);
    }

    auto offload_end = std::chrono::steady_clock::now();
    auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
    printf("Offload time: %.3f (ms)\n", offload_time * 1e-6f);

#ifdef VERIFY
    bool ok = true;
    nRarray *phi_h = (nRarray *)phi_host;
    nRarray *u_h = (nRarray *)u_host;
    for (int idx = 0; idx < nx; idx++)
      for (int idy = 0; idy < ny; idy++)
        for (int idz = 0; idz < nz; idz++) {
          if (fabs(phi_ref[idx][idy][idz] - phi_h[idx][idy][idz]) > 1e-3) {
            ok = false; printf("phi: %lf %lf\n", phi_ref[idx][idy][idz], phi_h[idx][idy][idz]);
          }
          if (fabs(u_ref[idx][idy][idz] - u_h[idx][idy][idz]) > 1e-3) {
            ok = false; printf("u: %lf %lf\n", u_ref[idx][idy][idz], u_h[idx][idy][idz]);
          }
        }
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(phi_ref);
    free(u_ref);
#endif

    free(phi_host);
    free(u_host);
  }
  Kokkos::finalize();
  return 0;
}
