/*
 * mcmd – Molecular Dynamics force kernel
 * Kokkos port from the OMP offload version.
 *
 * Original: Douglas M. Franz, University of South Florida (2017)
 *
 * Usage: ./main <num_atoms> <repeat>
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Atom data structure (same as OMP version)
// ---------------------------------------------------------------------------
typedef struct atom_t {
  double pos[3]    = {0, 0, 0};
  double eps       = 0;
  double sig       = 0;
  double charge    = 0;
  double f[3]      = {0, 0, 0};
  int    molid     = 0;
  int    frozen    = 0;
  double u[3]      = {0, 0, 0};
  double polar     = 0;
} d_atom;

// ---------------------------------------------------------------------------
// Force kernel (device)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void calculateForceKernel_body(
    Kokkos::View<d_atom*> atom_list,
    int N,
    double cutoffD,
    Kokkos::View<const double*> basis,
    Kokkos::View<const double*> reciprocal_basis,
    int pformD,
    double ewald_alpha,
    int kmax,
    int kspace,
    double polar_damp,
    int i)
{
  const d_atom anchoratom = atom_list(i);
  const int    pform      = pformD;
  const double damp       = polar_damp;
  const double alpha      = ewald_alpha;
  const double cutoff     = cutoffD;
  const double sqrtPI     = Kokkos::sqrt((double)M_PI);

  double rimg, rsq;
  double d[3], di[3], img[3], dimg[3], r, r2, ri, ri2;
  int q, j, n;
  double sig, eps, r6, s6;
  double u[3]  = {0, 0, 0};
  double af[3] = {0, 0, 0};
  double holder, chargeprod;

  // LJ forces
  if (pform == 0 || pform == 1 || pform == 2) {
    for (j = i + 1; j < N; j++) {
      if (anchoratom.molid == atom_list(j).molid) continue;
      if (anchoratom.frozen && atom_list(j).frozen) continue;

      sig = anchoratom.sig;
      if (sig != atom_list(j).sig) sig = 0.5 * (sig + atom_list(j).sig);
      eps = anchoratom.eps;
      if (eps != atom_list(j).eps) eps = Kokkos::sqrt(eps * atom_list(j).eps);
      if (sig == 0 || eps == 0) continue;

      for (n = 0; n < 3; n++) d[n] = anchoratom.pos[n] - atom_list(j).pos[n];
      for (n = 0; n < 3; n++) {
        img[n] = 0;
        for (q = 0; q < 3; q++) img[n] += reciprocal_basis(n*3+q) * d[q];
        img[n] = Kokkos::round(img[n]);
      }
      for (n = 0; n < 3; n++) {
        di[n] = 0;
        for (q = 0; q < 3; q++) di[n] += basis(n*3+q) * img[q];
        di[n] = d[n] - di[n];
      }
      r2 = ri2 = 0;
      for (n = 0; n < 3; n++) { r2 += d[n]*d[n]; ri2 += di[n]*di[n]; }
      r = Kokkos::sqrt(r2); ri = Kokkos::sqrt(ri2);
      if (ri != ri) { rimg = r; rsq = r2; for (n=0;n<3;n++) dimg[n]=d[n]; }
      else          { rimg = ri; rsq = ri2; for (n=0;n<3;n++) dimg[n]=di[n]; }

      if (rimg <= cutoff) {
        r6 = rsq * rsq * rsq;
        s6 = sig * sig; s6 = s6 * s6 * s6;
        for (n = 0; n < 3; n++) {
          holder = 24.0 * dimg[n] * eps *
                   (2.0*(s6*s6)/(r6*r6*rsq) - s6/(r6*rsq));
          Kokkos::atomic_add(&atom_list(j).f[n], -holder);
          af[n] += holder;
        }
      }
    }
    for (n = 0; n < 3; n++) Kokkos::atomic_add(&atom_list(i).f[n], af[n]);
  }

  // Electrostatics
  if (pform == 1 || pform == 2) {
    for (n = 0; n < 3; n++) af[n] = 0;
    double invV;
    int l[3];
    double k[3], k_sq, fourPI = 4.0 * M_PI;
    invV  =  basis(0) * (basis(4)*basis(8) - basis(7)*basis(5));
    invV += basis(3) * (basis(7)*basis(2) - basis(1)*basis(8));
    invV += basis(6) * (basis(1)*basis(5) - basis(5)*basis(2));
    invV = 1.0 / invV;

    for (j = 0; j < N; j++) {
      if (anchoratom.frozen && atom_list(j).frozen) continue;
      if (anchoratom.charge == 0 || atom_list(j).charge == 0) continue;
      if (i == j) continue;

      for (n = 0; n < 3; n++) d[n] = anchoratom.pos[n] - atom_list(j).pos[n];
      for (n = 0; n < 3; n++) {
        img[n] = 0;
        for (q = 0; q < 3; q++) img[n] += reciprocal_basis(n*3+q) * d[q];
        img[n] = Kokkos::round(img[n]);
      }
      for (n = 0; n < 3; n++) {
        di[n] = 0;
        for (q = 0; q < 3; q++) di[n] += basis(n*3+q) * img[q];
      }
      for (n = 0; n < 3; n++) di[n] = d[n] - di[n];
      r2 = ri2 = 0;
      for (n = 0; n < 3; n++) { r2 += d[n]*d[n]; ri2 += di[n]*di[n]; }
      r = Kokkos::sqrt(r2); ri = Kokkos::sqrt(ri2);
      if (ri != ri) { rimg=r; rsq=r2; for(n=0;n<3;n++) dimg[n]=d[n]; }
      else          { rimg=ri; rsq=ri2; for(n=0;n<3;n++) dimg[n]=di[n]; }

      // real-space
      if (rimg <= cutoff && (anchoratom.molid < atom_list(j).molid)) {
        chargeprod = anchoratom.charge * atom_list(j).charge;
        for (n = 0; n < 3; n++) u[n] = dimg[n] / rimg;
        for (n = 0; n < 3; n++) {
          holder = -((-2.0*chargeprod*alpha*Kokkos::exp(-alpha*alpha*rsq)) /
                     (sqrtPI*rimg) -
                     (chargeprod*erfc(alpha*rimg)/rsq)) * u[n];
          af[n] += holder;
          Kokkos::atomic_add(&atom_list(j).f[n], -holder);
        }
      }
      // k-space
      if (kspace && (anchoratom.molid < atom_list(j).molid)) {
        chargeprod = anchoratom.charge * atom_list(j).charge;
        for (n = 0; n < 3; n++) {
          for (l[0]=0; l[0]<=kmax; l[0]++) {
            for (l[1]=(!l[0]?0:-kmax); l[1]<=kmax; l[1]++) {
              for (l[2]=((!l[0]&&!l[1])?1:-kmax); l[2]<=kmax; l[2]++) {
                if (l[0]*l[0]+l[1]*l[1]+l[2]*l[2] > kmax*kmax) continue;
                int p;
                for (p=0; p<3; p++) {
                  k[p]=0;
                  for (q=0;q<3;q++) k[p]+=2.0*M_PI*reciprocal_basis(3*q+p)*l[q];
                }
                k_sq = k[0]*k[0]+k[1]*k[1]+k[2]*k[2];
                holder = chargeprod * invV * fourPI * k[n] *
                         Kokkos::exp(-k_sq/(4*alpha*alpha)) *
                         Kokkos::sin(k[0]*dimg[0]+k[1]*dimg[1]+k[2]*dimg[2]) /
                         k_sq * 2.0;
                af[n] += holder;
                Kokkos::atomic_add(&atom_list(j).f[n], -holder);
              }
            }
          }
        }
      }
    }
    for (n = 0; n < 3; n++) Kokkos::atomic_add(&atom_list(i).f[n], af[n]);
  }

  // Polarization
  if (pform == 2) {
    double common_factor, rv, rinv, r2v, r2inv, r3v, r3inv, r5inv, r7inv;
    double x2, y2, z2, xv, yv, zv;
    double udotu, ujdotr, uidotr;
    const double cc2inv = 1.0 / (cutoff * cutoff);
    double t1, t2, t3, p1, p2, p3, p4, p5;
    const double u_i[3] = {anchoratom.u[0], anchoratom.u[1], anchoratom.u[2]};
    double u_j[3];

    for (j = i + 1; j < N; j++) {
      double pair_af[3] = {0, 0, 0};
      if (anchoratom.molid == atom_list(j).molid) continue;

      for (n = 0; n < 3; n++) d[n] = anchoratom.pos[n] - atom_list(j).pos[n];
      for (n = 0; n < 3; n++) {
        img[n] = 0;
        for (q = 0; q < 3; q++) img[n] += reciprocal_basis(n*3+q)*d[q];
        img[n] = Kokkos::round(img[n]);
      }
      for (n = 0; n < 3; n++) {
        di[n] = 0;
        for (q = 0; q < 3; q++) di[n] += basis(n*3+q)*img[q];
      }
      for (n = 0; n < 3; n++) di[n] = d[n] - di[n];
      r2 = ri2 = 0;
      for (n = 0; n < 3; n++) { r2 += d[n]*d[n]; ri2 += di[n]*di[n]; }
      rv = Kokkos::sqrt(r2); ri = Kokkos::sqrt(ri2);
      if (ri != ri) { rimg=rv; rsq=r2; for(n=0;n<3;n++) dimg[n]=d[n]; }
      else          { rimg=ri; rsq=ri2; for(n=0;n<3;n++) dimg[n]=di[n]; }

      if (rimg > cutoff) continue;
      rv = rimg; xv=dimg[0]; yv=dimg[1]; zv=dimg[2];
      x2=xv*xv; y2=yv*yv; z2=zv*zv;
      r2v=rv*rv; r3v=r2v*rv;
      rinv=1./rv; r2inv=rinv*rinv; r3inv=r2inv*rinv;
      for (n = 0; n < 3; n++) u_j[n] = atom_list(j).u[n];

      if (atom_list(j).charge != 0 && anchoratom.polar != 0) {
        common_factor = atom_list(j).charge * r3inv;
        pair_af[0] += common_factor*((u_i[0]*(r2inv*(-2*x2+y2+z2)-cc2inv*(y2+z2)))+(u_i[1]*(r2inv*(-3*xv*yv)+cc2inv*xv*yv))+(u_i[2]*(r2inv*(-3*xv*zv)+cc2inv*xv*zv)));
        pair_af[1] += common_factor*(u_i[0]*(r2inv*(-3*xv*yv)+cc2inv*xv*yv)+u_i[1]*(r2inv*(-2*y2+x2+z2)-cc2inv*(x2+z2))+u_i[2]*(r2inv*(-3*yv*zv)+cc2inv*yv*zv));
        pair_af[2] += common_factor*(u_i[0]*(r2inv*(-3*xv*zv)+cc2inv*xv*zv)+u_i[1]*(r2inv*(-3*yv*zv)+cc2inv*yv*zv)+u_i[2]*(r2inv*(-2*z2+x2+y2)-cc2inv*(x2+y2)));
      }
      if (anchoratom.charge != 0 && atom_list(j).polar != 0) {
        common_factor = anchoratom.charge * r3inv;
        pair_af[0] -= common_factor*((u_j[0]*(r2inv*(-2*x2+y2+z2)-cc2inv*(y2+z2)))+(u_j[1]*(r2inv*(-3*xv*yv)+cc2inv*xv*yv))+(u_j[2]*(r2inv*(-3*xv*zv)+cc2inv*xv*zv)));
        pair_af[1] -= common_factor*(u_j[0]*(r2inv*(-3*xv*yv)+cc2inv*xv*yv)+u_j[1]*(r2inv*(-2*y2+x2+z2)-cc2inv*(x2+z2))+u_j[2]*(r2inv*(-3*yv*zv)+cc2inv*yv*zv));
        pair_af[2] -= common_factor*(u_j[0]*(r2inv*(-3*xv*zv)+cc2inv*xv*zv)+u_j[1]*(r2inv*(-3*yv*zv)+cc2inv*yv*zv)+u_j[2]*(r2inv*(-2*z2+x2+y2)-cc2inv*(x2+y2)));
      }
      if (anchoratom.polar != 0 && atom_list(j).polar != 0) {
        r5inv = r2inv * r3inv; r7inv = r5inv * r2inv;
        udotu = u_i[0]*u_j[0]+u_i[1]*u_j[1]+u_i[2]*u_j[2];
        uidotr = u_i[0]*dimg[0]+u_i[1]*dimg[1]+u_i[2]*dimg[2];
        ujdotr = u_j[0]*dimg[0]+u_j[1]*dimg[1]+u_j[2]*dimg[2];
        t1 = Kokkos::exp(-damp*rv);
        t2 = 1. + damp*rv + 0.5*damp*damp*r2v;
        t3 = t2 + damp*damp*damp*r3v/6.;
        p1 = 3*r5inv*udotu*(1.-t1*t2)-r7inv*15.*uidotr*ujdotr*(1.-t1*t3);
        p2 = 3*r5inv*ujdotr*(1.-t1*t3);
        p3 = 3*r5inv*uidotr*(1.-t1*t3);
        p4 = -udotu*r3inv*(-t1*(damp*rinv+damp*damp)+rinv*t1*damp*t2);
        p5 = 3*r5inv*uidotr*ujdotr*(-t1*(rinv*damp+damp*damp+0.5*rv*damp*damp*damp)+rinv*t1*damp*t3);
        pair_af[0] += p1*xv+p2*u_i[0]+p3*u_j[0]+p4*xv+p5*xv;
        pair_af[1] += p1*yv+p2*u_i[1]+p3*u_j[1]+p4*yv+p5*yv;
        pair_af[2] += p1*zv+p2*u_i[2]+p3*u_j[2]+p4*zv+p5*zv;
      }
      for (n = 0; n < 3; n++) {
        Kokkos::atomic_add(&atom_list(i).f[n],  pair_af[n]);
        Kokkos::atomic_add(&atom_list(j).f[n], -pair_af[n]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host wrapper that dispatches the Kokkos kernel
// ---------------------------------------------------------------------------
void force_kernel(int N, int pform, double cutoff,
                  double ewald_alpha, int ewald_kmax, int kspace_option,
                  double polar_damp,
                  const double *h_basis, const double *h_rbasis,
                  d_atom *h_atom_list)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = typename ExecSpace::memory_space;

  Kokkos::View<d_atom*>        d_atoms("atoms", N);
  Kokkos::View<double*>        d_basis("basis", 9);
  Kokkos::View<double*>        d_rbasis("rbasis", 9);

  auto h_atoms  = Kokkos::create_mirror_view(d_atoms);
  auto h_basis_v  = Kokkos::create_mirror_view(d_basis);
  auto h_rbasis_v = Kokkos::create_mirror_view(d_rbasis);

  for (int i = 0; i < N; i++) h_atoms(i) = h_atom_list[i];
  for (int i = 0; i < 9; i++) { h_basis_v(i) = h_basis[i]; h_rbasis_v(i) = h_rbasis[i]; }

  Kokkos::deep_copy(d_atoms,  h_atoms);
  Kokkos::deep_copy(d_basis,  h_basis_v);
  Kokkos::deep_copy(d_rbasis, h_rbasis_v);

  // Capture parameters by value
  int    pformD    = pform;
  double cutoffD   = cutoff;
  double alphaD    = ewald_alpha;
  int    kmaxD     = ewald_kmax;
  int    kspaceD   = kspace_option;
  double dampD     = polar_damp;

  Kokkos::parallel_for(
    "force_kernel",
    Kokkos::RangePolicy<ExecSpace>(0, N),
    KOKKOS_LAMBDA(int i) {
      calculateForceKernel_body(d_atoms, N, cutoffD,
                                Kokkos::View<const double*>(d_basis),
                                Kokkos::View<const double*>(d_rbasis),
                                pformD, alphaD, kmaxD, kspaceD, dampD, i);
    });
  Kokkos::fence();

  Kokkos::deep_copy(h_atoms, d_atoms);
  for (int i = 0; i < N; i++) h_atom_list[i] = h_atoms(i);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <num_atoms> <repeat>\n", argv[0]);
    return 1;
  }
  int N      = atoi(argv[1]);
  int repeat = atoi(argv[2]);

  // Simple identity basis (cubic box)
  double basis[9]   = {1,0,0, 0,1,0, 0,0,1};
  double rbasis[9]  = {1,0,0, 0,1,0, 0,0,1};

  d_atom *atoms = new d_atom[N];
  srand(42);
  for (int i = 0; i < N; i++) {
    atoms[i].pos[0]  = (double)rand()/RAND_MAX * 10.0;
    atoms[i].pos[1]  = (double)rand()/RAND_MAX * 10.0;
    atoms[i].pos[2]  = (double)rand()/RAND_MAX * 10.0;
    atoms[i].eps     = 0.1 + (double)rand()/RAND_MAX * 0.1;
    atoms[i].sig     = 2.0 + (double)rand()/RAND_MAX * 1.0;
    atoms[i].charge  = ((double)rand()/RAND_MAX - 0.5);
    atoms[i].molid   = i;   // each atom in its own molecule
    atoms[i].frozen  = 0;
    atoms[i].polar   = 0.0;
    for (int n = 0; n < 3; n++) atoms[i].f[n] = atoms[i].u[n] = 0;
  }

  Kokkos::initialize(argc, argv);
  {
    double total_time = 0.0;
    for (int r = 0; r < repeat; r++) {
      // Reset forces
      for (int i = 0; i < N; i++)
        for (int n = 0; n < 3; n++) atoms[i].f[n] = 0.0;

      auto t0 = std::chrono::steady_clock::now();
      force_kernel(N, 0 /*LJ only*/, 5.0, 0.2, 3, 0, 2.0,
                   basis, rbasis, atoms);
      auto t1 = std::chrono::steady_clock::now();
      total_time += std::chrono::duration<double>(t1 - t0).count();
    }
    printf("Average force kernel time: %g s\n", total_time / repeat);

    // Print a checksum
    double fsum = 0.0;
    for (int i = 0; i < N; i++)
      for (int n = 0; n < 3; n++) fsum += atoms[i].f[n];
    printf("Force checksum: %g\n", fsum);
  }
  Kokkos::finalize();

  delete[] atoms;
  return 0;
}
