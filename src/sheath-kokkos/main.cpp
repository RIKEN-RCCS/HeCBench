// 1D sheath PIC simulation - Kokkos port
// Based on https://www.particleincell.com/2016/cuda-pic/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

/*constants*/
#define EPS_0 8.85418782e-12
#define K     1.38065e-23
#define ME    9.10938215e-31
#define QE    1.602176565e-19
#define AMU   1.660538921e-27
#define EV_TO_K 11604.52

/*simulation parameters*/
#define PLASMA_DEN  1e16
#define NUM_IONS    500000
#define NUM_ELECTRONS 500000
#define DX  1e-4
#define NC  100
#define NUM_TS 1000
#define DT  1e-11
#define ELECTRON_TEMP 3.0
#define ION_TEMP 1.0

#define X0   0
#define XL   (NC * DX)
#define XMAX (X0 + XL)

struct Domain
{
  const int    ni   = NC + 1;
  const double x0   = X0;
  const double dx   = DX;
  const double xl   = XL;
  const double xmax = XMAX;

  double* phi;
  double* ef;
  double* rho;
  float*  ndi;
  float*  nde;
};

struct Particle
{
  double x;
  double v;
  bool   alive;
};

struct Species
{
  double mass;
  double charge;
  double spwt;
  int    np;
  int    np_alloc;
  Particle* part;
};

Domain domain;
FILE*  file_res;

double rnd() { return rand() / (double)RAND_MAX; }

double SampleVel(double v_th)
{
  const int M = 12;
  double sum = 0;
  for (int i = 0; i < M; i++) sum += rnd();
  return sqrt(0.5) * v_th * (sum - M / 2.0) / sqrt(M / 12.0);
}

void AddParticle(Species* species, double x, double v)
{
  if (species->np > species->np_alloc - 1)
  {
    printf("Too many particles!\n");
    exit(-1);
  }
  species->part[species->np].x     = x;
  species->part[species->np].v     = v;
  species->part[species->np].alive = true;
  species->np++;
}

void ComputeRho(Species* ions, Species* electrons)
{
  double* rho = domain.rho;
  for (int i = 0; i < domain.ni; i++)
    rho[i] = ions->charge * domain.ndi[i] + electrons->charge * domain.nde[i];
}

bool SolvePotential(double* phi, double* rho)
{
  double L2;
  double dx2 = domain.dx * domain.dx;
  phi[0] = phi[domain.ni - 1] = 0;
  for (int solver_it = 0; solver_it < 40000; solver_it++)
  {
    for (int i = 1; i < domain.ni - 1; i++)
    {
      double g = 0.5 * (phi[i-1] + phi[i+1] + dx2 * rho[i] / EPS_0);
      phi[i]   = phi[i] + 1.4 * (g - phi[i]);
    }
    if (solver_it % 25 == 0)
    {
      double sum = 0;
      for (int i = 1; i < domain.ni - 1; i++)
      {
        double R = -rho[i] / EPS_0 - (phi[i-1] - 2*phi[i] + phi[i+1]) / dx2;
        sum += R * R;
      }
      L2 = sqrt(sum) / domain.ni;
      if (L2 < 1e-4) return true;
    }
  }
  printf("Gauss-Seidel solver failed to converge, L2=%.3g!\n", L2);
  return false;
}

void ComputeEF(double* phi, double* ef,
               Kokkos::View<double*> ef_gpu)
{
  for (int i = 1; i < domain.ni - 1; i++)
    ef[i] = -(phi[i+1] - phi[i-1]) / (2 * domain.dx);
  ef[0]            = -(phi[1] - phi[0]) / domain.dx;
  ef[domain.ni-1]  = -(phi[domain.ni-1] - phi[domain.ni-2]) / domain.dx;

  auto h_ef = Kokkos::create_mirror_view(ef_gpu);
  for (int i = 0; i < domain.ni; i++) h_ef(i) = ef[i];
  Kokkos::deep_copy(ef_gpu, h_ef);
}

void ScatterSpecies(Species* species,
                    Kokkos::View<Particle*> species_gpu,
                    float* den,
                    Kokkos::View<float*> den_gpu,
                    double &time)
{
  int ni   = domain.ni;
  int size = species->np_alloc;

  Kokkos::deep_copy(den_gpu, 0.0f);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  Kokkos::parallel_for("scatter", size, KOKKOS_LAMBDA(int p) {
    if (species_gpu(p).alive) {
      double lc = (species_gpu(p).x - X0) / DX;
      int    i  = (int)lc;
      float  di = (float)(lc - i);
      Kokkos::atomic_add(&den_gpu(i),   1.f * (1.f - di));
      Kokkos::atomic_add(&den_gpu(i+1), 1.f * di);
    }
  });
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  time += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  auto h_den = Kokkos::create_mirror_view(den_gpu);
  Kokkos::deep_copy(h_den, den_gpu);

  for (int i = 0; i < ni; i++)
    den[i] = h_den(i) * (float)(species->spwt / domain.dx);
  den[0]      *= 2.0f;
  den[ni - 1] *= 2.0f;
}

void PushSpecies(Species* species,
                 Kokkos::View<Particle*> species_gpu,
                 Kokkos::View<double*> ef_gpu)
{
  double qm   = species->charge / species->mass;
  int    size = species->np_alloc;

  Kokkos::parallel_for("push", size, KOKKOS_LAMBDA(int p) {
    if (species_gpu(p).alive) {
      double lc  = (species_gpu(p).x - X0) / DX;
      int    i   = (int)lc;
      double di  = lc - i;
      double ef_val = ef_gpu(i) * (1.0 - di) + ef_gpu(i+1) * di;
      species_gpu(p).v += DT * qm * ef_val;
      species_gpu(p).x += DT * species_gpu(p).v;
      if (species_gpu(p).x < X0 || species_gpu(p).x >= XMAX)
        species_gpu(p).alive = false;
    }
  });
}

void RewindSpecies(Species* species,
                   Kokkos::View<Particle*> species_gpu,
                   Kokkos::View<double*> ef_gpu)
{
  double qm   = species->charge / species->mass;
  int    size = species->np_alloc;

  Kokkos::parallel_for("rewind", size, KOKKOS_LAMBDA(int p) {
    if (species_gpu(p).alive) {
      double lc  = (species_gpu(p).x - X0) / DX;
      int    i   = (int)lc;
      double di  = lc - i;
      double ef_val = ef_gpu(i) * (1.0 - di) + ef_gpu(i+1) * di;
      species_gpu(p).v -= 0.5 * DT * qm * ef_val;
    }
  });
}

void WriteResults(int ts)
{
  fprintf(file_res, "ZONE I=%d T=ZONE_%06d\n", domain.ni, ts);
  for (int i = 0; i < domain.ni; i++)
  {
    fprintf(file_res, "%g %g %g %g %g %g\n",
            i * domain.dx, domain.nde[i], domain.ndi[i],
            domain.rho[i], domain.phi[i], domain.ef[i]);
  }
  fflush(file_res);
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    double sp_time = 0.0;

    domain.phi = new double[domain.ni];
    domain.rho = new double[domain.ni];
    domain.ef  = new double[domain.ni];
    domain.nde = new float[domain.ni];
    domain.ndi = new float[domain.ni];

    double* phi = domain.phi;
    double* rho = domain.rho;
    double* ef  = domain.ef;
    float*  nde = domain.nde;
    float*  ndi = domain.ndi;

    memset(phi, 0, sizeof(double) * domain.ni);

    Species ions, electrons;

    ions.mass     = 16 * AMU;
    ions.charge   = QE;
    ions.spwt     = PLASMA_DEN * domain.xl / NUM_IONS;
    ions.np       = 0;
    ions.np_alloc = NUM_IONS;
    ions.part     = new Particle[NUM_IONS];

    electrons.mass     = ME;
    electrons.charge   = -QE;
    electrons.spwt     = PLASMA_DEN * domain.xl / NUM_ELECTRONS;
    electrons.np       = 0;
    electrons.np_alloc = NUM_ELECTRONS;
    electrons.part     = new Particle[NUM_ELECTRONS];

    srand(123);

    double delta_ions = domain.xl / NUM_IONS;
    double v_thi      = sqrt(2 * K * ION_TEMP * EV_TO_K / ions.mass);
    for (int p = 0; p < NUM_IONS; p++)
      AddParticle(&ions, domain.x0 + p * delta_ions, SampleVel(v_thi));

    double delta_electrons = domain.xl / NUM_ELECTRONS;
    double v_the           = sqrt(2 * K * ELECTRON_TEMP * EV_TO_K / electrons.mass);
    for (int p = 0; p < NUM_ELECTRONS; p++)
      AddParticle(&electrons, domain.x0 + p * delta_electrons, SampleVel(v_the));

    // Allocate device views
    Kokkos::View<float*>    nde_gpu("nde_gpu", domain.ni);
    Kokkos::View<float*>    ndi_gpu("ndi_gpu", domain.ni);
    Kokkos::View<double*>   ef_gpu("ef_gpu",  domain.ni);
    Kokkos::View<Particle*> ions_gpu("ions_gpu", NUM_IONS);
    Kokkos::View<Particle*> electrons_gpu("electrons_gpu", NUM_ELECTRONS);

    // Copy initial particles to device
    {
      auto h_ions = Kokkos::create_mirror_view(ions_gpu);
      for (int i = 0; i < NUM_IONS; i++) h_ions(i) = ions.part[i];
      Kokkos::deep_copy(ions_gpu, h_ions);

      auto h_elec = Kokkos::create_mirror_view(electrons_gpu);
      for (int i = 0; i < NUM_ELECTRONS; i++) h_elec(i) = electrons.part[i];
      Kokkos::deep_copy(electrons_gpu, h_elec);
    }

    ScatterSpecies(&ions,      ions_gpu,      ndi, ndi_gpu, sp_time);
    ScatterSpecies(&electrons, electrons_gpu, nde, nde_gpu, sp_time);

    ComputeRho(&ions, &electrons);
    SolvePotential(phi, rho);
    ComputeEF(phi, ef, ef_gpu);

    RewindSpecies(&ions,      ions_gpu,      ef_gpu);
    RewindSpecies(&electrons, electrons_gpu, ef_gpu);
    Kokkos::fence();

    file_res = fopen("result.dat", "w");
    fprintf(file_res, "VARIABLES = x nde ndi rho phi ef\n");
    WriteResults(0);

    auto t_start = std::chrono::steady_clock::now();

    for (int ts = 1; ts <= NUM_TS; ts++)
    {
      ScatterSpecies(&ions,      ions_gpu,      ndi, ndi_gpu, sp_time);
      ScatterSpecies(&electrons, electrons_gpu, nde, nde_gpu, sp_time);

      ComputeRho(&ions, &electrons);
      SolvePotential(phi, rho);
      ComputeEF(phi, ef, ef_gpu);

      PushSpecies(&electrons, electrons_gpu, ef_gpu);
      PushSpecies(&ions,      ions_gpu,      ef_gpu);
      Kokkos::fence();

      if (ts % 25 == 0)
      {
        double max_phi = fabs(phi[0]);
        for (int i = 0; i < domain.ni; i++)
          if (fabs(phi[i]) > max_phi) max_phi = fabs(phi[i]);
        printf("TS:%i\tdphi:%.3g\n", ts, max_phi - phi[0]);
      }

      if (ts % 1000 == 0)
        WriteResults(ts);
    }

    auto t_end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

    fclose(file_res);

    delete[] phi;
    delete[] rho;
    delete[] ef;
    delete[] nde;
    delete[] ndi;
    delete[] ions.part;
    delete[] electrons.part;

    printf("Total kernel execution time (scatter particles) : %.3g (s)\n", sp_time * 1e-9);
    printf("Total time for %d time steps: %.3g (s)\n", NUM_TS, time * 1e-9);
    printf("Time per time step: %.3g (ms)\n", (time * 1e-6) / NUM_TS);
  }
  Kokkos::finalize();
  return 0;
}
