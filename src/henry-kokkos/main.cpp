// Henry coefficient Monte Carlo simulation – Kokkos port
// Original OMP-target version: henry-omp/main.cpp

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#define NUMTHREADS 256

struct StructureAtom {
  double x, y, z;
  double epsilon;  // K
  double sigma;    // Å
};

// ─── Device functions ─────────────────────────────────────────────────────────

KOKKOS_INLINE_FUNCTION
double LCG_random_double(uint64_t* seed)
{
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

KOKKOS_INLINE_FUNCTION
double compute(double x, double y, double z,
               const StructureAtom* structureAtoms,
               int natoms, double L)
{
  const double T = 298.0;
  const double R = 8.314;

  double E = 0.0;
  for (int i = 0; i < natoms; i++) {
    double dx = x - structureAtoms[i].x;
    double dy = y - structureAtoms[i].y;
    double dz = z - structureAtoms[i].z;

    const double boxupper = 0.5 * L;
    const double boxlower = -boxupper;

    dx = (dx >  boxupper) ? dx - L : dx;
    dx = (dx <= boxlower) ? dx + L : dx;
    dy = (dy >  boxupper) ? dy - L : dy;
    dy = (dy <= boxlower) ? dy + L : dy;
    dz = (dz >  boxupper) ? dz - L : dz;
    dz = (dz <= boxlower) ? dz + L : dz;

    double rinv       = 1.0 / Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    double sig_ovr_r  = rinv * structureAtoms[i].sigma;
    double sig_ovr_r6 = Kokkos::pow(sig_ovr_r, 6.0);
    double sig_ovr_r12= sig_ovr_r6 * sig_ovr_r6;
    E += 4.0 * structureAtoms[i].epsilon * (sig_ovr_r12 - sig_ovr_r6);
  }
  return Kokkos::exp(-E / (R * T));
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: ./%s <material_file> <ninsertions_cycles>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Read crystal structure file
  std::ifstream materialfile(argv[1]);
  if (materialfile.fail()) {
    printf("Failed to import file %s.\n", argv[1]);
    return EXIT_FAILURE;
  }

  const int ncycles = atoi(argv[2]);

  std::map<std::string, double> epsilons, sigmas;
  epsilons["Zn"]=96.152688; epsilons["O"]=66.884614;
  epsilons["C"]=88.480032;  epsilons["H"]=57.276566;
  sigmas["Zn"]=3.095775; sigmas["O"]=3.424075;
  sigmas["C"]=3.580425;  sigmas["H"]=3.150565;

  std::string line;
  getline(materialfile, line);
  std::istringstream istream(line);
  double L;
  istream >> L;
  printf("L = %f\n", L);

  getline(materialfile, line); // waste

  getline(materialfile, line);
  int natoms;
  istream.str(line); istream.clear();
  istream >> natoms;
  printf("%d atoms\n", natoms);

  getline(materialfile, line); // waste

  // Read atoms into host array first
  std::vector<StructureAtom> h_atoms_vec(natoms);
  for (int i = 0; i < natoms; i++) {
    getline(materialfile, line);
    istream.str(line); istream.clear();
    int atomno;
    double xf, yf, zf;
    std::string element;
    istream >> atomno >> element >> xf >> yf >> zf;
    h_atoms_vec[i].x       = L * xf;
    h_atoms_vec[i].y       = L * yf;
    h_atoms_vec[i].z       = L * zf;
    h_atoms_vec[i].epsilon = epsilons[element];
    h_atoms_vec[i].sigma   = sigmas[element];
  }

  const int nBlocks           = 1024;
  const int insertionsPerCycle= nBlocks * NUMTHREADS;
  const int ninsertions       = ncycles * insertionsPerCycle;

  Kokkos::initialize(argc, argv);
  {
    // Device arrays
    Kokkos::View<StructureAtom*> d_atoms("structureAtoms", natoms);
    Kokkos::View<double*>        d_bf("boltzmannFactors", insertionsPerCycle);

    // Host mirror for atoms
    auto h_atoms = Kokkos::create_mirror_view(d_atoms);
    for (int i = 0; i < natoms; i++) h_atoms(i) = h_atoms_vec[i];
    Kokkos::deep_copy(d_atoms, h_atoms);

    // Host mirror for reduction
    auto h_bf = Kokkos::create_mirror_view(d_bf);

    double KH         = 0.0;
    double total_time = 0.0;

    for (int cycle = 0; cycle < ncycles; cycle++) {
      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for(
        "henry_mc",
        Kokkos::RangePolicy<>(0, insertionsPerCycle),
        KOKKOS_LAMBDA(int id) {
          uint64_t seed = (uint64_t)id;
          double x = L * LCG_random_double(&seed);
          double y = L * LCG_random_double(&seed);
          double z = L * LCG_random_double(&seed);
          d_bf(id) = compute(x, y, z, d_atoms.data(), natoms, L);
        });
      Kokkos::fence();

      auto end  = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end - start).count();

      // Copy Boltzmann factors back and sum on host
      Kokkos::deep_copy(h_bf, d_bf);
      for (int i = 0; i < insertionsPerCycle; i++)
        KH += h_bf(i);
    }

    KH /= ninsertions;
    const double R = 8.314, T = 298.0;
    KH /= (R * T);
    printf("Used %d blocks with %d threads each\n", nBlocks, NUMTHREADS);
    printf("Henry constant = %e mol/(m3 - Pa)\n", KH);
    printf("Number of actual insertions: %d\n", ninsertions);
    printf("Number of times we called the device kernel: %d\n", ncycles);
    printf("Average kernel execution time %f (s)\n", (total_time * 1e-9) / ncycles);
  }
  Kokkos::finalize();
  return EXIT_SUCCESS;
}
