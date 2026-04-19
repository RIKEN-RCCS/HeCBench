/*
  Kokkos port of the Henry coefficient benchmark.

  Computes the Henry coefficient KH = <e^{-E/(RT)}> / (RT) for methane
  adsorption in a nanoporous material using random Monte Carlo insertions.

  Original OMP version in src/henry-omp/main.cpp.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NUMTHREADS 256

// Plain struct usable on host and device
struct StructureAtom {
  double x;
  double y;
  double z;
  double epsilon;  // units: K
  double sigma;    // units: A
};

// temperature, Kelvin
const double T = 298.0;

// Universal gas constant, m3 - Pa / (K - mol)
const double R = 8.314;

KOKKOS_INLINE_FUNCTION
double LCG_random_double(uint64_t *seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

KOKKOS_INLINE_FUNCTION
double compute(double x, double y, double z,
               const StructureAtom *structureAtoms,
               int natoms, double L)
{
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

    double rinv = 1.0 / sqrt(dx*dx + dy*dy + dz*dz);
    double sig_ovr_r   = rinv * structureAtoms[i].sigma;
    double sig_ovr_r6  = pow(sig_ovr_r, 6.0);
    double sig_ovr_r12 = sig_ovr_r6 * sig_ovr_r6;
    E += 4.0 * structureAtoms[i].epsilon * (sig_ovr_r12 - sig_ovr_r6);
  }
  return exp(-E / (R * T));
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: ./%s <material file> <ninsertions>\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::ifstream materialfile(argv[1]);
  if (materialfile.fail()) {
    printf("Failed to import file %s.\n", argv[1]);
    return EXIT_FAILURE;
  }

  const int ncycles = atoi(argv[2]);

  std::map<std::string, double> epsilons;
  epsilons["Zn"] = 96.152688;
  epsilons["O"]  = 66.884614;
  epsilons["C"]  = 88.480032;
  epsilons["H"]  = 57.276566;

  std::map<std::string, double> sigmas;
  sigmas["Zn"] = 3.095775;
  sigmas["O"]  = 3.424075;
  sigmas["C"]  = 3.580425;
  sigmas["H"]  = 3.150565;

  std::string line;
  getline(materialfile, line);
  std::istringstream istream(line);

  double L;
  istream >> L;
  printf("L = %f\n", L);

  getline(materialfile, line);  // waste line

  getline(materialfile, line);
  int natoms;
  istream.str(line);
  istream.clear();
  istream >> natoms;
  printf("%d atoms\n", natoms);

  getline(materialfile, line);  // waste line

  // Temporary host array before copying to Kokkos View
  std::vector<StructureAtom> hostAtoms(natoms);
  for (int i = 0; i < natoms; i++) {
    getline(materialfile, line);
    istream.str(line);
    istream.clear();

    int atomno;
    double xf, yf, zf;
    std::string element;
    istream >> atomno >> element >> xf >> yf >> zf;

    hostAtoms[i].x = L * xf;
    hostAtoms[i].y = L * yf;
    hostAtoms[i].z = L * zf;
    hostAtoms[i].epsilon = epsilons[element];
    hostAtoms[i].sigma   = sigmas[element];
  }

  const int nBlocks           = 1024;
  const int insertionsPerCycle = nBlocks * NUMTHREADS;
  const int ninsertions        = ncycles * insertionsPerCycle;

  Kokkos::initialize(argc, argv);
  {
    // Copy structure atoms to device
    Kokkos::View<StructureAtom*> d_atoms("d_atoms", natoms);
    {
      auto h_atoms = Kokkos::create_mirror_view(d_atoms);
      for (int i = 0; i < natoms; i++) h_atoms(i) = hostAtoms[i];
      Kokkos::deep_copy(d_atoms, h_atoms);
    }

    // Boltzmann factors for one cycle, on device
    Kokkos::View<double*> d_bf("d_bf", insertionsPerCycle);
    // Mirror for host accumulation
    auto h_bf = Kokkos::create_mirror_view(d_bf);

    double total_time = 0.0;
    double KH = 0.0;

    for (int cycle = 0; cycle < ncycles; cycle++) {
      auto kstart = std::chrono::steady_clock::now();

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
      auto kend = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count() * 1e-9;

      // Copy Boltzmann factors back and accumulate on host
      Kokkos::deep_copy(h_bf, d_bf);
      for (int i = 0; i < insertionsPerCycle; i++)
        KH += h_bf(i);
    }

    KH = KH / ninsertions;
    KH = KH / (R * T);

    printf("Used %d blocks with %d threads each\n", nBlocks, NUMTHREADS);
    printf("Henry constant = %e mol/(m3 - Pa)\n", KH);
    printf("Number of actual insertions: %d\n", ninsertions);
    printf("Number of times we called the device kernel: %d\n", ncycles);
    printf("Average kernel execution time %f (s)\n", total_time / ncycles);
  }
  Kokkos::finalize();

  return EXIT_SUCCESS;
}
