#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// WKFUtils timer stubs (simplified)
typedef void* wkf_timerhandle;
static inline wkf_timerhandle wkf_timer_create() { return nullptr; }
static inline void wkf_timer_start(wkf_timerhandle) {}
static inline void wkf_timer_stop(wkf_timerhandle) {}
static inline double wkf_timer_time(wkf_timerhandle) { return 0.0; }

#define MAXATOMS 4000
#define UNROLLX    8
#define BLOCKSIZEX 8
#define BLOCKSIZEY 8
#define BLOCKSIZE  (BLOCKSIZEX * BLOCKSIZEY)

struct float4 { float x, y, z, w; };
struct int3   { int x, y, z; };

int copyatoms(float *atoms, int count, float zplane,
              Kokkos::View<float4*> d_atominfo)
{
  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer\n");
    return -1;
  }
  auto h_atominfo = Kokkos::create_mirror_view(d_atominfo);
  for (int i = 0; i < count; i++) {
    h_atominfo(i).x = atoms[i*4];
    h_atominfo(i).y = atoms[i*4+1];
    float dz = zplane - atoms[i*4+2];
    h_atominfo(i).z = dz*dz;
    h_atominfo(i).w = atoms[i*4+3];
  }
  Kokkos::deep_copy(d_atominfo, h_atominfo);
  return 0;
}

int initatoms(float **atombuf, int count, int3 volsize, float gridspacing) {
  float *atoms = (float*) malloc(count * 4 * sizeof(float));
  *atombuf = atoms;
  srand(2);
  float sx = gridspacing * volsize.x;
  float sy = gridspacing * volsize.y;
  float sz = gridspacing * volsize.z;
  for (int i = 0; i < count; i++) {
    atoms[i*4]   = ((float)rand()/RAND_MAX) * sx;
    atoms[i*4+1] = ((float)rand()/RAND_MAX) * sy;
    atoms[i*4+2] = ((float)rand()/RAND_MAX) * sz;
    atoms[i*4+3] = ((float)rand()/RAND_MAX) * 2.0f - 1.0f;
  }
  return 0;
}

int main(int argc, char **argv) {
  float *energy = nullptr, *atoms = nullptr;
  int3 volsize; volsize.x=768; volsize.y=768; volsize.z=1;
  float gridspacing = 0.1f;
  int atomcount = 1000000;

  printf("GPU accelerated coulombic potential microbenchmark\n");
  printf("Grid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);

  if (initatoms(&atoms, atomcount, volsize, gridspacing)) return -1;

  int volmem = volsize.x * volsize.y * volsize.z;
  energy = (float*) malloc(sizeof(float) * volmem);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float4*> d_atominfo("atominfo", MAXATOMS);
    Kokkos::View<float*>  d_energy("energy",   volmem);

    Kokkos::parallel_for("init_energy", volmem, KOKKOS_LAMBDA(int i) {
      d_energy(i) = 0.0f;
    });

    double copytotal = 0, runtotal = 0;
    int iterations = 0;
    const float gridspacing_u = gridspacing * BLOCKSIZEX;

    int atomstart;
    for (atomstart = 0; atomstart < atomcount; atomstart += MAXATOMS) {
      iterations++;
      int runatoms = atomcount - atomstart;
      if (runatoms > MAXATOMS) runatoms = MAXATOMS;

      auto t_copy0 = std::chrono::steady_clock::now();
      if (copyatoms(atoms + 4*atomstart, runatoms, 0*gridspacing, d_atominfo))
        return -1;
      auto t_copy1 = std::chrono::steady_clock::now();
      copytotal += std::chrono::duration<double>(t_copy1-t_copy0).count();

      auto t_run0 = std::chrono::steady_clock::now();

      int nx = volsize.x, ny = volsize.y;
      int nxu = nx / UNROLLX;
      Kokkos::parallel_for("clenergy",
        Kokkos::RangePolicy<>(0, ny * nxu),
        KOKKOS_LAMBDA(int idx) {
          int yindex = idx / nxu;
          int xindex = idx % nxu;
          unsigned int outaddr = yindex * nx + xindex;
          float coory = gridspacing * yindex;
          float coorx = gridspacing * xindex;

          float ev[8] = {0,0,0,0,0,0,0,0};

          for (int atomid = 0; atomid < runatoms; atomid++) {
            float dy   = coory - d_atominfo(atomid).y;
            float dyz2 = dy*dy + d_atominfo(atomid).z;
            float dx1 = coorx - d_atominfo(atomid).x;
            float w   = d_atominfo(atomid).w;
            ev[0] += w / Kokkos::sqrt(dx1*dx1 + dyz2);
            float dx2 = dx1 + gridspacing_u;
            ev[1] += w / Kokkos::sqrt(dx2*dx2 + dyz2);
            float dx3 = dx2 + gridspacing_u;
            ev[2] += w / Kokkos::sqrt(dx3*dx3 + dyz2);
            float dx4 = dx3 + gridspacing_u;
            ev[3] += w / Kokkos::sqrt(dx4*dx4 + dyz2);
            float dx5 = dx4 + gridspacing_u;
            ev[4] += w / Kokkos::sqrt(dx5*dx5 + dyz2);
            float dx6 = dx5 + gridspacing_u;
            ev[5] += w / Kokkos::sqrt(dx6*dx6 + dyz2);
            float dx7 = dx6 + gridspacing_u;
            ev[6] += w / Kokkos::sqrt(dx7*dx7 + dyz2);
            float dx8 = dx7 + gridspacing_u;
            ev[7] += w / Kokkos::sqrt(dx8*dx8 + dyz2);
          }

          d_energy(outaddr             ) += ev[0];
          d_energy(outaddr+1*BLOCKSIZEX) += ev[1];
          d_energy(outaddr+2*BLOCKSIZEX) += ev[2];
          d_energy(outaddr+3*BLOCKSIZEX) += ev[3];
          d_energy(outaddr+4*BLOCKSIZEX) += ev[4];
          d_energy(outaddr+5*BLOCKSIZEX) += ev[5];
          d_energy(outaddr+6*BLOCKSIZEX) += ev[6];
          d_energy(outaddr+7*BLOCKSIZEX) += ev[7];
        });
      Kokkos::fence();
      auto t_run1 = std::chrono::steady_clock::now();
      runtotal += std::chrono::duration<double>(t_run1-t_run0).count();
    }
    printf("Done\n");

    auto h_energy = Kokkos::create_mirror_view(d_energy);
    Kokkos::deep_copy(h_energy, d_energy);
    for (int i=0;i<volmem;i++) energy[i] = h_energy(i);

    for (int j=0;j<8;j++) {
      for (int i=0;i<8;i++) printf("[%d] %.1f ", j*volsize.x+i, energy[j*volsize.x+i]);
      printf("\n");
    }
    printf("Final calculation required %d iterations of %d atoms\n", iterations, MAXATOMS);
    printf("Copy time: %f seconds\n", copytotal);
    printf("Kernel time: %f seconds\n", runtotal);
  }
  Kokkos::finalize();

  free(atoms); free(energy);
  return 0;
}
