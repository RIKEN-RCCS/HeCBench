#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define F(x) (1.f + ((x) - 1.f) / 4.f)

const int   DIM             = 30;
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX =  5.12f;
const float OMEGA           =  0.5f;
const float c1              =  1.5f;
const float c2              =  1.5f;
const float phi             =  3.1415f;

float getRandom(float low, float high) {
  return low + float(((high - low) + 1.f) * rand() / ((float)RAND_MAX + 1.f));
}

float getRandomClamped(int seed) {
  srand(seed);
  return (float)rand() / (float)RAND_MAX;
}

float host_fitness_function(float x[]) {
  float res = 0.f;
  float y1 = F(x[0]), yn = F(x[DIM-1]);
  res += powf(sinf(phi*y1), 2.f) + powf(yn - 1, 2.f);
  for (int i = 0; i < DIM-1; i++) {
    float y = F(x[i]), yp = F(x[i+1]);
    res += powf(y - 1.f, 2.f) * (1.f + 10.f * powf(sinf(phi*yp), 2.f));
  }
  return res;
}

void pso_cpu(int p, int r, float* positions, float* velocities, float* pBests, float* gBest) {
  float tp1[DIM], tp2[DIM];
  for (int iter = 0; iter < r; iter++) {
    float rp = getRandomClamped(iter), rg = getRandomClamped(r - iter);
    for (int i = 0; i < p*DIM; i++) {
      velocities[i] = OMEGA*velocities[i] + c1*rp*(pBests[i]-positions[i]) + c2*rg*(gBest[i%DIM]-positions[i]);
      positions[i] += velocities[i];
    }
    for (int i = 0; i < p*DIM; i += DIM) {
      for (int j = 0; j < DIM; j++) { tp1[j]=positions[i+j]; tp2[j]=pBests[i+j]; }
      if (host_fitness_function(tp1) < host_fitness_function(tp2)) {
        for (int j = 0; j < DIM; j++) pBests[i+j] = tp1[j];
        if (host_fitness_function(tp1) < 130.f)
          for (int j = 0; j < DIM; j++) gBest[j] += tp1[j];
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of particles> <repeat>\n", argv[0]);
    return 1;
  }
  const int p = atoi(argv[1]);
  const int r = atoi(argv[2]);

  printf("Number of particles is %d\n", p);
  printf("Number of dimensions is %d\n", DIM);

  size_t size = (size_t)p * DIM;
  float* positions  = (float*)malloc(size * sizeof(float));
  float* velocities = (float*)malloc(size * sizeof(float));
  float* pBests     = (float*)malloc(size * sizeof(float));
  float* pBests_ref = (float*)malloc(size * sizeof(float));
  float* gBest      = (float*)malloc(DIM  * sizeof(float));

  srand(123);
  for (size_t i = 0; i < size; i++) {
    positions[i]  = getRandom(START_RANGE_MIN, START_RANGE_MAX);
    pBests[i]     = pBests_ref[i] = positions[i];
    velocities[i] = 0.f;
  }
  for (int i = 0; i < DIM; i++) gBest[i] = pBests[i];

  printf("\nExecute PSO on a device\n");

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_pos("positions",  size);
    Kokkos::View<float*> d_vel("velocities", size);
    Kokkos::View<float*> d_pb ("pBests",     size);
    Kokkos::View<float*> d_gb ("gBest",      DIM);

    auto h_pos = Kokkos::create_mirror_view(d_pos);
    auto h_vel = Kokkos::create_mirror_view(d_vel);
    auto h_pb  = Kokkos::create_mirror_view(d_pb);
    auto h_gb  = Kokkos::create_mirror_view(d_gb);

    for (size_t i = 0; i < size; i++) { h_pos(i)=positions[i]; h_vel(i)=velocities[i]; h_pb(i)=pBests[i]; }
    for (int i = 0; i < DIM;  i++) h_gb(i) = gBest[i];
    Kokkos::deep_copy(d_pos, h_pos);
    Kokkos::deep_copy(d_vel, h_vel);
    Kokkos::deep_copy(d_pb,  h_pb);
    Kokkos::deep_copy(d_gb,  h_gb);

    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < r; iter++) {
      float rp = getRandomClamped(iter);
      float rg = getRandomClamped(r - iter);

      Kokkos::parallel_for("updateParticle", (int)size, KOKKOS_LAMBDA(int i) {
        d_vel(i) = OMEGA*d_vel(i) + c1*rp*(d_pb(i)-d_pos(i)) + c2*rg*(d_gb(i%DIM)-d_pos(i));
        d_pos(i) += d_vel(i);
      });
      Kokkos::fence();

      Kokkos::parallel_for("updatePBest", p, KOKKOS_LAMBDA(int idx) {
        int base = idx * DIM;
        float tp1[DIM], tp2[DIM];
        for (int j = 0; j < DIM; j++) { tp1[j]=d_pos(base+j); tp2[j]=d_pb(base+j); }

        float f1 = 0.f, f2 = 0.f;
        {
          float y1=F(tp1[0]), yn=F(tp1[DIM-1]);
          f1 += powf(sinf(phi*y1),2.f) + powf(yn-1,2.f);
          for (int i=0;i<DIM-1;i++) { float y=F(tp1[i]),yp=F(tp1[i+1]); f1+=powf(y-1.f,2.f)*(1.f+10.f*powf(sinf(phi*yp),2.f)); }
        }
        {
          float y1=F(tp2[0]), yn=F(tp2[DIM-1]);
          f2 += powf(sinf(phi*y1),2.f) + powf(yn-1,2.f);
          for (int i=0;i<DIM-1;i++) { float y=F(tp2[i]),yp=F(tp2[i+1]); f2+=powf(y-1.f,2.f)*(1.f+10.f*powf(sinf(phi*yp),2.f)); }
        }
        if (f1 < f2) {
          for (int j = 0; j < DIM; j++) d_pb(base+j) = tp1[j];
          if (f1 < 130.f)
            for (int j = 0; j < DIM; j++) Kokkos::atomic_add(&d_gb(j), tp1[j]);
        }
      });
      Kokkos::fence();
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel execution time %f (us)\n", ns * 1e-3f / r);

    Kokkos::deep_copy(h_pb, d_pb);
    Kokkos::deep_copy(h_gb, d_gb);
    for (size_t i = 0; i < size; i++) pBests[i] = h_pb(i);
    for (int  i = 0; i < DIM;  i++) gBest[i]   = h_gb(i);
  }
  Kokkos::finalize();

  printf("Result=%f\n", host_fitness_function(gBest));

  printf("\nExecute PSO on a host. This may take a while for large problem size..\n");
  for (int i = 0; i < DIM; i++) gBest[i] = pBests_ref[i];
  pso_cpu(p, r, positions, velocities, pBests_ref, gBest);
  printf("Result=%f\n", host_fitness_function(gBest));

  bool ok = true;
  for (int i = 0; i < DIM; i++) {
    if (fabsf(pBests_ref[i] - pBests[i]) > 1e-3f) { printf("@%d %f %f\n", i, pBests_ref[i], pBests[i]); ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(positions); free(velocities); free(pBests); free(pBests_ref); free(gBest);
  return 0;
}
