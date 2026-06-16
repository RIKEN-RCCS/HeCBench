// Kokkos port of particlefilter-omp
// Particle filter simulation with synthetic video sequence.
// Args: -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>
#include <sys/time.h>

#define BLOCK_X     16
#define BLOCK_Y     16
#define PI          3.1415926535897932f
#define A           1103515245
#define C           12345
#define M           INT_MAX
#define SCALE_FACTOR 300.0f

static long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000LL) + tv.tv_usec;
}
static float elapsed_time(long long s, long long e) {
  return (float)(e - s) / 1000000.f;
}

static float roundFloat(float v) {
  int n = (int)v;
  return (v - n < 0.5f) ? (float)n : (float)(n + 1);
}

static float randu(int* seed, int idx) {
  int num = A * seed[idx] + C;
  seed[idx] = num % M;
  return fabsf(seed[idx] / (float)M);
}
static float randn(int* seed, int idx) {
  float u = randu(seed, idx), v = randu(seed, idx);
  return sqrtf(-2.f * logf(u)) * cosf(2.f * PI * v);
}

static void setIf(int tv, int nv, unsigned char* a, int X, int Y, int Z) {
  for (int x = 0; x < X; x++)
    for (int y = 0; y < Y; y++)
      for (int z = 0; z < Z; z++)
        if (a[x*Y*Z + y*Z + z] == tv) a[x*Y*Z + y*Z + z] = nv;
}
static void addNoise(unsigned char* a, int X, int Y, int Z, int* seed) {
  for (int x = 0; x < X; x++)
    for (int y = 0; y < Y; y++)
      for (int z = 0; z < Z; z++)
        a[x*Y*Z + y*Z + z] += (unsigned char)(5 * randn(seed, 0));
}
static void strelDisk(int* disk, int radius) {
  int diameter = radius * 2 - 1;
  for (int x = 0; x < diameter; x++)
    for (int y = 0; y < diameter; y++) {
      float d = sqrtf(powf((float)(x - radius + 1), 2) + powf((float)(y - radius + 1), 2));
      disk[x * diameter + y] = (d < radius) ? 1 : 0;
    }
}
static void dilate_matrix(unsigned char* m, int px, int py, int pz,
                           int X, int Y, int Z, int err) {
  int sx = px - err; if (sx < 0) sx = 0;
  int sy = py - err; if (sy < 0) sy = 0;
  int ex = px + err; if (ex > X) ex = X;
  int ey = py + err; if (ey > Y) ey = Y;
  for (int x = sx; x < ex; x++)
    for (int y = sy; y < ey; y++) {
      float d = sqrtf(powf((float)(x - px), 2) + powf((float)(y - py), 2));
      if (d < err) m[x*Y*Z + y*Z + pz] = 1;
    }
}
static void imdilate_disk(unsigned char* m, int X, int Y, int Z,
                           int err, unsigned char* nm) {
  for (int z = 0; z < Z; z++)
    for (int x = 0; x < X; x++)
      for (int y = 0; y < Y; y++)
        if (m[x*Y*Z + y*Z + z] == 1)
          dilate_matrix(nm, x, y, z, X, Y, Z, err);
}
static void getneighbors(int* se, int numOnes, int* nb, int radius) {
  int center = radius - 1, diameter = radius * 2 - 1, ny = 0;
  for (int x = 0; x < diameter; x++)
    for (int y = 0; y < diameter; y++)
      if (se[x*diameter + y]) {
        nb[ny*2]     = y - center;
        nb[ny*2 + 1] = x - center;
        ny++;
      }
}
static void videoSequence(unsigned char* I, int IszX, int IszY, int Nfr, int* seed) {
  int max_size = IszX * IszY * Nfr;
  int x0 = (int)roundFloat(IszY / 2.0f);
  int y0 = (int)roundFloat(IszX / 2.0f);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  for (int k = 1; k < Nfr; k++) {
    int xk = abs(x0 + (k - 1));
    int yk = abs(y0 - 2 * (k - 1));
    int pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size) pos = 0;
    I[pos] = 1;
  }

  unsigned char* newMatrix = (unsigned char*)calloc(IszX * IszY * Nfr, 1);
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  for (int x = 0; x < IszX; x++)
    for (int y = 0; y < IszY; y++)
      for (int k = 0; k < Nfr; k++)
        I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
  free(newMatrix);

  setIf(0, 100, I, IszX, IszY, Nfr);
  setIf(1, 228, I, IszX, IszY, Nfr);
  addNoise(I, IszX, IszY, Nfr, seed);
}

int particleFilter(unsigned char* I, int IszX, int IszY, int Nfr,
                   int* seed, int Nparticles)
{
  const int max_size = IszX * IszY * Nfr;

  float xe = roundFloat(IszY / 2.0f);
  float ye = roundFloat(IszX / 2.0f);

  const int radius   = 5;
  const int diameter = radius * 2 - 1;
  int* disk   = (int*)calloc(diameter * diameter, sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  for (int x = 0; x < diameter; x++)
    for (int y = 0; y < diameter; y++)
      if (disk[x*diameter + y]) countOnes++;
  int* objxy = (int*)calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);

  float* weights = (float*)calloc(Nparticles, sizeof(float));
  for (int x = 0; x < Nparticles; x++) weights[x] = 1.f / Nparticles;

  float* arrayX = (float*)calloc(Nparticles, sizeof(float));
  float* arrayY = (float*)calloc(Nparticles, sizeof(float));
  float* xj     = (float*)calloc(Nparticles, sizeof(float));
  float* yj     = (float*)calloc(Nparticles, sizeof(float));
  float* CDF    = (float*)calloc(Nparticles, sizeof(float));
  float* u      = (float*)calloc(Nparticles, sizeof(float));

  for (int x = 0; x < Nparticles; x++) { xj[x] = xe; yj[x] = ye; }

  long long offload_start = get_time();

  {
    // ---- Device views ----
    Kokkos::View<unsigned char*> I_d("I_d",   IszX * IszY * Nfr);
    Kokkos::View<int*>           objxy_d("objxy_d", countOnes * 2);
    Kokkos::View<float*>         arrayX_d("arrayX_d", Nparticles);
    Kokkos::View<float*>         arrayY_d("arrayY_d", Nparticles);
    Kokkos::View<float*>         xj_d("xj_d", Nparticles);
    Kokkos::View<float*>         yj_d("yj_d", Nparticles);
    Kokkos::View<float*>         weights_d("weights_d", Nparticles);
    Kokkos::View<int*>           seed_d("seed_d", Nparticles);
    Kokkos::View<float*>         CDF_d("CDF_d", Nparticles);
    Kokkos::View<float*>         u_d("u_d",   Nparticles);

    // Upload
    {
      auto hI    = Kokkos::create_mirror_view(I_d);
      auto hOxy  = Kokkos::create_mirror_view(objxy_d);
      auto hXj   = Kokkos::create_mirror_view(xj_d);
      auto hYj   = Kokkos::create_mirror_view(yj_d);
      auto hW    = Kokkos::create_mirror_view(weights_d);
      auto hSeed = Kokkos::create_mirror_view(seed_d);

      for (int i = 0; i < IszX * IszY * Nfr; i++) hI(i)   = I[i];
      for (int i = 0; i < countOnes * 2;     i++) hOxy(i) = objxy[i];
      for (int i = 0; i < Nparticles;        i++) {
        hXj(i) = xj[i]; hYj(i) = yj[i];
        hW(i)  = weights[i];
        hSeed(i) = seed[i];
      }
      Kokkos::deep_copy(I_d,       hI);
      Kokkos::deep_copy(objxy_d,   hOxy);
      Kokkos::deep_copy(xj_d,      hXj);
      Kokkos::deep_copy(yj_d,      hYj);
      Kokkos::deep_copy(weights_d, hW);
      Kokkos::deep_copy(seed_d,    hSeed);
    }

    long long start = get_time();

    for (int k = 1; k < Nfr; k++) {
      // ---- Likelihood kernel ----
      // Updates arrayX, arrayY (add noise), computes weights *= exp(likelihood)
      Kokkos::parallel_for("likelihood", Nparticles,
        KOKKOS_LAMBDA(int i) {
          arrayX_d(i) = xj_d(i);
          arrayY_d(i) = yj_d(i);

          // Advance seed twice for x noise
          seed_d(i) = (A * seed_d(i) + C) % M;
          float su = fabsf(seed_d(i) / (float)M);
          seed_d(i) = (A * seed_d(i) + C) % M;
          float sv = fabsf(seed_d(i) / (float)M);
          arrayX_d(i) += 1.0f + 5.0f * (sqrtf(-2.0f * logf(su)) * cosf(2.0f * PI * sv));

          // Advance seed twice for y noise
          seed_d(i) = (A * seed_d(i) + C) % M;
          su = fabsf(seed_d(i) / (float)M);
          seed_d(i) = (A * seed_d(i) + C) % M;
          sv = fabsf(seed_d(i) / (float)M);
          arrayY_d(i) += -2.0f + 2.0f * (sqrtf(-2.0f * logf(su)) * cosf(2.0f * PI * sv));

          // Compute likelihood from neighbourhood pixels
          int iX = (int)arrayX_d(i);
          int iY = (int)arrayY_d(i);
          int rnd_iX = (arrayX_d(i) - iX < 0.5f) ? iX : iX + 1;
          int rnd_iY = (arrayY_d(i) - iY < 0.5f) ? iY : iY + 1;

          float likelihoodSum = 0.0f;
          for (int y = 0; y < countOnes; y++) {
            int indX = rnd_iX + objxy_d(y * 2 + 1);
            int indY = rnd_iY + objxy_d(y * 2);
            int idx  = abs(indX * IszY * Nfr + indY * Nfr + k);
            if (idx >= max_size) idx = 0;
            float diff100 = I_d(idx) - 100;
            float diff228 = I_d(idx) - 228;
            likelihoodSum += (diff100 * diff100 - diff228 * diff228) / 50.0f;
          }
          float likelihood = likelihoodSum / countOnes - SCALE_FACTOR;
          weights_d(i) = (1.0f / Nparticles) * expf(likelihood);
        });
      Kokkos::fence();

      // ---- Reduction: sum of weights ----
      float totalWeight = 0.0f;
      Kokkos::parallel_reduce("sum_weights", Nparticles,
        KOKKOS_LAMBDA(int i, float& lsum) { lsum += weights_d(i); },
        totalWeight);

      // ---- Normalize weights ----
      Kokkos::parallel_for("normalize", Nparticles,
        KOKKOS_LAMBDA(int i) { weights_d(i) /= totalWeight; });
      Kokkos::fence();

      // ---- CDF + u on device (single work-item for sequential scan) ----
      Kokkos::parallel_for("cdf_u",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(int /*unused*/) {
          CDF_d(0) = weights_d(0);
          for (int x = 1; x < Nparticles; x++)
            CDF_d(x) = weights_d(x) + CDF_d(x - 1);

          seed_d(0) = (A * seed_d(0) + C) % M;
          float p   = fabsf(seed_d(0) / (float)M);
          seed_d(0) = (A * seed_d(0) + C) % M;
          float q   = fabsf(seed_d(0) / (float)M);
          u_d(0)    = (1.0f / Nparticles) *
                      (sqrtf(-2.0f * logf(p)) * cosf(2.0f * PI * q));
        });
      Kokkos::fence();

      Kokkos::parallel_for("u_spread", Nparticles,
        KOKKOS_LAMBDA(int i) {
          u_d(i) = u_d(0) + i / (float)Nparticles;
        });
      Kokkos::fence();

      // ---- Find index (resampling) ----
      Kokkos::parallel_for("resample", Nparticles,
        KOKKOS_LAMBDA(int i) {
          int index = Nparticles - 1;
          for (int x = 0; x < Nparticles; x++) {
            if (CDF_d(x) >= u_d(i)) { index = x; break; }
          }
          xj_d(i) = arrayX_d(index);
          yj_d(i) = arrayY_d(index);
        });
      Kokkos::fence();
    } // end frame loop

    long long end = get_time();
    printf("Average execution time of kernels: %f (s)\n",
           elapsed_time(start, end) / (Nfr - 1));

    // Copy results back
    {
      auto hAX = Kokkos::create_mirror_view(arrayX_d);
      auto hAY = Kokkos::create_mirror_view(arrayY_d);
      auto hW  = Kokkos::create_mirror_view(weights_d);
      Kokkos::deep_copy(hAX, arrayX_d);
      Kokkos::deep_copy(hAY, arrayY_d);
      Kokkos::deep_copy(hW,  weights_d);
      for (int i = 0; i < Nparticles; i++) {
        arrayX[i] = hAX(i);
        arrayY[i] = hAY(i);
        weights[i] = hW(i);
      }
    }
  }

  long long offload_end = get_time();
  printf("Device offloading time: %lf (s)\n", elapsed_time(offload_start, offload_end));

  xe = 0.f; ye = 0.f;
  for (int x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  float distance = sqrtf(powf(xe - roundFloat(IszY / 2.0f), 2) +
                          powf(ye - roundFloat(IszX / 2.0f), 2));

  FILE* fid = fopen("output.txt", "w+");
  if (fid) {
    fprintf(fid, "XE: %lf\n", (double)xe);
    fprintf(fid, "YE: %lf\n", (double)ye);
    fprintf(fid, "distance: %lf\n", (double)distance);
    fclose(fid);
  }

  free(disk); free(objxy); free(weights);
  free(arrayX); free(arrayY); free(xj); free(yj); free(CDF); free(u);
  return 0;
}

int main(int argc, char* argv[])
{
  const char* usage = "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  if (argc != 9) { printf("%s\n", usage); return 0; }
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") ||
      strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
    printf("%s\n", usage); return 0;
  }

  int IszX, IszY, Nfr, Nparticles;
  sscanf(argv[2], "%d", &IszX);
  sscanf(argv[4], "%d", &IszY);
  sscanf(argv[6], "%d", &Nfr);
  sscanf(argv[8], "%d", &Nparticles);

  if (IszX <= 0 || IszY <= 0 || Nfr <= 0 || Nparticles <= 0) {
    printf("All arguments must be positive\n"); return 0;
  }

  int* seed = (int*)calloc(Nparticles, sizeof(int));
  for (int i = 0; i < Nparticles; i++) seed[i] = i + 1;

  unsigned char* I = (unsigned char*)calloc(IszX * IszY * Nfr, 1);

  Kokkos::initialize(argc, argv);

  long long start = get_time();
  videoSequence(I, IszX, IszY, Nfr, seed);
  long long endVS = get_time();
  printf("VIDEO SEQUENCE TOOK %f (s)\n", elapsed_time(start, endVS));

  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
  long long endPF = get_time();
  printf("PARTICLE FILTER TOOK %f (s)\n", elapsed_time(endVS, endPF));
  printf("ENTIRE PROGRAM TOOK %f (s)\n",  elapsed_time(start, endPF));

  Kokkos::finalize();
  free(seed);
  free(I);
  return 0;
}
