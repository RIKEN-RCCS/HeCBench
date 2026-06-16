/*
 * FDTD3D Kokkos port.
 * Based on original NVIDIA SDK example.
 * Copyright 1993-2010 NVIDIA Corporation. All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <stdarg.h>
#include <Kokkos_Core.hpp>

// ============================================================
// Constants
// ============================================================
#define MEMORY_SIZE     134217728ULL
#define k_localWorkX    32
#define k_localWorkY    8
#define localWorkMaxX   32
#define localWorkMaxY   16
#define k_radius_default  4
#define k_dim_min         96
#define k_dim_max         376
#define k_dim_qa          248
#define k_timesteps_min   1
#define k_timesteps_max   10
#define k_timesteps_default 5

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

// ============================================================
// Simple logging/argument helpers
// ============================================================
static void shrLog(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);
}
static void shrLogEx(int /*mode*/, int /*err*/, const char * /*file*/) {}

static bool shrCheckCmdLineFlag(int argc, const char **argv, const char *name) {
  for (int i = 1; i < argc; i++)
    if (strncmp(argv[i] + 2, name, strlen(name)) == 0) return true;
  return false;
}

static bool shrGetCmdLineArgumentstr(int argc, const char **argv,
                                     const char *name, char **val) {
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-' && argv[i][1] == '-') {
      const char *p = argv[i] + 2;
      size_t nlen = strlen(name);
      if (strncmp(p, name, nlen) == 0 && p[nlen] == '=') {
        *val = (char *)(p + nlen + 1);
        return true;
      }
    }
  }
  return false;
}

#define LOGBOTH    3
#define ERRORMSG   8
#define STDERROR   ""

// ============================================================
// Reference implementation
// ============================================================
void generateRandomData(float *data, int dimx, int dimy, int dimz,
                        float lowerBound, float upperBound) {
  srand(0);
  for (int iz = 0; iz < dimz; iz++)
    for (int iy = 0; iy < dimy; iy++)
      for (int ix = 0; ix < dimx; ix++)
        *data++ = lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound);
}

bool fdtdReference(float *output, const float *input, const float *coeff,
                   int dimx, int dimy, int dimz, int radius, int timesteps) {
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = (size_t)outerDimx * outerDimy * outerDimz;
  const int stride_y = outerDimx;
  const int stride_z = stride_y * outerDimy;

  float *intermediate = (float *)calloc(volumeSize, sizeof(float));
  if (!intermediate) return false;

  const float *bufsrc;
  float *bufdst, *bufdstnext;
  if ((timesteps % 2) == 0) {
    bufsrc = input; bufdst = intermediate; bufdstnext = output;
  } else {
    bufsrc = input; bufdst = output; bufdstnext = intermediate;
  }

  shrLog(" Host FDTD loop\n");
  for (int it = 0; it < timesteps; it++) {
    shrLog("\tt = %d\n", it);
    const float *src = bufsrc;
    float *dst = bufdst;
    for (int iz = -radius; iz < dimz + radius; iz++)
      for (int iy = -radius; iy < dimy + radius; iy++)
        for (int ix = -radius; ix < dimx + radius; ix++) {
          if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz) {
            float value = (*src) * coeff[0];
            for (int ir = 1; ir <= radius; ir++) {
              value += coeff[ir] * (*(src + ir) + *(src - ir));
              value += coeff[ir] * (*(src + ir * stride_y) + *(src - ir * stride_y));
              value += coeff[ir] * (*(src + ir * stride_z) + *(src - ir * stride_z));
            }
            *dst = value;
          } else {
            *dst = *src;
          }
          ++dst; ++src;
        }
    float *tmp = bufdst;
    bufdst = bufdstnext;
    bufdstnext = tmp;
    bufsrc = tmp;
  }
  shrLog("\n");
  free(intermediate);
  return true;
}

bool compareData(const float *output, const float *reference,
                 int dimx, int dimy, int dimz, int radius,
                 float tolerance = 0.0001f) {
  for (int iz = -radius; iz < dimz + radius; iz++)
    for (int iy = -radius; iy < dimy + radius; iy++)
      for (int ix = -radius; ix < dimx + radius; ix++) {
        if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz) {
          float difference = fabsf(*reference - *output);
          float error = (*reference != 0) ? difference / *reference : difference;
          if (error > tolerance) {
            shrLog("Data error at point (%d,%d,%d)\t%f instead of %f\n",
                   ix, iy, iz, *output, *reference);
            return false;
          }
        }
        ++output; ++reference;
      }
  return true;
}

// ============================================================
// GPU FDTD via Kokkos TeamPolicy + scratch memory
// ============================================================
bool fdtdGPU(float *output, float *input, const float *coeff,
             int dimx, int dimy, int dimz, int radius, int timesteps,
             int argc, const char **argv) {
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = (size_t)outerDimx * outerDimy * outerDimz;

  // Pad to align inner data on 128-byte boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

  // Work group sizing
  const int userWorkSize = 256;
  const int workx = k_localWorkX;
  const int worky = userWorkSize / k_localWorkX;   // = 8

  const int teamX = (int)ceil((float)dimx / workx);
  const int teamY = (int)ceil((float)dimy / worky);
  const int numTeam = teamX * teamY;

  shrLog(" set thread size to %dx%d\n", workx, worky);
  shrLog(" set team size to %dx%d\n", teamX, teamY);
  shrLog(" GPU FDTD loop\n");

  // Tile dimensions (match OMP version): (localWorkMaxY + 2*radius) x (localWorkMaxX + 2*radius)
  const int tileRows = localWorkMaxY + 2 * k_radius_default; // 24
  const int tileCols = localWorkMaxX + 2 * k_radius_default; // 40
  const int tileSize = tileRows * tileCols;                   // 960 floats

  // Allocate device buffers (padded)
  Kokkos::View<float *> d_bufferIn("bufferIn",   paddedVolumeSize);
  Kokkos::View<float *> d_bufferOut("bufferOut", paddedVolumeSize);
  Kokkos::View<float *> d_coeff("coeff", radius + 1);

  {
    auto h_in  = Kokkos::create_mirror_view(d_bufferIn);
    auto h_out = Kokkos::create_mirror_view(d_bufferOut);
    auto h_c   = Kokkos::create_mirror_view(d_coeff);
    for (size_t i = 0; i < paddedVolumeSize; i++) h_in(i)  = 0.0f;
    for (size_t i = 0; i < paddedVolumeSize; i++) h_out(i) = 0.0f;
    memcpy(static_cast<float*>(h_in.data())  + padding, input, volumeSize * sizeof(float));
    memcpy(static_cast<float*>(h_out.data()) + padding, input, volumeSize * sizeof(float));
    for (int i = 0; i <= radius; i++) h_c(i) = coeff[i];
    Kokkos::deep_copy(d_bufferIn,  h_in);
    Kokkos::deep_copy(d_bufferOut, h_out);
    Kokkos::deep_copy(d_coeff, h_c);
  }

  // Alternate between two Views via raw pointers into them
  // We'll use a pair and swap via index
  Kokkos::View<float *> bufs[2] = {d_bufferIn, d_bufferOut};
  int cur = 0; // bufs[cur] = in, bufs[1-cur] = out

  using team_policy_t = Kokkos::TeamPolicy<>;
  using member_type   = team_policy_t::member_type;
  using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView   = Kokkos::View<float *, scratch_space, Kokkos::MemoryUnmanaged>;

  team_policy_t policy(numTeam, userWorkSize);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(tileSize * sizeof(float)));

  auto start = std::chrono::steady_clock::now();

  for (int it = 0; it < timesteps; it++) {
    auto d_in  = bufs[cur];
    auto d_out = bufs[1 - cur];
    const int pad_val   = padding;
    const int stride_y  = dimx + 2 * k_radius_default;
    const int stride_z  = stride_y * (dimy + 2 * k_radius_default);

    Kokkos::parallel_for(
        "fdtd3d", policy,
        KOKKOS_LAMBDA(const member_type &team) {
          ScratchView tile(team.team_scratch(0), tileRows * tileCols);

          const int tid   = team.team_rank();
          const int ltidx = tid % workx;
          const int ltidy = tid / workx;
          const int gtidx = (team.league_rank() % teamX) * workx + ltidx;
          const int gtidy = (team.league_rank() / teamX) * worky + ltidy;

          bool valid = (gtidx < dimx) && (gtidy < dimy);

          int inputIndex = k_radius_default * stride_y + k_radius_default + pad_val
                         + gtidy * stride_y + gtidx;

          float infront[k_radius_default];
          float behind[k_radius_default];
          float current;

          const int tx = ltidx + k_radius_default;
          const int ty = ltidy + k_radius_default;

          // Preload behind slices
          for (int i = k_radius_default - 2; i >= 0; i--) {
            behind[i] = d_in(inputIndex);
            inputIndex += stride_z;
          }
          current = d_in(inputIndex);
          int outputIndex = inputIndex;
          inputIndex += stride_z;

          // Preload infront slices
          for (int i = 0; i < k_radius_default; i++) {
            infront[i] = d_in(inputIndex);
            inputIndex += stride_z;
          }

          // Step through xy-planes
          for (int iz = 0; iz < dimz; iz++) {
            // Advance the z-pencil
            for (int i = k_radius_default - 1; i > 0; i--)
              behind[i] = behind[i - 1];
            behind[0] = current;
            current = infront[0];
            for (int i = 0; i < k_radius_default - 1; i++)
              infront[i] = infront[i + 1];
            infront[k_radius_default - 1] = d_in(inputIndex);
            inputIndex  += stride_z;
            outputIndex += stride_z;

            team.team_barrier();

            // Load halo above and below (y direction)
            if (ltidy < k_radius_default) {
              tile(ltidy * tileCols + tx) =
                d_in(outputIndex - k_radius_default * stride_y);
              tile((ltidy + worky + k_radius_default) * tileCols + tx) =
                d_in(outputIndex + worky * stride_y);
            }
            // Load halo left and right (x direction)
            if (ltidx < k_radius_default) {
              tile(ty * tileCols + ltidx) =
                d_in(outputIndex - k_radius_default);
              tile(ty * tileCols + ltidx + workx + k_radius_default) =
                d_in(outputIndex + workx);
            }
            tile(ty * tileCols + tx) = current;

            team.team_barrier();

            // Compute stencil
            float value = d_coeff(0) * current;
            for (int i = 1; i <= k_radius_default; i++) {
              value += d_coeff(i) * (infront[i - 1] + behind[i - 1]
                         + tile((ty - i) * tileCols + tx)
                         + tile((ty + i) * tileCols + tx)
                         + tile(ty * tileCols + (tx - i))
                         + tile(ty * tileCols + (tx + i)));
            }

            if (valid) d_out(outputIndex) = value;
          }
        });
    Kokkos::fence();

    cur = 1 - cur;  // toggle buffers
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / timesteps);

  // Copy result out (bufs[cur] is now "in" = most recent output)
  {
    auto h_in = Kokkos::create_mirror_view(bufs[cur]);
    Kokkos::deep_copy(h_in, bufs[cur]);
    memcpy(output, static_cast<float*>(h_in.data()) + padding, volumeSize * sizeof(float));
  }
  return true;
}

// ============================================================
// Main
// ============================================================
bool runTest(int argc, const char **argv) {
  bool ok = true;
  float *host_output = nullptr;
  float *device_output = nullptr;
  float *input = nullptr;
  float *coeff = nullptr;

  int dimx, dimy, dimz;
  int radius    = k_radius_default;
  int timesteps = k_timesteps_default;
  size_t volumeSize;
  uint64_t memsize = MEMORY_SIZE;

  // Determine default dimensions
  memsize /= 2;
  int defaultDim = (int)floor(pow(memsize / (2.0 * sizeof(float)), 1.0 / 3.0));
  int roundTarget = 128 / sizeof(float);
  defaultDim = defaultDim / roundTarget * roundTarget;
  defaultDim -= k_radius_default * 2;
  if (defaultDim < k_dim_min) { shrLog("Insufficient device memory\n"); ok = false; }
  else if (defaultDim > k_dim_max) defaultDim = k_dim_max;

  if (ok && shrCheckCmdLineFlag(argc, argv, "qatest"))
    defaultDim = MIN(defaultDim, k_dim_qa);

  // Parse dimensions
  char *dim_str = nullptr;
  dimx = shrGetCmdLineArgumentstr(argc, argv, "dimx", &dim_str) ? atoi(dim_str) : defaultDim;
  dimy = shrGetCmdLineArgumentstr(argc, argv, "dimy", &dim_str) ? atoi(dim_str) : defaultDim;
  dimz = shrGetCmdLineArgumentstr(argc, argv, "dimz", &dim_str) ? atoi(dim_str) : defaultDim;
  if (shrGetCmdLineArgumentstr(argc, argv, "timesteps", &dim_str))
    timesteps = atoi(dim_str);

  if (ok) {
    int outerDimx = dimx + 2 * radius;
    int outerDimy = dimy + 2 * radius;
    int outerDimz = dimz + 2 * radius;
    volumeSize = (size_t)outerDimx * outerDimy * outerDimz;
  }

  if (ok) {
    shrLog(" calloc host_output\n");
    host_output = (float *)calloc(volumeSize, sizeof(float));
    if (!host_output) { shrLog("Insufficient memory for host_output\n"); ok = false; }
  }
  if (ok) {
    shrLog(" malloc input\n");
    input = (float *)malloc(volumeSize * sizeof(float));
    if (!input) { shrLog("Insufficient memory for input\n"); ok = false; }
  }
  if (ok) {
    shrLog(" malloc coeff\n");
    coeff = (float *)malloc((radius + 1) * sizeof(float));
    if (!coeff) { shrLog("Insufficient memory for coeff\n"); ok = false; }
  }

  if (ok) {
    for (int i = 0; i <= radius; i++) coeff[i] = 0.1f;
  }

  if (ok) {
    shrLog(" generateRandomData\n\n");
    int outerDimx = dimx + 2 * radius;
    int outerDimy = dimy + 2 * radius;
    int outerDimz = dimz + 2 * radius;
    generateRandomData(input, outerDimx, outerDimy, outerDimz, 0.0f, 1.0f);
  }

  if (ok) {
    shrLog("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n",
           dimx, dimy, dimz, radius, timesteps);
  }

  if (ok) {
    shrLog("fdtdReference...\n");
    ok = fdtdReference(host_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
    shrLog("fdtdReference complete\n");
  }

  if (ok) {
    shrLog(" calloc device_output\n");
    device_output = (float *)calloc(volumeSize, sizeof(float));
    if (!device_output) { shrLog("Insufficient memory for device output\n"); ok = false; }
  }

  if (ok) {
    shrLog("fdtdGPU...\n");
    ok = fdtdGPU(device_output, input, coeff, dimx, dimy, dimz, radius, timesteps, argc, argv);
    shrLog("fdtdGPU complete\n");
  }

  if (ok) {
    float tolerance = 0.0001f;
    shrLog("\nCompareData (tolerance %f)...\n", tolerance);
    ok = compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);
  }

  if (input)         free(input);
  if (coeff)         free(coeff);
  if (host_output)   free(host_output);
  if (device_output) free(device_output);
  return ok;
}

void showHelp(int argc, const char **argv) {
  if (argc > 0) std::cout << std::endl << argv[0] << std::endl;
  std::cout << "\nSyntax:\n";
  std::cout << "    --dimx=<N>      Number of elements in x direction\n";
  std::cout << "    --dimy=<N>      Number of elements in y direction\n";
  std::cout << "    --dimz=<N>      Number of elements in z direction\n";
  std::cout << "    --timesteps=<N> Number of timesteps\n";
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    if (shrCheckCmdLineFlag(argc, (const char **)argv, "help")) {
      showHelp(argc, (const char **)argv);
    } else {
      bool result = runTest(argc, (const char **)argv);
      printf("%s\n", result ? "PASS" : "FAIL");
    }
  }
  Kokkos::finalize();
  return 0;
}
