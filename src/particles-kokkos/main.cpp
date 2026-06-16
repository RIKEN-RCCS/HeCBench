/*
 * Particles simulation - Kokkos port
 * Ported from particles-omp (NVIDIA CUDA SDK particles sample)
 */

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f
#define GRID_SIZE         64
#define NUM_PARTICLES     16384

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>

#define UMAD(a, b, c)  ( (a) * (b) + (c) )

struct uint3 { unsigned int x, y, z; };
struct int4  { int x, y, z, w; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

typedef struct {
  float3 colliderPos;
  float  colliderRadius;
  float3 gravity;
  float  globalDamping;
  float  particleRadius;
  uint3  gridSize;
  unsigned int numCells;
  float3 worldOrigin;
  float3 cellSize;
  unsigned int numBodies;
  unsigned int maxParticlesPerCell;
  float spring;
  float damping;
  float shear;
  float attraction;
  float boundaryDamping;
} simParams_t;

// ---------------------------------------------------------------------------
// Device-callable helpers
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
static void f4_add_eq(float4 &a, float4 b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

KOKKOS_INLINE_FUNCTION
static float4 f4_add(float4 a, float4 b)  { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }

KOKKOS_INLINE_FUNCTION
static float4 f4_sub(float4 a, float4 b)  { return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }

KOKKOS_INLINE_FUNCTION
static float4 f4_mulf(float4 a, float b)  { return {a.x*b, a.y*b, a.z*b, a.w*b}; }

KOKKOS_INLINE_FUNCTION
static void f4_mul_eq(float4 &a, float b) { a.x*=b; a.y*=b; a.z*=b; a.w*=b; }

KOKKOS_INLINE_FUNCTION
static int4 getGridPos(const float4 p, const simParams_t &params)
{
  int4 gp;
  gp.x = (int)floorf((p.x - params.worldOrigin.x) / params.cellSize.x);
  gp.y = (int)floorf((p.y - params.worldOrigin.y) / params.cellSize.y);
  gp.z = (int)floorf((p.z - params.worldOrigin.z) / params.cellSize.z);
  gp.w = 0;
  return gp;
}

KOKKOS_INLINE_FUNCTION
static unsigned int getGridHash(int4 gridPos, const simParams_t &params)
{
  gridPos.x = gridPos.x & (params.gridSize.x - 1);
  gridPos.y = gridPos.y & (params.gridSize.y - 1);
  gridPos.z = gridPos.z & (params.gridSize.z - 1);
  return UMAD(UMAD(gridPos.z, params.gridSize.y, gridPos.y), params.gridSize.x, gridPos.x);
}

KOKKOS_INLINE_FUNCTION
static float4 collideSpheres(
    float4 posA, float4 posB,
    float4 velA, float4 velB,
    float radiusA, float radiusB,
    float spring, float damping, float shear, float attraction)
{
  float4 relPos = {posB.x-posA.x, posB.y-posA.y, posB.z-posA.z, 0};
  float dist = sqrtf(relPos.x*relPos.x + relPos.y*relPos.y + relPos.z*relPos.z);
  float collideDist = radiusA + radiusB;
  float4 force = {0,0,0,0};
  if (dist < collideDist) {
    float4 norm = {relPos.x/dist, relPos.y/dist, relPos.z/dist, 0};
    float4 relVel = {velB.x-velA.x, velB.y-velA.y, velB.z-velA.z, 0};
    float relVelDotNorm = relVel.x*norm.x + relVel.y*norm.y + relVel.z*norm.z;
    float4 tanVel = {relVel.x - relVelDotNorm*norm.x,
                     relVel.y - relVelDotNorm*norm.y,
                     relVel.z - relVelDotNorm*norm.z, 0};
    float sf = -spring * (collideDist - dist);
    force = {sf*norm.x + damping*relVel.x + shear*tanVel.x + attraction*relPos.x,
             sf*norm.y + damping*relVel.y + shear*tanVel.y + attraction*relPos.y,
             sf*norm.z + damping*relVel.z + shear*tanVel.z + attraction*relPos.z, 0};
  }
  return force;
}

// ---------------------------------------------------------------------------
// Kokkos Views typedefs
// ---------------------------------------------------------------------------
using ViewF4  = Kokkos::View<float4*>;
using ViewUI  = Kokkos::View<unsigned int*>;
using MirrorF4 = ViewF4::HostMirror;
using MirrorUI = ViewUI::HostMirror;

// ---------------------------------------------------------------------------
// integrateSystem
// ---------------------------------------------------------------------------
void integrateSystem(ViewF4 d_Pos, ViewF4 d_Vel,
                     const simParams_t &params, float deltaTime,
                     unsigned int numParticles)
{
  Kokkos::parallel_for("integrateSystem", numParticles, KOKKOS_LAMBDA(int index) {
    float4 pos = d_Pos(index);
    float4 vel = d_Vel(index);
    pos.w = 1.0f;
    vel.w = 0.0f;

    float4 g = {params.gravity.x, params.gravity.y, params.gravity.z, 0};
    f4_add_eq(vel, f4_mulf(g, deltaTime));
    f4_mul_eq(vel, params.globalDamping);
    f4_add_eq(pos, f4_mulf(vel, deltaTime));

    // Collide with cube
    if (pos.x < -1.0f + params.particleRadius) { pos.x = -1.0f + params.particleRadius; vel.x *= params.boundaryDamping; }
    if (pos.x >  1.0f - params.particleRadius) { pos.x =  1.0f - params.particleRadius; vel.x *= params.boundaryDamping; }
    if (pos.y < -1.0f + params.particleRadius) { pos.y = -1.0f + params.particleRadius; vel.y *= params.boundaryDamping; }
    if (pos.y >  1.0f - params.particleRadius) { pos.y =  1.0f - params.particleRadius; vel.y *= params.boundaryDamping; }
    if (pos.z < -1.0f + params.particleRadius) { pos.z = -1.0f + params.particleRadius; vel.z *= params.boundaryDamping; }
    if (pos.z >  1.0f - params.particleRadius) { pos.z =  1.0f - params.particleRadius; vel.z *= params.boundaryDamping; }

    d_Pos(index) = pos;
    d_Vel(index) = vel;
  });
  Kokkos::fence();
}

// ---------------------------------------------------------------------------
// calcHash
// ---------------------------------------------------------------------------
void calcHash(ViewUI d_Hash, ViewUI d_Index, ViewF4 d_Pos,
              const simParams_t &params, unsigned int numParticles)
{
  Kokkos::parallel_for("calcHash", numParticles, KOKKOS_LAMBDA(int index) {
    float4 p = d_Pos(index);
    int4 gridPos = getGridPos(p, params);
    d_Hash(index)  = getGridHash(gridPos, params);
    d_Index(index) = index;
  });
  Kokkos::fence();
}

// ---------------------------------------------------------------------------
// bitonicSort - in-place bitonic sort on Kokkos Views (ascending, dir=0)
// Keys are d_Hash, values are d_Index.
// Sorts a power-of-2 array of length n.
// ---------------------------------------------------------------------------
void bitonicSort(ViewUI d_DstKey, ViewUI d_DstVal,
                 ViewUI d_SrcKey, ViewUI d_SrcVal,
                 unsigned int batch, unsigned int arrayLength, unsigned int dir)
{
  if (arrayLength < 2) return;

  // Copy src -> dst if they differ
  if (d_SrcKey.data() != d_DstKey.data()) {
    Kokkos::deep_copy(d_DstKey, d_SrcKey);
    Kokkos::deep_copy(d_DstVal, d_SrcVal);
  }

  const unsigned int n = arrayLength;
  for (unsigned int k = 2; k <= n; k <<= 1) {
    for (unsigned int j = k >> 1; j > 0; j >>= 1) {
      Kokkos::parallel_for("bitonicSort", n, KOKKOS_LAMBDA(int tid) {
        unsigned int ixj = tid ^ j;
        if (ixj > (unsigned int)tid) {
          bool ascending = ((tid & k) == 0) ^ (dir != 0);
          if ((d_DstKey(tid) > d_DstKey(ixj)) == ascending) {
            unsigned int tk = d_DstKey(tid); d_DstKey(tid) = d_DstKey(ixj); d_DstKey(ixj) = tk;
            unsigned int tv = d_DstVal(tid); d_DstVal(tid) = d_DstVal(ixj); d_DstVal(ixj) = tv;
          }
        }
      });
      Kokkos::fence();
    }
  }
}

// ---------------------------------------------------------------------------
// findCellBoundsAndReorder
// ---------------------------------------------------------------------------
void findCellBoundsAndReorder(
    ViewUI d_CellStart, ViewUI d_CellEnd,
    ViewF4 d_ReorderedPos, ViewF4 d_ReorderedVel,
    ViewUI d_Hash, ViewUI d_Index,
    ViewF4 d_Pos, ViewF4 d_Vel,
    unsigned int numParticles, unsigned int numCells)
{
  // Reset cell start to sentinel
  Kokkos::deep_copy(d_CellStart, 0xFFFFFFFFU);
  Kokkos::fence();

  Kokkos::parallel_for("findCellBounds", numParticles, KOKKOS_LAMBDA(int index) {
    unsigned int hash = d_Hash(index);

    if (index == 0) {
      d_CellStart(hash) = 0;
    } else if (hash != d_Hash(index - 1)) {
      d_CellEnd(d_Hash(index - 1)) = index;
      d_CellStart(hash) = index;
    }
    if (index == (int)numParticles - 1)
      d_CellEnd(hash) = numParticles;

    unsigned int sortedIndex = d_Index(index);
    d_ReorderedPos(index) = d_Pos(sortedIndex);
    d_ReorderedVel(index) = d_Vel(sortedIndex);
  });
  Kokkos::fence();
}

// ---------------------------------------------------------------------------
// collide
// ---------------------------------------------------------------------------
void collide(
    ViewF4 d_Vel,
    ViewF4 d_ReorderedPos,
    ViewF4 d_ReorderedVel,
    ViewUI d_Index,
    ViewUI d_CellStart,
    ViewUI d_CellEnd,
    const simParams_t &params,
    unsigned int numParticles,
    unsigned int /*numCells*/)
{
  Kokkos::parallel_for("collide", numParticles, KOKKOS_LAMBDA(int index) {
    float4 pos   = d_ReorderedPos(index);
    float4 vel   = d_ReorderedVel(index);
    float4 force = {0,0,0,0};

    int4 gridPos = getGridPos(pos, params);

    for (int z = -1; z <= 1; z++)
      for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++) {
          int4 t = {x, y, z, 0};
          int4 nb = {gridPos.x+t.x, gridPos.y+t.y, gridPos.z+t.z, 0};
          unsigned int hash = getGridHash(nb, params);
          unsigned int startI = d_CellStart(hash);
          if (startI == 0xFFFFFFFFU) continue;
          unsigned int endI = d_CellEnd(hash);
          for (unsigned int j = startI; j < endI; j++) {
            if (j == (unsigned int)index) continue;
            float4 pos2 = d_ReorderedPos(j);
            float4 vel2 = d_ReorderedVel(j);
            f4_add_eq(force, collideSpheres(
                pos, pos2, vel, vel2,
                params.particleRadius, params.particleRadius,
                params.spring, params.damping, params.shear, params.attraction));
          }
        }

    // Collide with cursor sphere
    float4 colliderPos = {params.colliderPos.x, params.colliderPos.y, params.colliderPos.z, 0};
    float4 zero4 = {0,0,0,0};
    f4_add_eq(force, collideSpheres(
        pos, colliderPos, vel, zero4,
        params.particleRadius, params.colliderRadius,
        params.spring, params.damping, params.shear, params.attraction));

    d_Vel(d_Index(index)) = f4_add(vel, force);
  });
  Kokkos::fence();
}

// ---------------------------------------------------------------------------
// Host init helpers
// ---------------------------------------------------------------------------
inline float frand() { return (float)rand() / (float)RAND_MAX; }

void initGrid(float *hPos, float *hVel, float particleRadius, float spacing,
              unsigned int numParticles)
{
  float jitter = particleRadius * 0.01f;
  unsigned int s = (int)ceilf(powf((float)numParticles, 1.0f/3.0f));
  unsigned int gs[3] = {s, s, s};

  srand(1973);
  for (unsigned int z = 0; z < gs[2]; z++)
    for (unsigned int y = 0; y < gs[1]; y++)
      for (unsigned int x = 0; x < gs[0]; x++) {
        unsigned int i = z*gs[0]*gs[1] + y*gs[1] + x;
        if (i < numParticles) {
          hPos[i*4+0] = spacing*x + particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
          hPos[i*4+1] = spacing*y + particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
          hPos[i*4+2] = spacing*z + particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
          hPos[i*4+3] = 1.0f;
          hVel[i*4+0] = hVel[i*4+1] = hVel[i*4+2] = hVel[i*4+3] = 0.0f;
        }
      }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <iterations>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int iterations = atoi(argv[1]);
    const float timestep = 0.5f;
    const float fParticleRadius = 0.023f;
    const float fColliderRadius = 0.17f;

    unsigned int numParticles = NUM_PARTICLES;
    unsigned int gridDim      = GRID_SIZE;

    uint3 gridSize;
    gridSize.x = gridSize.y = gridSize.z = gridDim;
    unsigned int numGridCells = gridSize.x * gridSize.y * gridSize.z;

    simParams_t params;
    params.gridSize       = gridSize;
    params.numCells       = numGridCells;
    params.numBodies      = numParticles;
    params.particleRadius = fParticleRadius;
    params.colliderPos    = {1.2f, -0.8f, 0.8f};
    params.colliderRadius = fColliderRadius;
    params.worldOrigin    = {1.0f, -1.0f, -1.0f};
    float cellSize = params.particleRadius * 2.0f;
    params.cellSize       = {cellSize, cellSize, cellSize};
    params.spring         = 0.5f;
    params.damping        = 0.02f;
    params.shear          = 0.1f;
    params.attraction     = 0.0f;
    params.boundaryDamping = -0.5f;
    params.gravity        = {0.0f, -0.0003f, 0.0f};
    params.globalDamping  = 1.0f;

    printf(" grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, numGridCells);
    printf(" particles: %d\n\n", numParticles);

    // Host arrays
    float *hPos          = new float[numParticles * 4];
    float *hVel          = new float[numParticles * 4];
    float *hReorderedPos = new float[numParticles * 4];
    float *hReorderedVel = new float[numParticles * 4];

    initGrid(hPos, hVel, params.particleRadius, params.particleRadius * 2.0f, numParticles);

    // Device views
    ViewF4 dPos("dPos", numParticles);
    ViewF4 dVel("dVel", numParticles);
    ViewF4 dReorderedPos("dReorderedPos", numParticles);
    ViewF4 dReorderedVel("dReorderedVel", numParticles);
    ViewUI dHash("dHash", numParticles);
    ViewUI dIndex("dIndex", numParticles);
    ViewUI dCellStart("dCellStart", numGridCells);
    ViewUI dCellEnd("dCellEnd", numGridCells);

    // Upload initial positions and velocities
    {
      auto hPosM = Kokkos::create_mirror_view(dPos);
      auto hVelM = Kokkos::create_mirror_view(dVel);
      for (unsigned int i = 0; i < numParticles; ++i) {
        hPosM(i) = ((float4*)hPos)[i];
        hVelM(i) = ((float4*)hVel)[i];
      }
      Kokkos::deep_copy(dPos, hPosM);
      Kokkos::deep_copy(dVel, hVelM);
    }

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
      integrateSystem(dPos, dVel, params, timestep, numParticles);

      calcHash(dHash, dIndex, dPos, params, numParticles);

      bitonicSort(dHash, dIndex, dHash, dIndex, 1, numParticles, 0);

      findCellBoundsAndReorder(
          dCellStart, dCellEnd,
          dReorderedPos, dReorderedVel,
          dHash, dIndex,
          dPos, dVel,
          numParticles, numGridCells);

      collide(dVel, dReorderedPos, dReorderedVel,
              dIndex, dCellStart, dCellEnd,
              params, numParticles, numGridCells);
    }

    auto end = std::chrono::steady_clock::now();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total execution time of %d loop iterations: %f (s)\n", iterations, ns * 1e-9f);
    printf("Average execution time of a loop iteration: %f (us)\n", (ns * 1e-3f) / iterations);

    delete[] hPos;
    delete[] hVel;
    delete[] hReorderedPos;
    delete[] hReorderedVel;
  }
  Kokkos::finalize();
  return 0;
}
