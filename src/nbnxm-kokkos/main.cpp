// Kokkos port of nbnxm (non-bonded neighbor list) CUDA benchmark.
// Computes Lennard-Jones + PME electrostatic interactions between atom clusters.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// Constants (from constants.h)
// ============================================================
static constexpr int c_nbnxnGpuNumClusterPerSupercluster = 8;
static constexpr int c_clSize                            = 8;
static constexpr int c_nbnxnGpuClusterpairSplit          = 2;
static constexpr int c_nbnxnGpuJgroupSize                = (32 / c_nbnxnGpuNumClusterPerSupercluster); // 4
static constexpr int c_nbnxnGpuExclSize                  = c_clSize * c_clSize / c_nbnxnGpuClusterpairSplit; // 32
static constexpr int c_splitClSize                       = c_clSize / c_nbnxnGpuClusterpairSplit; // 4
static constexpr int c_centralShiftIndex                 = 13; // c_numIvecs / 2 where c_numIvecs=27
static constexpr float c_nbnxnMinDistanceSquared         = 3.82e-07f;
static constexpr unsigned int NBNXN_INTERACTION_MASK_ALL = 0xffffffffU;
static constexpr unsigned superClInteractionMask         = ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);
static constexpr int DIM = 3;

// Grid / problem size
static constexpr int grid_z   = 3199;
static constexpr int block_x  = 8;
static constexpr int block_y  = 8;
static constexpr int NUM_ATOMS = (grid_z * c_nbnxnGpuNumClusterPerSupercluster + block_x) * c_clSize + block_y;

// Data structure sizes used in the original benchmark
static constexpr int NUM_CJ4  = 56881;
static constexpr int NUM_SCI  = 4806;
static constexpr int NUM_EXCL = 19205;
static constexpr int NUM_NBFP = 1024;
static constexpr int NUM_SHIFT = 45;
static constexpr int NUM_TYPES = 32;

// ============================================================
// Data structures
// ============================================================
struct nbnxn_im_ei_t {
  unsigned int imask   = 0U;
  int          excl_ind = 0;
};

struct nbnxn_cj4_t {
  int           cj[c_nbnxnGpuJgroupSize];
  nbnxn_im_ei_t imei[c_nbnxnGpuClusterpairSplit];
};

struct nbnxn_sci_t {
  int sci;
  int shift;
  int cj4_ind_start;
  int cj4_ind_end;
};

struct nbnxn_excl_t {
  unsigned int pair[c_nbnxnGpuExclSize];
};

// Simple float4 / float3 / float2 helpers (no CUDA dependency)
struct Float4 { float x, y, z, w; };
struct Float3 { float x, y, z;
  KOKKOS_INLINE_FUNCTION float& operator[](int i)       { return (&x)[i]; }
  KOKKOS_INLINE_FUNCTION float  operator[](int i) const { return (&x)[i]; }
};
struct Float2 { float x, y; };

// ============================================================
// PME correction polynomial (from original CUDA source)
// ============================================================
KOKKOS_INLINE_FUNCTION float pmeCorrF(float z2) {
  constexpr float FN6 = -1.7357322914161492954e-8f;
  constexpr float FN5 =  1.4703624142580877519e-6f;
  constexpr float FN4 = -0.000053401640219807709149f;
  constexpr float FN3 =  0.0010054721316683106153f;
  constexpr float FN2 = -0.019278317264888380590f;
  constexpr float FN1 =  0.069670166153766424023f;
  constexpr float FN0 = -0.75225204789749321333f;

  constexpr float FD4 = 0.0011193462567257629232f;
  constexpr float FD3 = 0.014866955030185295499f;
  constexpr float FD2 = 0.11583842382862377919f;
  constexpr float FD1 = 0.50736591960530292870f;
  constexpr float FD0 = 1.0f;

  const float z4       = z2 * z2;
  float       polyFD0  = FD4 * z4 + FD2;
  const float polyFD1  = FD3 * z4 + FD1;
  polyFD0 = polyFD0 * z4 + FD0;
  polyFD0 = polyFD1 * z2 + polyFD0;
  polyFD0 = 1.0f / polyFD0;

  float polyFN0 = FN6 * z4 + FN4;
  float polyFN1 = FN5 * z4 + FN3;
  polyFN0       = polyFN0 * z4 + FN2;
  polyFN1       = polyFN1 * z4 + FN1;
  polyFN0       = polyFN0 * z4 + FN0;
  polyFN0       = polyFN1 * z2 + polyFN0;

  return polyFN0 * polyFD0;
}

// ============================================================
// Kokkos View types
// ============================================================
using ViewFloat4  = Kokkos::View<Float4*>;
using ViewFloat3  = Kokkos::View<Float3*>;
using ViewFloat2  = Kokkos::View<Float2*>;
using ViewInt     = Kokkos::View<int*>;
using ViewCj4     = Kokkos::View<nbnxn_cj4_t*>;
using ViewSci     = Kokkos::View<nbnxn_sci_t*>;
using ViewExcl    = Kokkos::View<nbnxn_excl_t*>;

// ============================================================
// Helper initializers (matching CUDA benchmark setup)
// ============================================================
static nbnxn_cj4_t make_cj4(int id) {
  nbnxn_cj4_t v;
  for (int i = 0; i < c_nbnxnGpuJgroupSize; ++i)
    v.cj[i] = i + id;
  for (int i = 0; i < c_nbnxnGpuClusterpairSplit; ++i) {
    v.imei[i].imask    = 0U;
    v.imei[i].excl_ind = 0;
  }
  return v;
}

static nbnxn_sci_t make_sci(int id) {
  return {id, 0, 8 * id, 8 * id + 7};
}

static nbnxn_excl_t make_excl() {
  nbnxn_excl_t v;
  for (int i = 0; i < c_nbnxnGpuExclSize; ++i)
    v.pair[i] = 7;
  return v;
}

// ============================================================
// Main NBNXM kernel (Kokkos TeamPolicy version)
//
// Mapping from CUDA:
//   CUDA block (bidx)  → Kokkos league_rank()
//   CUDA tidxi/tidxj   → derived from team_rank()
//   __shared__ memory  → team scratch memory (level 0)
//   atomicAdd          → Kokkos::atomic_add
// ============================================================
void runNbnxmKernel(
    const ViewFloat4&  a_xq,
    const ViewFloat3&  a_f,       // output (read-modify-write via atomic)
    const ViewFloat3&  a_shiftVec,
    const ViewFloat3&  a_fShift,  // output
    const ViewCj4&     a_cj4,
    const ViewSci&     a_sci,
    const ViewExcl&    a_excl,
    const ViewInt&     a_atomTypes,
    const ViewFloat2&  a_nbfp,
    int   numTypes,
    float rCoulombSq,
    float ewaldBeta,
    float epsFac,
    bool  calcShift,
    // mutable output views (separate to allow atomic)
    Kokkos::View<Float3*> f_out,
    Kokkos::View<Float3*> fShift_out)
{
  const int num_sci = static_cast<int>(a_sci.extent(0));

  // Scratch memory per team: stores i-cluster positions (xq) and atom types.
  // c_nbnxnGpuNumClusterPerSupercluster * c_clSize entries of Float4 + int.
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using ScratchFloat4 = Kokkos::View<Float4*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using ScratchInt    = Kokkos::View<int*,    ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const int nIClust = c_nbnxnGpuNumClusterPerSupercluster * c_clSize;
  const int scratch_total =
      ScratchFloat4::shmem_size(nIClust) + ScratchInt::shmem_size(nIClust);

  using policy_t = Kokkos::TeamPolicy<>;
  using member_t = policy_t::member_type;

  // Team size: c_clSize * c_clSize = 64 (matches CUDA 8x8 block)
  const int team_size = c_clSize * c_clSize;

  Kokkos::parallel_for("nbnxm_kernel",
    policy_t(num_sci, team_size)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_total)),
    KOKKOS_LAMBDA(const member_t& team) {
      const int bidx  = team.league_rank();
      const int tidx  = team.team_rank();
      const int tidxi = tidx % c_clSize;   // 0..7
      const int tidxj = tidx / c_clSize;   // 0..7

      // Scratch memory views (mapped to shared / fast memory)
      ScratchFloat4 sm_xq(team.team_scratch(0), nIClust);
      ScratchInt    sm_atomTypeI(team.team_scratch(0), nIClust);

      const nbnxn_sci_t nbSci     = a_sci(bidx);
      const int         sci       = nbSci.sci;
      const int         cij4Start = nbSci.cj4_ind_start;
      const int         cij4End   = nbSci.cj4_ind_end;
      const int         sciShift  = nbSci.shift;

      // Load i-cluster positions into scratch memory
      // In CUDA: loop over i=0..c_nbnxnGpuNumClusterPerSupercluster/c_clSize (=1)
      // since c_nbnxnGpuNumClusterPerSupercluster == c_clSize, i is always 0
      {
        const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj;
        const int ai = ci * c_clSize + tidxi;
        Float4 xqi = a_xq(ai);
        const Float3 shift = a_shiftVec(sciShift);
        xqi.x += shift.x;
        xqi.y += shift.y;
        xqi.z += shift.z;
        xqi.w *= epsFac;
        sm_xq[tidxj * c_clSize + tidxi]      = xqi;
        sm_atomTypeI[tidxj * c_clSize + tidxi] = a_atomTypes(ai);
      }
      team.team_barrier();

      const float beta2 = ewaldBeta * ewaldBeta;
      const float beta3 = ewaldBeta * ewaldBeta * ewaldBeta;

      // Per-thread force accumulators for each i-cluster (8 values)
      float fCiBufX[c_nbnxnGpuNumClusterPerSupercluster] = {};
      float fCiBufY[c_nbnxnGpuNumClusterPerSupercluster] = {};
      float fCiBufZ[c_nbnxnGpuNumClusterPerSupercluster] = {};

      const bool nonSelfInteraction =
          !(sciShift == c_centralShiftIndex && tidxj <= tidxi);

      const int prunedClusterPairSize = c_clSize * c_splitClSize; // 32
      const int imeiIdx               = tidx / prunedClusterPairSize;

      // Loop over j4 entries
      for (int j4 = cij4Start; j4 < cij4End; ++j4) {
        unsigned imask = a_cj4(j4).imei[imeiIdx].imask;
        if (!imask) continue;

        const int      wexclIdx = a_cj4(j4).imei[imeiIdx].excl_ind;
        const unsigned wexcl    = a_excl(wexclIdx).pair[tidx & (prunedClusterPairSize - 1)];

        for (int jm = 0; jm < c_nbnxnGpuJgroupSize; ++jm) {
          const bool maskSet = imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster));
          if (!maskSet) continue;

          unsigned  maskJI = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
          const int cj     = a_cj4(j4).cj[jm];
          const int aj     = cj * c_clSize + tidxj;

          const Float4 xqj = a_xq(aj);
          const float  xj  = xqj.x, yj = xqj.y, zj = xqj.z;
          const float  qj  = xqj.w;
          const int    atomTypeJ = a_atomTypes(aj);

          float fCjX = 0.f, fCjY = 0.f, fCjZ = 0.f;

          for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; ++i) {
            if (imask & maskJI) {
              const int   ci  = sci * c_nbnxnGpuNumClusterPerSupercluster + i;
              const Float4 xqi = sm_xq[i * c_clSize + tidxi];
              const float  xi  = xqi.x, yi = xqi.y, zi = xqi.z;

              const float rvx = xi - xj;
              const float rvy = yi - yj;
              const float rvz = zi - zj;
              float        r2 = rvx*rvx + rvy*rvy + rvz*rvz;

              const float pairExclMask = (wexcl & maskJI) ? 1.0f : 0.0f;
              const bool  notExcluded  = (nonSelfInteraction | (ci != cj));

              if ((r2 < rCoulombSq) && notExcluded) {
                const float qi     = xqi.w;
                const int   atomTypeI = sm_atomTypeI[i * c_clSize + tidxi];
                const Float2 c6c12 = a_nbfp(numTypes * atomTypeI + atomTypeJ);
                const float  c6    = c6c12.x;
                const float  c12   = c6c12.y;

                r2 = Kokkos::max(r2, c_nbnxnMinDistanceSquared);

                const float rInv  = 1.0f / Kokkos::sqrt(r2);
                const float r2Inv = rInv * rInv;
                float r6Inv = r2Inv * r2Inv * r2Inv;
                r6Inv *= pairExclMask;
                float fInvR = r6Inv * (c12 * r6Inv - c6) * r2Inv;
                fInvR += qi * qj * (pairExclMask * r2Inv * rInv + pmeCorrF(beta2 * r2) * beta3);

                const float fx = rvx * fInvR;
                const float fy = rvy * fInvR;
                const float fz = rvz * fInvR;

                fCjX -= fx;  fCjY -= fy;  fCjZ -= fz;
                fCiBufX[i] += fx;
                fCiBufY[i] += fy;
                fCiBufZ[i] += fz;
              }
            }
            maskJI += maskJI;
          } // i-loop

          // Reduce j-forces into global memory atomically
          Kokkos::atomic_add(&f_out(aj).x, fCjX);
          Kokkos::atomic_add(&f_out(aj).y, fCjY);
          Kokkos::atomic_add(&f_out(aj).z, fCjZ);
        } // jm-loop
      } // j4-loop

      // Reduce i-forces into global memory atomically
      for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; ++i) {
        const int aidx = (sci * c_nbnxnGpuNumClusterPerSupercluster + i) * c_clSize + tidxi;
        Kokkos::atomic_add(&f_out(aidx).x, fCiBufX[i]);
        Kokkos::atomic_add(&f_out(aidx).y, fCiBufY[i]);
        Kokkos::atomic_add(&f_out(aidx).z, fCiBufZ[i]);

        if (calcShift && sciShift != c_centralShiftIndex) {
          Kokkos::atomic_add(&fShift_out(sciShift).x, fCiBufX[i]);
          Kokkos::atomic_add(&fShift_out(sciShift).y, fCiBufY[i]);
          Kokkos::atomic_add(&fShift_out(sciShift).z, fCiBufZ[i]);
        }
      }
    });
}

// ============================================================
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    // ---- Allocate device views ----
    ViewFloat4 a_xq      ("xq",       NUM_ATOMS);
    ViewFloat3 a_f_base  ("f_base",   NUM_ATOMS);
    ViewFloat3 shiftVec  ("shiftVec", NUM_SHIFT);
    ViewFloat3 fShift_base("fShift",  NUM_SHIFT);
    ViewCj4    cj4       ("cj4",      NUM_CJ4);
    ViewSci    sci       ("sci",      NUM_SCI);
    ViewExcl   excl      ("excl",     NUM_EXCL);
    ViewInt    atomTypes ("atomTypes",NUM_ATOMS);
    ViewFloat2 nbfp      ("nbfp",     NUM_NBFP);

    // ---- Host mirrors for initialization ----
    auto h_xq       = Kokkos::create_mirror_view(a_xq);
    auto h_f        = Kokkos::create_mirror_view(a_f_base);
    auto h_shiftVec = Kokkos::create_mirror_view(shiftVec);
    auto h_fShift   = Kokkos::create_mirror_view(fShift_base);
    auto h_cj4      = Kokkos::create_mirror_view(cj4);
    auto h_sci      = Kokkos::create_mirror_view(sci);
    auto h_excl     = Kokkos::create_mirror_view(excl);
    auto h_atomTypes= Kokkos::create_mirror_view(atomTypes);
    auto h_nbfp     = Kokkos::create_mirror_view(nbfp);

    // Initialize (matching CUDA benchmark)
    for (int i = 0; i < NUM_ATOMS;  ++i) h_xq(i)        = {1.0f, 0.5f, 0.25f, 0.125f};
    for (int i = 0; i < NUM_ATOMS;  ++i) h_f(i)         = {1.0f, 0.5f, 0.25f};
    for (int i = 0; i < NUM_SHIFT;  ++i) h_shiftVec(i)  = {1.0f, 0.5f, 0.25f};
    for (int i = 0; i < NUM_SHIFT;  ++i) h_fShift(i)    = {1.0f, 0.5f, 0.25f};
    for (int i = 0; i < NUM_CJ4;    ++i) h_cj4(i)       = make_cj4(i);
    for (int i = 0; i < NUM_SCI;    ++i) h_sci(i)        = make_sci(i);
    for (int i = 0; i < NUM_EXCL;   ++i) h_excl(i)       = make_excl();
    for (int i = 0; i < NUM_ATOMS;  ++i) h_atomTypes(i)  = (i % 2);
    for (int i = 0; i < NUM_NBFP;   ++i) h_nbfp(i)       = {0.5f, 0.25f};

    Kokkos::deep_copy(a_xq,       h_xq);
    Kokkos::deep_copy(a_f_base,   h_f);
    Kokkos::deep_copy(shiftVec,   h_shiftVec);
    Kokkos::deep_copy(fShift_base,h_fShift);
    Kokkos::deep_copy(cj4,        h_cj4);
    Kokkos::deep_copy(sci,        h_sci);
    Kokkos::deep_copy(excl,       h_excl);
    Kokkos::deep_copy(atomTypes,  h_atomTypes);
    Kokkos::deep_copy(nbfp,       h_nbfp);

    // Mutable output views (reset each run)
    Kokkos::View<Float3*> f_out    ("f_out",     NUM_ATOMS);
    Kokkos::View<Float3*> fShift_out("fShift_out",NUM_SHIFT);

    const float rCoulombSq = 1.0f;
    const float ewaldBeta  = 3.12341f;
    const float epsFac     = 138.935f;

    // ---- Warm-up ----
    Kokkos::deep_copy(f_out,     a_f_base);
    Kokkos::deep_copy(fShift_out,fShift_base);
    runNbnxmKernel(a_xq, a_f_base, shiftVec, fShift_base,
                   cj4, sci, excl, atomTypes, nbfp,
                   NUM_TYPES, rCoulombSq, ewaldBeta, epsFac, false,
                   f_out, fShift_out);
    Kokkos::fence();

    // ---- Benchmark (w/o shift) ----
    Kokkos::deep_copy(f_out,     a_f_base);
    Kokkos::deep_copy(fShift_out,fShift_base);

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
      runNbnxmKernel(a_xq, a_f_base, shiftVec, fShift_base,
                     cj4, sci, excl, atomTypes, nbfp,
                     NUM_TYPES, rCoulombSq, ewaldBeta, epsFac, false,
                     f_out, fShift_out);
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel execution time (w/o shift): %f (us)\n", (ns * 1e-3) / repeat);

#ifdef DEBUG
    {
      auto hf = Kokkos::create_mirror_view(f_out);
      Kokkos::deep_copy(hf, f_out);
      float s0 = 0, s1 = 0, s2 = 0;
      for (int i = 0; i < NUM_ATOMS; ++i) { s0 += hf(i).x; s1 += hf(i).y; s2 += hf(i).z; }
      printf("Checksum (f_out): %f %f %f\n", s0, s1, s2);
    }
#endif

    // ---- Benchmark (w/ shift) ----
    Kokkos::deep_copy(f_out,     a_f_base);
    Kokkos::deep_copy(fShift_out,fShift_base);

    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
      runNbnxmKernel(a_xq, a_f_base, shiftVec, fShift_base,
                     cj4, sci, excl, atomTypes, nbfp,
                     NUM_TYPES, rCoulombSq, ewaldBeta, epsFac, true,
                     f_out, fShift_out);
    }
    Kokkos::fence();
    t1 = std::chrono::steady_clock::now();
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel execution time (w/ shift): %f (us)\n", (ns * 1e-3) / repeat);

#ifdef DEBUG
    {
      auto hf = Kokkos::create_mirror_view(f_out);
      Kokkos::deep_copy(hf, f_out);
      float s0 = 0, s1 = 0, s2 = 0;
      for (int i = 0; i < NUM_ATOMS; ++i) { s0 += hf(i).x; s1 += hf(i).y; s2 += hf(i).z; }
      printf("Checksum (f_out): %f %f %f\n", s0, s1, s2);
      auto hfs = Kokkos::create_mirror_view(fShift_out);
      Kokkos::deep_copy(hfs, fShift_out);
      s0 = 0; s1 = 0; s2 = 0;
      for (int i = 0; i < NUM_SHIFT; ++i) { s0 += hfs(i).x; s1 += hfs(i).y; s2 += hfs(i).z; }
      printf("Checksum (fShift_out): %f %f %f\n", s0, s1, s2);
    }
#endif
  }
  Kokkos::finalize();
  return 0;
}
