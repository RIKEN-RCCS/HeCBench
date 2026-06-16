#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include "bude.h"

typedef std::chrono::high_resolution_clock::time_point TimePoint;

struct Params {
  size_t natlig;
  size_t natpro;
  size_t ntypes;
  size_t nposes;

  std::vector<Atom>    protein;
  std::vector<Atom>    ligand;
  std::vector<FFParams> forcefield;
  std::array<std::vector<float>, 6> poses;

  size_t iterations;
  size_t wgSize;
  std::string deckDir;

  friend std::ostream &operator<<(std::ostream &os, const Params &p) {
    os << "natlig:      " << p.natlig   << "\n"
       << "natpro:      " << p.natpro   << "\n"
       << "ntypes:      " << p.ntypes   << "\n"
       << "nposes:      " << p.nposes   << "\n"
       << "iterations:  " << p.iterations << "\n"
       << "posesPerWI:  " << NUM_TD_PER_THREAD << "\n"
       << "wgSize:      " << p.wgSize   << "\n";
    return os;
  }
};

// ──────────────────────────────────────────────────────────────────────────────
// Energy kernel constants
// ──────────────────────────────────────────────────────────────────────────────
#define ZERO    0.0f
#define QUARTER 0.25f
#define HALF    0.5f
#define ONE     1.0f
#define TWO     2.0f
#define FOUR    4.0f
#define CNSTNT 45.0f

#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f

// ──────────────────────────────────────────────────────────────────────────────
// Kokkos fasten_main using TeamPolicy + team scratch memory for forcefield
// ──────────────────────────────────────────────────────────────────────────────
void fasten_main(
    const size_t teams,
    const int    block,
    const size_t ntypes,
    const size_t nposes,
    const size_t natlig,
    const size_t natpro,
    const Kokkos::View<const Atom*>     &d_protein,
    const Kokkos::View<const Atom*>     &d_ligand,
    const Kokkos::View<const float*>    &d_t0,
    const Kokkos::View<const float*>    &d_t1,
    const Kokkos::View<const float*>    &d_t2,
    const Kokkos::View<const float*>    &d_t3,
    const Kokkos::View<const float*>    &d_t4,
    const Kokkos::View<const float*>    &d_t5,
    const Kokkos::View<const FFParams*> &d_forcefield,
    const Kokkos::View<float*>          &d_etotals)
{
  using TeamPolicy  = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchFF   = Kokkos::View<FFParams*,
                        Kokkos::DefaultExecutionSpace::scratch_memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const size_t scratch_size = ScratchFF::shmem_size(ntypes);

  Kokkos::parallel_for("fasten_main",
    TeamPolicy((int)teams, block)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const member_type &team) {

      ScratchFF local_forcefield(team.team_scratch(0), ntypes);

      const int lid    = team.team_rank();
      const int gid    = team.league_rank();
      const int lrange = team.team_size();

      // Cooperatively load forcefield into shared (scratch) memory
      for (int i = lid; (size_t)i < ntypes; i += lrange)
        local_forcefield[i] = d_forcefield[i];

      // Compute the pose index for this thread
      int ix = gid * lrange * NUM_TD_PER_THREAD + lid;
      ix = (ix < (int)nposes) ? ix : (int)nposes - NUM_TD_PER_THREAD;

      float etot[NUM_TD_PER_THREAD];
      float3 lpos[NUM_TD_PER_THREAD];
      float4 transform[NUM_TD_PER_THREAD][3];

      for (int i = 0; i < NUM_TD_PER_THREAD; i++) {
        const int index = ix + i * lrange;
        const float sx = sinf(d_t0[index]);
        const float cx = cosf(d_t0[index]);
        const float sy = sinf(d_t1[index]);
        const float cy = cosf(d_t1[index]);
        const float sz = sinf(d_t2[index]);
        const float cz = cosf(d_t2[index]);

        transform[i][0].x = cy*cz;
        transform[i][0].y = sx*sy*cz - cx*sz;
        transform[i][0].z = cx*sy*cz + sx*sz;
        transform[i][0].w = d_t3[index];
        transform[i][1].x = cy*sz;
        transform[i][1].y = sx*sy*sz + cx*cz;
        transform[i][1].z = cx*sy*sz - sx*cz;
        transform[i][1].w = d_t4[index];
        transform[i][2].x = -sy;
        transform[i][2].y = sx*cy;
        transform[i][2].z = cx*cy;
        transform[i][2].w = d_t5[index];
        etot[i] = ZERO;
      }

      team.team_barrier();

      // Loop over ligand atoms
      for (size_t il = 0; il < natlig; ++il) {
        const Atom l_atom = d_ligand[il];
        const FFParams l_params = local_forcefield[l_atom.type];
        const bool lhphb_ltz = l_params.hphb < ZERO;
        const bool lhphb_gtz = l_params.hphb > ZERO;
        const float4 linitpos = {l_atom.x, l_atom.y, l_atom.z, ONE};

        for (int i = 0; i < NUM_TD_PER_THREAD; i++) {
          lpos[i].x = transform[i][0].w
                    + linitpos.x*transform[i][0].x
                    + linitpos.y*transform[i][0].y
                    + linitpos.z*transform[i][0].z;
          lpos[i].y = transform[i][1].w
                    + linitpos.x*transform[i][1].x
                    + linitpos.y*transform[i][1].y
                    + linitpos.z*transform[i][1].z;
          lpos[i].z = transform[i][2].w
                    + linitpos.x*transform[i][2].x
                    + linitpos.y*transform[i][2].y
                    + linitpos.z*transform[i][2].z;
        }

        for (size_t ip = 0; ip < natpro; ++ip) {
          const Atom p_atom    = d_protein[ip];
          const FFParams p_params = local_forcefield[p_atom.type];

          const float radij    = p_params.radius + l_params.radius;
          const float r_radij  = 1.f / radij;
          const float elcdst   = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
          const float elcdst1  = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
          const bool  type_E   = (p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E);

          const bool phphb_ltz = p_params.hphb < ZERO;
          const bool phphb_gtz = p_params.hphb > ZERO;
          const bool phphb_nz  = p_params.hphb != ZERO;
          const float p_hphb   = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
          const float l_hphb   = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
          const float distdslv = phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST)
                                           : (lhphb_ltz ? NPPDIST  : -FLT_MAX);
          const float r_distdslv = 1.f / distdslv;
          const float chrg_init  = l_params.elsc * p_params.elsc;
          const float dslv_init  = p_hphb + l_hphb;

          for (int i = 0; i < NUM_TD_PER_THREAD; i++) {
            const float x = lpos[i].x - p_atom.x;
            const float y = lpos[i].y - p_atom.y;
            const float z = lpos[i].z - p_atom.z;
            const float distij  = sqrtf(x*x + y*y + z*z);
            const float distbb  = distij - radij;
            const bool  zone1   = (distbb < ZERO);

            etot[i] += (ONE - (distij * r_radij)) * (zone1 ? 2*HARDNESS : ZERO);

            float chrg_e = chrg_init
              * ((zone1 ? 1 : (ONE - distbb*elcdst1)) * (distbb < elcdst ? 1 : ZERO));
            const float neg_chrg_e = -fabsf(chrg_e);
            chrg_e = type_E ? neg_chrg_e : chrg_e;
            etot[i] += chrg_e * CNSTNT;

            const float coeff  = ONE - (distbb * r_distdslv);
            float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : ZERO);
            dslv_e *= (zone1 ? 1 : coeff);
            etot[i] += dslv_e;
          }
        } // protein atoms
      } // ligand atoms

      const int td_base = gid * lrange * NUM_TD_PER_THREAD + lid;
      if (td_base < (int)nposes) {
        for (int i = 0; i < NUM_TD_PER_THREAD; i++)
          d_etotals[td_base + i * lrange] = etot[i] * HALF;
      }
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────
double elapsedMillis(const TimePoint &start, const TimePoint &end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) * 1e-6;
}

void printTimings(const Params &params, double millis) {
  double ms      = millis / params.iterations;
  double runtime = ms * 1e-3;
  double ops_per_wg = NUM_TD_PER_THREAD * 27 + params.natlig
      * (3 + NUM_TD_PER_THREAD*18 + params.natpro*(11 + NUM_TD_PER_THREAD*30))
      + NUM_TD_PER_THREAD;
  double total_ops = ops_per_wg * ((double)params.nposes / NUM_TD_PER_THREAD);
  double gflops    = (total_ops / runtime) / 1e9;
  double interactions = (double)params.nposes * params.natlig * params.natpro;
  std::cout.precision(3);
  std::cout << std::fixed;
  std::cout << "- Total kernel time:    " << millis << " ms\n";
  std::cout << "- Average kernel time:   " << ms     << " ms\n";
  std::cout << "- Interactions/s: " << (interactions / runtime / 1e9) << " billion\n";
  std::cout << "- GFLOP/s:        " << gflops << "\n";
}

template<typename T>
std::vector<T> readNStruct(const std::string &path) {
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) throw std::invalid_argument("Bad file: " + path);
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount(); s.clear(); s.seekg(0, std::ios::beg);
  std::vector<T> xs(len / sizeof(T));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();
  return xs;
}

Params loadParameters(const std::vector<std::string> &args) {
  Params params = {};
  params.iterations = DEFAULT_ITERS;
  params.nposes     = DEFAULT_NPOSES;
  params.wgSize     = DEFAULT_WGSIZE;
  params.deckDir    = DATA_DIR;

  const auto readParam = [&args](size_t &current,
      const std::string &arg,
      const std::initializer_list<std::string> &matches,
      const std::function<void(std::string)> &handle) {
    if (!matches.size()) return false;
    if (std::find(matches.begin(), matches.end(), arg) != matches.end()) {
      if (current + 1 < args.size()) { current++; handle(args[current]); }
      else {
        std::cerr << "option requires a value\n"; std::exit(EXIT_FAILURE);
      }
      return true;
    }
    return false;
  };

  const auto bindInt = [](const std::string &param, size_t &dest, const std::string &name) {
    try {
      auto v = std::stol(param);
      if (v < 0) { std::cerr << "positive integer required for " << name << "\n"; std::exit(EXIT_FAILURE); }
      dest = v;
    } catch (...) { std::cerr << "bad integer for " << name << "\n"; std::exit(EXIT_FAILURE); }
  };

  for (size_t i = 0; i < args.size(); ++i) {
    using namespace std::placeholders;
    const auto arg = args[i];
    if (readParam(i, arg, {"--iterations","-i"}, std::bind(bindInt,_1,std::ref(params.iterations),"iterations"))) continue;
    if (readParam(i, arg, {"--numposes",  "-n"}, std::bind(bindInt,_1,std::ref(params.nposes),    "numposes"  ))) continue;
    if (readParam(i, arg, {"--wgsize",    "-w"}, std::bind(bindInt,_1,std::ref(params.wgSize),    "wgsize"    ))) continue;
    if (readParam(i, arg, {"--deck"}, [&](const std::string &p){ params.deckDir = p; })) continue;
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: ./main [--deck DIR] [--iterations I] [--numposes N] [--wgsize W]\n";
      std::exit(EXIT_SUCCESS);
    }
    std::cout << "Unrecognized argument '" << arg << "'\n"; std::exit(EXIT_FAILURE);
  }

  params.ligand     = readNStruct<Atom>    (params.deckDir + FILE_LIGAND);
  params.natlig     = params.ligand.size();
  params.protein    = readNStruct<Atom>    (params.deckDir + FILE_PROTEIN);
  params.natpro     = params.protein.size();
  params.forcefield = readNStruct<FFParams>(params.deckDir + FILE_FORCEFIELD);
  params.ntypes     = params.forcefield.size();

  auto poses = readNStruct<float>(params.deckDir + FILE_POSES);
  if (poses.size() / 6 != params.nposes)
    throw std::invalid_argument("Bad poses: " + std::to_string(poses.size()));

  for (size_t i = 0; i < 6; ++i) {
    params.poses[i].resize(params.nposes);
    std::copy(std::next(poses.cbegin(), (long)(i * params.nposes)),
              std::next(poses.cbegin(), (long)(i * params.nposes + params.nposes)),
              params.poses[i].begin());
  }
  return params;
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: wrap a host std::vector in an unmanaged View, deep_copy to device View
// ──────────────────────────────────────────────────────────────────────────────
template<typename T>
Kokkos::View<T*> toDevice(const std::vector<T> &v, const std::string &name) {
  Kokkos::View<T*> d(name, v.size());
  auto h = Kokkos::create_mirror_view(d);
  std::copy(v.begin(), v.end(), h.data());
  Kokkos::deep_copy(d, h);
  return d;
}

// ──────────────────────────────────────────────────────────────────────────────
std::vector<float> runKernel(Params &params) {
  std::vector<float> energies(params.nposes, 0.f);

  // Upload all static data once
  auto d_protein    = toDevice<Atom>    (params.protein,    "protein");
  auto d_ligand     = toDevice<Atom>    (params.ligand,     "ligand");
  auto d_t0         = toDevice<float>   (params.poses[0],   "t0");
  auto d_t1         = toDevice<float>   (params.poses[1],   "t1");
  auto d_t2         = toDevice<float>   (params.poses[2],   "t2");
  auto d_t3         = toDevice<float>   (params.poses[3],   "t3");
  auto d_t4         = toDevice<float>   (params.poses[4],   "t4");
  auto d_t5         = toDevice<float>   (params.poses[5],   "t5");
  auto d_forcefield = toDevice<FFParams>(params.forcefield, "forcefield");

  Kokkos::View<float*> d_results("results", params.nposes);

  const size_t global = (params.nposes + NUM_TD_PER_THREAD - 1) / NUM_TD_PER_THREAD;
  const size_t teams  = (global + params.wgSize - 1) / params.wgSize;
  const int    block  = (int)params.wgSize;

  // Const views for the kernel
  Kokkos::View<const Atom*>     d_prot_c  = d_protein;
  Kokkos::View<const Atom*>     d_lig_c   = d_ligand;
  Kokkos::View<const float*>    d_t0_c    = d_t0;
  Kokkos::View<const float*>    d_t1_c    = d_t1;
  Kokkos::View<const float*>    d_t2_c    = d_t2;
  Kokkos::View<const float*>    d_t3_c    = d_t3;
  Kokkos::View<const float*>    d_t4_c    = d_t4;
  Kokkos::View<const float*>    d_t5_c    = d_t5;
  Kokkos::View<const FFParams*> d_ff_c    = d_forcefield;

  // Warm-up pass
  fasten_main(teams, block, params.ntypes, params.nposes, params.natlig, params.natpro,
              d_prot_c, d_lig_c, d_t0_c, d_t1_c, d_t2_c, d_t3_c, d_t4_c, d_t5_c,
              d_ff_c, d_results);
  Kokkos::fence();

  auto kernelStart = std::chrono::high_resolution_clock::now();
  for (size_t iter = 0; iter < params.iterations; ++iter) {
    fasten_main(teams, block, params.ntypes, params.nposes, params.natlig, params.natpro,
                d_prot_c, d_lig_c, d_t0_c, d_t1_c, d_t2_c, d_t3_c, d_t4_c, d_t5_c,
                d_ff_c, d_results);
  }
  Kokkos::fence();
  auto kernelEnd = std::chrono::high_resolution_clock::now();

  printTimings(params, elapsedMillis(kernelStart, kernelEnd));

  // Copy results to host
  auto h_results = Kokkos::create_mirror_view(d_results);
  Kokkos::deep_copy(h_results, d_results);
  std::copy(h_results.data(), h_results.data() + params.nposes, energies.begin());

  return energies;
}

// ──────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto args   = std::vector<std::string>(argv + 1, argv + argc);
    auto params = loadParameters(args);

    std::cout << "Poses     : " << params.nposes     << "\n"
              << "Iterations: " << params.iterations << "\n"
              << "Ligands   : " << params.natlig      << "\n"
              << "Proteins  : " << params.natpro      << "\n"
              << "Deck      : " << params.deckDir     << "\n"
              << "Types     : " << params.ntypes      << "\n"
              << "WG        : " << params.wgSize       << "\n";

    auto energies = runKernel(params);

    // Validate
    std::ifstream refEnergies(params.deckDir + FILE_REF_ENERGIES);
    size_t nRefPoses = params.nposes;
    if (params.nposes > REF_NPOSES) {
      std::cout << "Only validating the first " << REF_NPOSES << " poses.\n";
      nRefPoses = REF_NPOSES;
    }
    std::string line;
    float maxdiff = 0.0f;
    for (size_t i = 0; i < nRefPoses; i++) {
      if (!std::getline(refEnergies, line))
        throw std::logic_error("ran out of ref energies");
      float e = std::stof(line);
      if (std::fabs(e) < 1.f && std::fabs(energies[i]) < 1.f) continue;
      float diff = std::fabs(e - energies[i]) / e;
      if (diff > maxdiff) maxdiff = diff;
    }
    std::cout << "Largest difference was " << std::setprecision(3)
              << (100 * maxdiff) << "%.\n\n";
    refEnergies.close();
  }
  Kokkos::finalize();
  return 0;
}
