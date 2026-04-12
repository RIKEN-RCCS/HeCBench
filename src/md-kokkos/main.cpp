#include <cassert>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <list>
#include <iostream>
#include <Kokkos_Core.hpp>

// ─── Vec4 struct replaces double4 / float4 ───────────────────────────────────
struct Vec4 { double x, y, z, w; };

// ─── Problem constants ────────────────────────────────────────────────────────
static const double cutsq       = 13.5;
static const int    maxNeighbors = 128;
static const int    domainEdge   = 20;
static const double lj1         = 1.5;
static const double lj2         = 2.0;

// ─── Neighbour-list utilities (CPU, same as reference) ───────────────────────
template <class T>
static inline void insertInOrder(std::list<T>& currDist,
                                  std::list<int>& currList,
                                  int j, T distIJ, int maxN)
{
  if (distIJ > currDist.back()) return;
  typename std::list<T>::iterator   it;
  typename std::list<int>::iterator it2 = currList.begin();
  for (it = currDist.begin(); it != currDist.end(); ++it) {
    if (distIJ < *it) {
      currDist.insert(it, distIJ);
      currList.insert(it2, j);
      currList.resize(maxN);
      currDist.resize(maxN);
      return;
    }
    ++it2;
  }
}

static int buildNeighborList(int nAtom, const Vec4 *pos, int *neighborList)
{
  int totalPairs = 0;
  for (int i = 0; i < nAtom; i++) {
    std::list<int>    currList(maxNeighbors, -1);
    std::list<double> currDist(maxNeighbors, FLT_MAX);
    for (int j = 0; j < nAtom; j++) {
      if (i == j) continue;
      double dx = pos[i].x - pos[j].x;
      double dy = pos[i].y - pos[j].y;
      double dz = pos[i].z - pos[j].z;
      double d2 = dx*dx + dy*dy + dz*dz;
      insertInOrder<double>(currDist, currList, j, d2, maxNeighbors);
    }
    int idx = 0;
    auto dit = currDist.begin();
    for (auto nit = currList.begin(); nit != currList.end(); ++nit, ++dit, ++idx) {
      neighborList[idx * nAtom + i] = *nit;
      if (*dit < cutsq) totalPairs++;
    }
  }
  return totalPairs;
}

// ─── Reference (CPU) force computation for correctness check ─────────────────
static bool checkResults(const Vec4 *d_force, const Vec4 *position,
                          const int *neighborList, int nAtom)
{
  double max_error = 0.0;
  for (int i = 0; i < nAtom; i++) {
    Vec4 ipos = position[i];
    double fx = 0, fy = 0, fz = 0;
    for (int j = 0; j < maxNeighbors; j++) {
      int jidx = neighborList[j * nAtom + i];
      Vec4 jpos = position[jidx];
      double delx = ipos.x - jpos.x;
      double dely = ipos.y - jpos.y;
      double delz = ipos.z - jpos.z;
      double r2 = delx*delx + dely*dely + delz*delz;
      if (r2 > 0 && r2 < cutsq) {
        double r2inv = 1.0 / r2;
        double r6inv = r2inv * r2inv * r2inv;
        double force = r2inv * r6inv * (lj1 * r6inv - lj2);
        fx += delx * force;
        fy += dely * force;
        fz += delz * force;
      }
    }
    assert(!std::isnan(d_force[i].x));
    assert(!std::isnan(d_force[i].y));
    assert(!std::isnan(d_force[i].z));
    double ex = std::fabs(fx - d_force[i].x);
    double ey = std::fabs(fy - d_force[i].y);
    double ez = std::fabs(fz - d_force[i].z);
    if (ex > max_error) max_error = ex;
    if (ey > max_error) max_error = ey;
    if (ez > max_error) max_error = ez;
  }
  std::cout << "Max error between host and device: " << max_error << "\n";
  return true;
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <class size (0-3)> <iterations>\n";
    return 1;
  }
  int sizeClass = atoi(argv[1]);
  int iteration = atoi(argv[2]);
  const int probSizes[] = { 12288, 24576, 36864, 73728 };
  assert(sizeClass >= 0 && sizeClass < 4);
  assert(iteration >= 0);

  int nAtom = probSizes[sizeClass];

  Vec4 *position     = (Vec4 *)malloc(nAtom * sizeof(Vec4));
  Vec4 *h_force      = (Vec4 *)malloc(nAtom * sizeof(Vec4));
  int  *neighborList = (int  *)malloc(maxNeighbors * nAtom * sizeof(int));

  std::cout << "Initializing test problem (this can take several minutes for large problems).\n";
  srand(123);
  for (int i = 0; i < nAtom; i++) {
    position[i].x = rand() % domainEdge;
    position[i].y = rand() % domainEdge;
    position[i].z = rand() % domainEdge;
    position[i].w = 0.0;
  }
  std::cout << "Finished.\n";

  int totalPairs = buildNeighborList(nAtom, position, neighborList);
  std::cout << totalPairs << " of " << nAtom * maxNeighbors
            << " pairs within cutoff distance = "
            << 100.0 * (double)totalPairs / (nAtom * maxNeighbors) << " %\n";

  Kokkos::initialize(argc, argv);
  {
    // ── Device views ──────────────────────────────────────────────────────
    // Use separate x,y,z arrays (struct-of-arrays) for cleaner Kokkos code
    Kokkos::View<double*> d_px("px", nAtom);
    Kokkos::View<double*> d_py("py", nAtom);
    Kokkos::View<double*> d_pz("pz", nAtom);
    Kokkos::View<double*> d_fx("fx", nAtom);
    Kokkos::View<double*> d_fy("fy", nAtom);
    Kokkos::View<double*> d_fz("fz", nAtom);
    Kokkos::View<int*>    d_nl("nl", maxNeighbors * nAtom);

    auto h_px = Kokkos::create_mirror_view(d_px);
    auto h_py = Kokkos::create_mirror_view(d_py);
    auto h_pz = Kokkos::create_mirror_view(d_pz);
    auto h_fx = Kokkos::create_mirror_view(d_fx);
    auto h_fy = Kokkos::create_mirror_view(d_fy);
    auto h_fz = Kokkos::create_mirror_view(d_fz);
    auto h_nl = Kokkos::create_mirror_view(d_nl);

    for (int i = 0; i < nAtom; i++) {
      h_px(i) = position[i].x;
      h_py(i) = position[i].y;
      h_pz(i) = position[i].z;
    }
    for (int i = 0; i < maxNeighbors * nAtom; i++) h_nl(i) = neighborList[i];

    Kokkos::deep_copy(d_px, h_px);
    Kokkos::deep_copy(d_py, h_py);
    Kokkos::deep_copy(d_pz, h_pz);
    Kokkos::deep_copy(d_nl, h_nl);

    int nAtom_c = nAtom;
    const double lj1_c = lj1, lj2_c = lj2, cutsq_c = cutsq;

    // ── Lambda for MD force computation ───────────────────────────────────
    auto run_md = [&]() {
      Kokkos::parallel_for("md", nAtom_c, KOKKOS_LAMBDA(int i) {
        double ix = d_px(i), iy = d_py(i), iz = d_pz(i);
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int j = 0; j < maxNeighbors; j++) {
          int jidx = d_nl(j * nAtom_c + i);
          double delx = ix - d_px(jidx);
          double dely = iy - d_py(jidx);
          double delz = iz - d_pz(jidx);
          double r2 = delx*delx + dely*dely + delz*delz;
          if (r2 > 0.0 && r2 < cutsq_c) {
            double r2inv = 1.0 / r2;
            double r6inv = r2inv * r2inv * r2inv;
            double forceC = r2inv * r6inv * (lj1_c * r6inv - lj2_c);
            fx += delx * forceC;
            fy += dely * forceC;
            fz += delz * forceC;
          }
        }
        d_fx(i) = fx;
        d_fy(i) = fy;
        d_fz(i) = fz;
      });
    };

    // Warmup + correctness check
    run_md();
    Kokkos::fence();

    Kokkos::deep_copy(h_fx, d_fx);
    Kokkos::deep_copy(h_fy, d_fy);
    Kokkos::deep_copy(h_fz, d_fz);
    for (int i = 0; i < nAtom; i++) {
      h_force[i].x = h_fx(i);
      h_force[i].y = h_fy(i);
      h_force[i].z = h_fz(i);
      h_force[i].w = 0.0;
    }

    std::cout << "Performing Correctness Check (may take several minutes)\n";
    checkResults(h_force, position, neighborList, nAtom);

    // ── Timed benchmark ───────────────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iteration; it++) run_md();
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9;
    std::cout << "Average kernel execution time " << elapsed / iteration << " (s)\n";
  }
  Kokkos::finalize();

  free(position);
  free(h_force);
  free(neighborList);
  return 0;
}
