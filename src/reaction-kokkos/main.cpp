#include <Kokkos_Core.hpp>
#include <chrono>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <new>

using FloatView = Kokkos::View<float*>;
using TeamPolicy = Kokkos::TeamPolicy<>;
using MemberType = TeamPolicy::member_type;
using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using ScratchPad = Kokkos::View<float*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// -----------------------------------------------------------------------
// Utility: build initial concentrations (from reaction-cuda/util.h)
// -----------------------------------------------------------------------
static float uniform_dist() {
  static std::mt19937 rng(123);
  static std::uniform_real_distribution<> nd(0.0, 1.0);
  return (float)nd(rng);
}

void build_input_central_cube(
    unsigned int ncells, unsigned int mx, unsigned int my, unsigned int mz,
    float *a, float *b,
    float a0, float b0, float ca, float cb, float delta)
{
  for (unsigned int i = 0; i < ncells; i++) {
    a[i] = a0 + uniform_dist() * delta;
    b[i] = b0 + uniform_dist() * delta;
  }
  const unsigned int cbsz = 5;
  for (unsigned int z = mz/2 - cbsz; z < mz/2 + cbsz; z++)
    for (unsigned int y = my/2 - cbsz; y < my/2 + cbsz; y++)
      for (unsigned int x = mx/2 - cbsz; x < mx/2 + cbsz; x++) {
        a[z * mx * my + y * mx + x] = ca + uniform_dist() * delta;
        b[z * mx * my + y * mx + x] = cb + uniform_dist() * delta;
      }
}

void stats(const float *a, const float *b, unsigned int ncells) {
  float minA = 100.f, minB = 100.f, maxA = 0.f, maxB = 0.f;
  for (unsigned int i = 0; i < ncells; i++) {
    minA = std::min(minA, a[i]); minB = std::min(minB, b[i]);
    maxA = std::max(maxA, a[i]); maxB = std::max(maxB, b[i]);
  }
  printf("  Components A | B \n");
  printf("  Min = %12.6f | %12.6f\n", minA, minB);
  printf("  Max = %12.6f | %12.6f\n", maxA, maxB);
}

// -----------------------------------------------------------------------
// Gray-Scott reaction rate
// -----------------------------------------------------------------------
void reaction_gray_scott(
    FloatView fx, FloatView fy,
    FloatView drx, FloatView dry,
    unsigned int ncells, float d_c1, float d_c2)
{
  Kokkos::parallel_for("reaction_gs", Kokkos::RangePolicy<>(0, (int)ncells),
    KOKKOS_LAMBDA(const int i) {
      float r = fx[i] * fy[i] * fy[i];
      drx[i] = -r + d_c1 * (1.f - fx[i]);
      dry[i] = r - (d_c1 + d_c2) * fy[i];
    });
}

// -----------------------------------------------------------------------
// Second derivative in X, periodic boundary conditions
// Teams: my*mz/pencils,  threads: mx*pencils
// Scratch: pencils * (mx + 2) floats
// -----------------------------------------------------------------------
void derivative_x2_pbc(
    FloatView f, FloatView df,
    unsigned int mx, unsigned int my, unsigned int mz, unsigned int pencils)
{
  const int nteams   = (int)(my * mz / pencils);
  const int nthreads = (int)(mx * pencils);
  const int scratch_size = ScratchPad::shmem_size(pencils * (mx + 2));

  Kokkos::parallel_for("deriv_x2_pbc",
    TeamPolicy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const MemberType &team) {
      ScratchPad s_f(team.team_scratch(0), pencils * (mx + 2));

      const int offset   = 1;
      const int tr       = team.team_rank();
      const int lr       = team.league_rank();
      const int threadIdx_x = (int)(tr % mx);
      const int threadIdx_y = (int)(tr / mx);
      const int blockIdx_x  = (int)(lr % (my / pencils));
      const int blockIdx_y  = (int)(lr / (my / pencils));

      const int i  = threadIdx_x;
      const int j  = blockIdx_x * (int)pencils + threadIdx_y;
      const int k  = blockIdx_y;
      const int si = i + offset;
      const int sj = threadIdx_y;

      const int globalIdx = k * (int)(mx * my) + j * (int)mx + i;
      s_f[sj * (int)(mx + 2 * offset) + si] = f[globalIdx];

      team.team_barrier();

      if (i < offset) {
        s_f[sj * (int)(mx + 2) + si - offset] = s_f[sj * (int)(mx + 2) + si + (int)mx - offset];
        s_f[sj * (int)(mx + 2) + si + (int)mx] = s_f[sj * (int)(mx + 2) + si];
      }

      team.team_barrier();

      df[globalIdx] = s_f[sj*(int)(mx+2) + si+1]
                    - 2.f * s_f[sj*(int)(mx+2) + si]
                    + s_f[sj*(int)(mx+2) + si-1];
    });
}

// -----------------------------------------------------------------------
// Second derivative in X, zero-flux boundary conditions
// Teams: my*mz/pencils,  threads: mx*pencils
// Scratch: pencils * mx floats
// -----------------------------------------------------------------------
void derivative_x2_zeroflux(
    FloatView f, FloatView df,
    unsigned int mx, unsigned int my, unsigned int mz, unsigned int pencils)
{
  const int nteams   = (int)(my * mz / pencils);
  const int nthreads = (int)(mx * pencils);
  const int scratch_size = ScratchPad::shmem_size(pencils * mx);

  Kokkos::parallel_for("deriv_x2_zf",
    TeamPolicy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const MemberType &team) {
      ScratchPad s_f(team.team_scratch(0), pencils * (int)mx);

      const int tr = team.team_rank();
      const int lr = team.league_rank();
      const int threadIdx_x = (int)(tr % mx);
      const int threadIdx_y = (int)(tr / mx);
      const int blockIdx_x  = (int)(lr % (my / pencils));
      const int blockIdx_y  = (int)(lr / (my / pencils));

      const int i  = threadIdx_x;
      const int j  = blockIdx_x * (int)pencils + threadIdx_y;
      const int k  = blockIdx_y;
      const int sj = threadIdx_y;

      const int globalIdx = k * (int)(mx * my) + j * (int)mx + i;
      s_f[sj * (int)mx + i] = f[globalIdx];

      team.team_barrier();

      if (i == 0)
        df[globalIdx] = s_f[sj*(int)mx + i+1] - s_f[sj*(int)mx + i];
      else if (i == (int)(mx - 1))
        df[globalIdx] = s_f[sj*(int)mx + i-1] - s_f[sj*(int)mx + i];
      else
        df[globalIdx] = s_f[sj*(int)mx + i+1]
                      - 2.f * s_f[sj*(int)mx + i]
                      + s_f[sj*(int)mx + i-1];
    });
}

// -----------------------------------------------------------------------
// Second derivative in Y, periodic boundary conditions
// Teams: mx*mz/pencils,  threads: my*pencils
// Scratch: (my+2)*pencils floats
// -----------------------------------------------------------------------
void derivative_y2_pbc(
    FloatView f, FloatView df,
    unsigned int mx, unsigned int my, unsigned int mz, unsigned int pencils)
{
  const int nteams   = (int)(mx * mz / pencils);
  const int nthreads = (int)(my * pencils);
  const int scratch_size = ScratchPad::shmem_size((my + 2) * pencils);

  Kokkos::parallel_for("deriv_y2_pbc",
    TeamPolicy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const MemberType &team) {
      ScratchPad s_f(team.team_scratch(0), ((int)my + 2) * (int)pencils);

      const int offset = 1;
      const int tr = team.team_rank();
      const int lr = team.league_rank();
      const int threadIdx_x = (int)(tr % pencils);
      const int threadIdx_y = (int)(tr / pencils);
      const int blockIdx_x  = (int)(lr % (mx / pencils));
      const int blockIdx_y  = (int)(lr / (mx / pencils));

      const int i  = blockIdx_x * (int)pencils + threadIdx_x;
      const int j  = threadIdx_y;
      const int k  = blockIdx_y;
      const int si = threadIdx_x;
      const int sj = j + offset;

      const int globalIdx = k * (int)(mx * my) + j * (int)mx + i;
      s_f[sj * (int)pencils + si] = f[globalIdx];

      team.team_barrier();

      if (j < offset) {
        s_f[(sj - offset) * (int)pencils + si] = s_f[(sj + (int)my - offset) * (int)pencils + si];
        s_f[(sj + (int)my) * (int)pencils + si] = s_f[sj * (int)pencils + si];
      }

      team.team_barrier();

      df[globalIdx] = s_f[(sj+1)*(int)pencils + si]
                    - 2.f * s_f[sj*(int)pencils + si]
                    + s_f[(sj-1)*(int)pencils + si];
    });
}

// -----------------------------------------------------------------------
// Second derivative in Y, zero-flux boundary conditions
// Teams: mx*mz/pencils,  threads: my*pencils
// Scratch: my*pencils floats
// -----------------------------------------------------------------------
void derivative_y2_zeroflux(
    FloatView f, FloatView df,
    unsigned int mx, unsigned int my, unsigned int mz, unsigned int pencils)
{
  const int nteams   = (int)(mx * mz / pencils);
  const int nthreads = (int)(my * pencils);
  const int scratch_size = ScratchPad::shmem_size(my * pencils);

  Kokkos::parallel_for("deriv_y2_zf",
    TeamPolicy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const MemberType &team) {
      ScratchPad s_f(team.team_scratch(0), (int)my * (int)pencils);

      const int tr = team.team_rank();
      const int lr = team.league_rank();
      const int threadIdx_x = (int)(tr % pencils);
      const int threadIdx_y = (int)(tr / pencils);
      const int blockIdx_x  = (int)(lr % (mx / pencils));
      const int blockIdx_y  = (int)(lr / (mx / pencils));

      const int i  = blockIdx_x * (int)pencils + threadIdx_x;
      const int j  = threadIdx_y;
      const int k  = blockIdx_y;
      const int si = threadIdx_x;

      const int globalIdx = k * (int)(mx * my) + j * (int)mx + i;
      s_f[j * (int)pencils + si] = f[globalIdx];

      team.team_barrier();

      if (j == 0)
        df[globalIdx] = s_f[(j+1)*(int)pencils + si] - s_f[j*(int)pencils + si];
      else if (j == (int)(my - 1))
        df[globalIdx] = s_f[(j-1)*(int)pencils + si] - s_f[j*(int)pencils + si];
      else
        df[globalIdx] = s_f[(j+1)*(int)pencils + si]
                      - 2.f * s_f[j*(int)pencils + si]
                      + s_f[(j-1)*(int)pencils + si];
    });
}

// -----------------------------------------------------------------------
// Second derivative in Z, periodic boundary conditions
// Teams: mx*my/pencils,  threads: mz*pencils
// Scratch: (mz+2)*pencils floats
// -----------------------------------------------------------------------
void derivative_z2_pbc(
    FloatView f, FloatView df,
    unsigned int mx, unsigned int my, unsigned int mz, unsigned int pencils)
{
  const int nteams   = (int)(mx * my / pencils);
  const int nthreads = (int)(mz * pencils);
  const int scratch_size = ScratchPad::shmem_size((mz + 2) * pencils);

  Kokkos::parallel_for("deriv_z2_pbc",
    TeamPolicy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const MemberType &team) {
      ScratchPad s_f(team.team_scratch(0), ((int)mz + 2) * (int)pencils);

      const int offset = 1;
      const int tr = team.team_rank();
      const int lr = team.league_rank();
      const int threadIdx_x = (int)(tr % pencils);
      const int threadIdx_y = (int)(tr / pencils);
      const int blockIdx_x  = (int)(lr % (mx / pencils));
      const int blockIdx_y  = (int)(lr / (mx / pencils));

      const int i  = blockIdx_x * (int)pencils + threadIdx_x;
      const int j  = blockIdx_y;
      const int k  = threadIdx_y;
      const int si = threadIdx_x;
      const int sk = k + offset;

      const int globalIdx = k * (int)(mx * my) + j * (int)mx + i;
      s_f[sk * (int)pencils + si] = f[globalIdx];

      team.team_barrier();

      if (k < offset) {
        s_f[(sk - offset) * (int)pencils + si] = s_f[(sk + (int)mz - offset) * (int)pencils + si];
        s_f[(sk + (int)mz) * (int)pencils + si] = s_f[sk * (int)pencils + si];
      }

      team.team_barrier();

      df[globalIdx] = s_f[(sk+1)*(int)pencils + si]
                    - 2.f * s_f[sk*(int)pencils + si]
                    + s_f[(sk-1)*(int)pencils + si];
    });
}

// -----------------------------------------------------------------------
// Second derivative in Z, zero-flux boundary conditions
// Teams: mx*my/pencils,  threads: mz*pencils
// Scratch: mz*pencils floats
// -----------------------------------------------------------------------
void derivative_z2_zeroflux(
    FloatView f, FloatView df,
    unsigned int mx, unsigned int my, unsigned int mz, unsigned int pencils)
{
  const int nteams   = (int)(mx * my / pencils);
  const int nthreads = (int)(mz * pencils);
  const int scratch_size = ScratchPad::shmem_size(mz * pencils);

  Kokkos::parallel_for("deriv_z2_zf",
    TeamPolicy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const MemberType &team) {
      ScratchPad s_f(team.team_scratch(0), (int)mz * (int)pencils);

      const int tr = team.team_rank();
      const int lr = team.league_rank();
      const int threadIdx_x = (int)(tr % pencils);
      const int threadIdx_y = (int)(tr / pencils);
      const int blockIdx_x  = (int)(lr % (mx / pencils));
      const int blockIdx_y  = (int)(lr / (mx / pencils));

      const int i  = blockIdx_x * (int)pencils + threadIdx_x;
      const int j  = blockIdx_y;
      const int k  = threadIdx_y;
      const int si = threadIdx_x;

      const int globalIdx = k * (int)(mx * my) + j * (int)mx + i;
      s_f[k * (int)pencils + si] = f[globalIdx];

      team.team_barrier();

      if (k == 0)
        df[globalIdx] = s_f[(k+1)*(int)pencils + si] - s_f[k*(int)pencils + si];
      else if (k == (int)(mz - 1))
        df[globalIdx] = s_f[(k-1)*(int)pencils + si] - s_f[k*(int)pencils + si];
      else
        df[globalIdx] = s_f[(k+1)*(int)pencils + si]
                      - 2.f * s_f[k*(int)pencils + si]
                      + s_f[(k-1)*(int)pencils + si];
    });
}

// -----------------------------------------------------------------------
// Construct Laplacian from three second-derivative components
// -----------------------------------------------------------------------
void construct_laplacian(
    FloatView df, FloatView dfx, FloatView dfy, FloatView dfz,
    unsigned int ncells, float d_diffcon)
{
  Kokkos::parallel_for("laplacian", Kokkos::RangePolicy<>(0, (int)ncells),
    KOKKOS_LAMBDA(const int i) {
      df[i] = d_diffcon * (dfx[i] + dfy[i] + dfz[i]);
    });
}

// -----------------------------------------------------------------------
// Time-step integration update
// -----------------------------------------------------------------------
void update(
    FloatView x, FloatView y,
    FloatView ddx, FloatView ddy,
    FloatView drx, FloatView dry,
    unsigned int ncells, float d_dt)
{
  Kokkos::parallel_for("update", Kokkos::RangePolicy<>(0, (int)ncells),
    KOKKOS_LAMBDA(const int i) {
      x[i] += (ddx[i] + drx[i]) * d_dt;
      y[i] += (ddy[i] + dry[i]) * d_dt;
    });
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <timesteps>\n", argv[0]);
    return 1;
  }
  unsigned int timesteps = (unsigned int)atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    unsigned int mx = 128, my = 128, mz = 128;
    unsigned int ncells  = mx * my * mz;
    unsigned int pencils = 2;
    bool zeroflux = true;

    float Da = 0.16f, Db = 0.08f;
    float dt = 0.25f, dx = 0.5f;
    float c1 = 0.0392f, c2 = 0.0649f;

    printf("Starting time-integration\n");
    printf("Constructing initial concentrations...\n");

    float *h_a = new float[ncells];
    float *h_b = new float[ncells];
    build_input_central_cube(ncells, mx, my, mz, h_a, h_b,
                             1.0f, 0.0f, 0.5f, 0.25f, 0.05f);

    // Allocate device views
    FloatView d_a("a",   ncells), d_b("b",   ncells);
    FloatView d_dx2("dx2", ncells), d_dy2("dy2", ncells), d_dz2("dz2", ncells);
    FloatView d_ra("ra",  ncells), d_rb("rb",  ncells);
    FloatView d_da("da",  ncells), d_db("db",  ncells);

    // Upload initial concentrations
    auto hm_a = Kokkos::create_mirror_view(d_a);
    auto hm_b = Kokkos::create_mirror_view(d_b);
    for (unsigned int i = 0; i < ncells; i++) { hm_a(i) = h_a[i]; hm_b(i) = h_b[i]; }
    Kokkos::deep_copy(d_a, hm_a);
    Kokkos::deep_copy(d_b, hm_b);
    // Zero auxiliary arrays
    Kokkos::deep_copy(d_dx2, 0.f); Kokkos::deep_copy(d_dy2, 0.f);
    Kokkos::deep_copy(d_dz2, 0.f); Kokkos::deep_copy(d_ra,  0.f);
    Kokkos::deep_copy(d_rb,  0.f); Kokkos::deep_copy(d_da,  0.f);
    Kokkos::deep_copy(d_db,  0.f);

    float diffcon_a = Da / (dx * dx);
    float diffcon_b = Db / (dx * dx);

    auto start = std::chrono::system_clock::now();

    for (unsigned int t = 0; t < timesteps; t++) {
      if (zeroflux) {
        derivative_x2_zeroflux(d_a, d_dx2, mx, my, mz, pencils);
        derivative_y2_zeroflux(d_a, d_dy2, mx, my, mz, pencils);
        derivative_z2_zeroflux(d_a, d_dz2, mx, my, mz, pencils);
      } else {
        derivative_x2_pbc(d_a, d_dx2, mx, my, mz, pencils);
        derivative_y2_pbc(d_a, d_dy2, mx, my, mz, pencils);
        derivative_z2_pbc(d_a, d_dz2, mx, my, mz, pencils);
      }
      construct_laplacian(d_da, d_dx2, d_dy2, d_dz2, ncells, diffcon_a);

      if (zeroflux) {
        derivative_x2_zeroflux(d_b, d_dx2, mx, my, mz, pencils);
        derivative_y2_zeroflux(d_b, d_dy2, mx, my, mz, pencils);
        derivative_z2_zeroflux(d_b, d_dz2, mx, my, mz, pencils);
      } else {
        derivative_x2_pbc(d_b, d_dx2, mx, my, mz, pencils);
        derivative_y2_pbc(d_b, d_dy2, mx, my, mz, pencils);
        derivative_z2_pbc(d_b, d_dz2, mx, my, mz, pencils);
      }
      construct_laplacian(d_db, d_dx2, d_dy2, d_dz2, ncells, diffcon_b);

      reaction_gray_scott(d_a, d_b, d_ra, d_rb, ncells, c1, c2);
      update(d_a, d_b, d_da, d_db, d_ra, d_rb, ncells, dt);
    }
    Kokkos::fence();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("timesteps: %d\n", timesteps);
    printf("Total kernel execution time:     %12.3f s\n\n", elapsed.count());

    // Copy results back for stats
    Kokkos::deep_copy(hm_a, d_a);
    Kokkos::deep_copy(hm_b, d_b);
    for (unsigned int i = 0; i < ncells; i++) { h_a[i] = hm_a(i); h_b[i] = hm_b(i); }

    stats(h_a, h_b, ncells);

    delete[] h_a;
    delete[] h_b;
  }
  Kokkos::finalize();
  return 0;
}
