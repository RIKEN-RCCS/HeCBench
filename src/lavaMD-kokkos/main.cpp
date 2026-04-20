// LavaMD – Molecular Dynamics
// Kokkos port from the OpenMP target version.
// Usage: ./main -boxes1d <n>   (default n=4)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>

// ---------- types & constants ----------
typedef float fp;

#define NUMBER_PAR_PER_BOX 100
#define NUMBER_THREADS     128

struct FOUR_VECTOR { fp v, x, y, z; };
struct THREE_VECTOR { fp x, y, z; };

struct nei_str {
  int x, y, z, number;
  long offset;
};

struct box_str {
  int x, y, z, number;
  long offset;
  int nn;
  nei_str nei[26];
};

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))

// ---------- main ----------
int main(int argc, char* argv[])
{
  int boxes1d = 4;  // default

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "-boxes1d") == 0) {
      boxes1d = atoi(argv[i+1]);
      if (boxes1d <= 0) { fprintf(stderr, "boxes1d must be > 0\n"); return 1; }
    }
  }

  printf("WG size = %d\n", NUMBER_THREADS);
  printf("boxes1d = %d\n", boxes1d);

  long number_boxes = (long)boxes1d * boxes1d * boxes1d;
  long space_elem   = number_boxes * NUMBER_PAR_PER_BOX;

  // Build box array on host
  box_str* box_h = (box_str*)malloc(number_boxes * sizeof(box_str));
  {
    int nh = 0;
    for (int i = 0; i < boxes1d; i++)
    for (int j = 0; j < boxes1d; j++)
    for (int k = 0; k < boxes1d; k++) {
      box_h[nh].x = k; box_h[nh].y = j; box_h[nh].z = i;
      box_h[nh].number = nh;
      box_h[nh].offset = (long)nh * NUMBER_PAR_PER_BOX;
      box_h[nh].nn = 0;
      for (int l = -1; l < 2; l++)
      for (int m = -1; m < 2; m++)
      for (int n = -1; n < 2; n++) {
        if ((i+l>=0 && j+m>=0 && k+n>=0) &&
            (i+l<boxes1d && j+m<boxes1d && k+n<boxes1d) &&
            !(l==0 && m==0 && n==0)) {
          nei_str& nb = box_h[nh].nei[box_h[nh].nn];
          nb.x = k+n; nb.y = j+m; nb.z = i+l;
          nb.number = (i+l)*boxes1d*boxes1d + (j+m)*boxes1d + (k+n);
          nb.offset = (long)nb.number * NUMBER_PAR_PER_BOX;
          box_h[nh].nn++;
        }
      }
      nh++;
    }
  }

  // Particle data
  FOUR_VECTOR* rv_h = (FOUR_VECTOR*)malloc(space_elem * sizeof(FOUR_VECTOR));
  fp*          qv_h = (fp*)         malloc(space_elem * sizeof(fp));
  FOUR_VECTOR* fv_h = (FOUR_VECTOR*)malloc(space_elem * sizeof(FOUR_VECTOR));

  srand(2);
  for (long i = 0; i < space_elem; i++) {
    rv_h[i].v = (fp)((rand()%10)+1) / 10.f;
    rv_h[i].x = (fp)((rand()%10)+1) / 10.f;
    rv_h[i].y = (fp)((rand()%10)+1) / 10.f;
    rv_h[i].z = (fp)((rand()%10)+1) / 10.f;
    qv_h[i]   = (fp)((rand()%10)+1) / 10.f;
    fv_h[i]   = {0.f, 0.f, 0.f, 0.f};
  }

  const fp alpha = 0.5f;

  Kokkos::initialize(argc, argv);
  {
    using DevView4 = Kokkos::View<FOUR_VECTOR*>;
    using DevViewF = Kokkos::View<fp*>;
    using DevViewB = Kokkos::View<box_str*>;

    DevViewB d_box("box", number_boxes);
    DevView4 d_rv ("rv",  space_elem);
    DevViewF d_qv ("qv",  space_elem);
    DevView4 d_fv ("fv",  space_elem);

    // Copy to device
    {
      auto hb = Kokkos::create_mirror_view(d_box);
      auto hr = Kokkos::create_mirror_view(d_rv);
      auto hq = Kokkos::create_mirror_view(d_qv);
      auto hf = Kokkos::create_mirror_view(d_fv);
      for (long i = 0; i < number_boxes; i++) hb(i) = box_h[i];
      for (long i = 0; i < space_elem;   i++) {
        hr(i) = rv_h[i];
        hq(i) = qv_h[i];
        hf(i) = fv_h[i];
      }
      Kokkos::deep_copy(d_box, hb);
      Kokkos::deep_copy(d_rv,  hr);
      Kokkos::deep_copy(d_qv,  hq);
      Kokkos::deep_copy(d_fv,  hf);
    }

    // Scratch memory types (level 0)
    using ScrView4 = Kokkos::View<FOUR_VECTOR*,
                       Kokkos::DefaultExecutionSpace::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
    using ScrViewF = Kokkos::View<fp*,
                       Kokkos::DefaultExecutionSpace::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;

    size_t scratch_bytes =
        ScrView4::shmem_size(NUMBER_PAR_PER_BOX) * 2 +  // rA + rB
        ScrViewF::shmem_size(NUMBER_PAR_PER_BOX);        // qB

    // Use AUTO so Kokkos picks a legal team size for the current backend.
    // NUMBER_THREADS is only used as a hint / for informational output.
    auto policy = Kokkos::TeamPolicy<>((int)number_boxes, Kokkos::AUTO)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    auto t0 = std::chrono::steady_clock::now();

    Kokkos::parallel_for("lavaMD", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        int bx = team.league_rank();

        ScrView4 rA(team.team_scratch(0), NUMBER_PAR_PER_BOX);
        ScrView4 rB(team.team_scratch(0), NUMBER_PAR_PER_BOX);
        ScrViewF qB(team.team_scratch(0), NUMBER_PAR_PER_BOX);

        fp a2 = 2.f * alpha * alpha;
        int first_i = (int)d_box(bx).offset;

        // Load home-box positions into scratch
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUMBER_PAR_PER_BOX),
          [&](int t) { rA(t) = d_rv(first_i + t); });
        team.team_barrier();

        int num_nei = 1 + d_box(bx).nn;
        for (int k = 0; k < num_nei; k++) {
          int ptr     = (k == 0) ? bx : d_box(bx).nei[k-1].number;
          int first_j = (int)d_box(ptr).offset;

          // Load neighbor-box particles
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUMBER_PAR_PER_BOX),
            [&](int t) {
              rB(t) = d_rv(first_j + t);
              qB(t) = d_qv(first_j + t);
            });
          team.team_barrier();

          // Compute pairwise interactions
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUMBER_PAR_PER_BOX),
            [&](int wtx) {
              FOUR_VECTOR ra = rA(wtx);
              fp fv_v = 0.f, fv_x = 0.f, fv_y = 0.f, fv_z = 0.f;
              for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                FOUR_VECTOR rb = rB(j);
                fp r2  = ra.v + rb.v - (ra.x*rb.x + ra.y*rb.y + ra.z*rb.z);
                fp u2  = a2 * r2;
                fp vij = expf(-u2);
                fp fs  = 2.f * vij;
                fp q   = qB(j);
                fv_v += q * vij;
                fv_x += q * fs * (ra.x - rb.x);
                fv_y += q * fs * (ra.y - rb.y);
                fv_z += q * fs * (ra.z - rb.z);
              }
              // Each wtx is unique within the team -> no race condition
              d_fv(first_i + wtx).v += fv_v;
              d_fv(first_i + wtx).x += fv_x;
              d_fv(first_i + wtx).y += fv_y;
              d_fv(first_i + wtx).z += fv_z;
            });
          team.team_barrier();
        }
      });
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() * 1e-6;
    printf("Kernel execution time: %.6f s\n", secs);

    // Copy result back
    auto hf = Kokkos::create_mirror_view(d_fv);
    Kokkos::deep_copy(hf, d_fv);
    for (long i = 0; i < space_elem; i++) fv_h[i] = hf(i);
  }
  Kokkos::finalize();

#ifdef DEBUG
  int offset = 395;
  for (int g = 0; g < 10; g++)
    printf("g=%d %f %f %f %f\n", g,
           fv_h[offset+g].v, fv_h[offset+g].x,
           fv_h[offset+g].y, fv_h[offset+g].z);
#endif

  free(box_h); free(rv_h); free(qv_h); free(fv_h);
  return 0;
}
