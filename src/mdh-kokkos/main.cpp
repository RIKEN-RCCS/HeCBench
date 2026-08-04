/*
 * Multiple Debye-Hückel (MDH) electrostatics kernel
 * Kokkos port from the OMP offload version.
 *
 * Original: David Gohara / John Stone (APBS / UIUC)
 */

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>

// -----------------------------------------------------------------------
// Data generation
// -----------------------------------------------------------------------
static void gendata(float *ax, float *ay, float *az,
                    float *gx, float *gy, float *gz,
                    float *charge, float *size,
                    int natom, int ngrid)
{
  printf("Generating Data..\n");
  for (int i = 0; i < natom; i++) {
    ax[i]     = (float)rand() / (float)RAND_MAX;
    ay[i]     = (float)rand() / (float)RAND_MAX;
    az[i]     = (float)rand() / (float)RAND_MAX;
    charge[i] = (float)rand() / (float)RAND_MAX;
    size[i]   = (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < ngrid; i++) {
    gx[i] = (float)rand() / (float)RAND_MAX;
    gy[i] = (float)rand() / (float)RAND_MAX;
    gz[i] = (float)rand() / (float)RAND_MAX;
  }
  printf("Done generating inputs.\n\n");
}

static void print_total(const float *arr, int ngrid)
{
  double accum = 0.0;
  for (int i = 0; i < ngrid; i++) accum += arr[i];
  printf("Accumulated value: %1.7g\n", accum);
}

static void compare(const float *arr, const float *arr2, int ngrid)
{
  bool ok = true;
  for (int i = 0; i < ngrid; i++) {
    if (fabsf(arr[i] - arr2[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

// -----------------------------------------------------------------------
// CPU reference kernel
// -----------------------------------------------------------------------
static void run_cpu_kernel(int itmax, int ngrid, int natom,
                           const float *ax, const float *ay, const float *az,
                           const float *gx, const float *gy, const float *gz,
                           const float *charge, const float *size,
                           float xkappa, float pre1, float *val)
{
  using HostExec = Kokkos::DefaultHostExecutionSpace;
  auto t0 = std::chrono::steady_clock::now();

  for (int n = 0; n < itmax; n++) {
    Kokkos::parallel_for(
      "mdh_host_reference",
      Kokkos::RangePolicy<HostExec>(0, ngrid),
      KOKKOS_LAMBDA(const int igrid) {
        float sum = 0.0f;
        for (int iatom = 0; iatom < natom; iatom++) {
          float dx = gx[igrid] - ax[iatom];
          float dy = gy[igrid] - ay[iatom];
          float dz = gz[igrid] - az[iatom];
          float dist = sqrtf(dx*dx + dy*dy + dz*dz);
          sum += pre1 * (charge[iatom] / dist) *
                 expf(-xkappa * (dist - size[iatom])) /
                 (1.0f + xkappa * size[iatom]);
        }
        val[igrid] = sum;
      });
  }
  HostExec().fence();

  auto t1 = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  printf("Average CPU kernel time: %1.12g s\n", elapsed / itmax);
}

// -----------------------------------------------------------------------
// Kokkos GPU kernel
// -----------------------------------------------------------------------
static void run_gpu_kernel(int wgsize, int itmax,
                           int ngrid, int natom, int ngadj,
                           const float *h_ax, const float *h_ay, const float *h_az,
                           const float *h_gx, const float *h_gy, const float *h_gz,
                           const float *h_charge, const float *h_size,
                           float xkappa, float pre1, float *h_val)
{
  using ExecSpace  = Kokkos::DefaultExecutionSpace;
  using MemSpace   = typename ExecSpace::memory_space;
  using ViewF1     = Kokkos::View<float*, MemSpace>;
  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using Member     = TeamPolicy::member_type;

  // Device views
  ViewF1 d_ax    ("ax",     natom);
  ViewF1 d_ay    ("ay",     natom);
  ViewF1 d_az    ("az",     natom);
  ViewF1 d_charge("charge", natom);
  ViewF1 d_size  ("size",   natom);
  ViewF1 d_gx    ("gx",     ngadj);
  ViewF1 d_gy    ("gy",     ngadj);
  ViewF1 d_gz    ("gz",     ngadj);
  ViewF1 d_val   ("val",    ngadj);

  // Host mirrors
  auto h_ax_v     = Kokkos::create_mirror_view(d_ax);
  auto h_ay_v     = Kokkos::create_mirror_view(d_ay);
  auto h_az_v     = Kokkos::create_mirror_view(d_az);
  auto h_charge_v = Kokkos::create_mirror_view(d_charge);
  auto h_size_v   = Kokkos::create_mirror_view(d_size);
  auto h_gx_v     = Kokkos::create_mirror_view(d_gx);
  auto h_gy_v     = Kokkos::create_mirror_view(d_gy);
  auto h_gz_v     = Kokkos::create_mirror_view(d_gz);

  for (int i = 0; i < natom; i++) {
    h_ax_v(i) = h_ax[i]; h_ay_v(i) = h_ay[i]; h_az_v(i) = h_az[i];
    h_charge_v(i) = h_charge[i]; h_size_v(i) = h_size[i];
  }
  for (int i = 0; i < ngadj; i++) {
    h_gx_v(i) = h_gx[i]; h_gy_v(i) = h_gy[i]; h_gz_v(i) = h_gz[i];
  }

  Kokkos::deep_copy(d_ax, h_ax_v);
  Kokkos::deep_copy(d_ay, h_ay_v);
  Kokkos::deep_copy(d_az, h_az_v);
  Kokkos::deep_copy(d_charge, h_charge_v);
  Kokkos::deep_copy(d_size, h_size_v);
  Kokkos::deep_copy(d_gx, h_gx_v);
  Kokkos::deep_copy(d_gy, h_gy_v);
  Kokkos::deep_copy(d_gz, h_gz_v);

  Kokkos::fence();
  auto t0 = std::chrono::steady_clock::now();

  TeamPolicy policy = (wgsize > 0) ? TeamPolicy(ngrid, wgsize)
                                   : TeamPolicy(ngrid, Kokkos::AUTO());

  for (int n = 0; n < itmax; n++) {
    Kokkos::parallel_for(
      "mdh",
      policy,
      KOKKOS_LAMBDA(const Member& team) {
        const int igrid = team.league_rank();
        float sum = 0.0f;
        const float l_gx = d_gx(igrid);
        const float l_gy = d_gy(igrid);
        const float l_gz = d_gz(igrid);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, natom), [&](const int iatom, float& local_sum) {
          float dx = l_gx - d_ax(iatom);
          float dy = l_gy - d_ay(iatom);
          float dz = l_gz - d_az(iatom);
          float dist = sqrtf(dx*dx + dy*dy + dz*dz);
          local_sum += pre1 * (d_charge(iatom) / dist) *
                 expf(-xkappa * (dist - d_size(iatom))) /
                 (1.0f + xkappa * d_size(iatom));
        }, sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          d_val(igrid) = sum;
        });
      });
  }

  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  printf("Average kernel time on the device: %1.12g s\n", elapsed / itmax);

  // Copy result back
  auto h_val_v = Kokkos::create_mirror_view(d_val);
  Kokkos::deep_copy(h_val_v, d_val);
  for (int i = 0; i < ngrid; i++) h_val[i] = h_val_v(i);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
static void usage() {
  printf("Optional flags:\n");
  printf("  -itmax N    loop test N times (default 100)\n");
  printf("  -wgsize N   workgroup size hint (default 256)\n");
}

int main(int argc, const char **argv)
{
  int itmax  = 100;
  int wgsize = 256;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-itmax")  && i+1 < argc) { itmax  = atoi(argv[++i]); continue; }
    if (!strcmp(argv[i], "-wgsize") && i+1 < argc) { wgsize = atoi(argv[++i]); continue; }
    if (!strcmp(argv[i], "-h")) { usage(); return 0; }
  }
  printf("Run parameters:\n  kernel loop count: %d\n  workgroup size:    %d\n", itmax, wgsize);

  const int natom = 5877;
  const int ngrid = 134918;
  const int ngadj = ngrid + (512 - (ngrid & 511));

  const float pre1   = 4.46184985145e19f;
  const float xkappa = 0.0735516324639f;

  float *ax     = (float*)calloc(natom, sizeof(float));
  float *ay     = (float*)calloc(natom, sizeof(float));
  float *az     = (float*)calloc(natom, sizeof(float));
  float *charge = (float*)calloc(natom, sizeof(float));
  float *size   = (float*)calloc(natom, sizeof(float));
  float *gx     = (float*)calloc(ngadj, sizeof(float));
  float *gy     = (float*)calloc(ngadj, sizeof(float));
  float *gz     = (float*)calloc(ngadj, sizeof(float));
  float *val_cpu = (float*)calloc(ngadj, sizeof(float));
  float *val_gpu = (float*)calloc(ngadj, sizeof(float));

  gendata(ax, ay, az, gx, gy, gz, charge, size, natom, ngrid);

  Kokkos::initialize(argc, const_cast<char**>(argv));
  {
    // One host pass is enough for validation: repeating this deterministic
    // kernel only overwrites the same output values.
    run_cpu_kernel(1, ngrid, natom, ax, ay, az, gx, gy, gz, charge, size,
                   xkappa, pre1, val_cpu);
    print_total(val_cpu, ngrid);

    // Device kernel
    run_gpu_kernel(wgsize, itmax, ngrid, natom, ngadj,
                   ax, ay, az, gx, gy, gz, charge, size,
                   xkappa, pre1, val_gpu);
    print_total(val_gpu, ngrid);
    compare(val_cpu, val_gpu, ngrid);
  }
  Kokkos::finalize();

  free(ax); free(ay); free(az);
  free(charge); free(size);
  free(gx); free(gy); free(gz);
  free(val_cpu); free(val_gpu);
  return 0;
}
