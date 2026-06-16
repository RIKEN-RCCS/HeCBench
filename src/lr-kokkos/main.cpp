#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <chrono>
#include <vector>

// ---- Constants ------------------------------------------------------------
#define TEMP_SIZE           96453
#define TEMP_WORKGROUP_SIZE 63
#define TEMP_WORKGROUP_NBR  (TEMP_SIZE / TEMP_WORKGROUP_SIZE)
#define TEMP_FILENAME       "assets/temperature.txt"
#define RESULT_FILENAME     "assets/_results.txt"

// ---- Types ----------------------------------------------------------------
struct alignas(8) float2  { float x, y; };
struct alignas(16) float4 { float x, y, z, w; };

typedef float2 data_t;
typedef float4 sum_t;
typedef float2 rsquared_t;

struct result_t {
  float  a0, a1;
  int    rsquared;
  double ktime;
};

struct results_t {
  result_t iterative;
  result_t parallelized;
};

struct linear_param_t {
  int    repeat;
  char  *filename;
  size_t size;
  size_t wg_size;
  size_t wg_count;
};

#define PRINT_RESULT(title, result) \
  printf("\t%s\n\t--------\n\t| Equation: y = %.3fx + %.3f\n\n", \
    title, result.a1, result.a0);

#define WRITE_RESULT(file, result) \
  fprintf(file, "%.3f   %.3f   %.3f   %d\n", \
    result.a0, result.a1, result.ktime, result.rsquared);

// ---- File reading ---------------------------------------------------------
static void create_dataset(linear_param_t *params, data_t *dataset) {
  FILE *fp = fopen(params->filename, "r");
  if (!fp) { perror("Failed to load dataset file"); exit(1); }
  char buf[1024];
  for (size_t i = 0; i < params->size && fgets(buf, 1024, fp); i++) {
    char *token = strtok(buf, "\t");
    dataset[i].x = atof(token);
    token = strtok(NULL, "\t");
    dataset[i].y = atof(token);
  }
  fclose(fp);
}

// ---- Iterative (CPU) regression -------------------------------------------
static void iterative_r_squared(linear_param_t *params, data_t *dataset,
                                 sum_t *sumset, result_t *response) {
  float mean = sumset->y / params->size;
  float dist_x = 0.f, dist_y = 0.f;
  for (int i = 0; i < (int)params->size; i++) {
    dist_x += powf(dataset[i].y - mean, 2.f);
    float ye = dataset[i].x * response->a1 + response->a0;
    dist_y += powf(ye - mean, 2.f);
  }
  response->rsquared = (int)(dist_y / dist_x * 100);
}

void iterative_regression(linear_param_t *params, data_t *dataset, result_t *response) {
  sum_t s = {0, 0, 0, 0};
  for (int i = 0; i < (int)params->size; i++) {
    s.x += dataset[i].x;
    s.y += dataset[i].y;
    s.z += dataset[i].x * dataset[i].y;
    s.w += powf(dataset[i].x, 2.f);
  }
  double det = params->size * s.w - (double)(s.x * s.x);
  response->a0 = (float)((s.y * s.w - s.x * s.z) / det);
  response->a1 = (float)((params->size * s.z - s.x * s.y) / det);
  iterative_r_squared(params, dataset, &s, response);
}

// ---- Kokkos team-based parallel reduction ---------------------------------

// linear_regression kernel: each team reduces TEMP_WORKGROUP_SIZE data points
// into sum_x, sum_y, sum_xy, sum_x2
static void linear_regression_kokkos(
    int nTeams, Kokkos::View<data_t *> d_dataset,
    Kokkos::View<sum_t *> d_results) {

  const int wg_size = TEMP_WORKGROUP_SIZE;
  // Scratch: wg_size sum_t values
  const int scratch_bytes = wg_size * (int)sizeof(sum_t);

  using TeamPolicy  = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView  = Kokkos::View<sum_t *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(nTeams, wg_size, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  Kokkos::parallel_for(
      "linear_regression", policy,
      KOKKOS_LAMBDA(const member_type &team) {
        int blk_id  = team.league_rank();
        int loc_id  = team.team_rank();
        int glob_id = blk_id * wg_size + loc_id;

        ScratchView interns(team.team_scratch(0), wg_size);

        interns(loc_id).x = d_dataset(glob_id).x;
        interns(loc_id).y = d_dataset(glob_id).y;
        interns(loc_id).z = d_dataset(glob_id).x * d_dataset(glob_id).y;
        interns(loc_id).w = d_dataset(glob_id).x * d_dataset(glob_id).x;
        team.team_barrier();

        for (int i = wg_size / 2, old_i = wg_size; i > 0; old_i = i, i /= 2) {
          if (loc_id < i) {
            interns(loc_id).x += interns(loc_id + i).x;
            interns(loc_id).y += interns(loc_id + i).y;
            interns(loc_id).z += interns(loc_id + i).z;
            interns(loc_id).w += interns(loc_id + i).w;
            if (loc_id == (i - 1) && old_i % 2 != 0) {
              interns(loc_id).x += interns(old_i - 1).x;
              interns(loc_id).y += interns(old_i - 1).y;
              interns(loc_id).z += interns(old_i - 1).z;
              interns(loc_id).w += interns(old_i - 1).w;
            }
          }
          team.team_barrier();
        }
        if (loc_id == 0) d_results(blk_id) = interns(0);
      });
  Kokkos::fence();
}

// rsquared kernel: each team computes actual vs estimated variance
static void rsquared_kokkos(
    int nTeams, Kokkos::View<data_t *> d_dataset,
    float mean, float2 equation,
    Kokkos::View<rsquared_t *> d_results) {

  const int wg_size = TEMP_WORKGROUP_SIZE;
  const int scratch_bytes = wg_size * (int)sizeof(rsquared_t);

  using TeamPolicy  = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView  = Kokkos::View<rsquared_t *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(nTeams, wg_size, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  Kokkos::parallel_for(
      "rsquared", policy,
      KOKKOS_LAMBDA(const member_type &team) {
        int blk_id  = team.league_rank();
        int loc_id  = team.team_rank();
        int glob_id = blk_id * wg_size + loc_id;

        ScratchView dist(team.team_scratch(0), wg_size);

        dist(loc_id).x = powf(d_dataset(glob_id).y - mean, 2.f);
        float ye = d_dataset(glob_id).x * equation.y + equation.x;
        dist(loc_id).y = powf(ye - mean, 2.f);
        team.team_barrier();

        for (int i = wg_size / 2, old_i = wg_size; i > 0; old_i = i, i /= 2) {
          if (loc_id < i) {
            dist(loc_id).x += dist(loc_id + i).x;
            dist(loc_id).y += dist(loc_id + i).y;
            if (loc_id == (i - 1) && old_i % 2 != 0) {
              dist(loc_id).x += dist(old_i - 1).x;
              dist(loc_id).y += dist(old_i - 1).y;
            }
          }
          team.team_barrier();
        }
        if (loc_id == 0) d_results(blk_id) = dist(0);
      });
  Kokkos::fence();
}

// ---- CPU fallback (mirrors OMP version) -----------------------------------
static void linear_regressionCPU(data_t *dataset, sum_t *result,
                                  int gpu_groups, int total_groups) {
  const int wg_size = TEMP_WORKGROUP_SIZE;
  std::vector<sum_t> interns(wg_size);
  for (int grp = gpu_groups; grp < total_groups; grp++) {
    for (int loc = wg_size - 1; loc >= 0; loc--) {
      int glob = loc + grp * wg_size;
      interns[loc].x = dataset[glob].x;
      interns[loc].y = dataset[glob].y;
      interns[loc].z = dataset[glob].x * dataset[glob].y;
      interns[loc].w = dataset[glob].x * dataset[glob].x;
      for (int i = wg_size / 2, old_i = wg_size; i > 0; old_i = i, i /= 2) {
        if (loc < i) {
          interns[loc].x += interns[loc + i].x;
          interns[loc].y += interns[loc + i].y;
          interns[loc].z += interns[loc + i].z;
          interns[loc].w += interns[loc + i].w;
          if (loc == (i - 1) && old_i % 2 != 0) {
            interns[loc].x += interns[old_i - 1].x;
            interns[loc].y += interns[old_i - 1].y;
            interns[loc].z += interns[old_i - 1].z;
            interns[loc].w += interns[old_i - 1].w;
          }
        }
      }
      if (loc == 0) result[grp] = interns[0];
    }
  }
}

static void rsquaredCPU(data_t *dataset, float mean, float2 equation,
                        rsquared_t *result, int gpu_groups, int total_groups) {
  const int wg_size = TEMP_WORKGROUP_SIZE;
  std::vector<rsquared_t> dist(wg_size);
  for (int grp = gpu_groups; grp < total_groups; grp++) {
    for (int loc = wg_size - 1; loc >= 0; loc--) {
      int glob = loc + grp * wg_size;
      dist[loc].x = powf(dataset[glob].y - mean, 2.f);
      float ye = dataset[glob].x * equation.y + equation.x;
      dist[loc].y = powf(ye - mean, 2.f);
      for (int i = wg_size / 2, old_i = wg_size; i > 0; old_i = i, i /= 2) {
        if (loc < i) {
          dist[loc].x += dist[loc + i].x;
          dist[loc].y += dist[loc + i].y;
          if (loc == (i - 1) && old_i % 2 != 0) {
            dist[loc].x += dist[old_i - 1].x;
            dist[loc].y += dist[old_i - 1].y;
          }
        }
      }
      if (loc == 0) result[grp] = dist[0];
    }
  }
}

// ---- r_squared wrapper ----------------------------------------------------
static void r_squared(linear_param_t *params, data_t *dataset,
                      Kokkos::View<data_t *> d_dataset,
                      sum_t *linreg, result_t *response,
                      int cpu_offset) {
  const size_t wg_size  = params->wg_size;
  const size_t wg_count = params->wg_count;
  const size_t size     = params->size;

  size_t globalWorkSize = size;
  if (size % wg_size) globalWorkSize += wg_size - (size % wg_size);

  size_t cpu_global = cpu_offset * globalWorkSize / 100;
  if (cpu_global % wg_size != 0)
    cpu_global = (1 + cpu_global / wg_size) * wg_size;
  size_t gpu_global = globalWorkSize - cpu_global;
  int    gpu_groups = (int)(gpu_global / wg_size);

  float   mean     = linreg->y / params->size;
  float2  equation = {response->a0, response->a1};

  std::vector<rsquared_t> h_results(wg_count, {0.f, 0.f});

  if (gpu_groups > 0) {
    Kokkos::View<rsquared_t *> d_results("d_rsq", wg_count);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < params->repeat; i++)
      rsquared_kokkos(gpu_groups, d_dataset, mean, equation, d_results);
    auto t1   = std::chrono::steady_clock::now();
    response->ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    auto h_d = Kokkos::create_mirror_view(d_results);
    Kokkos::deep_copy(h_d, d_results);
    for (int i = 0; i < gpu_groups; i++) h_results[i] = h_d(i);
  }

  if (cpu_offset > 0)
    rsquaredCPU(dataset, mean, equation, h_results.data(),
                gpu_groups, (int)(globalWorkSize / wg_size));

  rsquared_t final_result = {0.f, 0.f};
  for (size_t i = 0; i < wg_count; i++) {
    final_result.x += h_results[i].x;
    final_result.y += h_results[i].y;
  }
  response->rsquared = (int)(final_result.y / final_result.x * 100);
}

// ---- Parallelized regression ----------------------------------------------
void parallelized_regression(linear_param_t *params, data_t *dataset,
                              result_t *response, int cpu_offset) {
  const size_t wg_size  = params->wg_size;
  const size_t wg_count = params->wg_count;
  const size_t size     = params->size;

  size_t globalWorkSize = size;
  if (size % wg_size) globalWorkSize += wg_size - (size % wg_size);

  size_t cpu_global = cpu_offset * globalWorkSize / 100;
  if (cpu_global % wg_size != 0)
    cpu_global = (1 + cpu_global / wg_size) * wg_size;
  size_t gpu_global = globalWorkSize - cpu_global;
  int    gpu_groups = (int)(gpu_global / wg_size);

  // Copy dataset to device
  Kokkos::View<data_t *> d_dataset("d_dataset", size);
  {
    auto h_d = Kokkos::create_mirror_view(d_dataset);
    memcpy(h_d.data(), dataset, size * sizeof(data_t));
    Kokkos::deep_copy(d_dataset, h_d);
  }

  std::vector<sum_t> h_results(wg_count, {0.f, 0.f, 0.f, 0.f});

  if (gpu_groups > 0) {
    Kokkos::View<sum_t *> d_results("d_linreg", wg_count);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < params->repeat; i++)
      linear_regression_kokkos(gpu_groups, d_dataset, d_results);
    auto t1 = std::chrono::steady_clock::now();
    response->ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    auto h_d = Kokkos::create_mirror_view(d_results);
    Kokkos::deep_copy(h_d, d_results);
    for (int i = 0; i < gpu_groups; i++) h_results[i] = h_d(i);
  }

  if (cpu_offset > 0)
    linear_regressionCPU(dataset, h_results.data(),
                         gpu_groups, (int)(globalWorkSize / wg_size));

  sum_t final_result = {0.f, 0.f, 0.f, 0.f};
  for (size_t i = 0; i < wg_count; i++) {
    final_result.x += h_results[i].x;
    final_result.y += h_results[i].y;
    final_result.z += h_results[i].z;
    final_result.w += h_results[i].w;
  }

  double denom = (double)size * final_result.w -
                 (double)(final_result.x * final_result.x);
  response->a0 = (float)((final_result.y * final_result.w -
                           final_result.x * final_result.z) / denom);
  response->a1 = (float)(((double)size * final_result.z -
                           final_result.x * final_result.y) / denom);

  r_squared(params, dataset, d_dataset, &final_result, response, cpu_offset);
}

// ---- Main -----------------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: linear <repeat> <cpu_offset>\n");
    printf("Device execution only when cpu_offset is 0\n");
    printf("Host execution only when cpu_offset is 100\n");
    return 1;
  }
  int repeat     = atoi(argv[1]);
  int cpu_offset = atoi(argv[2]);
  printf("CPU offset: %d\n", cpu_offset);

  Kokkos::initialize(argc, argv);
  {
    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    linear_param_t params;
    params.repeat   = repeat;
    params.filename = (char *)TEMP_FILENAME;
    params.size     = TEMP_SIZE;
    params.wg_size  = TEMP_WORKGROUP_SIZE;
    params.wg_count = TEMP_WORKGROUP_NBR;

    data_t *dataset = (data_t *)malloc(sizeof(data_t) * params.size);
    create_dataset(&params, dataset);

    results_t results = {{0}};
    results.parallelized.ktime = 0;

    parallelized_regression(&params, dataset, &results.parallelized, cpu_offset);
    iterative_regression(&params, dataset, &results.iterative);

    free(dataset);

    gettimeofday(&tv1, NULL);
    double elapsed = (tv1.tv_sec - tv0.tv_sec) * 1000.0 +
                     (tv1.tv_usec - tv0.tv_usec) * 1e-3;
    printf("Total execution time: %lf ms\n", elapsed);
    printf("Average kernel execution time: %lf us\n",
           results.parallelized.ktime * 1e-3 / repeat);

    // Write results
    FILE *fout = fopen(RESULT_FILENAME, "a");
    if (fout) {
      WRITE_RESULT(fout, results.parallelized);
      WRITE_RESULT(fout, results.iterative);
      fclose(fout);
    }

    printf("\n> TEMPERATURE REGRESSION (%d)\n\n", TEMP_SIZE);
    PRINT_RESULT("Parallelized", results.parallelized);
    PRINT_RESULT("Iterative",    results.iterative);
  }
  Kokkos::finalize();
  return 0;
}
