#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

template<typename T>
void verify(const T* cpu_out, const T* gpu_out, int64_t n)
{
  int error = memcmp(cpu_out, gpu_out, n * sizeof(T));
  if (error) {
    for (int64_t i = 0; i < n; i++) {
      if (cpu_out[i] != gpu_out[i]) {
        printf("@%zu: %lf != %lf\n", (size_t)i, (double)cpu_out[i], (double)gpu_out[i]);
        break;
      }
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
}

// TeamPolicy-based parallel scan (with bank conflicts)
template<typename T, int N>
void run_scan(
    Kokkos::View<T*> d_in,
    Kokkos::View<T*> d_out,
    int64_t num_blocks,
    int repeat,
    double &elapsed_ns)
{
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;
  // Scratch pad: store N elements of type T
  using ScratchT = Kokkos::View<int64_t*,
                    Kokkos::DefaultExecutionSpace::scratch_memory_space,
                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  int team_size = N / 2;
  // Use up to 16*CU teams, stride over num_blocks
  int64_t nteams = num_blocks; // each team handles one block
  if (nteams > 65536) nteams = 65536;

  int scratch_size = ScratchT::shmem_size(N);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {
    Kokkos::parallel_for("scan",
      team_policy(nteams, team_size).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const member_type& team) {
        ScratchT temp(team.team_scratch(0), N);
        int thid = team.team_rank();

        for (int64_t bid = team.league_rank(); bid < num_blocks; bid += team.league_size()) {
          auto gi = d_in.data() + bid * N;
          auto go = d_out.data() + bid * N;

          temp(2*thid)   = (int64_t)gi[2*thid];
          temp(2*thid+1) = (int64_t)gi[2*thid+1];

          int offset = 1;
          for (int d = N >> 1; d > 0; d >>= 1) {
            team.team_barrier();
            if (thid < d) {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              temp(bi) += temp(ai);
            }
            offset *= 2;
          }

          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            temp(N-1) = 0;
          });

          for (int d = 1; d < N; d *= 2) {
            offset >>= 1;
            team.team_barrier();
            if (thid < d) {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              int64_t t = temp(ai);
              temp(ai) = temp(bi);
              temp(bi) += t;
            }
          }
          team.team_barrier();

          go[2*thid]   = (T)temp(2*thid);
          go[2*thid+1] = (T)temp(2*thid+1);
        }
      });
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  elapsed_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

// TeamPolicy-based parallel scan (bank conflict aware)
template<typename T, int N>
void run_scan_bcao(
    Kokkos::View<T*> d_in,
    Kokkos::View<T*> d_out,
    int64_t num_blocks,
    int repeat,
    double &elapsed_ns)
{
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;
  using ScratchT = Kokkos::View<int64_t*,
                    Kokkos::DefaultExecutionSpace::scratch_memory_space,
                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  int team_size = N / 2;
  int64_t nteams = num_blocks;
  if (nteams > 65536) nteams = 65536;

  // bcao needs N + OFFSET(N) + extra
  int padded = N + OFFSET(N) + 2;
  int scratch_size = ScratchT::shmem_size(padded);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {
    Kokkos::parallel_for("scan_bcao",
      team_policy(nteams, team_size).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const member_type& team) {
        ScratchT temp(team.team_scratch(0), padded);
        int thid = team.team_rank();

        for (int64_t bid = team.league_rank(); bid < num_blocks; bid += team.league_size()) {
          auto gi = d_in.data() + bid * N;
          auto go = d_out.data() + bid * N;

          int a = thid;
          int b = a + (N/2);
          int oa = OFFSET(a);
          int ob = OFFSET(b);

          temp(a + oa) = (int64_t)gi[a];
          temp(b + ob) = (int64_t)gi[b];

          int offset = 1;
          for (int d = N >> 1; d > 0; d >>= 1) {
            team.team_barrier();
            if (thid < d) {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              ai += OFFSET(ai);
              bi += OFFSET(bi);
              temp(bi) += temp(ai);
            }
            offset *= 2;
          }

          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            temp(N-1 + OFFSET(N-1)) = 0;
          });

          for (int d = 1; d < N; d *= 2) {
            offset >>= 1;
            team.team_barrier();
            if (thid < d) {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              ai += OFFSET(ai);
              bi += OFFSET(bi);
              int64_t t = temp(ai);
              temp(ai) = temp(bi);
              temp(bi) += t;
            }
          }
          team.team_barrier();

          go[a] = (T)temp(a + oa);
          go[b] = (T)temp(b + ob);
        }
      });
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  elapsed_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

template <typename T, int N>
void runTest(const int64_t n, const int repeat, bool timing = false)
{
  int64_t num_blocks = (n + N - 1) / N;
  int64_t nelems = num_blocks * N;
  int64_t bytes = nelems * sizeof(T);

  T *in      = (T*) malloc(bytes);
  T *cpu_out = (T*) malloc(bytes);
  T *gpu_out = (T*) malloc(bytes);

  srand(123);
  for (int64_t i = 0; i < nelems; i++) in[i] = (T)(rand() % 5 + 1);

  // CPU reference
  T *t_in  = in;
  T *t_out = cpu_out;
  for (int64_t b = 0; b < num_blocks; b++) {
    t_out[0] = 0;
    for (int i = 1; i < N; i++)
      t_out[i] = t_out[i-1] + t_in[i-1];
    t_out += N;
    t_in  += N;
  }

  Kokkos::View<T*> d_in("d_in", nelems);
  Kokkos::View<T*> d_out("d_out", nelems);

  auto h_in = Kokkos::create_mirror_view(d_in);
  for (int64_t i = 0; i < nelems; i++) h_in(i) = in[i];
  Kokkos::deep_copy(d_in, h_in);

  double elapsed_ns = 0.0, bcao_elapsed_ns = 0.0;

  run_scan<T, N>(d_in, d_out, num_blocks, repeat, elapsed_ns);

  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);
  for (int64_t i = 0; i < nelems; i++) gpu_out[i] = h_out(i);

  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of scan (w/  bank conflicts): %f (us)\n",
           sizeof(T), (elapsed_ns * 1e-3) / repeat);
  } else {
    verify(cpu_out, gpu_out, nelems);
  }

  run_scan_bcao<T, N>(d_in, d_out, num_blocks, repeat, bcao_elapsed_ns);

  Kokkos::deep_copy(h_out, d_out);
  for (int64_t i = 0; i < nelems; i++) gpu_out[i] = h_out(i);

  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of scan (w/o bank conflicts): %f (us). ",
           sizeof(T), (bcao_elapsed_ns * 1e-3) / repeat);
    printf("Reduce the time by %.1f%%\n",
           (elapsed_ns - bcao_elapsed_ns) * 1.0 / elapsed_ns * 100.0);
  } else {
    verify(cpu_out, gpu_out, nelems);
  }

  free(in);
  free(cpu_out);
  free(gpu_out);
}

template<int N>
void run(const int64_t n, const int repeat) {
  for (int i = 0; i < 2; i++) {
    bool report_timing = i > 0;
    printf("\nThe number of elements to scan in a thread block: %d\n", N);
    runTest< char, N>(n, repeat, report_timing);
    runTest<short, N>(n, repeat, report_timing);
    runTest<  int, N>(n, repeat, report_timing);
    runTest< long, N>(n, repeat, report_timing);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    run< 128>(n, repeat);
    run< 256>(n, repeat);
    run< 512>(n, repeat);
    run<1024>(n, repeat);
    run<2048>(n, repeat);
  }
  Kokkos::finalize();
  return 0;
}
