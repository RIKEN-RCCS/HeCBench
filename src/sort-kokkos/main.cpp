#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <Kokkos_Core.hpp>

typedef unsigned int T;

template<typename T>
struct vec4 {
  T x, y, z, w;
};

void verifySort(const T *keys, const size_t size)
{
  bool passed = true;
  for (size_t i = 0; i < size - 1; i++) {
    if (keys[i] > keys[i + 1]) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "PASS" : "FAIL") << std::endl;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <problem size> <number of passes>\n.", argv[0]);
    return -1;
  }

  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  int probSizes[4] = { 1, 8, 32, 64 };
  size_t size = probSizes[select];
  size = (size * 1024 * 1024) / sizeof(T);

  unsigned int bytes = size * sizeof(T);

  T* idata = (T*) malloc(bytes);
  T* odata = (T*) malloc(bytes);

  std::cout << "Initializing host memory." << std::endl;
  for (size_t i = 0; i < size; i++) {
    idata[i] = i % 16;
    odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size << std::endl;

  const size_t local_wsize  = 256;
  const size_t global_wsize = 16384;
  const size_t num_work_groups = global_wsize / local_wsize;  // 64
  const int radix_width = 4;
  const int num_digits  = 16;

  Kokkos::initialize(argc, argv);
  {
    using ScratchT = Kokkos::View<T*,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryUnmanaged>;
    using member_type = Kokkos::TeamPolicy<>::member_type;

    // Device views
    Kokkos::View<T*> d_idata("d_idata", size);
    Kokkos::View<T*> d_odata("d_odata", size);
    Kokkos::View<T*> d_isums("d_isums", num_work_groups * num_digits);

    // Copy input to device
    {
      auto h = Kokkos::create_mirror_view(d_idata);
      for (size_t i = 0; i < size; i++) h(i) = idata[i];
      Kokkos::deep_copy(d_idata, h);
    }

    // Scratch sizes per team for each kernel
    size_t scratch_reduce      = ScratchT::shmem_size(local_wsize);
    size_t scratch_top_scan    = ScratchT::shmem_size(local_wsize * 2 + 1);
    size_t scratch_bottom_scan = ScratchT::shmem_size(local_wsize * 2)
                               + ScratchT::shmem_size(16)
                               + ScratchT::shmem_size(16);

    auto policy_reduce =
        Kokkos::TeamPolicy<>((int)num_work_groups, (int)local_wsize)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_reduce));
    auto policy_top_scan =
        Kokkos::TeamPolicy<>((int)num_work_groups, (int)local_wsize)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_top_scan));
    auto policy_bottom_scan =
        Kokkos::TeamPolicy<>((int)num_work_groups, (int)local_wsize)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_bottom_scan));

    double time = 0.0;

    for (int k = 0; k < passes; k++) {
      auto start = std::chrono::steady_clock::now();

      for (unsigned int shift = 0; shift < sizeof(T) * 8; shift += radix_width) {
        bool even = ((shift / radix_width) % 2 == 0);
        auto d_in  = even ? d_idata : d_odata;
        auto d_out = even ? d_odata : d_idata;

        // ---- reduce kernel ----
        // Counts digit occurrences per work-group using tree reduction.
        // Shared: lmem[local_wsize]
        Kokkos::parallel_for("sort_reduce", policy_reduce,
            KOKKOS_LAMBDA(const member_type& team) {
              ScratchT lmem(team.team_scratch(0), (int)local_wsize);

              int group_range = team.league_size();
              int group       = team.league_rank();
              int local_range = team.team_size();
              int tid         = team.team_rank();

              int region_size = ((int)(size / 4) / group_range) * 4;
              int block_start = group * region_size;
              int block_stop  = (group == group_range - 1)
                                    ? (int)size
                                    : block_start + region_size;

              int i = block_start + tid;
              int digit_counts[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

              while (i < block_stop) {
                digit_counts[(d_in(i) >> shift) & 0xFU]++;
                i += local_range;
              }

              for (int d = 0; d < 16; d++) {
                lmem(tid) = digit_counts[d];
                team.team_barrier();

                for (unsigned int s = (unsigned)local_range / 2; s > 0; s >>= 1) {
                  if (tid < (int)s) lmem(tid) += lmem(tid + (int)s);
                  team.team_barrier();
                }

                if (tid == 0)
                  d_isums(d * group_range + group) = lmem(0);
              }
            });

        // ---- top_scan kernel ----
        // Exclusive scan across work-group digit counts (isums).
        // Shared: lmem[local_wsize*2] + s_seed at lmem[local_wsize*2]
        Kokkos::parallel_for("sort_top_scan", policy_top_scan,
            KOKKOS_LAMBDA(const member_type& team) {
              // lmem[0..2*local_wsize-1] = scan workspace
              // lmem[2*local_wsize]      = s_seed (team-shared scalar)
              ScratchT lmem(team.team_scratch(0), (int)(local_wsize * 2 + 1));

              int local_range = team.team_size();
              int lid         = team.team_rank();

              if (lid == 0) lmem((int)(local_wsize * 2)) = 0;  // s_seed = 0
              team.team_barrier();

              // last_thread: thread whose lid+1 == num_work_groups
              int last_thread = (lid < (int)num_work_groups &&
                                 (lid + 1) == (int)num_work_groups) ? 1 : 0;

              for (int d = 0; d < 16; d++) {
                T val = 0;
                if (lid < (int)num_work_groups)
                  val = d_isums((int)num_work_groups * d + lid);

                // Exclusive scan via Hillis-Steele in scratch memory
                int idx = lid;
                lmem(idx) = 0;
                idx += local_range;
                lmem(idx) = val;
                team.team_barrier();

                for (int stride = 1; stride < local_range; stride *= 2) {
                  T t = lmem(idx - stride);
                  team.team_barrier();
                  lmem(idx) += t;
                  team.team_barrier();
                }
                T res = lmem(idx - 1);  // exclusive prefix for this thread

                if (lid < (int)num_work_groups)
                  d_isums((int)num_work_groups * d + lid) =
                      res + lmem((int)(local_wsize * 2));  // res + s_seed
                team.team_barrier();

                if (last_thread)
                  lmem((int)(local_wsize * 2)) += res + val;  // s_seed accumulate
                team.team_barrier();
              }
            });

        // ---- bottom_scan kernel ----
        // Scatter elements to output using scanned seeds.
        // Shared: lmem[local_wsize*2], l_scanned_seeds[16], l_block_counts[16]
        Kokkos::parallel_for("sort_bottom_scan", policy_bottom_scan,
            KOKKOS_LAMBDA(const member_type& team) {
              ScratchT lmem(team.team_scratch(0), (int)(local_wsize * 2));
              ScratchT l_scanned_seeds(team.team_scratch(0), 16);
              ScratchT l_block_counts(team.team_scratch(0), 16);

              int group_range = team.league_size();
              int group       = team.league_rank();
              int local_range = team.team_size();
              int lid         = team.team_rank();

              int histogram[16];

              int n4          = (int)size / 4;
              int region_size = n4 / group_range;
              int block_start = group * region_size;
              int block_stop  = (group == group_range - 1)
                                    ? n4
                                    : block_start + region_size;

              int ei     = block_start + lid;
              int window = block_start;

              if (lid < 16) {
                l_block_counts(lid)   = 0;
                l_scanned_seeds(lid)  = d_isums(lid * group_range + group);
              }
              team.team_barrier();

              while (window < block_stop) {
                for (int q = 0; q < 16; q++) histogram[q] = 0;

                vec4<T> val_4, key_4;
                if (ei < block_stop) {
                  val_4.x = d_in(4 * ei);
                  val_4.y = d_in(4 * ei + 1);
                  val_4.z = d_in(4 * ei + 2);
                  val_4.w = d_in(4 * ei + 3);

                  key_4.x = (val_4.x >> shift) & 0xFU;
                  key_4.y = (val_4.y >> shift) & 0xFU;
                  key_4.z = (val_4.z >> shift) & 0xFU;
                  key_4.w = (val_4.w >> shift) & 0xFU;

                  histogram[key_4.x]++;
                  histogram[key_4.y]++;
                  histogram[key_4.z]++;
                  histogram[key_4.w]++;
                }

                // Exclusive scan of per-thread digit counts across the team
                for (int digit = 0; digit < 16; digit++) {
                  int idx = lid;
                  lmem(idx) = 0;
                  idx += local_range;
                  lmem(idx) = histogram[digit];
                  team.team_barrier();

                  for (int stride = 1; stride < local_range; stride *= 2) {
                    T t = lmem(idx - stride);
                    team.team_barrier();
                    lmem(idx) += t;
                    team.team_barrier();
                  }
                  histogram[digit] = lmem(idx - 1);
                  team.team_barrier();
                }

                if (ei < block_stop) {
                  int address;
                  address = histogram[key_4.x] + l_scanned_seeds(key_4.x)
                            + l_block_counts(key_4.x);
                  d_out(address) = val_4.x;
                  histogram[key_4.x]++;

                  address = histogram[key_4.y] + l_scanned_seeds(key_4.y)
                            + l_block_counts(key_4.y);
                  d_out(address) = val_4.y;
                  histogram[key_4.y]++;

                  address = histogram[key_4.z] + l_scanned_seeds(key_4.z)
                            + l_block_counts(key_4.z);
                  d_out(address) = val_4.z;
                  histogram[key_4.z]++;

                  address = histogram[key_4.w] + l_scanned_seeds(key_4.w)
                            + l_block_counts(key_4.w);
                  d_out(address) = val_4.w;
                  histogram[key_4.w]++;
                }
                team.team_barrier();

                // Thread with highest lid accumulates block histogram
                if (lid == local_range - 1) {
                  for (int q = 0; q < 16; q++)
                    l_block_counts(q) += histogram[q];
                }
                team.team_barrier();

                window += local_range;
                ei     += local_range;
              }
            });
      }  // shift loop

      Kokkos::fence();
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }  // passes loop

    printf("Average elapsed time per pass %lf (s)\n", time * 1e-9 / passes);

    // Copy result (odata device view matches the OMP map(from: odata) semantics)
    {
      auto h = Kokkos::create_mirror_view(d_odata);
      Kokkos::deep_copy(h, d_odata);
      for (size_t i = 0; i < size; i++) odata[i] = h(i);
    }
  }
  Kokkos::finalize();

  verifySort(odata, size);

  free(idata);
  free(odata);
  return 0;
}
