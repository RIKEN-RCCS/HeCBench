/*
 * GPU Quicksort – Kokkos port
 * Original: Copyright (c) 2014-2019, Intel Corporation
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>
#include <climits>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

// ─── type alias ────────────────────────────────────────────────────────────
typedef unsigned int uint;
using ExecSpace   = Kokkos::DefaultExecutionSpace;
using MemSpace    = ExecSpace::memory_space;
using ScratchSpace = ExecSpace::scratch_memory_space;
using TeamPolicy  = Kokkos::TeamPolicy<ExecSpace>;
using Member      = TeamPolicy::member_type;

template<typename T>
using ScratchView = Kokkos::View<T*, ScratchSpace, Kokkos::MemoryUnmanaged>;
template<typename T>
using DevView     = Kokkos::View<T*, MemSpace>;

// ─── constants ─────────────────────────────────────────────────────────────
#define QUICKSORT_BLOCK_SIZE        1024
#define GQSORT_LOCAL_WORKGROUP_SIZE  128
#define LQSORT_LOCAL_WORKGROUP_SIZE  256
#define SORT_THRESHOLD               512
#define EMPTY_RECORD                  42
#define WS_LEN (QUICKSORT_BLOCK_SIZE / SORT_THRESHOLD)   // 2

// ─── data structures ────────────────────────────────────────────────────────
template <class T>
struct work_record {
  uint start, end, direction;
  T    pivot;
  KOKKOS_INLINE_FUNCTION
  work_record() : start(0), end(0), direction(EMPTY_RECORD), pivot(T(0)) {}
  KOKKOS_INLINE_FUNCTION
  work_record(uint s, uint e, T p, uint d) : start(s), end(e), direction(d), pivot(p) {}
};

struct parent_record {
  uint sstart, send, oldstart, oldend, blockcount;
  KOKKOS_INLINE_FUNCTION
  parent_record() : sstart(0), send(0), oldstart(0), oldend(0), blockcount(0) {}
  KOKKOS_INLINE_FUNCTION
  parent_record(uint ss, uint se, uint os, uint oe, uint bc)
    : sstart(ss), send(se), oldstart(os), oldend(oe), blockcount(bc) {}
};

template <class T>
struct block_record {
  uint start, end, direction, parent;
  T    pivot;
  KOKKOS_INLINE_FUNCTION
  block_record() : start(0), end(0), direction(EMPTY_RECORD), parent(0), pivot(T(0)) {}
  KOKKOS_INLINE_FUNCTION
  block_record(uint s, uint e, T p, uint d, uint prnt)
    : start(s), end(e), direction(d), parent(prnt), pivot(p) {}
};

struct workstack_record {
  uint start, end, direction;
};

// ─── helper device functions ────────────────────────────────────────────────
template <typename T, typename P>
KOKKOS_INLINE_FUNCTION
T Select(T a, T b, P c) { return c ? b : a; }

KOKKOS_INLINE_FUNCTION
uint median(uint x1, uint x2, uint x3) {
  if (x1 < x2) {
    if (x2 < x3) return x2;
    return Select(x1, x3, x1 < x3);
  } else {
    if (x1 < x3) return x1;
    return Select(x2, x3, x2 < x3);
  }
}

template <class T>
T median_host(T x1, T x2, T x3) {
  if (x1 < x2) {
    if (x2 < x3) return x2;
    return (x1 < x3) ? x3 : x1;
  } else {
    if (x1 < x3) return x1;
    return (x2 < x3) ? x2 : x3;
  }
}

KOKKOS_INLINE_FUNCTION
void plus_prescan(uint* a, uint* b) {
  uint av = *a, bv = *b;
  *a = bv;
  *b = bv + av;
}

// ─── bitonic_sort ────────────────────────────────────────────────────────────
// Sorts 2*LQSORT_LOCAL_WORKGROUP_SIZE elements in-place (reinterprets as uint)
template <typename T>
KOKKOS_INLINE_FUNCTION
void bitonic_sort(T* sh_data, const uint localid, const Member& team)
{
  for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
    for (uint j = ulevel; j > 0; j >>= 1) {
      uint pos = 2*localid - (localid & (j - 1));
      uint direction = localid & ulevel;
      uint av = (uint)sh_data[pos], bv = (uint)sh_data[pos + j];
      const bool sortThem = av > bv;
      const uint greater = Select(bv, av, sortThem);
      const uint lesser  = Select(av, bv, sortThem);
      sh_data[pos]     = (T)Select(lesser, greater, direction);
      sh_data[pos + j] = (T)Select(greater, lesser, direction);
      team.team_barrier();
    }
  }
  for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
    uint pos = 2*localid - (localid & (j - 1));
    uint av = (uint)sh_data[pos], bv = (uint)sh_data[pos + j];
    const bool sortThem = av > bv;
    sh_data[pos]     = (T)Select(av, bv, sortThem);
    sh_data[pos + j] = (T)Select(bv, av, sortThem);
    team.team_barrier();
  }
}

// ─── sort_threshold ──────────────────────────────────────────────────────────
template <typename T>
KOKKOS_INLINE_FUNCTION
void sort_threshold(T* data_in, T* data_out,
                    uint start, uint end, T* temp,
                    uint localid, const Member& team)
{
  uint tsum = end - start;
  if (tsum == SORT_THRESHOLD) {
    bitonic_sort(data_in + start, localid, team);
    for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE)
      data_out[start + i] = data_in[start + i];
  } else if (tsum > 1) {
    for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      if (i < tsum)
        temp[i] = data_in[start + i];
      else
        temp[i] = (T)UINT_MAX;
    }
    team.team_barrier();
    bitonic_sort(temp, localid, team);
    for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE)
      data_out[start + i] = temp[i];
  } else if (tsum == 1 && localid == 0) {
    data_out[start] = data_in[start];
  }
}

// ─── gqsort kernel ──────────────────────────────────────────────────────────
template <class T>
void gqsort_kokkos(DevView<T>& d, DevView<T>& dn,
                   DevView<block_record<T>>& blocksb,
                   DevView<parent_record>& parentsb,
                   DevView<work_record<T>>& result,
                   int blocks_size)
{
  // scratch: lt[GWS+1], gt[GWS+1], shared_scalars[4] (ltsum,gtsum,lbeg,gbeg)
  const size_t scratch_size =
    ScratchView<uint>::shmem_size(GQSORT_LOCAL_WORKGROUP_SIZE + 1) * 2 +
    ScratchView<uint>::shmem_size(4);

  Kokkos::parallel_for(
    "gqsort",
    TeamPolicy(blocks_size, GQSORT_LOCAL_WORKGROUP_SIZE)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<uint> lt(team.team_scratch(0), GQSORT_LOCAL_WORKGROUP_SIZE + 1);
      ScratchView<uint> gt(team.team_scratch(0), GQSORT_LOCAL_WORKGROUP_SIZE + 1);
      // sh[0]=ltsum, sh[1]=gtsum, sh[2]=lbeg, sh[3]=gbeg
      ScratchView<uint> sh(team.team_scratch(0), 4);

      const uint blockid = team.league_rank();
      const uint localid = team.team_rank();

      block_record<T> block = blocksb[blockid];
      uint start = block.start, end = block.end;
      T    pivot = block.pivot;
      uint direction = block.direction;

      T *s, *sn;
      if (direction == 1) { s = d.data();  sn = dn.data(); }
      else                 { s = dn.data(); sn = d.data();  }

      lt[localid] = gt[localid] = 0;
      team.team_barrier();

      uint ltp = 0, gtp = 0;
      for (uint i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
        T tmp = s[i];
        if (tmp < pivot) ltp++;
        if (tmp > pivot) gtp++;
      }
      lt[localid] = ltp;
      gt[localid] = gtp;
      team.team_barrier();

      // up-sweep
      for (uint i = 1; i < GQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
        uint n = 2*i - 1;
        if ((localid & n) == n) {
          lt[localid] += lt[localid - i];
          gt[localid] += gt[localid - i];
        }
        team.team_barrier();
      }
      uint n = GQSORT_LOCAL_WORKGROUP_SIZE - 1;
      if ((localid & n) == n) {
        lt[GQSORT_LOCAL_WORKGROUP_SIZE] = sh[0] = lt[localid]; // ltsum
        gt[GQSORT_LOCAL_WORKGROUP_SIZE] = sh[1] = gt[localid]; // gtsum
        lt[localid] = 0;
        gt[localid] = 0;
      }
      // down-sweep
      for (uint i = GQSORT_LOCAL_WORKGROUP_SIZE / 2; i >= 1; i >>= 1) {
        n = 2*i - 1;
        if ((localid & n) == n) {
          plus_prescan(&lt[localid - i], &lt[localid]);
          plus_prescan(&gt[localid - i], &gt[localid]);
        }
        team.team_barrier();
      }

      // atomic allocation in parent
      if (localid == 0) {
        uint ltsum = sh[0], gtsum = sh[1];
        sh[2] = Kokkos::atomic_fetch_add(&parentsb[block.parent].sstart, ltsum);
        uint old_send = Kokkos::atomic_fetch_sub(&parentsb[block.parent].send, gtsum);
        sh[3] = old_send - gtsum; // gbeg
      }
      team.team_barrier();

      uint lfrom = sh[2] + lt[localid];
      uint gfrom = sh[3] + gt[localid];

      for (uint i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
        T tmp = s[i];
        if (tmp < pivot) sn[lfrom++] = tmp;
        if (tmp > pivot) sn[gfrom++] = tmp;
      }
      team.team_barrier();

      if (localid == 0) {
        uint old_bc = Kokkos::atomic_fetch_sub(&parentsb[block.parent].blockcount, 1u);
        if (old_bc == 0) {
          uint sstart   = parentsb[block.parent].sstart;
          uint send_val = parentsb[block.parent].send;
          uint oldstart = parentsb[block.parent].oldstart;
          uint oldend   = parentsb[block.parent].oldend;

          for (uint i = sstart; i < send_val; i++)
            d[i] = pivot;

          T lpivot = sn[oldstart];
          T gpivot = sn[oldend - 1];
          if (oldstart < sstart)
            lpivot = (T)median((uint)lpivot, (uint)sn[(oldstart+sstart)>>1], (uint)sn[sstart-1]);
          if (send_val < oldend)
            gpivot = (T)median((uint)sn[send_val], (uint)sn[(oldend+send_val)>>1], (uint)gpivot);

          uint dir2 = direction ^ 1;
          result[2*blockid]   = work_record<T>(oldstart, sstart,  lpivot, dir2);
          result[2*blockid+1] = work_record<T>(send_val, oldend,  gpivot, dir2);
        }
      }
    }
  );
  Kokkos::fence();
}

// ─── lqsort kernel ──────────────────────────────────────────────────────────
template <class T>
void lqsort_kokkos(DevView<T>& d, DevView<T>& dn,
                   DevView<work_record<T>>& seqs,
                   int done_size)
{
  const size_t scratch_size =
    ScratchView<workstack_record>::shmem_size(WS_LEN + 1) +
    ScratchView<int>::shmem_size(1) +
    ScratchView<T>::shmem_size(QUICKSORT_BLOCK_SIZE) * 2 +
    ScratchView<T>::shmem_size(SORT_THRESHOLD) +
    ScratchView<uint>::shmem_size(LQSORT_LOCAL_WORKGROUP_SIZE + 1) * 2 +
    ScratchView<uint>::shmem_size(2);  // ltsum, gtsum

  Kokkos::parallel_for(
    "lqsort",
    TeamPolicy(done_size, LQSORT_LOCAL_WORKGROUP_SIZE)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<workstack_record> workstack(team.team_scratch(0), WS_LEN + 1);
      ScratchView<int>  ws_ptr (team.team_scratch(0), 1);
      ScratchView<T>    mys    (team.team_scratch(0), QUICKSORT_BLOCK_SIZE);
      ScratchView<T>    mysn   (team.team_scratch(0), QUICKSORT_BLOCK_SIZE);
      ScratchView<T>    temp   (team.team_scratch(0), SORT_THRESHOLD);
      ScratchView<uint> lt     (team.team_scratch(0), LQSORT_LOCAL_WORKGROUP_SIZE + 1);
      ScratchView<uint> gt     (team.team_scratch(0), LQSORT_LOCAL_WORKGROUP_SIZE + 1);
      ScratchView<uint> lgtsum (team.team_scratch(0), 2);  // [0]=ltsum, [1]=gtsum

      const uint blockid = team.league_rank();
      const uint localid = team.team_rank();

      work_record<T> block = seqs[blockid];
      const uint d_offset = block.start;
      uint start = 0;
      uint end   = block.end - d_offset;

      // initialise workstack
      if (localid == 0) {
        ws_ptr[0] = 0;
        workstack[0] = { start, end, 1u };
      }
      // load data into shared memory
      if (block.direction == 1) {
        for (uint i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE)
          mys[i] = d[i + d_offset];
      } else {
        for (uint i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE)
          mys[i] = dn[i + d_offset];
      }
      team.team_barrier();

      while (ws_ptr[0] >= 0) {
        // pop
        workstack_record wr = workstack[ws_ptr[0]];
        start     = wr.start;
        end       = wr.end;
        uint direction = wr.direction;
        team.team_barrier();
        if (localid == 0) {
          ws_ptr[0]--;
          lgtsum[0] = lgtsum[1] = 0;
        }

        T *s, *sn;
        if (direction == 1) { s = mys.data();  sn = mysn.data(); }
        else                 { s = mysn.data(); sn = mys.data();  }

        lt[localid] = gt[localid] = 0;
        uint ltp = 0, gtp = 0;
        team.team_barrier();

        uint pivot = (uint)s[start];
        if (start < end)
          pivot = median(pivot, (uint)s[(start+end)>>1], (uint)s[end-1]);

        for (uint i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
          uint tmp = (uint)s[i];
          if (tmp < pivot) ltp++;
          if (tmp > pivot) gtp++;
        }
        lt[localid] = ltp;
        gt[localid] = gtp;
        team.team_barrier();

        // up-sweep
        for (uint i = 1; i < LQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
          uint n = 2*i - 1;
          if ((localid & n) == n) {
            lt[localid] += lt[localid - i];
            gt[localid] += gt[localid - i];
          }
          team.team_barrier();
        }
        uint n = LQSORT_LOCAL_WORKGROUP_SIZE - 1;
        if ((localid & n) == n) {
          lt[LQSORT_LOCAL_WORKGROUP_SIZE] = lgtsum[0] = lt[localid];
          gt[LQSORT_LOCAL_WORKGROUP_SIZE] = lgtsum[1] = gt[localid];
          lt[localid] = 0;
          gt[localid] = 0;
        }
        // down-sweep
        for (uint i = LQSORT_LOCAL_WORKGROUP_SIZE / 2; i >= 1; i >>= 1) {
          n = 2*i - 1;
          if ((localid & n) == n) {
            plus_prescan(&lt[localid - i], &lt[localid]);
            plus_prescan(&gt[localid - i], &gt[localid]);
          }
          team.team_barrier();
        }

        uint lfrom = start + lt[localid];
        uint gfrom = end - gt[localid + 1];

        for (uint i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
          uint tmp = (uint)s[i];
          if (tmp < pivot) sn[lfrom++] = (T)tmp;
          if (tmp > pivot) sn[gfrom++] = (T)tmp;
        }
        team.team_barrier();

        uint ltsum = lgtsum[0], gtsum = lgtsum[1];
        // write pivot values directly to d
        for (uint i = start + ltsum + localid; i < end - gtsum; i += LQSORT_LOCAL_WORKGROUP_SIZE)
          d[i + d_offset] = (T)pivot;
        team.team_barrier();

        // handle left partition
        if (ltsum <= SORT_THRESHOLD) {
          sort_threshold(sn, d.data() + d_offset, start, start + ltsum, temp.data(), localid, team);
        } else {
          if (localid == 0) {
            int wp = ws_ptr[0] + 1;
            ws_ptr[0] = wp;
            workstack[wp] = { start, start + ltsum, direction ^ 1 };
          }
          team.team_barrier();
        }

        // handle right partition
        if (gtsum <= SORT_THRESHOLD) {
          sort_threshold(sn, d.data() + d_offset, end - gtsum, end, temp.data(), localid, team);
        } else {
          if (localid == 0) {
            int wp = ws_ptr[0] + 1;
            ws_ptr[0] = wp;
            workstack[wp] = { end - gtsum, end, direction ^ 1 };
          }
          team.team_barrier();
        }
      } // while
    }
  );
  Kokkos::fence();
}

// ─── host-side helper ────────────────────────────────────────────────────────
size_t optp(size_t s, double k, size_t m) {
  return (size_t)std::pow(2.0, std::floor(std::log(s*k + m) / std::log(2.0) + 0.5));
}

// ─── GPUQSort ────────────────────────────────────────────────────────────────
template <class T>
void GPUQSort(size_t size, T* h_d, T* h_dn)
{
  const size_t padded = (size / 64 + 1) * 64;

  // device views
  DevView<T> dev_d ("d",  padded);
  DevView<T> dev_dn("dn", padded);
  auto h_d_mirror  = Kokkos::create_mirror_view(dev_d);
  auto h_dn_mirror = Kokkos::create_mirror_view(dev_dn);
  std::memcpy(h_d_mirror.data(),  h_d,  padded * sizeof(T));
  std::memcpy(h_dn_mirror.data(), h_dn, padded * sizeof(T));
  Kokkos::deep_copy(dev_d,  h_d_mirror);
  Kokkos::deep_copy(dev_dn, h_dn_mirror);

  const size_t MAXSEQ = optp(size, 0.00009516, 203);
  const size_t MAX_SIZE = 12 * std::max(MAXSEQ, (size_t)QUICKSORT_BLOCK_SIZE);

  uint startpivot = (uint)median_host(h_d[0], h_d[size/2], h_d[size-1]);

  std::vector<work_record<T>>  work, done, news;
  std::vector<parent_record>   parent_records;
  std::vector<block_record<T>> blocks;
  work.reserve(MAX_SIZE); done.reserve(MAX_SIZE); news.reserve(MAX_SIZE);
  parent_records.reserve(MAX_SIZE); blocks.reserve(MAX_SIZE);

  work.push_back(work_record<T>(0, size, (T)startpivot, 1));

  while (!work.empty()) {
    size_t blocksize = 0;
    for (auto& it : work)
      blocksize += std::max((it.end - it.start) / MAXSEQ, (size_t)1);

    for (auto& it : work) {
      uint bstart = it.start, bend = it.end;
      T    pivot  = it.pivot;
      uint dir    = it.direction;
      uint bcnt   = (bend - bstart + blocksize - 1) / blocksize;
      parent_record prnt(bstart, bend, bstart, bend, bcnt - 1);
      parent_records.push_back(prnt);
      for (uint i = 0; i < bcnt - 1; i++) {
        uint bs = bstart + blocksize * i;
        blocks.push_back(block_record<T>(bs, bs + blocksize, pivot, dir, parent_records.size()-1));
      }
      blocks.push_back(block_record<T>(bstart + blocksize*(bcnt-1), bend, pivot, dir, parent_records.size()-1));
    }

    news.resize(blocks.size() * 2);

    // copy blocks and parents to device
    DevView<block_record<T>>  d_blocks ("blocks",  blocks.size());
    DevView<parent_record>    d_parents("parents", parent_records.size());
    DevView<work_record<T>>   d_result ("result",  news.size());

    {
      auto hb = Kokkos::create_mirror_view(d_blocks);
      std::memcpy(hb.data(), blocks.data(), blocks.size() * sizeof(block_record<T>));
      Kokkos::deep_copy(d_blocks, hb);
    }
    {
      auto hp = Kokkos::create_mirror_view(d_parents);
      std::memcpy(hp.data(), parent_records.data(), parent_records.size() * sizeof(parent_record));
      Kokkos::deep_copy(d_parents, hp);
    }
    // zero-init result (set direction=EMPTY_RECORD)
    Kokkos::parallel_for("init_result", news.size(), KOKKOS_LAMBDA(int i){
      d_result[i] = work_record<T>();
    });

    gqsort_kokkos(dev_d, dev_dn, d_blocks, d_parents, d_result, (int)blocks.size());

    {
      auto hr = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(hr, d_result);
      std::memcpy(news.data(), hr.data(), news.size() * sizeof(work_record<T>));
    }

    work.clear(); parent_records.clear(); blocks.clear();
    for (auto& r : news) {
      if (r.direction != EMPTY_RECORD) {
        if (r.end - r.start <= QUICKSORT_BLOCK_SIZE) {
          if (r.end - r.start > 0) done.push_back(r);
        } else {
          work.push_back(r);
        }
      }
    }
    news.clear();
  }

  for (auto& it : work)
    if (it.end - it.start > 0) done.push_back(it);

  if (!done.empty()) {
    DevView<work_record<T>> d_seqs("seqs", done.size());
    {
      auto hs = Kokkos::create_mirror_view(d_seqs);
      std::memcpy(hs.data(), done.data(), done.size() * sizeof(work_record<T>));
      Kokkos::deep_copy(d_seqs, hs);
    }
    lqsort_kokkos(dev_d, dev_dn, d_seqs, (int)done.size());
  }

  // copy d back to host
  Kokkos::deep_copy(h_d_mirror, dev_d);
  std::memcpy(h_d, h_d_mirror.data(), padded * sizeof(T));
}

// ─── CPU quicksort (for verification baseline) ───────────────────────────────
template <class T>
T* cpu_partition(T* left, T* right, T pivot) {
  T temp = *right; *right = pivot; *left = temp;
  T* store = left;
  for (T* p = left; p != right; p++) {
    if (*p < pivot) { temp = *store; *store = *p; *p = temp; store++; }
  }
  temp = *store; *store = pivot; *right = temp;
  return store;
}
template <class T>
void cpu_quicksort(T* data, int left, int right) {
  T* store = cpu_partition(data + left, data + right, data[left]);
  int nr = store - data, nl = nr + 1;
  if (left  < nr)  (nr - left  > 32) ? cpu_quicksort(data, left, nr)  : std::sort(data+left, data+nr+1);
  if (nl < right) (right - nl > 32) ? cpu_quicksort(data, nl, right) : std::sort(data+nl,   data+right+1);
}

double seconds() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec / 1e9;
}

template <class T>
int test(uint arraySize, unsigned int NUM_ITERATIONS, const std::string& type_name)
{
  printf("\n--------------------------------------------------------------------\n");
  printf("Allocating array size of %d (data type: %s)\n", arraySize, type_name.c_str());
  T* pArray     = (T*)aligned_alloc(4096, ((arraySize*sizeof(T))/64+1)*64);
  T* pArrayCopy = (T*)aligned_alloc(4096, ((arraySize*sizeof(T))/64+1)*64);

  std::generate(pArray, pArray + arraySize, [](){ static uint i = 0; return (T)(++i); });
  std::shuffle(pArray, pArray + arraySize, std::mt19937(19937));

  std::vector<T> original(arraySize);
  std::copy(pArray, pArray + arraySize, original.begin());

  std::vector<double> times(NUM_ITERATIONS);
  double avgTime = 0.0;
  uint failures = 0;

  for (uint k = 0; k < NUM_ITERATIONS; k++) {
    std::copy(original.begin(), original.end(), pArray);
    std::vector<T> verify(pArray, pArray + arraySize);

    double t0 = seconds();
    GPUQSort(arraySize, pArray, pArrayCopy);
    double t1 = seconds();
    times[k] = t1 - t0;
    avgTime  += times[k];
    printf("Time to sort: %.3f ms\n", times[k]*1000.0);

    std::sort(verify.begin(), verify.end());
    if (!std::equal(verify.begin(), verify.end(), pArray)) {
      fprintf(stderr, "MISMATCH at iteration %u\n", k);
      failures++;
    }
  }
  printf("Number of failures: %u / %u\n", failures, NUM_ITERATIONS);
  avgTime /= NUM_ITERATIONS;
  printf("Average Time: %.3f ms\n", avgTime * 1000.0);
  printf("-------done--------------------------------------------------------\n");
  free(pArray);
  free(pArrayCopy);
  return failures ? 1 : 0;
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <iterations> <width> <height>\n", argv[0]);
    return -1;
  }
  unsigned int iters  = atoi(argv[1]);
  unsigned int width  = atoi(argv[2]);
  unsigned int height = atoi(argv[3]);
  unsigned int sz     = width * height;

  Kokkos::initialize(argc, argv);
  {
    test<uint>  (sz, iters, "uint");
    test<float> (sz, iters, "float");
    test<double>(sz, iters, "double");
  }
  Kokkos::finalize();
  return 0;
}
