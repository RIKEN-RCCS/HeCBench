#include <Kokkos_Core.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <climits>
#include <sys/types.h>
#include "StringSearch.h"

// MAX_PATTERN_LENGTH for scratch allocation (ENABLE_2ND_LEVEL_FILTER requires 32)
#define MAX_PAT_LEN 32

using ExecSpace   = Kokkos::DefaultExecutionSpace;
using ScratchSpace = ExecSpace::scratch_memory_space;
template<typename T>
using ScratchView = Kokkos::View<T*, ScratchSpace, Kokkos::MemoryUnmanaged>;

KOKKOS_INLINE_FUNCTION
int compare_dev(const uchar* text, const uchar* pattern, uint length)
{
  for (uint l = 0; l < length; ++l)
    if (TOLOWER(text[l]) != pattern[l]) return 0;
  return 1;
}

int verify(uint* resultCount, uint workGroupCount,
    uint* result, uint searchLenPerWG,
    std::vector<uint>& cpuResults)
{
  uint count = resultCount[0];
  for (uint i = 1; i < workGroupCount; ++i) {
    uint found = resultCount[i];
    if (found > 0) {
      memcpy(result + count, result + i * searchLenPerWG, found * sizeof(uint));
      count += found;
    }
  }
  std::sort(result, result + count);
  std::cout << "Device: found " << count << " times\n";

  bool pass = (count == cpuResults.size());
  pass = pass && std::equal(result, result + count, cpuResults.begin());
  if (pass) { std::cout << "Passed!\n\n"; return 0; }
  else       { std::cout << "Failed\n\n";  return -1; }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <path to file> <substring> <repeat>\n", argv[0]);
    return -1;
  }
  std::string file   = argv[1];
  std::string subStr = argv[2];
  int iterations     = atoi(argv[3]);

  if (iterations < 1) {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    exit(0);
  }
  if (file.empty()) {
    std::cout << "\n Error: Input File not specified...\n";
    return -1;
  }

  std::ifstream textFile(file.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
  if (!textFile.is_open()) {
    std::cout << "\n Unable to open file: " << file << "\n";
    return -1;
  }

  uint   textLength = (uint)textFile.tellg();
  uchar *text       = (uchar*)malloc(textLength + 1);
  memset(text, 0, textLength + 1);
  textFile.seekg(0, std::ios::beg);
  if (!textFile.read((char*)text, textLength)) {
    std::cout << "\n Reading file failed\n";
    textFile.close();
    return -1;
  }
  textFile.close();

  uint subStrLength = (uint)subStr.length();
  if (subStrLength == 0) {
    std::cout << "\nError: Sub-String not specified...\n";
    return -1;
  }
  if (textLength < subStrLength) {
    std::cout << "\nText size less than search pattern ("
              << textLength << " < " << subStrLength << ")\n";
    return -1;
  }

#ifdef ENABLE_2ND_LEVEL_FILTER
  if (subStrLength != 1 && subStrLength <= 16) {
    std::cout << "\nSearch pattern size should be longer than 16\n";
    return -1;
  }
#endif

  std::cout << "Search Pattern : " << subStr << "\n";

  // CPU reference
  std::vector<uint> cpuResults;
  uint last = subStrLength - 1;
  uint badCharSkip[UCHAR_MAX + 1];
  for (uint scan = 0; scan <= UCHAR_MAX; ++scan) badCharSkip[scan] = subStrLength;
  for (uint scan = 0; scan < last; ++scan) {
    badCharSkip[toupper(subStr[scan])] = last - scan;
    badCharSkip[tolower(subStr[scan])] = last - scan;
  }
  uint curPos = 0;
  while ((textLength - curPos) > last) {
    int p = (int)last;
    uint scan;
    for (scan = last + curPos; COMPARE(text[scan], subStr[p--]); scan -= 1) {
      if (scan == curPos) { cpuResults.push_back(curPos); break; }
    }
    curPos += (scan == curPos) ? 1 : badCharSkip[text[last + curPos]];
  }
  std::cout << "CPU: found " << cpuResults.size() << " times\n";

  const uchar *pattern       = (const uchar*)subStr.c_str();
  uint totalSearchPos        = textLength - subStrLength + 1;
  uint searchLenPerWG        = SEARCH_BYTES_PER_WORKITEM * LOCAL_SIZE;
  uint workGroupCount        = (totalSearchPos + searchLenPerWG - 1) / searchLenPerWG;
  uint patternLength         = subStrLength;
  uint maxSearchLength       = searchLenPerWG;

  uint *resultCount = (uint*)malloc(workGroupCount * sizeof(uint));
  uint *result      = (uint*)malloc((textLength - subStrLength + 1) * sizeof(uint));

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<uchar*> text_d("text", textLength);
    Kokkos::View<uchar*> pattern_d("pattern", subStrLength);
    Kokkos::View<uint*>  resultCount_d("resultCount", workGroupCount);
    Kokkos::View<uint*>  result_d("result", textLength - subStrLength + 1);

    {
      auto text_hv    = Kokkos::View<uchar*, Kokkos::HostSpace,
                                     Kokkos::MemoryUnmanaged>(text, textLength);
      auto pattern_hv = Kokkos::View<uchar*, Kokkos::HostSpace,
                                     Kokkos::MemoryUnmanaged>(pattern, subStrLength);
      Kokkos::deep_copy(text_d,    text_hv);
      Kokkos::deep_copy(pattern_d, pattern_hv);
    }

    double time = 0.0;

    if (subStrLength == 1) {
      std::cout << "\nRun only Naive-Kernel version of String Search for pattern size = 1\n";
      std::cout << "\nExecuting String search naive for " << iterations << " iterations\n";

      size_t scratch_bytes =
          ScratchView<uchar>::shmem_size(1) +
          ScratchView<uint>::shmem_size(1);   // groupSuccessCounter

      auto policy = Kokkos::TeamPolicy<>(workGroupCount, LOCAL_SIZE)
                        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      using member_type = Kokkos::TeamPolicy<>::member_type;

      auto start = std::chrono::steady_clock::now();

      for (int it = 0; it < iterations; ++it) {
        Kokkos::parallel_for("naive_search", policy,
          KOKKOS_LAMBDA(member_type const& team) {
            const int localIdx  = team.team_rank();
            const int localSize = team.team_size();
            const int groupIdx  = team.league_rank();

            ScratchView<uchar> localPattern(team.team_scratch(0), 1);
            ScratchView<uint>  gSucc(team.team_scratch(0), 1);  // groupSuccessCounter

            uint lastSearchIdx  = textLength - patternLength + 1;
            uint beginSearchIdx = (uint)groupIdx * maxSearchLength;
            uint endSearchIdx   = beginSearchIdx + maxSearchLength;

            if (beginSearchIdx <= lastSearchIdx) {
              if (endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;

              for (int idx = localIdx; idx < (int)patternLength; idx += localSize)
                localPattern(idx) = TOLOWER(pattern_d(idx));

              if (localIdx == 0) gSucc(0) = 0;
              team.team_barrier();

              for (uint sp = beginSearchIdx + (uint)localIdx; sp < endSearchIdx; sp += (uint)localSize) {
                if (compare_dev(text_d.data() + sp, localPattern.data(), patternLength) == 1) {
                  int count = (int)Kokkos::atomic_fetch_add(&gSucc(0), 1u);
                  result_d(beginSearchIdx + (uint)count) = sp;
                }
              }

              team.team_barrier();
              if (localIdx == 0) resultCount_d(groupIdx) = gSucc(0);
            }
          });
      }
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      {
        auto rc_hv = Kokkos::View<uint*, Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>(resultCount, workGroupCount);
        auto r_hv  = Kokkos::View<uint*, Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>(result, textLength - subStrLength + 1);
        Kokkos::deep_copy(rc_hv, resultCount_d);
        Kokkos::deep_copy(r_hv,  result_d);
      }

      verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
    }

    if (subStrLength > 1) {
      std::cout << "\nExecuting String search with load balance for "
                << iterations << " iterations\n";

      size_t scratch_bytes =
          ScratchView<uchar>::shmem_size(MAX_PAT_LEN) +
          ScratchView<uint>::shmem_size(LOCAL_SIZE * 2) +  // stack1
          ScratchView<uint>::shmem_size(LOCAL_SIZE * 2) +  // stack2
          ScratchView<uint>::shmem_size(3);                  // counters: stack1Ctr,stack2Ctr,groupSuccCtr

      auto policy = Kokkos::TeamPolicy<>(workGroupCount, LOCAL_SIZE)
                        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      using member_type = Kokkos::TeamPolicy<>::member_type;

      auto start = std::chrono::steady_clock::now();

      for (int it = 0; it < iterations; ++it) {
        Kokkos::parallel_for("balanced_search", policy,
          KOKKOS_LAMBDA(member_type const& team) {
            const int localIdx  = team.team_rank();
            const int localSize = team.team_size();
            const int groupIdx  = team.league_rank();

            ScratchView<uchar> localPattern(team.team_scratch(0), MAX_PAT_LEN);
            ScratchView<uint>  stack1(team.team_scratch(0), LOCAL_SIZE * 2);
            ScratchView<uint>  stack2(team.team_scratch(0), LOCAL_SIZE * 2);
            // counters(0)=stack1Counter, counters(1)=stack2Counter, counters(2)=groupSuccessCounter
            ScratchView<uint>  counters(team.team_scratch(0), 3);

            if (localIdx == 0) {
              counters(0) = 0;
              counters(1) = 0;
              counters(2) = 0;
            }

            uint lastSearchIdx  = textLength - patternLength + 1;
            uint beginSearchIdx = (uint)groupIdx * maxSearchLength;
            uint endSearchIdx   = beginSearchIdx + maxSearchLength;

            if (beginSearchIdx <= lastSearchIdx) {
              if (endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;
              uint searchLength = endSearchIdx - beginSearchIdx;

              for (uint idx = (uint)localIdx; idx < patternLength; idx += (uint)localSize)
                localPattern(idx) = TOLOWER(pattern_d(idx));

              team.team_barrier();

              uchar first  = localPattern(0);
              uchar second = localPattern(1);
              int   stringPos   = localIdx;
              int   stackPos    = 0;
              int   revStackPos = 0;

              while (true) {
                // Level-1: two-character filter; push candidate positions to stack1
                if ((uint)stringPos < searchLength) {
                  if (first  == TOLOWER(text_d(beginSearchIdx + (uint)stringPos)) &&
                      second == TOLOWER(text_d(beginSearchIdx + (uint)stringPos + 1))) {
                    stackPos = (int)Kokkos::atomic_fetch_add(&counters(0), 1u);
                    stack1((uint)stackPos) = (uint)stringPos;
                  }
                }

                stringPos += localSize;

                team.team_barrier();
                uint stackSize = counters(0);
                team.team_barrier();

                if ((stackSize < (uint)localSize) &&
                    ((((uint)stringPos / (uint)localSize) * (uint)localSize) < searchLength))
                  continue;

#ifdef ENABLE_2ND_LEVEL_FILTER
                // Level-2: roll over 8 more bytes from stack1 positions into stack2
                if ((uint)localIdx < stackSize) {
                  revStackPos = (int)Kokkos::atomic_fetch_sub(&counters(0), 1u) - 1;
                  int pos = (int)stack1((uint)revStackPos);
                  bool status =
                      (localPattern(2) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 2))) &&
                      (localPattern(3) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 3))) &&
                      (localPattern(4) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 4))) &&
                      (localPattern(5) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 5))) &&
                      (localPattern(6) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 6))) &&
                      (localPattern(7) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 7))) &&
                      (localPattern(8) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 8))) &&
                      (localPattern(9) == TOLOWER(text_d(beginSearchIdx + (uint)pos + 9)));
                  if (status) {
                    stackPos = (int)Kokkos::atomic_fetch_add(&counters(1), 1u);
                    stack2((uint)stackPos) = (uint)pos;
                  }
                }

                team.team_barrier();
                stackSize = counters(1);
                team.team_barrier();

                if ((stackSize < (uint)localSize) &&
                    ((((uint)stringPos / (uint)localSize) * (uint)localSize) < searchLength))
                  continue;
#endif

                // Level-3: full match check on remaining stack positions
                if ((uint)localIdx < stackSize) {
#ifdef ENABLE_2ND_LEVEL_FILTER
                  revStackPos = (int)Kokkos::atomic_fetch_sub(&counters(1), 1u) - 1;
                  int pos = (int)stack2((uint)revStackPos);
                  if (compare_dev(text_d.data() + beginSearchIdx + (uint)pos + 10,
                                  localPattern.data() + 10, patternLength - 10) == 1)
#else
                  revStackPos = (int)Kokkos::atomic_fetch_sub(&counters(0), 1u) - 1;
                  int pos = (int)stack1((uint)revStackPos);
                  if (compare_dev(text_d.data() + beginSearchIdx + (uint)pos + 2,
                                  localPattern.data() + 2, patternLength - 2) == 1)
#endif
                  {
                    int count = (int)Kokkos::atomic_fetch_add(&counters(2), 1u);
                    result_d(beginSearchIdx + (uint)count) = beginSearchIdx + (uint)pos;
                  }
                }

                team.team_barrier();
                if ((((uint)stringPos / (uint)localSize) * (uint)localSize >= searchLength) &&
                    (counters(0) == 0u) && (counters(1) == 0u))
                  break;
              }  // while

              if (localIdx == 0) resultCount_d(groupIdx) = counters(2);
            }
          });
      }
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      {
        auto rc_hv = Kokkos::View<uint*, Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>(resultCount, workGroupCount);
        auto r_hv  = Kokkos::View<uint*, Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>(result, textLength - subStrLength + 1);
        Kokkos::deep_copy(rc_hv, resultCount_d);
        Kokkos::deep_copy(r_hv,  result_d);
      }

      verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
    }

    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);
  }
  Kokkos::finalize();

  free(text);
  free(result);
  free(resultCount);
  return 0;
}
