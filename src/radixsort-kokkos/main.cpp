/*
 * GPU Radix Sort – Kokkos port
 * Original: Copyright 1993-2010 NVIDIA Corporation
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>

// ─── type helpers ─────────────────────────────────────────────────────────────
struct uint2 { unsigned int x, y; };
struct uint4 {
  unsigned int x, y, z, w;
  KOKKOS_INLINE_FUNCTION uint4 operator+(const uint4& o) const { return {x+o.x,y+o.y,z+o.z,w+o.w}; }
  KOKKOS_INLINE_FUNCTION uint4 operator-(const uint4& o) const { return {x-o.x,y-o.y,z-o.z,w-o.w}; }
  KOKKOS_INLINE_FUNCTION uint4& operator+=(const uint4& o){ x+=o.x;y+=o.y;z+=o.z;w+=o.w; return *this; }
};

using ExecSpace    = Kokkos::DefaultExecutionSpace;
using MemSpace     = ExecSpace::memory_space;
using ScratchSpace = ExecSpace::scratch_memory_space;
using TeamPolicy   = Kokkos::TeamPolicy<ExecSpace>;
using Member       = TeamPolicy::member_type;
using UintView     = Kokkos::View<unsigned int*, MemSpace>;

template<typename T>
using ScratchView = Kokkos::View<T*, ScratchSpace, Kokkos::MemoryUnmanaged>;

// ─── constants ────────────────────────────────────────────────────────────────
static const unsigned int WARP_SIZE   = 32;
static const unsigned int bitStep     = 4;
static const unsigned int CTA_SIZE    = 128;
static const int WORKGROUP_SIZE       = 256;
static const unsigned int MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE = 1024;
static const unsigned int MAX_LOCAL_GROUP_SIZE = 256;
static const unsigned int MAX_BATCH_ELEMENTS  = 64 * 1048576;
static const unsigned int MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
static const unsigned int MAX_LARGE_ARRAY_SIZE = 4u * WORKGROUP_SIZE * WORKGROUP_SIZE;

// ─── scan device helpers ──────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
unsigned int warpScanInclusive(const unsigned int idata,
                                volatile unsigned int* l_Data,
                                const unsigned int size)
{
  int lid = Kokkos::impl_get_thread_num();   // NOTE: non-portable; use thread rank
  // We receive lid via a wrapper – see scan4 below
  (void)lid;
  return idata; // placeholder; actual implementation below using passed lid
}

// ─── scan helpers (all take explicit thread index) ────────────────────────────
KOKKOS_INLINE_FUNCTION
unsigned int warpScanInclusiveIdx(const unsigned int idata,
                                   volatile unsigned int* l_Data,
                                   const unsigned int size,
                                   const int lid)
{
  unsigned int pos = 2 * lid - (lid & (size - 1));
  l_Data[pos] = 0;
  pos += size;
  l_Data[pos] = idata;
  if (0 <= 4) { if ((unsigned)(pos)     < 2*WORKGROUP_SIZE) l_Data[pos] += l_Data[pos -  1]; }
  if (1 <= 4) { if ((unsigned)(pos)     < 2*WORKGROUP_SIZE) l_Data[pos] += l_Data[pos -  2]; }
  if (2 <= 4) { if ((unsigned)(pos)     < 2*WORKGROUP_SIZE) l_Data[pos] += l_Data[pos -  4]; }
  if (3 <= 4) { if ((unsigned)(pos)     < 2*WORKGROUP_SIZE) l_Data[pos] += l_Data[pos -  8]; }
  if (4 <= 4) { if ((unsigned)(pos)     < 2*WORKGROUP_SIZE) l_Data[pos] += l_Data[pos - 16]; }
  return l_Data[pos];
}

KOKKOS_INLINE_FUNCTION
unsigned int warpScanExclusiveIdx(const unsigned int idata,
                                   unsigned int* l_Data,
                                   const unsigned int size,
                                   const int lid)
{
  return warpScanInclusiveIdx(idata, (volatile unsigned int*)l_Data, size, lid) - idata;
}

KOKKOS_INLINE_FUNCTION
unsigned int scan1InclusiveIdx(const unsigned int idata,
                                unsigned int* l_Data,
                                const unsigned int size,
                                const int lid,
                                const Member& team)
{
  const unsigned int LOG2_WARP_SIZE = 5;
  if (size > WARP_SIZE) {
    unsigned int warpResult = warpScanInclusiveIdx(idata, (volatile unsigned int*)l_Data, WARP_SIZE, lid);
    team.team_barrier();
    if ((lid & (WARP_SIZE - 1)) == WARP_SIZE - 1)
      l_Data[lid >> LOG2_WARP_SIZE] = warpResult;
    team.team_barrier();
    if (lid < (WORKGROUP_SIZE / WARP_SIZE)) {
      unsigned int val = l_Data[lid];
      l_Data[lid] = warpScanExclusiveIdx(val, l_Data, size >> LOG2_WARP_SIZE, lid);
    }
    team.team_barrier();
    return warpResult + l_Data[lid >> LOG2_WARP_SIZE];
  } else {
    return warpScanInclusiveIdx(idata, (volatile unsigned int*)l_Data, size, lid);
  }
}

KOKKOS_INLINE_FUNCTION
unsigned int scan1ExclusiveIdx(const unsigned int idata,
                                unsigned int* l_Data,
                                const unsigned int size,
                                const int lid,
                                const Member& team)
{
  return scan1InclusiveIdx(idata, l_Data, size, lid, team) - idata;
}

KOKKOS_INLINE_FUNCTION
uint4 scan4Inclusive(uint4 data4, unsigned int* l_Data,
                     const unsigned int size, const int lid, const Member& team)
{
  data4.y += data4.x; data4.z += data4.y; data4.w += data4.z;
  unsigned int val = scan1InclusiveIdx(data4.w, l_Data, size / 4, lid, team) - data4.w;
  uint4 v4 = { val, val, val, val };
  return data4 + v4;
}

KOKKOS_INLINE_FUNCTION
uint4 scan4Exclusive(uint4 data4, unsigned int* l_Data,
                     const unsigned int size, const int lid, const Member& team)
{
  return scan4Inclusive(data4, l_Data, size, lid, team) - data4;
}

// ─── scanwarp (as in original RadixSort_kernels, takes explicit lid) ──────────
KOKKOS_INLINE_FUNCTION
unsigned int scanwarp(unsigned int val, volatile unsigned int* sData,
                      const int maxlevel, const int localId)
{
  int idx = 2 * localId - (localId & (WARP_SIZE - 1));
  sData[idx] = 0;
  idx += WARP_SIZE;
  sData[idx] = val;
  if (0 <= maxlevel) { sData[idx] += sData[idx -  1]; }
  if (1 <= maxlevel) { sData[idx] += sData[idx -  2]; }
  if (2 <= maxlevel) { sData[idx] += sData[idx -  4]; }
  if (3 <= maxlevel) { sData[idx] += sData[idx -  8]; }
  if (4 <= maxlevel) { sData[idx] += sData[idx - 16]; }
  return sData[idx] - val;
}

KOKKOS_INLINE_FUNCTION
uint4 scan4_rsx(const uint4 idata, unsigned int* ptr,
                const int idx, const Member& team)
{
  uint4 val4 = idata;
  unsigned int sum[3];
  sum[0] = val4.x;
  sum[1] = val4.y + sum[0];
  sum[2] = val4.z + sum[1];
  unsigned int val = val4.w + sum[2];

  val = scanwarp(val, ptr, 4, idx);
  team.team_barrier();

  if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
    ptr[idx >> 5] = val + val4.w + sum[2];
  team.team_barrier();

  if (idx < (int)WARP_SIZE)
    ptr[idx] = scanwarp(ptr[idx], ptr, 2, idx);
  team.team_barrier();

  val += ptr[idx >> 5];
  val4.x = val;
  val4.y = val + sum[0];
  val4.z = val + sum[1];
  val4.w = val + sum[2];
  return val4;
}

KOKKOS_INLINE_FUNCTION
uint4 rank4_rsx(const uint4 preds, unsigned int* sMem, unsigned int* numtrue,
                const int localId, const int localSize, const Member& team)
{
  uint4 address = scan4_rsx(preds, sMem, localId, team);
  if (localId == localSize - 1)
    numtrue[0] = address.w + preds.w;
  team.team_barrier();

  uint4 rank;
  int idx = localId * 4;
  rank.x = preds.x ? address.x : numtrue[0] + idx     - address.x;
  rank.y = preds.y ? address.y : numtrue[0] + idx + 1 - address.y;
  rank.z = preds.z ? address.z : numtrue[0] + idx + 2 - address.z;
  rank.w = preds.w ? address.w : numtrue[0] + idx + 3 - address.w;
  return rank;
}

// ─── radixSortBlocksKeysOnly ──────────────────────────────────────────────────
void radixSortBlocksKeysOnly(UintView d_keys, UintView d_tempKeys,
                              unsigned int nbits, unsigned int startbit,
                              unsigned int numElements)
{
  unsigned int totalBlocks = numElements / 4 / CTA_SIZE;
  // scratch: sMem[4*CTA_SIZE] + numtrue[1]
  size_t scratch = ScratchView<unsigned int>::shmem_size(4 * CTA_SIZE + 1);

  Kokkos::parallel_for(
    "radixSortBlocks",
    TeamPolicy(totalBlocks, CTA_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<unsigned int> sMem   (team.team_scratch(0), 4 * CTA_SIZE);
      ScratchView<unsigned int> numtrue(team.team_scratch(0), 1);

      int localId   = team.team_rank();
      int localSize = team.team_size();
      int globalId  = team.league_rank() * localSize + localId;

      // load 4 keys per thread
      unsigned int* keys_u4 = d_keys.data();
      uint4 key;
      key.x = keys_u4[4*globalId + 0];
      key.y = keys_u4[4*globalId + 1];
      key.z = keys_u4[4*globalId + 2];
      key.w = keys_u4[4*globalId + 3];

      team.team_barrier();

      for (unsigned int shift = startbit; shift < startbit + nbits; ++shift) {
        uint4 lsb;
        lsb.x = !((key.x >> shift) & 1u);
        lsb.y = !((key.y >> shift) & 1u);
        lsb.z = !((key.z >> shift) & 1u);
        lsb.w = !((key.w >> shift) & 1u);

        uint4 r = rank4_rsx(lsb, sMem.data(), numtrue.data(), localId, localSize, team);

        sMem[(r.x & 3) * localSize + (r.x >> 2)] = key.x;
        sMem[(r.y & 3) * localSize + (r.y >> 2)] = key.y;
        sMem[(r.z & 3) * localSize + (r.z >> 2)] = key.z;
        sMem[(r.w & 3) * localSize + (r.w >> 2)] = key.w;
        team.team_barrier();

        key.x = sMem[localId];
        key.y = sMem[localId +     localSize];
        key.z = sMem[localId + 2 * localSize];
        key.w = sMem[localId + 3 * localSize];
        team.team_barrier();
      }

      unsigned int* tkeys = d_tempKeys.data();
      tkeys[4*globalId + 0] = key.x;
      tkeys[4*globalId + 1] = key.y;
      tkeys[4*globalId + 2] = key.z;
      tkeys[4*globalId + 3] = key.w;
    }
  );
  Kokkos::fence();
}

// ─── findRadixOffsets ─────────────────────────────────────────────────────────
void findRadixOffsets(UintView d_tempKeys, UintView d_counters,
                      UintView d_blockOffsets,
                      unsigned int startbit, unsigned int numElements)
{
  unsigned int totalBlocks = numElements / 2 / CTA_SIZE;
  size_t scratch = ScratchView<unsigned int>::shmem_size(16) +
                   ScratchView<unsigned int>::shmem_size(2 * CTA_SIZE);

  Kokkos::parallel_for(
    "findRadixOffsets",
    TeamPolicy(totalBlocks, CTA_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<unsigned int> sStartPointers(team.team_scratch(0), 16);
      ScratchView<unsigned int> sRadix1       (team.team_scratch(0), 2 * CTA_SIZE);

      unsigned int groupId   = team.league_rank();
      unsigned int localId   = team.team_rank();
      unsigned int groupSize = team.team_size();
      unsigned int globalId  = groupId * groupSize + localId;

      // load 2 keys per thread
      unsigned int* tkeys = d_tempKeys.data();
      unsigned int r2x = tkeys[2*globalId];
      unsigned int r2y = tkeys[2*globalId + 1];

      sRadix1[2 * localId]     = (r2x >> startbit) & 0xF;
      sRadix1[2 * localId + 1] = (r2y >> startbit) & 0xF;

      if (localId < 16) sStartPointers[localId] = 0;
      team.team_barrier();

      if (localId > 0 && sRadix1[localId] != sRadix1[localId - 1])
        sStartPointers[sRadix1[localId]] = localId;
      if (sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
      team.team_barrier();

      if (localId < 16)
        d_blockOffsets[groupId * 16 + localId] = sStartPointers[localId];
      team.team_barrier();

      if (localId > 0 && sRadix1[localId] != sRadix1[localId - 1])
        sStartPointers[sRadix1[localId - 1]] = localId - sStartPointers[sRadix1[localId - 1]];
      if (sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
        sStartPointers[sRadix1[localId + groupSize - 1]] =
          localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
      if (localId == groupSize - 1)
        sStartPointers[sRadix1[2*groupSize - 1]] =
          2*groupSize - sStartPointers[sRadix1[2*groupSize - 1]];
      team.team_barrier();

      if (localId < 16)
        d_counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
  );
  Kokkos::fence();
}

// ─── reorderDataKeysOnly ──────────────────────────────────────────────────────
void reorderDataKeysOnly(UintView d_keys, UintView d_tempKeys,
                          UintView d_blockOffsets, UintView d_countersSum,
                          UintView d_counters,
                          unsigned int startbit, unsigned int numElements)
{
  unsigned int totalBlocks = numElements / 2 / CTA_SIZE;
  size_t scratch = ScratchView<unsigned int>::shmem_size(16) * 2 +
                   ScratchView<unsigned int>::shmem_size(2 * CTA_SIZE);

  Kokkos::parallel_for(
    "reorderData",
    TeamPolicy(totalBlocks, CTA_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<unsigned int> sOffsets     (team.team_scratch(0), 16);
      ScratchView<unsigned int> sBlockOffsets(team.team_scratch(0), 16);
      ScratchView<unsigned int> sKeys1       (team.team_scratch(0), 2 * CTA_SIZE);

      unsigned int groupId   = team.league_rank();
      unsigned int localId   = team.team_rank();
      unsigned int groupSize = team.team_size();
      unsigned int globalId  = groupId * groupSize + localId;

      // load 2 keys
      sKeys1[2*localId]   = d_tempKeys[2*globalId];
      sKeys1[2*localId+1] = d_tempKeys[2*globalId+1];

      if (localId < 16) {
        sOffsets[localId]      = d_countersSum[localId * totalBlocks + groupId];
        sBlockOffsets[localId] = d_blockOffsets[groupId * 16 + localId];
      }
      team.team_barrier();

      unsigned int radix = (sKeys1[localId] >> startbit) & 0xF;
      unsigned int go = sOffsets[radix] + localId - sBlockOffsets[radix];
      if (go < numElements)
        d_keys[go] = sKeys1[localId];

      radix = (sKeys1[localId + groupSize] >> startbit) & 0xF;
      go = sOffsets[radix] + localId + groupSize - sBlockOffsets[radix];
      if (go < numElements)
        d_keys[go] = sKeys1[localId + groupSize];
    }
  );
  Kokkos::fence();
}

// ─── scan functions ───────────────────────────────────────────────────────────
static unsigned int iSnapUp(unsigned int dividend, unsigned int divisor) {
  return (dividend % divisor == 0) ? dividend : dividend - dividend % divisor + divisor;
}
unsigned int factorRadix2(unsigned int& log2L, unsigned int L) {
  if (!L) { log2L = 0; return 0; }
  for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
  return L;
}

void scanExclusiveLocal1(UintView d_Dst, UintView d_Src,
                          unsigned int n, unsigned int size)
{
  size_t localWorkSize  = WORKGROUP_SIZE;
  size_t globalWorkSize = (n * size) / 4;
  unsigned int totalBlocks = (unsigned int)(globalWorkSize / localWorkSize);
  size_t scratch = ScratchView<unsigned int>::shmem_size(2 * WORKGROUP_SIZE);

  Kokkos::parallel_for(
    "scanLocal1",
    TeamPolicy(totalBlocks, WORKGROUP_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<unsigned int> l_Data(team.team_scratch(0), 2 * WORKGROUP_SIZE);
      int lid = team.team_rank();
      int i   = team.league_rank() * team.team_size() + lid;

      unsigned int* src4 = d_Src.data();
      uint4 idata4 = { src4[4*i], src4[4*i+1], src4[4*i+2], src4[4*i+3] };
      uint4 odata4 = scan4Exclusive(idata4, l_Data.data(), size, lid, team);
      unsigned int* dst4 = d_Dst.data();
      dst4[4*i]   = odata4.x; dst4[4*i+1] = odata4.y;
      dst4[4*i+2] = odata4.z; dst4[4*i+3] = odata4.w;
    }
  );
  Kokkos::fence();
}

void scanExclusiveLocal2(UintView d_Buf, UintView d_Dst, UintView d_Src,
                          unsigned int n, unsigned int size)
{
  unsigned int elements = n * size;
  unsigned int totalBlocks = (unsigned int)(iSnapUp(elements, WORKGROUP_SIZE) / WORKGROUP_SIZE);
  size_t scratch = ScratchView<unsigned int>::shmem_size(2 * WORKGROUP_SIZE);

  Kokkos::parallel_for(
    "scanLocal2",
    TeamPolicy(totalBlocks, WORKGROUP_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<unsigned int> l_Data(team.team_scratch(0), 2 * WORKGROUP_SIZE);
      int lid = team.team_rank();
      int i   = team.league_rank() * team.team_size() + lid;

      unsigned int data = 0;
      if ((unsigned)i < elements)
        data = d_Dst[(4*WORKGROUP_SIZE - 1) + (4*WORKGROUP_SIZE) * i] +
               d_Src[(4*WORKGROUP_SIZE - 1) + (4*WORKGROUP_SIZE) * i];

      unsigned int odata = scan1ExclusiveIdx(data, l_Data.data(), size, lid, team);
      if ((unsigned)i < elements) d_Buf[i] = odata;
    }
  );
  Kokkos::fence();
}

void uniformUpdate(UintView d_Dst, UintView d_Buf, unsigned int n)
{
  size_t scratch = ScratchView<unsigned int>::shmem_size(1);
  Kokkos::parallel_for(
    "uniformUpdate",
    TeamPolicy(n, WORKGROUP_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team)
    {
      ScratchView<unsigned int> buf(team.team_scratch(0), 1);
      int lid     = team.team_rank();
      int groupId = team.league_rank();
      int i       = groupId * team.team_size() + lid;

      unsigned int* dst4 = d_Dst.data();
      uint4 data4 = { dst4[4*i], dst4[4*i+1], dst4[4*i+2], dst4[4*i+3] };
      if (lid == 0) buf[0] = d_Buf[groupId];
      team.team_barrier();
      uint4 bv = { buf[0], buf[0], buf[0], buf[0] };
      data4 += bv;
      dst4[4*i]   = data4.x; dst4[4*i+1] = data4.y;
      dst4[4*i+2] = data4.z; dst4[4*i+3] = data4.w;
    }
  );
  Kokkos::fence();
}

void scanExclusiveLarge(UintView d_Dst, UintView d_Src, UintView d_Buf,
                         unsigned int batchSize, unsigned int arrayLength,
                         unsigned int /*numElements*/)
{
  scanExclusiveLocal1(d_Dst, d_Src,
                      (batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
                      4 * WORKGROUP_SIZE);
  scanExclusiveLocal2(d_Buf, d_Dst, d_Src,
                      batchSize, arrayLength / (4 * WORKGROUP_SIZE));
  uniformUpdate(d_Dst, d_Buf,
                (batchSize * arrayLength) / (4 * WORKGROUP_SIZE));
}

// ─── radixSortKeys ────────────────────────────────────────────────────────────
void radixSortStepKeysOnly(UintView d_keys, UintView d_tempKeys,
                            UintView d_counters, UintView d_blockOffsets,
                            UintView d_countersSum, UintView d_buffer,
                            unsigned int nbits, unsigned int startbit,
                            unsigned int numElements, unsigned int batchSize)
{
  radixSortBlocksKeysOnly(d_keys, d_tempKeys, nbits, startbit, numElements);
  findRadixOffsets(d_tempKeys, d_counters, d_blockOffsets, startbit, numElements);
  unsigned int arrayLength = numElements / 2 / CTA_SIZE * 16;
  scanExclusiveLarge(d_countersSum, d_counters, d_buffer,
                     batchSize, arrayLength, numElements);
  reorderDataKeysOnly(d_keys, d_tempKeys, d_blockOffsets, d_countersSum, d_counters,
                      startbit, numElements);
}

void radixSortKeys(UintView d_keys, UintView d_tempKeys,
                   UintView d_counters, UintView d_blockOffsets,
                   UintView d_countersSum, UintView d_buffer,
                   unsigned int numElements, unsigned int keyBits,
                   unsigned int batchSize)
{
  int i = 0;
  while (keyBits > (unsigned)i * bitStep) {
    radixSortStepKeysOnly(d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum,
                          d_buffer, bitStep, i * bitStep, numElements, batchSize);
    i++;
  }
}

// ─── verification helpers ──────────────────────────────────────────────────────
void makeRandomUintVector(unsigned int* a, unsigned int n, unsigned int keybits) {
  int keyshiftmask = (keybits > 16) ? (1 << (keybits - 16)) - 1 : 0;
  int keymask      = (keybits < 16) ? (1 << keybits) - 1 : 0xffff;
  srand(95123);
  for (unsigned int i = 0; i < n; i++)
    a[i] = ((rand() & keyshiftmask) << 16) | (rand() & keymask);
}

bool verifySortUint(unsigned int* sorted, unsigned int* unsorted, unsigned int len) {
  for (unsigned int i = 0; i < len - 1; i++) {
    if (sorted[i] > sorted[i+1]) {
      printf("Unordered key[%u]: %u > key[%u]: %u\n", i, sorted[i], i+1, sorted[i+1]);
      return false;
    }
  }
  return true;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, const char** argv)
{
  if (argc != 2) { printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  const int numIterations = atoi(argv[1]);

  const unsigned int numElements = 128*128*128*2;
  const int keybits  = 32;
  const int batchSize = 1;
  const unsigned int numBlocks =
    ((numElements % (CTA_SIZE * 4)) == 0) ?
    numElements / (CTA_SIZE * 4) : numElements / (CTA_SIZE * 4) + 1;

  unsigned int arrayLength = numElements / 2 / CTA_SIZE * 16;
  unsigned int log2L;
  assert(factorRadix2(log2L, arrayLength) == 1);
  assert(arrayLength >= MIN_LARGE_ARRAY_SIZE && arrayLength <= MAX_LARGE_ARRAY_SIZE);
  assert(arrayLength > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE);
  assert((batchSize * arrayLength) <= (int)MAX_BATCH_ELEMENTS);

  unsigned int* h_keys        = (unsigned int*)malloc(numElements * sizeof(unsigned int));
  unsigned int* h_tempKeys    = (unsigned int*)malloc(numElements * sizeof(unsigned int));

  makeRandomUintVector(h_keys, numElements, keybits);
  memcpy(h_tempKeys, h_keys, numElements * sizeof(unsigned int));

  Kokkos::initialize(argc, (char**)argv);
  {
    // device buffers
    UintView d_keys       ("d_keys",        numElements);
    UintView d_tempKeys   ("d_tempKeys",    numElements);
    UintView d_counters   ("d_counters",    WARP_SIZE * numBlocks);
    UintView d_countersSum("d_countersSum", WARP_SIZE * numBlocks);
    UintView d_blockOff   ("d_blockOff",    WARP_SIZE * numBlocks);
    UintView d_buffer     ("d_buffer",      arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE);

    // copy keys to device
    auto hm = Kokkos::create_mirror_view(d_keys);
    memcpy(hm.data(), h_keys, numElements * sizeof(unsigned int));
    Kokkos::deep_copy(d_keys, hm);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < numIterations; i++) {
      // reset keys each iteration
      Kokkos::deep_copy(d_keys, hm);
      radixSortKeys(d_keys, d_tempKeys, d_counters, d_blockOff, d_countersSum,
                    d_buffer, numElements, keybits, batchSize);
    }
    auto end  = std::chrono::steady_clock::now();
    auto ns   = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of radixsort: %f (s)\n", (ns * 1e-9) / numIterations);

    // copy result back
    Kokkos::deep_copy(hm, d_keys);
    memcpy(h_keys, hm.data(), numElements * sizeof(unsigned int));
  }
  Kokkos::finalize();

  bool passed = verifySortUint(h_keys, h_tempKeys, numElements);
  free(h_keys); free(h_tempKeys);
  printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
