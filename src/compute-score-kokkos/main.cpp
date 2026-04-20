// Copyright (C) 2013-2018 Altera Corporation.
// Kokkos port of compute-score-omp/main.cpp.
// The OMP target teams kernel is replaced with Kokkos::TeamPolicy.
// The OMP target parallel for reduction kernel uses Kokkos::parallel_for.

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "options.h"
#include "scoped_ptrs.h"

using namespace aocl_utils;

#define MANUAL_VECTOR      8
#define NUM_THREADS_PER_WG 64
#define BLOOM_1            5
#define BLOOM_2            0x7FFFF
#define BLOOM_SIZE         14
#define docEndingTag       0xFFFFFFFF

typedef unsigned int  uint;
typedef unsigned long ulong;

// ---- tunable parameters ---------------------------------------------------
uint block_size      = 64;
uint repeat          = 100;
uint total_num_docs  = 256 * 1024;
uint total_doc_size  = 0;
uint total_doc_size_no_padding = 0;

// ---- Host buffers ----------------------------------------------------------
scoped_aligned_ptr<uint>  h_docWordFrequencies_dimm1;
scoped_aligned_ptr<uint>  h_docWordFrequencies_dimm2;
scoped_aligned_ptr<ulong> h_profileWeights;
scoped_aligned_ptr<ulong> h_docInfo;
scoped_aligned_ptr<uint>  h_isWordInProfileHash;
scoped_aligned_ptr<uint>  h_startingDocID;
scoped_aligned_ptr<uint>  h_numItemsPerThread;
scoped_aligned_ptr<ulong> h_profileScore;
scoped_aligned_ptr<uint>  h_docSizes;

static uint m_z = 1, m_w = 1;
static uint rand_desh() {
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return (m_z << 16) + m_w;
}
double sampleNormal() {
    double u = ((double)rand() / RAND_MAX) * 2 - 1;
    double v = ((double)rand() / RAND_MAX) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    return u * sqrt(-2 * log(r) / r);
}
#define DOC_LEN_SIGMA 100
#define AVG_DOC_LEN   350
uint get_doc_length() {
    int len = (int)(sampleNormal() * DOC_LEN_SIGMA + AVG_DOC_LEN);
    if (len < 10) len = 10;
    return (uint)len;
}

double getCurrentTimestamp() {
    timespec a;
    clock_gettime(CLOCK_MONOTONIC, &a);
    return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
}

void setupData() {
    h_startingDocID.reset(total_num_docs);
    h_numItemsPerThread.reset(total_num_docs);
    h_profileScore.reset(total_num_docs);
    h_docInfo.reset(total_num_docs);
    h_docSizes.reset(total_num_docs);

    total_doc_size = 0;
    total_doc_size_no_padding = 0;

    for (uint i = 0; i < total_num_docs; i++) {
        uint unpadded_size = get_doc_length();
        uint size = unpadded_size & ~(2 * block_size - 1);
        if (unpadded_size & (2 * block_size - 1)) size += 2 * block_size;
        h_startingDocID[i]      = total_doc_size / 2;
        h_numItemsPerThread[i]  = size / (2 * block_size);
        ulong start_line = total_doc_size / (2 * block_size);
        ulong end_line   = start_line + size / (2 * block_size) - 1;
        total_doc_size          += size;
        total_doc_size_no_padding += unpadded_size;
        h_docSizes[i]    = unpadded_size;
        h_profileScore[i] = (ulong)-1;
        h_docInfo[i]      = (start_line << 32) | end_line;
    }

    h_isWordInProfileHash.reset(1L << BLOOM_SIZE);
    h_docWordFrequencies_dimm1.reset(total_doc_size / 2);
    h_docWordFrequencies_dimm2.reset(total_doc_size / 2);

    printf("Creating Documents total_terms=%d (no_pad=%d)\n",
           total_doc_size, total_doc_size_no_padding);

    for (uint i = 0; i < total_doc_size / 2; i++) {
        h_docWordFrequencies_dimm1[i] = docEndingTag;
        h_docWordFrequencies_dimm2[i] = docEndingTag;
    }
    for (uint doci = 0; doci < total_num_docs; doci++) {
        uint start = h_startingDocID[doci];
        uint size  = h_docSizes[doci];
        for (uint i = 0; i < size / 2; i++) {
            h_docWordFrequencies_dimm1[start + i] =
                ((rand_desh() % ((1L << 24) - 1)) << 8) | ((rand_desh() % 254) + 1);
            h_docWordFrequencies_dimm2[start + i] =
                ((rand_desh() % ((1L << 24) - 1)) << 8) | ((rand_desh() % 254) + 1);
        }
        if (size % 2) {
            h_docWordFrequencies_dimm1[start + size / 2] =
                ((rand_desh() % ((1L << 24) - 1)) << 8) | ((rand_desh() % 254) + 1);
        }
    }

    h_profileWeights.reset(1L << 24);
    for (uint i = 0; i < (1L << BLOOM_SIZE); i++) h_isWordInProfileHash[i] = 0;
    printf("Creating Profile\n");
    for (uint i = 0; i < (1L << 24); i++) h_profileWeights[i] = 0;
    for (uint i = 0; i < 16384; i++) {
        uint entry = rand_desh() % (1 << 24);
        h_profileWeights[entry] = 10;
        uint hash1 = entry >> BLOOM_1;
        h_isWordInProfileHash[hash1 >> 5] |= 1 << (hash1 & 0x1f);
        uint hash2 = entry & BLOOM_2;
        h_isWordInProfileHash[hash2 >> 5] |= 1 << (hash2 & 0x1f);
    }
}

// ---- Device-callable multiply function ------------------------------------
KOKKOS_INLINE_FUNCTION
ulong mulfp(ulong weight, uint freq) {
    uint part1 = weight & 0xFFFFF;
    uint part2 = (weight >> 24) & 0xFFFF;
    return (ulong)(part1 * freq) + (((ulong)(part2 * freq)) << 24);
}

// ---- Reference CPU computation -------------------------------------------
void runOnCPU(const Kokkos::View<ulong*>::HostMirror &h_profileScore_result) {
    scoped_aligned_ptr<ulong> cpu_profileScore;
    cpu_profileScore.reset(total_num_docs);
    uint total = 0, falsies = 0;

    for (uint doci = 0; doci < total_num_docs; doci++) {
        cpu_profileScore[doci] = 0;
        uint start = h_startingDocID[doci];
        uint size  = h_docSizes[doci];

        for (uint i = 0; i < size / 2 + (size % 2); i++) {
            uint curr = h_docWordFrequencies_dimm1[start + i];
            uint freq = curr & 0xff, wid = curr >> 8;
            uint h1 = wid >> BLOOM_1, h2 = wid & BLOOM_2;
            bool in1 = (h_isWordInProfileHash[h1 >> 5] >> (h1 & 0x1f)) & 1;
            bool in2 = (h_isWordInProfileHash[h2 >> 5] >> (h2 & 0x1f)) & 1;
            if (in1 && in2) {
                total++;
                if (h_profileWeights[wid] == 0) falsies++;
                cpu_profileScore[doci] += h_profileWeights[wid] * (ulong)freq;
            }
        }
        for (uint i = 0; i < size / 2; i++) {
            uint curr = h_docWordFrequencies_dimm2[start + i];
            uint freq = curr & 0xff, wid = curr >> 8;
            uint h1 = wid >> BLOOM_1, h2 = wid & BLOOM_2;
            bool in1 = (h_isWordInProfileHash[h1 >> 5] >> (h1 & 0x1f)) & 1;
            bool in2 = (h_isWordInProfileHash[h2 >> 5] >> (h2 & 0x1f)) & 1;
            if (in1 && in2) {
                total++;
                if (h_profileWeights[wid] == 0) falsies++;
                cpu_profileScore[doci] += h_profileWeights[wid] * (ulong)freq;
            }
        }
    }
    printf("total_access=%d falsies=%d percentage=%f hit=%g\n",
           total, falsies, total * 1.0f / total_doc_size,
           (total - falsies) * 1.0f / total_doc_size);
    for (uint doci = 0; doci < total_num_docs; doci++) {
        if (cpu_profileScore[doci] != h_profileScore_result(doci)) {
            printf("FAILED: doc[%u] CPU=%lu Device=%lu\n",
                   doci, cpu_profileScore[doci], h_profileScore_result(doci));
            return;
        }
    }
    printf("Verification: PASS\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    Options options(argc, argv);
    if (options.has("n")) total_num_docs = options.get<uint>("n");
    printf("Total number of documents: %u\n", total_num_docs);
    if (options.has("p")) repeat = options.get<uint>("p");
    printf("Kernel execution count: %u\n", repeat);

    srand(2);
    printf("RAND_MAX: %d\n", RAND_MAX);
    printf("Allocating and setting up data\n");
    setupData();

    Kokkos::initialize(argc, argv);
    {
        // local_size = block_size / MANUAL_VECTOR  (64/8 = 8 threads per team)
        const uint local_size  = block_size / MANUAL_VECTOR;
        // global_size = total_doc_size/2 / MANUAL_VECTOR / local_size
        //             = total_doc_size / (2 * MANUAL_VECTOR * local_size)
        //             = total_doc_size / (2 * block_size)
        const size_t global_size = (size_t)total_doc_size / 2 / MANUAL_VECTOR / local_size;
        const size_t partial_size = total_doc_size / (2 * block_size);

        // ---- Device Views (copy-to) ----------------------------------------
        Kokkos::View<uint*>  d_docWF1("d_wf1", total_doc_size / 2);
        Kokkos::View<uint*>  d_docWF2("d_wf2", total_doc_size / 2);
        Kokkos::View<ulong*> d_profileWeights("d_pw", 1L << 24);
        Kokkos::View<uint*>  d_isWordInProfileHash("d_bloom", 1L << BLOOM_SIZE);
        Kokkos::View<ulong*> d_docInfo("d_docinfo", total_num_docs);
        Kokkos::View<uint*>  d_partialSums1("d_ps1", partial_size);
        Kokkos::View<uint*>  d_partialSums2("d_ps2", partial_size);
        Kokkos::View<ulong*> d_profileScore("d_score", total_num_docs);

        // host mirrors for copy-in
        {
            auto hm1 = Kokkos::create_mirror_view(d_docWF1);
            auto hm2 = Kokkos::create_mirror_view(d_docWF2);
            auto hpw = Kokkos::create_mirror_view(d_profileWeights);
            auto hbl = Kokkos::create_mirror_view(d_isWordInProfileHash);
            auto hdi = Kokkos::create_mirror_view(d_docInfo);

            for (uint i = 0; i < total_doc_size / 2; i++) {
                hm1(i) = h_docWordFrequencies_dimm1[i];
                hm2(i) = h_docWordFrequencies_dimm2[i];
            }
            for (uint i = 0; i < (1u << 24); i++)  hpw(i) = h_profileWeights[i];
            for (uint i = 0; i < (1u << BLOOM_SIZE); i++) hbl(i) = h_isWordInProfileHash[i];
            for (uint i = 0; i < total_num_docs; i++) hdi(i) = h_docInfo[i];

            Kokkos::deep_copy(d_docWF1, hm1);
            Kokkos::deep_copy(d_docWF2, hm2);
            Kokkos::deep_copy(d_profileWeights, hpw);
            Kokkos::deep_copy(d_isWordInProfileHash, hbl);
            Kokkos::deep_copy(d_docInfo, hdi);
        }

        const double start_time = getCurrentTimestamp();

        for (uint iter = 0; iter < repeat; iter++) {
            // ----------------------------------------------------------------
            // Kernel 1: flat parallel_for — each work item processes
            //   local_size × MANUAL_VECTOR words from dimm1 + dimm2 and
            //   accumulates one partial sum for the enclosing document block.
            // Replaces:  #pragma omp target teams num_teams(global_size)
            //                              thread_limit(local_size)
            // ----------------------------------------------------------------
            Kokkos::parallel_for("score_team", (int)global_size,
                KOKKOS_LAMBDA(int team_id) {
                    ulong team_total = 0;

                    for (int tid = 0; tid < (int)local_size; tid++) {
                        int gid = team_id * (int)local_size + tid;

                        // process MANUAL_VECTOR entries from dimm1
                        for (uint i = 0; i < MANUAL_VECTOR; i++) {
                            uint curr  = d_docWF1[gid * MANUAL_VECTOR + i];
                            uint freq  = curr & 0xff;
                            uint wid   = curr >> 8;
                            bool is_end = (curr == docEndingTag);
                            uint h1 = wid >> BLOOM_1;
                            uint h2 = wid & BLOOM_2;
                            bool ma = !is_end &&
                                ((d_isWordInProfileHash[h1 >> 5] >> (h1 & 0x1f)) & 0x1) &&
                                ((d_isWordInProfileHash[h2 >> 5] >> (h2 & 0x1f)) & 0x1);
                            if (ma) team_total += mulfp(d_profileWeights[wid], freq);
                        }

                        // process MANUAL_VECTOR entries from dimm2
                        for (uint i = 0; i < MANUAL_VECTOR; i++) {
                            uint curr  = d_docWF2[gid * MANUAL_VECTOR + i];
                            uint freq  = curr & 0xff;
                            uint wid   = curr >> 8;
                            bool is_end = (curr == docEndingTag);
                            uint h1 = wid >> BLOOM_1;
                            uint h2 = wid & BLOOM_2;
                            bool ma = !is_end &&
                                ((d_isWordInProfileHash[h1 >> 5] >> (h1 & 0x1f)) & 0x1) &&
                                ((d_isWordInProfileHash[h2 >> 5] >> (h2 & 0x1f)) & 0x1);
                            if (ma) team_total += mulfp(d_profileWeights[wid], freq);
                        }
                    }

                    d_partialSums1[team_id] = (uint)(team_total >> 32);
                    d_partialSums2[team_id] = (uint)(team_total & 0xFFFFFFFF);
                });
            Kokkos::fence();

            // ----------------------------------------------------------------
            // Kernel 2: reduce partial sums per document
            // Replaces:  #pragma omp target teams distribute parallel for
            // ----------------------------------------------------------------
            Kokkos::parallel_for("score_reduce", (int)total_num_docs,
                KOKKOS_LAMBDA(int gid) {
                    ulong info  = d_docInfo[gid];
                    unsigned start = (unsigned)(info >> 32);
                    unsigned end   = (unsigned)(info & 0xFFFFFFFF);
                    ulong total = 0;
                    for (unsigned i = start; i <= end; i++) {
                        ulong upper = d_partialSums1[i];
                        ulong lower = d_partialSums2[i];
                        total += (upper << 32) | lower;
                    }
                    d_profileScore[gid] = total;
                });
            Kokkos::fence();
        }

        const double end_time = getCurrentTimestamp();
        double elapsed = (end_time - start_time) / repeat;
        printf("======================================================\n");
        printf("Kernel Time = %f ms (averaged over %u times)\n", elapsed * 1000.0, repeat);
        printf("Throughput = %f\n", total_doc_size_no_padding / elapsed / 1.0e6f);

        // ---- Copy result back and verify ----------------------------------
        auto h_score_m = Kokkos::create_mirror_view(d_profileScore);
        Kokkos::deep_copy(h_score_m, d_profileScore);
        // Update host array for verification
        for (uint i = 0; i < total_num_docs; i++)
            h_profileScore[i] = h_score_m(i);

        printf("Done\n");
        runOnCPU(h_score_m);
    }
    Kokkos::finalize();
    return 0;
}
