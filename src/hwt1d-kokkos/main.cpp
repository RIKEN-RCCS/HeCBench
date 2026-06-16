/*
 * Port of hwt1d-omp to Kokkos.
 * Uses Kokkos::TeamPolicy with team-level scratch memory to replace the
 * OpenMP target teams + shared memory approach.
 *
 * Usage: ./main <signal_length> <repeat>
 *   signal_length : must be a power of 2 (e.g. 1048576)
 *   repeat        : number of benchmark iterations
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>

// ── Helper: rounds up to next power of 2 ─────────────────────────────────────
template<typename T>
T roundToPowerOf2(T val)
{
    int bytes = sizeof(T);
    val--;
    for (int i = 0; i < bytes; i++) val |= val >> (1 << i);
    return val + 1;
}

// ── getLevels: returns log2 of length ────────────────────────────────────────
static int getLevels(unsigned int length, unsigned int *levels)
{
    for (unsigned int i = 0; i < 24; ++i) {
        if (length == (1U << i)) { *levels = i; return 0; }
    }
    return 1;  // not a recognised power of 2 (or > 2^23)
}

// ── CPU reference: full Haar DWT ──────────────────────────────────────────────
static void calApproxFinalOnHost(float *inData, float *hOutData,
                                 unsigned int signalLength)
{
    std::vector<float> tmp(inData, inData + signalLength);

    for (unsigned int i = 0; i < signalLength; i++)
        tmp[i] /= sqrtf((float)signalLength);

    unsigned int length = signalLength;
    while (length > 1u) {
        for (unsigned int i = 0; i < length / 2; i++) {
            float d0 = tmp[2*i], d1 = tmp[2*i+1];
            hOutData[i]            = (d0 + d1) / sqrtf(2.f);
            hOutData[length/2 + i] = (d0 - d1) / sqrtf(2.f);
        }
        memcpy(tmp.data(), hOutData, signalLength * sizeof(float));
        length >>= 1;
    }
}

// ── Scratch memory types ──────────────────────────────────────────────────────
using ExeSpace   = Kokkos::DefaultExecutionSpace;
using TeamPolicy = Kokkos::TeamPolicy<ExeSpace>;
using MemberType = TeamPolicy::member_type;
using ScratchSpace = ExeSpace::scratch_memory_space;
using ScratchView  = Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged>;

// ── One pass of the device Haar kernel ───────────────────────────────────────
static void runKernelPass(Kokkos::View<float*>& d_inData,
                          Kokkos::View<float*>& d_dOutData,
                          Kokkos::View<float*>& d_dPartialOutData,
                          unsigned int curSignalLength,
                          unsigned int groupSize,
                          unsigned int curLevels,
                          unsigned int totalLevels,
                          int          levelsDone)
{
    const int teams = (int)((curSignalLength >> 1) / groupSize);

    // We always allocate 512 floats of scratch (matches the original lmem[512])
    const int scratch_size = ScratchView::shmem_size(512);
    auto policy = TeamPolicy(teams, (int)groupSize)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    const unsigned int csl   = curSignalLength;
    const unsigned int tl    = totalLevels;
    const unsigned int cl    = curLevels;
    const int          ld    = levelsDone;
    const int          maxLD = 9;

    Kokkos::parallel_for("hwt1d_pass", policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            ScratchView lmem(team.team_scratch(0), 512);

            const int localId  = team.team_rank();
            const int groupId  = team.league_rank();
            const int localSize = team.team_size();

            // Load two input elements per thread
            float t0 = d_inData[groupId * localSize * 2 + localId];
            float t1 = d_inData[groupId * localSize * 2 + localSize + localId];

            if (ld == 0) {
                float r = 1.f / sqrtf((float)csl);
                t0 *= r;  t1 *= r;
            }
            lmem[localId]           = t0;
            lmem[localSize + localId] = t1;

            team.team_barrier();

            unsigned int levels       = (tl > (unsigned)maxLD) ? (unsigned)maxLD : tl;
            unsigned int activeThreads = (1u << levels) / 2u;
            unsigned int midOutPos    = csl / 2u;
            const float  rsqrt2       = 0.7071067811865475f;

            for (unsigned int i = 0; i < levels; ++i) {
                float data0 = 0.f, data1 = 0.f;
                if ((unsigned)localId < activeThreads) {
                    data0 = lmem[2 * localId];
                    data1 = lmem[2 * localId + 1];
                }
                team.team_barrier();

                if ((unsigned)localId < activeThreads) {
                    lmem[localId] = (data0 + data1) * rsqrt2;
                    unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
                    d_dOutData[globalPos] = (data0 - data1) * rsqrt2;
                    midOutPos >>= 1;
                }
                activeThreads >>= 1;
                team.team_barrier();
            }

            if (localId == 0)
                d_dPartialOutData[groupId] = lmem[0];
        });

    Kokkos::fence();
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <signal_length> <repeat>\n";
        return 1;
    }
    unsigned int signalLength = (unsigned int)atoi(argv[1]);
    const int    iterations   = atoi(argv[2]);

    signalLength = roundToPowerOf2<unsigned int>(signalLength);

    unsigned int actualLevels = 0;
    if (getLevels(signalLength, &actualLevels) != 0) {
        std::cerr << "signalLength > 2^23 not supported\n";
        return 1;
    }

    // Host input signal
    std::vector<float> inData(signalLength);
    srand(2);
    for (unsigned int i = 0; i < signalLength; i++)
        inData[i] = (float)(rand() % 10);

    std::vector<float> hOutData(signalLength, 0.f);

    // Device Views
    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<float*> d_inData        ("d_inData",         signalLength);
        Kokkos::View<float*> d_dOutData      ("d_dOutData",       signalLength);
        Kokkos::View<float*> d_dPartialOutData("d_dPartialOutData", signalLength);

        // Host mirrors for output
        auto h_dOutData       = Kokkos::create_mirror_view(d_dOutData);
        auto h_dPartialOutData= Kokkos::create_mirror_view(d_dPartialOutData);

        Kokkos::deep_copy(d_dOutData,        0.f);
        Kokkos::deep_copy(d_dPartialOutData, 0.f);

        std::cout << "Executing kernel for " << iterations << " iterations\n";
        std::cout << "-------------------------------------------\n";

        // Keep original inData for re-loading between benchmark iterations
        std::vector<float> inDataOrig = inData;

        Kokkos::fence();
        auto t0 = std::chrono::steady_clock::now();

        for (int rep = 0; rep < iterations; rep++) {
            // Reset
            inData = inDataOrig;

            unsigned int levels = actualLevels;
            int levelsDone  = 0;
            int one         = 1;

            while ((unsigned int)levelsDone < actualLevels) {
                const int maxLD = 9;
                unsigned int curLevels =
                    (levels < (unsigned)maxLD) ? levels : (unsigned)maxLD;

                unsigned int curSignalLength =
                    (levelsDone == 0) ? signalLength : (unsigned)(one << levels);

                unsigned int groupSize = (1u << curLevels) / 2u;

                // Upload current inData slice
                {
                    auto hm = Kokkos::create_mirror_view(d_inData);
                    for (unsigned int i = 0; i < signalLength; i++) hm(i) = inData[i];
                    Kokkos::deep_copy(d_inData, hm);
                }

                runKernelPass(d_inData, d_dOutData, d_dPartialOutData,
                              curSignalLength, groupSize, curLevels, levels, levelsDone);

                Kokkos::deep_copy(h_dOutData,        d_dOutData);
                Kokkos::deep_copy(h_dPartialOutData, d_dPartialOutData);

                // Copy host mirrors back to std::vectors for host processing
                std::vector<float> dOutData(signalLength);
                std::vector<float> dPartialOutData(signalLength);
                for (unsigned int i = 0; i < signalLength; i++) {
                    dOutData[i]        = h_dOutData(i);
                    dPartialOutData[i] = h_dPartialOutData(i);
                }

                if (levels <= (unsigned)maxLD) {
                    dOutData[0] = dPartialOutData[0];
                    memcpy(hOutData.data(), dOutData.data(),
                           (size_t)(one << curLevels) * sizeof(float));
                    memcpy(hOutData.data() + (one << curLevels),
                           dOutData.data() + (one << curLevels),
                           (signalLength - (one << curLevels)) * sizeof(float));
                    break;
                } else {
                    levels -= (unsigned)maxLD;
                    memcpy(hOutData.data(), dOutData.data(),
                           curSignalLength * sizeof(float));
                    memcpy(inData.data(), dPartialOutData.data(),
                           (size_t)(one << levels) * sizeof(float));
                    levelsDone += maxLD;
                }
            }
        }

        Kokkos::fence();
        double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "Average device offload time "
                  << elapsed / iterations << " (s)\n";

        // Verify against CPU reference
        calApproxFinalOnHost(inDataOrig.data(), hOutData.data(), signalLength);

        // Re-run a single pass to get device output for comparison
        {
            inData = inDataOrig;
            unsigned int levels = actualLevels;
            int levelsDone = 0;
            int one = 1;

            std::vector<float> dOutFinal(signalLength, 0.f);
            Kokkos::deep_copy(d_dOutData,        0.f);
            Kokkos::deep_copy(d_dPartialOutData, 0.f);

            while ((unsigned int)levelsDone < actualLevels) {
                const int maxLD = 9;
                unsigned int curLevels =
                    (levels < (unsigned)maxLD) ? levels : (unsigned)maxLD;
                unsigned int curSignalLength =
                    (levelsDone == 0) ? signalLength : (unsigned)(one << levels);
                unsigned int groupSize = (1u << curLevels) / 2u;

                {
                    auto hm = Kokkos::create_mirror_view(d_inData);
                    for (unsigned int i = 0; i < signalLength; i++) hm(i) = inData[i];
                    Kokkos::deep_copy(d_inData, hm);
                }

                runKernelPass(d_inData, d_dOutData, d_dPartialOutData,
                              curSignalLength, groupSize, curLevels, levels, levelsDone);
                Kokkos::deep_copy(h_dOutData,        d_dOutData);
                Kokkos::deep_copy(h_dPartialOutData, d_dPartialOutData);

                std::vector<float> dOut(signalLength), dPart(signalLength);
                for (unsigned int i = 0; i < signalLength; i++) {
                    dOut[i]  = h_dOutData(i);
                    dPart[i] = h_dPartialOutData(i);
                }

                if (levels <= (unsigned)maxLD) {
                    dOut[0] = dPart[0];
                    dOutFinal = dOut;
                    break;
                } else {
                    levels -= (unsigned)maxLD;
                    memcpy(inData.data(), dPart.data(),
                           (size_t)(one << levels) * sizeof(float));
                    levelsDone += maxLD;
                }
            }

            bool ok = true;
            for (unsigned int i = 0; i < signalLength; i++) {
                if (fabsf(dOutFinal[i] - hOutData[i]) > 0.1f) { ok = false; break; }
            }
            std::cout << (ok ? "PASS" : "FAIL") << "\n";
        }
    }
    Kokkos::finalize();
    return 0;
}
