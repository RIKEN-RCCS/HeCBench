/*
 * Port of hmm-omp to Kokkos.
 * Uses Kokkos::parallel_for for the Viterbi GPU kernel with ping-pong Views
 * for maxProb, and a 2D flattened View for the path table.
 */

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

static int initHMM(float *initProb, float *mtState, float *mtObs,
                   const int nState, const int nEmit)
{
    for (int i = 0; i < nState; i++) initProb[i] = (float)rand();
    float sum = 0.0f;
    for (int i = 0; i < nState; i++) sum += initProb[i];
    for (int i = 0; i < nState; i++) initProb[i] /= sum;

    for (int i = 0; i < nState; i++)
        for (int j = 0; j < nState; j++) {
            mtState[i*nState+j] = (float)rand();
            mtState[i*nState+j] /= RAND_MAX;
        }

    for (int i = 0; i < nEmit; i++)
        for (int j = 0; j < nState; j++)
            mtObs[i*nState+j] = (float)rand();

    for (int j = 0; j < nState; j++) {
        float s = 0.0f;
        for (int i = 0; i < nEmit; i++) s += mtObs[i*nState+j];
        for (int i = 0; i < nEmit; i++) mtObs[i*nState+j] /= s;
    }
    return 1;
}

// Pure CPU reference implementation
static int ViterbiCPU(float &viterbiProb, int *viterbiPath, int *obs,
                      const int nObs, float *initProb, float *mtState,
                      const int nState, float *mtEmit)
{
    std::vector<float> maxProbOld(initProb, initProb + nState);
    std::vector<float> maxProbNew(nState);
    std::vector<std::vector<int>> path(nObs-1, std::vector<int>(nState));

    for (int t = 1; t < nObs; t++) {
        for (int iState = 0; iState < nState; iState++) {
            float maxProb = 0.0f;
            int maxState = -1;
            for (int preState = 0; preState < nState; preState++) {
                float p = maxProbOld[preState] + mtState[iState*nState + preState];
                if (p > maxProb) { maxProb = p; maxState = preState; }
            }
            maxProbNew[iState] = maxProb + mtEmit[obs[t]*nState + iState];
            path[t-1][iState] = maxState;
        }
        std::swap(maxProbOld, maxProbNew);
    }

    float maxProb = 0.0f;
    int maxState = -1;
    for (int i = 0; i < nState; i++)
        if (maxProbOld[i] > maxProb) { maxProb = maxProbOld[i]; maxState = i; }
    viterbiProb = maxProb;
    viterbiPath[nObs-1] = maxState;
    for (int t = nObs-2; t >= 0; t--)
        viterbiPath[t] = path[t][viterbiPath[t+1]];
    return 1;
}

// Kokkos-based Viterbi GPU kernel
static int ViterbiGPU(float &viterbiProb, int *viterbiPath, int *obs,
                      const int nObs, const int nState, const int nEmit,
                      float *h_initProb, float *h_mtState, float *h_mtEmit)
{
    // Device views
    Kokkos::View<float*> d_mtState("d_mtState", (size_t)nState * nState);
    Kokkos::View<float*> d_mtEmit ("d_mtEmit",  (size_t)nEmit  * nState);
    Kokkos::View<int*>   d_obs    ("d_obs",      nObs);
    Kokkos::View<float*> d_probA  ("d_probA",    nState);  // ping
    Kokkos::View<float*> d_probB  ("d_probB",    nState);  // pong
    Kokkos::View<int*>   d_path   ("d_path",     (size_t)(nObs-1) * nState);

    // Host mirrors for bulk upload
    {
        auto hm = Kokkos::create_mirror_view(d_mtState);
        for (int i = 0; i < nState*nState; i++) hm(i) = h_mtState[i];
        Kokkos::deep_copy(d_mtState, hm);
    }
    {
        auto hm = Kokkos::create_mirror_view(d_mtEmit);
        for (int i = 0; i < nEmit*nState; i++) hm(i) = h_mtEmit[i];
        Kokkos::deep_copy(d_mtEmit, hm);
    }
    {
        auto hm = Kokkos::create_mirror_view(d_obs);
        for (int i = 0; i < nObs; i++) hm(i) = obs[i];
        Kokkos::deep_copy(d_obs, hm);
    }
    {
        auto hm = Kokkos::create_mirror_view(d_probA);
        for (int i = 0; i < nState; i++) hm(i) = h_initProb[i];
        Kokkos::deep_copy(d_probA, hm);
    }

    auto start = std::chrono::steady_clock::now();

    // old = d_probA, new = d_probB; swap each step
    auto old_prob = d_probA;
    auto new_prob = d_probB;

    for (int t = 1; t < nObs; t++) {
        // Capture current Views and t by value
        auto cur_old = old_prob;
        auto cur_new = new_prob;
        auto cur_path = d_path;
        auto cur_mtState = d_mtState;
        auto cur_mtEmit  = d_mtEmit;
        auto cur_obs     = d_obs;
        const int ns = nState;
        const int step = t;

        Kokkos::parallel_for("viterbi_step", ns,
            KOKKOS_LAMBDA(int iState) {
                float maxProb = 0.0f;
                int   maxState = -1;
                for (int pre = 0; pre < ns; pre++) {
                    float p = cur_old[pre] + cur_mtState[iState*ns + pre];
                    if (p > maxProb) { maxProb = p; maxState = pre; }
                }
                cur_new[iState] = maxProb + cur_mtEmit[cur_obs[step]*ns + iState];
                cur_path[(step-1)*ns + iState] = maxState;
            });

        std::swap(old_prob, new_prob);
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    printf("Device execution time of Viterbi iterations %f (s)\n",
           std::chrono::duration<double>(end - start).count());

    // The last written probabilities are in old_prob after final swap
    auto h_maxProb = Kokkos::create_mirror_view(old_prob);
    auto h_path    = Kokkos::create_mirror_view(d_path);
    Kokkos::deep_copy(h_maxProb, old_prob);
    Kokkos::deep_copy(h_path, d_path);

    // Find final max-probability state
    float maxProb = 0.0f;
    int maxState = -1;
    for (int i = 0; i < nState; i++)
        if (h_maxProb(i) > maxProb) { maxProb = h_maxProb(i); maxState = i; }
    viterbiProb = maxProb;

    // Backtrace
    viterbiPath[nObs-1] = maxState;
    for (int t = nObs-2; t >= 0; t--)
        viterbiPath[t] = h_path(t*nState + viterbiPath[t+1]);

    return 1;
}

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        const int nState = 4096;
        const int nEmit  = 4096;
        const int nObs   = 500;

        std::vector<float> initProb(nState);
        std::vector<float> mtState((size_t)nState * nState);
        std::vector<float> mtEmit ((size_t)nEmit  * nState);
        initHMM(initProb.data(), mtState.data(), mtEmit.data(), nState, nEmit);

        std::vector<int> obs(nObs);
        for (int i = 0; i < nObs; i++) obs[i] = i % 15;

        std::vector<int> viterbiPathCPU(nObs), viterbiPathGPU(nObs);
        float viterbiProbCPU = 0.0f, viterbiProbGPU = 0.0f;

        printf("# of states = %d\n# of possible observations = %d\n"
               "Size of observational sequence = %d\n\n", nState, nEmit, nObs);

        printf("\nCompute Viterbi path on GPU (Kokkos)\n");
        ViterbiGPU(viterbiProbGPU, viterbiPathGPU.data(), obs.data(),
                   nObs, nState, nEmit,
                   initProb.data(), mtState.data(), mtEmit.data());

        printf("\nCompute Viterbi path on CPU\n");
        ViterbiCPU(viterbiProbCPU, viterbiPathCPU.data(), obs.data(),
                   nObs, initProb.data(), mtState.data(), nState, mtEmit.data());

        bool pass = true;
        for (int i = 0; i < nObs; i++)
            if (viterbiPathCPU[i] != viterbiPathGPU[i]) { pass = false; break; }
        printf("%s\n", pass ? "Success" : "Fail");
    }
    Kokkos::finalize();
    return 0;
}
