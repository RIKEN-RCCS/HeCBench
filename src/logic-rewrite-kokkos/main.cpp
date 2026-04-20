// Kokkos port of logic-rewrite-cuda: AIG (And-Inverter Graph) rewriting.
// Original CUDA sources: rewrite.cu, rewrite.h, common.h
//
// The core parallel operation is cut enumeration: for each AND node at the
// current level, enumerate all feasible 4-input cuts from its two fanin cut
// sets.  Nodes at the same level are independent and can be processed in
// parallel.
//
// Key CUDA→Kokkos mapping:
//   __device__ function      → KOKKOS_INLINE_FUNCTION
//   __global__ kernel        → Kokkos::parallel_for
//   cudaMalloc/cudaMemcpy    → Kokkos::View / Kokkos::deep_copy
//   __managed__ int N        → plain int on host, passed into functor

#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

// -------------------------------------------------------------------------
// Constants (from rewrite.h / rewrite.cu)
// -------------------------------------------------------------------------
static constexpr int CUT_SET_SIZE = 8;   // max cuts stored per node
static constexpr int MAX_LEAVES   = 4;   // max leaves per cut

#define ID(i, j) ((i) * CUT_SET_SIZE + (j))

// -------------------------------------------------------------------------
// Data structures
// -------------------------------------------------------------------------
struct Cut {
    int  sign;
    int  leaves[MAX_LEAVES];
    unsigned truthtable : 16;
    unsigned value      : 11;
    unsigned nLeaves    : 4;
    bool     used;
};

// -------------------------------------------------------------------------
// Device helper functions (all must be KOKKOS_INLINE_FUNCTION)
// -------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
int CutFindValue(const Cut* cut, const int* nRef) {
    int value = 0, nOnes = 0;
    for (int i = 0; i < (int)cut->nLeaves; i++) {
        value += nRef[cut->leaves[i]];
        nOnes += (nRef[cut->leaves[i]] == 1);
    }
    if (cut->nLeaves < 2) return 1001;
    if (value > 1000) value = 1000;
    if (nOnes > 3) value = 5 - nOnes;
    return value;
}

KOKKOS_INLINE_FUNCTION
int CountOnes(unsigned uWord) {
    uWord = (uWord & 0x55555555u) + ((uWord >> 1) & 0x55555555u);
    uWord = (uWord & 0x33333333u) + ((uWord >> 2) & 0x33333333u);
    uWord = (uWord & 0x0F0F0F0Fu) + ((uWord >> 4) & 0x0F0F0F0Fu);
    uWord = (uWord & 0x00FF00FFu) + ((uWord >> 8) & 0x00FF00FFu);
    return (int)((uWord & 0x0000FFFFu) + (uWord >> 16));
}

// Find a slot in the cut set for a new cut (evict least-valuable if full).
KOKKOS_INLINE_FUNCTION
int FindCut(int idx, Cut* cuts) {
    for (int i = 0; i < CUT_SET_SIZE; i++)
        if (!cuts[ID(idx, i)].used) return i;
    int ans = -1;
    for (int i = 0; i < CUT_SET_SIZE; i++)
        if (cuts[ID(idx, i)].nLeaves > 2) {
            if (ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                ans = i;
        }
    if (ans == -1)
        for (int i = 0; i < CUT_SET_SIZE; i++)
            if (cuts[ID(idx, i)].nLeaves == 2) {
                if (ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    if (ans == -1)
        for (int i = 0; i < CUT_SET_SIZE; i++)
            if (cuts[ID(idx, i)].nLeaves < 2) {
                if (ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    cuts[ID(idx, ans)].used = false;
    return ans;
}

// Merge two sorted leaf-sets; return 1 on success, 0 if merged size > MAX_LEAVES.
KOKKOS_INLINE_FUNCTION
int MergeCutOrdered(Cut a, Cut b, Cut* out) {
    int c = 0, i = 0, k = 0;
    for (c = 0; c < MAX_LEAVES; c++) {
        if (k == (int)b.nLeaves) {
            if (i == (int)a.nLeaves) { out->nLeaves = c; return 1; }
            out->leaves[c] = a.leaves[i++]; continue;
        }
        if (i == (int)a.nLeaves) {
            out->leaves[c] = b.leaves[k++]; continue;
        }
        if (a.leaves[i] < b.leaves[k]) { out->leaves[c] = a.leaves[i++]; continue; }
        if (a.leaves[i] > b.leaves[k]) { out->leaves[c] = b.leaves[k++]; continue; }
        out->leaves[c] = a.leaves[i++]; k++;
    }
    if (i < (int)a.nLeaves || k < (int)b.nLeaves) return 0;
    out->nLeaves = c;
    return 1;
}

KOKKOS_INLINE_FUNCTION
int MergeCut(Cut a, Cut b, Cut* out) {
    if (a.nLeaves >= b.nLeaves) {
        if (!MergeCutOrdered(a, b, out)) return 0;
    } else {
        if (!MergeCutOrdered(b, a, out)) return 0;
    }
    out->sign = a.sign | b.sign;
    out->used = true;
    return 1;
}

// Return 1 if all leaves of a are present in b (a dominates b).
KOKKOS_INLINE_FUNCTION
int CutDominance(Cut a, Cut b) {
    for (int i = 0; i < (int)a.nLeaves; i++) {
        int ok = 0;
        for (int j = 0; j < (int)b.nLeaves; j++)
            if (b.leaves[j] == a.leaves[i]) ok = 1;
        if (!ok) return 0;
    }
    return 1;
}

// Remove dominated cuts; return 1 if cut[id] itself is dominated.
KOKKOS_INLINE_FUNCTION
int CutFilter(Cut* cut, int id) {
    for (int i = 0; i < CUT_SET_SIZE; i++) if (i != id && cut[i].used) {
        if (cut[i].nLeaves > cut[id].nLeaves) {
            if ((cut[i].sign & cut[id].sign) != cut[id].sign) continue;
            if (CutDominance(cut[id], cut[i])) cut[i].used = false;
        } else {
            if ((cut[i].sign & cut[id].sign) != cut[i].sign) continue;
            if (CutDominance(cut[i], cut[id])) { cut[id].used = false; return 1; }
        }
    }
    return 0;
}

KOKKOS_INLINE_FUNCTION
int CutTruthPhase(Cut x, Cut cut) {
    int phase = 0;
    for (int i = 0; i < (int)cut.nLeaves; i++)
        for (int j = 0; j < (int)x.nLeaves; j++)
            if (x.leaves[j] == cut.leaves[i]) phase |= 1 << i;
    return phase;
}

KOKKOS_INLINE_FUNCTION
int CutTruthSwapAdjacentVars(int uTruth, int iVar) {
    if (iVar == 0) return (uTruth & 0x99999999) | ((uTruth & 0x22222222) << 1) | ((uTruth & 0x44444444) >> 1);
    if (iVar == 1) return (uTruth & 0xC3C3C3C3) | ((uTruth & 0x0C0C0C0C) << 2) | ((uTruth & 0x30303030) >> 2);
    if (iVar == 2) return (uTruth & 0xF00FF00F) | ((uTruth & 0x00F000F0) << 4) | ((uTruth & 0x0F000F00) >> 4);
    return 0;
}

KOKKOS_INLINE_FUNCTION
int CutTruthStretch(int tt, int nVar, int phase) {
    int Var = nVar - 1;
    for (int i = 3; i >= 0; i--) if (phase >> i & 1) {
        for (int k = Var; k < i; k++) tt = CutTruthSwapAdjacentVars(tt, k);
        Var--;
    }
    return tt;
}

KOKKOS_INLINE_FUNCTION
int CutTruthtable(Cut cut, Cut a, Cut b, int aC, int bC) {
    int tt0 = aC ? ~a.truthtable : a.truthtable;
    int tt1 = bC ? ~b.truthtable : b.truthtable;
    tt0 = CutTruthStretch(tt0, a.nLeaves, CutTruthPhase(a, cut));
    tt1 = CutTruthStretch(tt1, b.nLeaves, CutTruthPhase(b, cut));
    return tt0 & tt1;
}

KOKKOS_INLINE_FUNCTION
int CutTruthShrink(int uTruth, int nVars, int Phase) {
    int Var = 0;
    for (int i = 0; i < 4; i++) if (Phase & (1 << i)) {
        for (int k = i - 1; k >= Var; k--) uTruth = CutTruthSwapAdjacentVars(uTruth, k);
        Var++;
    }
    return uTruth;
}

KOKKOS_INLINE_FUNCTION
int MinimizeCutSupport(Cut* cut) {
    const int masks[4][2] = {
        {0x5555, 0xAAAA}, {0x3333, 0xCCCC}, {0x0F0F, 0xF0F0}, {0x00FF, 0xFF00}
    };
    int phase = 0, truth = cut->truthtable & 0xFFFF, nLeaves = cut->nLeaves;
    for (int i = 0; i < (int)cut->nLeaves; i++)
        if ((truth & masks[i][0]) == ((truth & masks[i][1]) >> (1 << i)))
            nLeaves--;
        else
            phase |= 1 << i;
    if (nLeaves == (int)cut->nLeaves) return 0;
    truth = CutTruthShrink(truth, cut->nLeaves, phase);
    cut->truthtable = truth & 0xFFFF;
    cut->sign  = 0;
    int k = 0;
    for (int i = 0; i < (int)cut->nLeaves; i++) if (phase >> i & 1) {
        cut->leaves[k++] = cut->leaves[i];
        cut->sign |= 1 << (31 & cut->leaves[i]);
    }
    cut->nLeaves = nLeaves;
    return 1;
}

// -------------------------------------------------------------------------
// Kokkos functor: assign trivial (single-leaf) cuts to primary inputs.
// -------------------------------------------------------------------------
struct InputsCutKernel {
    Kokkos::View<const int*> nRef;
    Kokkos::View<Cut*>       cuts;
    int nPIs;

    KOKKOS_INLINE_FUNCTION
    void operator()(int idx) const {
        if (idx < 1 || idx > nPIs) return;
        Cut* cut       = &cuts[ID(idx, 0)];
        cut->used      = true;
        cut->sign      = 1 << (idx & 31);
        cut->truthtable = 0xAAAA;
        cut->nLeaves   = 1;
        cut->leaves[0] = idx;
        cut->value     = (unsigned)CutFindValue(cut, nRef.data());
    }
};

// -------------------------------------------------------------------------
// Kokkos functor: enumerate cuts for one topological level of AND nodes.
// Mirrors CutEnumerate<<<...>>> kernel in rewrite.cu.
// -------------------------------------------------------------------------
struct CutEnumerateKernel {
    Kokkos::View<const int*> fanin0;
    Kokkos::View<const int*> fanin1;
    Kokkos::View<const int*> isC0;
    Kokkos::View<const int*> isC1;
    Kokkos::View<const int*> nRef;
    Kokkos::View<Cut*>       cuts;
    int delta;
    int n;      // # of nodes in this level

    KOKKOS_INLINE_FUNCTION
    void operator()(int tidx) const {
        if (tidx >= n) return;
        int idx = tidx + delta + 1;

        // Trivial (identity) cut for this node
        Cut* cut        = &cuts[ID(idx, 0)];
        cut->used       = true;
        cut->sign       = 1 << (idx & 31);
        cut->truthtable = 0xAAAA;
        cut->nLeaves    = 1;
        cut->leaves[0]  = idx;
        cut->value      = (unsigned)CutFindValue(cut, nRef.data());

        int in0 = fanin0[idx] >> 1;
        int in1 = fanin1[idx] >> 1;

        Cut* cutsPtr = cuts.data();

        // Pairwise merge of cuts from both fanins
        for (int i = 0; i < CUT_SET_SIZE; i++) {
            if (!cutsPtr[ID(in0, i)].used) continue;
            for (int j = 0; j < CUT_SET_SIZE; j++) {
                if (!cutsPtr[ID(in1, j)].used) continue;

                Cut a = cutsPtr[ID(in0, i)];
                Cut b = cutsPtr[ID(in1, j)];

                if (CountOnes((unsigned)(a.sign | b.sign)) > MAX_LEAVES) continue;

                int cutId = FindCut(idx, cutsPtr);
                Cut newCut;
                if (!MergeCut(a, b, &newCut)) continue;
                cutsPtr[ID(idx, cutId)] = newCut;

                if (CutFilter(&cutsPtr[ID(idx, 0)], cutId)) continue;

                cutsPtr[ID(idx, cutId)].truthtable = (unsigned)(
                    0xFFFF & CutTruthtable(cutsPtr[ID(idx, cutId)], a, b,
                                          isC0[idx], isC1[idx]));

                if (MinimizeCutSupport(&cutsPtr[ID(idx, cutId)]))
                    CutFilter(&cutsPtr[ID(idx, 0)], cutId);

                cutsPtr[ID(idx, cutId)].value =
                    (unsigned)CutFindValue(&cutsPtr[ID(idx, cutId)], nRef.data());

                if (cutsPtr[ID(idx, cutId)].nLeaves < 2) return;
            }
        }
    }
};

// -------------------------------------------------------------------------
// Build a random AIG: nPIs primary inputs + nNodes AND gates arranged in
// a balanced tree (each AND takes two earlier nodes as fanins).
// -------------------------------------------------------------------------
static void buildRandomAIG(int nPIs, int nNodes,
                            std::vector<int>& fanin0,
                            std::vector<int>& fanin1,
                            std::vector<int>& isC0,
                            std::vector<int>& isC1,
                            std::vector<int>& level,
                            std::vector<int>& nRef) {
    int nObjs = 1 + nPIs + nNodes; // const-1 + PIs + AND nodes
    fanin0.assign(nObjs, 0);
    fanin1.assign(nObjs, 0);
    isC0  .assign(nObjs, 0);
    isC1  .assign(nObjs, 0);
    level .assign(nObjs, 0);
    nRef  .assign(nObjs, 0);

    std::mt19937 rng(123);

    for (int i = 0; i < nNodes; i++) {
        int nodeId = 1 + nPIs + i;
        // Choose two distinct fanin IDs from already-created objects
        int maxFanin = nodeId - 1;
        std::uniform_int_distribution<int> fdist(1, maxFanin);
        int f0 = fdist(rng);
        int f1 = fdist(rng);
        if (f0 == f1) f1 = (f1 % maxFanin) + 1;

        fanin0[nodeId] = f0 * 2 + (rng() & 1);
        fanin1[nodeId] = f1 * 2 + (rng() & 1);
        isC0  [nodeId] = fanin0[nodeId] & 1;
        isC1  [nodeId] = fanin1[nodeId] & 1;
        level [nodeId] = 1 + std::max(level[f0], level[f1]);
        nRef  [f0]++;
        nRef  [f1]++;
    }
}

// -------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int nPIs   = 64;
        int nNodes = 1024;
        if (argc > 1) nNodes = atoi(argv[1]);

        printf("AIG rewrite cut enumeration: %d PIs, %d AND nodes\n",
               nPIs, nNodes);

        std::vector<int> fanin0, fanin1, isC0_h, isC1_h, level_h, nRef_h;
        buildRandomAIG(nPIs, nNodes, fanin0, fanin1, isC0_h, isC1_h,
                       level_h, nRef_h);

        int nObjs = (int)fanin0.size();

        // Gather level-order node lists
        int maxLevel = *std::max_element(level_h.begin(), level_h.end());
        std::vector<std::vector<int>> levelNodes(maxLevel + 1);
        for (int i = 1 + nPIs; i < nObjs; i++)
            levelNodes[level_h[i]].push_back(i);

        // Allocate device views
        Kokkos::View<int*>  d_fanin0("fanin0", nObjs);
        Kokkos::View<int*>  d_fanin1("fanin1", nObjs);
        Kokkos::View<int*>  d_isC0  ("isC0",   nObjs);
        Kokkos::View<int*>  d_isC1  ("isC1",   nObjs);
        Kokkos::View<int*>  d_nRef  ("nRef",   nObjs);
        Kokkos::View<Cut*>  d_cuts  ("cuts",   nObjs * CUT_SET_SIZE);

        // Upload
        {
            auto hm_f0   = Kokkos::create_mirror_view(d_fanin0);
            auto hm_f1   = Kokkos::create_mirror_view(d_fanin1);
            auto hm_c0   = Kokkos::create_mirror_view(d_isC0);
            auto hm_c1   = Kokkos::create_mirror_view(d_isC1);
            auto hm_ref  = Kokkos::create_mirror_view(d_nRef);
            for (int i = 0; i < nObjs; i++) {
                hm_f0(i)  = fanin0[i];
                hm_f1(i)  = fanin1[i];
                hm_c0(i)  = isC0_h[i];
                hm_c1(i)  = isC1_h[i];
                hm_ref(i) = nRef_h[i];
            }
            Kokkos::deep_copy(d_fanin0, hm_f0);
            Kokkos::deep_copy(d_fanin1, hm_f1);
            Kokkos::deep_copy(d_isC0,   hm_c0);
            Kokkos::deep_copy(d_isC1,   hm_c1);
            Kokkos::deep_copy(d_nRef,   hm_ref);
        }
        Kokkos::deep_copy(d_cuts, Cut{});

        // ---- Phase 1: Assign trivial cuts to PIs ----
        {
            InputsCutKernel k;
            k.nRef = d_nRef;
            k.cuts = d_cuts;
            k.nPIs = nPIs;
            Kokkos::parallel_for("InputsCuts",
                                 Kokkos::RangePolicy<>(1, nPIs + 1), k);
            Kokkos::fence();
        }

        // ---- Phase 2: Level-by-level cut enumeration ----
        printf("Enumerating cuts over %d levels ...\n", maxLevel);

        auto t0 = std::chrono::steady_clock::now();

        for (int lev = 1; lev <= maxLevel; lev++) {
            const std::vector<int>& nodes = levelNodes[lev];
            if (nodes.empty()) continue;

            // delta = index of first node at this level - 1
            int delta = nodes[0] - 1;
            int cnt   = (int)nodes.size();

            CutEnumerateKernel k;
            k.fanin0 = d_fanin0;
            k.fanin1 = d_fanin1;
            k.isC0   = d_isC0;
            k.isC1   = d_isC1;
            k.nRef   = d_nRef;
            k.cuts   = d_cuts;
            k.delta  = delta;
            k.n      = cnt;

            Kokkos::parallel_for("CutEnumerate",
                                 Kokkos::RangePolicy<>(0, cnt), k);
            Kokkos::fence();
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        printf("Cut enumeration time: %.6f (s)\n", elapsed);

        // Count total valid cuts as a correctness check
        auto hm_cuts = Kokkos::create_mirror_view_and_copy(
                           Kokkos::HostSpace{}, d_cuts);
        long long totalCuts = 0;
        for (int i = 0; i < nObjs; i++)
            for (int j = 0; j < CUT_SET_SIZE; j++)
                if (hm_cuts(ID(i, j)).used) ++totalCuts;
        printf("Total valid cuts: %lld\n", totalCuts);
    }
    Kokkos::finalize();
    return 0;
}
