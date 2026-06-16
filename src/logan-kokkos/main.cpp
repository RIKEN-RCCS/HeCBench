// Kokkos port of logan-cuda benchmark
// LOGAN: x-drop seed-and-extend alignment algorithm
// Original: A. Zeni, G. Guidi
#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>
#include <cstring>

#define NOW std::chrono::high_resolution_clock::now()

// ---------------------------------------------------------------------------
// Scoring scheme
// ---------------------------------------------------------------------------
struct ScoringSchemeL {
  int match_score;
  int mismatch_score;
  int gap_extend_score;
  int gap_open_score;

  ScoringSchemeL()
    : match_score(1), mismatch_score(-1), gap_extend_score(-1), gap_open_score(-1) {}

  ScoringSchemeL(int match, int mismatch, int gap_extend, int gap_open)
    : match_score(match), mismatch_score(mismatch),
      gap_extend_score(gap_extend), gap_open_score(gap_open) {}
};

KOKKOS_INLINE_FUNCTION int scoreGap(const ScoringSchemeL& me) {
  return me.gap_open_score;
}
KOKKOS_INLINE_FUNCTION int scoreMatch(const ScoringSchemeL& me) {
  return me.match_score;
}
KOKKOS_INLINE_FUNCTION int scoreMismatch(const ScoringSchemeL& me) {
  return me.mismatch_score;
}
inline int scoreGapExtend(const ScoringSchemeL& me) { return me.gap_extend_score; }
inline int scoreGapOpen  (const ScoringSchemeL& me) { return me.gap_open_score;   }
KOKKOS_INLINE_FUNCTION int score(const ScoringSchemeL& me, char h, char v) {
  return (h == v) ? me.match_score : me.mismatch_score;
}

// ---------------------------------------------------------------------------
// Seed
// ---------------------------------------------------------------------------
template<class T>
KOKKOS_INLINE_FUNCTION const T& max_logan(const T& a, const T& b) {
  return (a < b) ? b : a;
}
template<class T>
KOKKOS_INLINE_FUNCTION const T& min_logan(const T& a, const T& b) {
  return (a < b) ? a : b;
}

struct SeedL {
  int beginPositionH, beginPositionV;
  int endPositionH,   endPositionV;
  int seedLength;
  int lowerDiagonal, upperDiagonal;
  int beginDiagonal,  endDiagonal;
  int score;

  KOKKOS_INLINE_FUNCTION SeedL()
    : beginPositionH(0), beginPositionV(0),
      endPositionH(0),   endPositionV(0),
      lowerDiagonal(0),  upperDiagonal(0), score(0) {}

  KOKKOS_INLINE_FUNCTION SeedL(int bH, int bV, int sLen)
    : beginPositionH(bH), beginPositionV(bV),
      endPositionH(bH + sLen), endPositionV(bV + sLen),
      lowerDiagonal(bH - bV),  upperDiagonal(bH - bV),
      beginDiagonal(bH - bV),  endDiagonal((bH+sLen) - (bV+sLen)),
      score(0) {}

  KOKKOS_INLINE_FUNCTION SeedL(const SeedL& o)
    : beginPositionH(o.beginPositionH), beginPositionV(o.beginPositionV),
      endPositionH(o.endPositionH),     endPositionV(o.endPositionV),
      lowerDiagonal(o.lowerDiagonal),   upperDiagonal(o.upperDiagonal),
      beginDiagonal(o.beginDiagonal),   endDiagonal(o.endDiagonal),
      score(0) {}
};

KOKKOS_INLINE_FUNCTION int getBeginPositionH(const SeedL& s) { return s.beginPositionH; }
KOKKOS_INLINE_FUNCTION int getBeginPositionV(const SeedL& s) { return s.beginPositionV; }
KOKKOS_INLINE_FUNCTION int getEndPositionH  (const SeedL& s) { return s.endPositionH;   }
KOKKOS_INLINE_FUNCTION int getEndPositionV  (const SeedL& s) { return s.endPositionV;   }
KOKKOS_INLINE_FUNCTION int getLowerDiagonal (const SeedL& s) { return s.lowerDiagonal;  }
KOKKOS_INLINE_FUNCTION int getUpperDiagonal (const SeedL& s) { return s.upperDiagonal;  }
KOKKOS_INLINE_FUNCTION int getBeginDiagonal (const SeedL& s) { return s.beginDiagonal;  }
KOKKOS_INLINE_FUNCTION int getEndDiagonal   (const SeedL& s) { return s.endDiagonal;    }

KOKKOS_INLINE_FUNCTION void setBeginPositionH(SeedL& s, int v) { s.beginPositionH = v; }
KOKKOS_INLINE_FUNCTION void setBeginPositionV(SeedL& s, int v) { s.beginPositionV = v; }
KOKKOS_INLINE_FUNCTION void setEndPositionH  (SeedL& s, int v) { s.endPositionH   = v; }
KOKKOS_INLINE_FUNCTION void setEndPositionV  (SeedL& s, int v) { s.endPositionV   = v; }
KOKKOS_INLINE_FUNCTION void setLowerDiagonal (SeedL& s, int v) { s.lowerDiagonal  = v; }
KOKKOS_INLINE_FUNCTION void setUpperDiagonal (SeedL& s, int v) { s.upperDiagonal  = v; }
KOKKOS_INLINE_FUNCTION void setBeginDiagonal (SeedL& s, int v) { s.beginDiagonal  = v; }
KOKKOS_INLINE_FUNCTION void setEndDiagonal   (SeedL& s, int v) { s.endDiagonal    = v; }

// ---------------------------------------------------------------------------
// Extension direction
// ---------------------------------------------------------------------------
enum ExtensionDirectionL {
  EXTEND_NONEL  = 0,
  EXTEND_LEFTL  = 1,
  EXTEND_RIGHTL = 2,
  EXTEND_BOTHL  = 3
};

// DP scoring constants (linear gap)
static constexpr int MATCH    =  1;
static constexpr int MISMATCH = -1;
static constexpr int GAP_EXT  = -1;
static constexpr int GAP_OPEN = -1;
static constexpr short UNDEF  = -32767;
static constexpr short MIN_VAL= -32768;

// ---------------------------------------------------------------------------
// Update seed after extension
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void updateExtendedSeedL(SeedL& seed,
                                                  ExtensionDirectionL dir,
                                                  int cols, int rows,
                                                  int lowerDiag, int upperDiag) {
  if (dir == EXTEND_LEFTL) {
    int beginDiag = seed.beginDiagonal;
    if (getLowerDiagonal(seed) > beginDiag + lowerDiag)
      setLowerDiagonal(seed, beginDiag + lowerDiag);
    if (getUpperDiagonal(seed) < beginDiag + upperDiag)
      setUpperDiagonal(seed, beginDiag + upperDiag);
    seed.beginPositionH -= rows;
    seed.beginPositionV -= cols;
  } else {
    int endDiag = seed.endDiagonal;
    if (getUpperDiagonal(seed) < endDiag - lowerDiag)
      setUpperDiagonal(seed, endDiag - lowerDiag);
    if (getLowerDiagonal(seed) > endDiag - upperDiag)
      setLowerDiagonal(seed, endDiag - upperDiag);
    seed.endPositionH += rows;
    seed.endPositionV += cols;
  }
}

// ---------------------------------------------------------------------------
// Diagonal DP helpers
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void calcExtendedLowerDiag(int& lowerDiag,
                                                    int minCol, int antiDiagNo) {
  int minRow = antiDiagNo - minCol;
  if (minCol - minRow < lowerDiag) lowerDiag = minCol - minRow;
}

KOKKOS_INLINE_FUNCTION void calcExtendedUpperDiag(int& upperDiag,
                                                    int maxCol, int antiDiagNo) {
  int maxRow = antiDiagNo + 1 - maxCol;
  if (maxCol - 1 - maxRow > upperDiag) upperDiag = maxCol - 1 - maxRow;
}

KOKKOS_INLINE_FUNCTION void initAntiDiag3(short* ad3, int& a3size,
                                            int offset, int maxCol,
                                            int antiDiagNo, int minScore,
                                            int gapCost, short undef) {
  a3size = maxCol + 1 - offset;
  ad3[0] = undef;
  ad3[maxCol - offset] = undef;
  if (antiDiagNo * gapCost > minScore) {
    if (offset == 0)             ad3[0]             = (short)(antiDiagNo * gapCost);
    if (antiDiagNo - maxCol == 0) ad3[maxCol-offset] = (short)(antiDiagNo * gapCost);
  }
}

// Compute one full anti-diagonal (sequential version: no parallel threads)
KOKKOS_INLINE_FUNCTION void computeAntidiag_seq(const short* ad1,
                                                  const short* ad2,
                                                        short* ad3,
                                                  const char* querySeg,
                                                  const char* dbSeg,
                                                  int best, int scoreDropOff,
                                                  int cols, int rows,
                                                  int minCol, int maxCol,
                                                  int antiDiagNo,
                                                  int offset1, int offset2,
                                                  ExtensionDirectionL /*dir*/) {
  for (int col = minCol; col < maxCol; col++) {
    int queryPos = col - 1;
    int dbPos    = col + rows - antiDiagNo - 1;

    int tmp = max_logan((int)ad2[col-offset2],
                        (int)ad2[col-offset2-1]) + GAP_EXT;
    int sc  = (querySeg[queryPos] == dbSeg[dbPos]) ? MATCH : MISMATCH;
    tmp     = max_logan((int)ad1[col-offset1-1] + sc, tmp);

    ad3[col - minCol + 1] = (tmp < best - scoreDropOff) ? UNDEF : (short)tmp;
  }
}

// ---------------------------------------------------------------------------
// Single-alignment x-drop extension (one direction, sequential)
// Returns the extension score (UNDEF if nothing extended)
// Writes updated seed back.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION int extendOneDirection(SeedL& seed,
                                               const char* querySeg,
                                               const char* dbSeg,
                                               int qLen, int tLen,
                                               ExtensionDirectionL dir,
                                               int scoreDropOff,
                                               short* antidiag_buf) {
  // cols = query direction length + 1, rows = target direction length + 1
  int cols = qLen + 1;
  int rows = tLen + 1;
  if (rows <= 1 || cols <= 1) return UNDEF;

  // We use three rotating antidiag arrays, each of length (max(cols,rows)+2)
  const int ADBUF = cols + rows + 4;
  short* ad1 = antidiag_buf + 0*ADBUF;
  short* ad2 = antidiag_buf + 1*ADBUF;
  short* ad3 = antidiag_buf + 2*ADBUF;

  // init
  int a1size = 0, a2size = 1, a3size = 2;
  ad2[0] = 0;
  ad3[0] = (short)GAP_EXT;
  ad3[1] = (short)GAP_EXT;

  int antiDiagNo = 1;
  int best = 0;
  int lowerDiag = 0, upperDiag = 0;
  int minCol = 1, maxCol = 2;
  int offset1 = 0, offset2 = 0, offset3 = 0;

  while (minCol < maxCol) {
    ++antiDiagNo;

    // Rotate: ad2→ad1, ad3→ad2, ad1→ad3
    short* t = ad1; ad1 = ad2; ad2 = ad3; ad3 = t;
    int tl = a1size; a1size = a2size; a2size = a3size; a3size = tl;
    offset1 = offset2;
    offset2 = offset3;
    offset3 = minCol - 1;

    initAntiDiag3(ad3, a3size, offset3, maxCol, antiDiagNo,
                  best - scoreDropOff, GAP_EXT, UNDEF);

    // Compute the new antidiagonal sequentially
    computeAntidiag_seq(ad1, ad2, ad3, querySeg, dbSeg,
                        best, scoreDropOff, cols, rows,
                        minCol, maxCol, antiDiagNo,
                        offset1, offset2, dir);

    // Find best in ad3
    short antiDiagBest = UNDEF;
    for (int i = 0; i < a3size; i++)
      if (ad3[i] > antiDiagBest) antiDiagBest = ad3[i];
    if (antiDiagBest > best) best = antiDiagBest;

    // Advance minCol
    while (minCol - offset3 < a3size &&
           ad3[minCol - offset3] == UNDEF &&
           minCol - offset2 - 1 < a2size &&
           ad2[minCol - offset2 - 1] == UNDEF)
      ++minCol;

    // Retract maxCol
    while (maxCol - offset3 > 0 &&
           ad3[maxCol - offset3 - 1] == UNDEF &&
           ad2[maxCol - offset2 - 1] == UNDEF)
      --maxCol;
    ++maxCol;

    calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
    calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

    minCol = max_logan(minCol, antiDiagNo + 2 - rows);
    maxCol = min_logan(maxCol, cols);
  }

  // Determine longest extension
  int longestCol   = a3size + offset3 - 2;
  int longestRow   = antiDiagNo - longestCol;
  short longestScore = ad3[longestCol - offset3];

  if (longestScore == UNDEF) {
    if (ad2[a2size-2] != UNDEF) {
      longestCol   = a2size + offset2 - 2;
      longestRow   = antiDiagNo - 1 - longestCol;
      longestScore = ad2[longestCol - offset2];
    } else if (a2size > 2 && ad2[a2size-3] != UNDEF) {
      longestCol   = a2size + offset2 - 3;
      longestRow   = antiDiagNo - 1 - longestCol;
      longestScore = ad2[longestCol - offset2];
    }
  }

  if (longestScore == UNDEF) {
    for (int i = 0; i < a1size; i++) {
      if (ad1[i] > longestScore) {
        longestScore = ad1[i];
        longestCol   = i + offset1;
        longestRow   = antiDiagNo - 2 - longestCol;
      }
    }
  }

  if (longestScore != UNDEF)
    updateExtendedSeedL(seed, dir, longestCol, longestRow, lowerDiag, upperDiag);

  return (int)longestScore;
}

// ---------------------------------------------------------------------------
// Kokkos extendSeedL: process batch of alignments in parallel
// ---------------------------------------------------------------------------
void extendSeedL_kokkos(std::vector<SeedL>& seeds,
                         std::vector<std::string>& target,
                         std::vector<std::string>& query,
                         int XDrop, int kmer_length,
                         int* res, int numAlignments) {
  // ---- Compute prefix offsets for packed sequence arrays ----
  std::vector<int> offLeftQ(numAlignments), offLeftT(numAlignments);
  std::vector<int> offRightQ(numAlignments), offRightT(numAlignments);

  int antLenLeft = 0, antLenRight = 0;
  for (int j = 0; j < numAlignments; j++) {
    offLeftQ[j]  = getBeginPositionV(seeds[j]);
    offLeftT[j]  = getBeginPositionH(seeds[j]);
    offRightQ[j] = (int)query[j].size()  - getEndPositionV(seeds[j]);
    offRightT[j] = (int)target[j].size() - getEndPositionH(seeds[j]);
    antLenLeft  = std::max(antLenLeft,  std::min(offLeftQ[j],  offLeftT[j]));
    antLenRight = std::max(antLenRight, std::min(offRightQ[j], offRightT[j]));
  }

  // Prefix sums (cumulative lengths)
  std::vector<int> prefixLQ(numAlignments), prefixLT(numAlignments);
  std::vector<int> prefixRQ(numAlignments), prefixRT(numAlignments);
  std::partial_sum(offLeftQ.begin(),  offLeftQ.end(),  prefixLQ.begin());
  std::partial_sum(offLeftT.begin(),  offLeftT.end(),  prefixLT.begin());
  std::partial_sum(offRightQ.begin(), offRightQ.end(), prefixRQ.begin());
  std::partial_sum(offRightT.begin(), offRightT.end(), prefixRT.begin());

  int totalLQ = prefixLQ[numAlignments-1];
  int totalLT = prefixLT[numAlignments-1];
  int totalRQ = prefixRQ[numAlignments-1];
  int totalRT = prefixRT[numAlignments-1];

  // Pack sequence prefixes/suffixes into flat arrays
  std::vector<char> prefQ(totalLQ), prefT(totalLT);
  std::vector<char> suffQ(totalRQ), suffT(totalRT);

  // alignment 0
  std::reverse_copy(query[0].c_str(),  query[0].c_str()  + offLeftQ[0],  prefQ.data());
  std::memcpy(prefT.data(), target[0].c_str(), offLeftT[0]);
  std::memcpy(suffQ.data(), query[0].c_str()  + getEndPositionV(seeds[0]), offRightQ[0]);
  std::reverse_copy(target[0].c_str() + getEndPositionH(seeds[0]),
                    target[0].c_str() + getEndPositionH(seeds[0]) + offRightT[0],
                    suffT.data());

  for (int j = 1; j < numAlignments; j++) {
    std::reverse_copy(query[j].c_str(), query[j].c_str() + (prefixLQ[j]-prefixLQ[j-1]),
                      prefQ.data() + prefixLQ[j-1]);
    std::memcpy(prefT.data() + prefixLT[j-1], target[j].c_str(),
                prefixLT[j]-prefixLT[j-1]);
    std::memcpy(suffQ.data() + prefixRQ[j-1],
                query[j].c_str() + getEndPositionV(seeds[j]),
                prefixRQ[j]-prefixRQ[j-1]);
    std::reverse_copy(
      target[j].c_str() + getEndPositionH(seeds[j]),
      target[j].c_str() + getEndPositionH(seeds[j]) + (prefixRT[j]-prefixRT[j-1]),
      suffT.data() + prefixRT[j-1]);
  }

  // ---- Copy to Kokkos Views ----
  Kokkos::View<char*>  d_prefQ("prefQ", totalLQ > 0 ? totalLQ : 1);
  Kokkos::View<char*>  d_prefT("prefT", totalLT > 0 ? totalLT : 1);
  Kokkos::View<char*>  d_suffQ("suffQ", totalRQ > 0 ? totalRQ : 1);
  Kokkos::View<char*>  d_suffT("suffT", totalRT > 0 ? totalRT : 1);
  Kokkos::View<int*>   d_offLQ("offLQ", numAlignments);
  Kokkos::View<int*>   d_offLT("offLT", numAlignments);
  Kokkos::View<int*>   d_offRQ("offRQ", numAlignments);
  Kokkos::View<int*>   d_offRT("offRT", numAlignments);
  Kokkos::View<SeedL*> d_seeds_l("seeds_l", numAlignments);
  Kokkos::View<SeedL*> d_seeds_r("seeds_r", numAlignments);
  Kokkos::View<int*>   d_scL("scL", numAlignments);
  Kokkos::View<int*>   d_scR("scR", numAlignments);

  // Anti-diagonal scratch: each alignment needs 3 * (maxAntLen+4) shorts
  // Use separate buffers per alignment to avoid conflicts
  int adbuf_left  = antLenLeft  + 4;
  int adbuf_right = antLenRight + 4;
  // Worst-case: max of left/right offsets
  int adbuf_max_left  = 0, adbuf_max_right = 0;
  for (int j = 0; j < numAlignments; j++) {
    int bl = offLeftQ[j]  + offLeftT[j]  + 4;
    int br = offRightQ[j] + offRightT[j] + 4;
    if (bl > adbuf_max_left)  adbuf_max_left  = bl;
    if (br > adbuf_max_right) adbuf_max_right = br;
  }
  if (adbuf_max_left  == 0) adbuf_max_left  = 4;
  if (adbuf_max_right == 0) adbuf_max_right = 4;

  Kokkos::View<short*> d_ant_l("ant_l", (size_t)numAlignments * adbuf_max_left  * 3);
  Kokkos::View<short*> d_ant_r("ant_r", (size_t)numAlignments * adbuf_max_right * 3);

  // Host mirrors
  auto h_prefQ   = Kokkos::create_mirror_view(d_prefQ);
  auto h_prefT   = Kokkos::create_mirror_view(d_prefT);
  auto h_suffQ   = Kokkos::create_mirror_view(d_suffQ);
  auto h_suffT   = Kokkos::create_mirror_view(d_suffT);
  auto h_offLQ   = Kokkos::create_mirror_view(d_offLQ);
  auto h_offLT   = Kokkos::create_mirror_view(d_offLT);
  auto h_offRQ   = Kokkos::create_mirror_view(d_offRQ);
  auto h_offRT   = Kokkos::create_mirror_view(d_offRT);
  auto h_seeds_l = Kokkos::create_mirror_view(d_seeds_l);
  auto h_seeds_r = Kokkos::create_mirror_view(d_seeds_r);

  for (int i = 0; i < totalLQ; i++) h_prefQ(i) = prefQ[i];
  for (int i = 0; i < totalLT; i++) h_prefT(i) = prefT[i];
  for (int i = 0; i < totalRQ; i++) h_suffQ(i) = suffQ[i];
  for (int i = 0; i < totalRT; i++) h_suffT(i) = suffT[i];
  for (int j = 0; j < numAlignments; j++) {
    h_offLQ(j) = prefixLQ[j];
    h_offLT(j) = prefixLT[j];
    h_offRQ(j) = prefixRQ[j];
    h_offRT(j) = prefixRT[j];
    h_seeds_l(j) = seeds[j];
    h_seeds_r(j) = seeds[j];
  }

  Kokkos::deep_copy(d_prefQ,   h_prefQ);
  Kokkos::deep_copy(d_prefT,   h_prefT);
  Kokkos::deep_copy(d_suffQ,   h_suffQ);
  Kokkos::deep_copy(d_suffT,   h_suffT);
  Kokkos::deep_copy(d_offLQ,   h_offLQ);
  Kokkos::deep_copy(d_offLT,   h_offLT);
  Kokkos::deep_copy(d_offRQ,   h_offRQ);
  Kokkos::deep_copy(d_offRT,   h_offRT);
  Kokkos::deep_copy(d_seeds_l, h_seeds_l);
  Kokkos::deep_copy(d_seeds_r, h_seeds_r);

  auto start_c = NOW;

  // ---- LEFT extension kernel ----
  {
    char*  pQ   = d_prefQ.data();
    char*  pT   = d_prefT.data();
    int*   oLQ  = d_offLQ.data();
    int*   oLT  = d_offLT.data();
    SeedL* sL   = d_seeds_l.data();
    int*   scL  = d_scL.data();
    short* ant  = d_ant_l.data();
    int    abL  = adbuf_max_left;

    Kokkos::parallel_for("logan_left", numAlignments,
      KOKKOS_LAMBDA(const int myId) {
        const char* querySeg = (myId == 0) ? pQ : pQ + oLQ[myId-1];
        const char* dbSeg    = (myId == 0) ? pT : pT + oLT[myId-1];
        int qLen = (myId == 0) ? oLQ[myId] : oLQ[myId] - oLQ[myId-1];
        int tLen = (myId == 0) ? oLT[myId] : oLT[myId] - oLT[myId-1];

        short* buf = ant + (size_t)myId * abL * 3;
        scL[myId] = extendOneDirection(sL[myId], querySeg, dbSeg,
                                       qLen, tLen,
                                       EXTEND_LEFTL, XDrop, buf);
      });
    Kokkos::fence();
  }

  // ---- RIGHT extension kernel ----
  {
    char*  sQ   = d_suffQ.data();
    char*  sT   = d_suffT.data();
    int*   oRQ  = d_offRQ.data();
    int*   oRT  = d_offRT.data();
    SeedL* sR   = d_seeds_r.data();
    int*   scR  = d_scR.data();
    short* ant  = d_ant_r.data();
    int    abR  = adbuf_max_right;

    Kokkos::parallel_for("logan_right", numAlignments,
      KOKKOS_LAMBDA(const int myId) {
        const char* querySeg = (myId == 0) ? sQ : sQ + oRQ[myId-1];
        const char* dbSeg    = (myId == 0) ? sT : sT + oRT[myId-1];
        int qLen = (myId == 0) ? oRQ[myId] : oRQ[myId] - oRQ[myId-1];
        int tLen = (myId == 0) ? oRT[myId] : oRT[myId] - oRT[myId-1];

        short* buf = ant + (size_t)myId * abR * 3;
        scR[myId] = extendOneDirection(sR[myId], querySeg, dbSeg,
                                       qLen, tLen,
                                       EXTEND_RIGHTL, XDrop, buf);
      });
    Kokkos::fence();
  }

  auto end_c = NOW;
  std::chrono::duration<double> compute = end_c - start_c;
  std::cout << "Device only time [seconds]:\t" << compute.count() << std::endl;

  // ---- Copy results back ----
  auto h_scL    = Kokkos::create_mirror_view(d_scL);
  auto h_scR    = Kokkos::create_mirror_view(d_scR);
  auto h_sL_out = Kokkos::create_mirror_view(d_seeds_l);
  auto h_sR_out = Kokkos::create_mirror_view(d_seeds_r);
  Kokkos::deep_copy(h_scL,    d_scL);
  Kokkos::deep_copy(h_scR,    d_scR);
  Kokkos::deep_copy(h_sL_out, d_seeds_l);
  Kokkos::deep_copy(h_sR_out, d_seeds_r);

  for (int i = 0; i < numAlignments; i++) {
    int sl = h_scL(i), sr = h_scR(i);
    res[i] = (sl == UNDEF ? 0 : sl) + (sr == UNDEF ? 0 : sr) + kmer_length;
    setEndPositionH(seeds[i], getEndPositionH(h_sR_out(i)));
    setEndPositionV(seeds[i], getEndPositionV(h_sR_out(i)));
    std::cout << res[i] << "\n";
  }
}

// ---------------------------------------------------------------------------
// Top-level LOGAN entry point (matches original API)
// ---------------------------------------------------------------------------
std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) result.push_back(item);
  return result;
}

char basecomplement(char n) {
  switch (n) {
    case 'A': return 'T'; case 'T': return 'A';
    case 'G': return 'C'; case 'C': return 'G';
  }
  return 'N';
}

static constexpr int BATCH_SIZE = 30000;

void LOGAN(std::vector<std::vector<std::string>>& alignments,
           int ksize, int xdrop, int AlignmentsToBePerformed) {
  std::vector<int>    posV(AlignmentsToBePerformed);
  std::vector<int>    posH(AlignmentsToBePerformed);
  std::vector<SeedL>  seeds(AlignmentsToBePerformed);
  std::vector<std::string> seqsV(AlignmentsToBePerformed);
  std::vector<std::string> seqsH(AlignmentsToBePerformed);
  std::vector<ScoringSchemeL> penalties(AlignmentsToBePerformed);
  ScoringSchemeL sscheme(1, -1, -1, -1);

  for (int i = 0; i < AlignmentsToBePerformed; i++) {
    posV[i]    = std::stoi(alignments[i][1]);
    posH[i]    = std::stoi(alignments[i][3]);
    seqsV[i]   = alignments[i][0];
    seqsH[i]   = alignments[i][2];
    std::string strand = alignments[i][4];

    if (strand == "c") {
      std::transform(seqsH[i].begin(), seqsH[i].end(),
                     seqsH[i].begin(), basecomplement);
      posH[i] = (int)seqsH[i].size() - posH[i] - ksize;
    }
    penalties[i] = sscheme;
    seeds[i]     = SeedL(posH[i], posV[i], ksize);
  }

  int numLocal = BATCH_SIZE;
  for (int i = 0; i < AlignmentsToBePerformed; i += BATCH_SIZE) {
    if (AlignmentsToBePerformed < i + BATCH_SIZE)
      numLocal = AlignmentsToBePerformed - i;

    std::vector<std::string> tgt_b(seqsH.begin()+i, seqsH.begin()+i+numLocal);
    std::vector<std::string> qry_b(seqsV.begin()+i, seqsV.begin()+i+numLocal);
    std::vector<SeedL>       sds_b(seeds.begin()+i, seeds.begin()+i+numLocal);

    std::vector<int> resv(numLocal);
    extendSeedL_kokkos(sds_b, tgt_b, qry_b, xdrop, ksize,
                       resv.data(), numLocal);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 4) {
      std::cerr << "Usage: " << argv[0]
                << " <alignments_file> <kmer_size> <xdrop>\n";
      Kokkos::finalize();
      return 1;
    }

    std::ifstream input(argv[1]);
    if (!input.is_open()) {
      std::cerr << "Cannot open file: " << argv[1] << "\n";
      Kokkos::finalize();
      return 1;
    }

    int ksize = std::atoi(argv[2]);
    int xdrop = std::atoi(argv[3]);

    uint64_t AlignmentsToBePerformed =
      std::count(std::istreambuf_iterator<char>(input),
                 std::istreambuf_iterator<char>(), '\n');
    input.seekg(0, std::ios_base::beg);

    std::vector<std::string> entries;
    for (uint64_t i = 0; i < AlignmentsToBePerformed; i++) {
      std::string line;
      std::getline(input, line);
      entries.push_back(line);
    }
    input.close();

    std::vector<std::vector<std::string>> alignments(AlignmentsToBePerformed);
    for (uint64_t i = 0; i < AlignmentsToBePerformed; i++)
      alignments[i] = split(entries[i], '\t');

    auto start = NOW;
    LOGAN(alignments, ksize, xdrop, (int)AlignmentsToBePerformed);
    auto end = NOW;
    std::chrono::duration<double> tot = end - start;
    std::cout << "Total execution time [seconds]:\t" << tot.count() << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
