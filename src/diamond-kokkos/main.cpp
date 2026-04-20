/*
 * Diamond protein masking benchmark – standalone Kokkos port
 *
 * Standalone version: generates N synthetic protein sequences of length 33,
 * then runs the tantan-based repeat masking algorithm in parallel.
 * The core algorithms (calcRepeatProbs, maskProbableLetters) are ported
 * from diamond-omp/masking.cpp to Kokkos device lambdas.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define SEQ_LEN        33
#define MATRIX_SIZE    SEQ_LEN   // simplified: use seq-len sized probability matrix
#define MAX_REPEAT_OFFSET 50
#define SCALE_STEP     16
#define MIN_MASK_PROB  0.5

KOKKOS_INLINE_FUNCTION
double firstRepeatOffsetProb(double probMult, int maxRepeatOffset) {
  if (probMult < 1.0 || probMult > 1.0)
    return (1.0 - probMult) / (1.0 - Kokkos::pow(probMult, (double)maxRepeatOffset));
  return 1.0 / maxRepeatOffset;
}

// Returns non-zero on numeric inaccuracy
KOKKOS_INLINE_FUNCTION
int calcRepeatProbs(
    float* letterProbs,
    const unsigned char* seqBeg,
    int size,
    int maxRepeatOffset,
    const double* likelihoodRatioMatrix,
    double b2b, double f2f0, double f2b,
    double b2fLast_inv,
    const double* pow_lkp,
    double* foregroundProbs,
    int scaleStepSize,
    double* scaleFactors)
{
  double backgroundProb = 1.0;
  for (int k = 0; k < size; k++) {
    int v0    = seqBeg[k] % MATRIX_SIZE;
    int k_cap = k < maxRepeatOffset ? k : maxRepeatOffset;
    int pad1  = k_cap - 1;
    int pad2  = maxRepeatOffset - k_cap;
    int pad3  = k - k_cap;
    double accu = 0;
    for (int i = 0; i < k; i++) {
      int idx1 = pad1 - i, idx2 = pad2 + i, idx3 = pad3 + i;
      int v1   = seqBeg[idx3] % MATRIX_SIZE;
      accu += foregroundProbs[idx1];
      foregroundProbs[idx1] = (f2f0 * foregroundProbs[idx1]
                               + backgroundProb * pow_lkp[idx2])
                              * likelihoodRatioMatrix[v0*MATRIX_SIZE + v1];
    }
    backgroundProb = backgroundProb * b2b + accu * f2b;
    if (k % scaleStepSize == scaleStepSize - 1) {
      double scale = 1.0 / backgroundProb;
      scaleFactors[k / scaleStepSize] = scale;
      for (int i = 0; i < k_cap; i++) foregroundProbs[i] *= scale;
      backgroundProb = 1.0;
    }
    letterProbs[k] = (float)backgroundProb;
  }

  double accu2 = 0;
  for (int i = 0; i < maxRepeatOffset; i++) {
    accu2 += foregroundProbs[i];
    foregroundProbs[i] = f2b;
  }
  double fTot = backgroundProb * b2b + accu2 * f2b;
  backgroundProb = b2b;
  double fTot_inv = 1.0 / fTot;

  for (int k = size - 1; k >= 0; k--) {
    double nonRepeatProb = letterProbs[k] * backgroundProb * fTot_inv;
    letterProbs[k] = 1.f - (float)nonRepeatProb;
    int k_cap = k < maxRepeatOffset ? k : maxRepeatOffset;
    if (k % scaleStepSize == scaleStepSize - 1) {
      double scale = scaleFactors[k / scaleStepSize];
      for (int i = 0; i < k_cap; i++) foregroundProbs[i] *= scale;
      backgroundProb *= scale;
    }
    double c0 = f2b * backgroundProb;
    int v0    = seqBeg[k] % MATRIX_SIZE;
    double a2 = 0;
    for (int i = 0; i < k_cap; i++) {
      int v1 = seqBeg[k-(i+1)] % MATRIX_SIZE;
      double f = foregroundProbs[i] * likelihoodRatioMatrix[v0*MATRIX_SIZE + v1];
      a2 += pow_lkp[k_cap-(i+1)] * f;
      foregroundProbs[i] = c0 + f2f0 * f;
    }
    double p = (k > maxRepeatOffset) ? 1.0 : pow_lkp[maxRepeatOffset - k] * b2fLast_inv;
    backgroundProb = b2b * backgroundProb + a2 * p;
  }
  double bTot = backgroundProb;
  return (fabs(fTot - bTot) > fmax(fTot, bTot) / 1e6) ? 1 : 0;
}

KOKKOS_INLINE_FUNCTION
void maskProbableLetters(int size, unsigned char* seqBeg,
    const float* probabilities, const unsigned char* maskTable) {
  for (int i = 0; i < size; i++)
    if (probabilities[i] >= (float)MIN_MASK_PROB)
      seqBeg[i] = maskTable[seqBeg[i] % MATRIX_SIZE];
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <num_sequences> <repeat>\n", argv[0]);
    return 1;
  }
  const int numSeq = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int total  = numSeq * SEQ_LEN;

  // Build a simple likelihood ratio matrix (identity-like, scaled)
  // Using lambda=0.324 and a simple diagonal-dominant matrix
  const double lambda = 0.324032;
  std::vector<double> h_probMat(MATRIX_SIZE * MATRIX_SIZE);
  srand(42);
  for (int i = 0; i < MATRIX_SIZE; i++)
    for (int j = 0; j < MATRIX_SIZE; j++) {
      // Diagonal gets score 5, off-diagonal gets -1 (simplified BLOSUM-like)
      int score = (i == j) ? 5 : -1;
      h_probMat[i*MATRIX_SIZE + j] = exp(lambda * score);
    }

  // Build a mask table: map each amino acid to a mask char ('X' = 23 in standard encoding)
  std::vector<unsigned char> h_maskTable(MATRIX_SIZE);
  for (int i = 0; i < MATRIX_SIZE; i++) h_maskTable[i] = (unsigned char)23; // 'X'

  // Generate random sequences
  std::vector<unsigned char> h_seqs(total);
  srand(123);
  for (int i = 0; i < total; i++) h_seqs[i] = (unsigned char)(rand() % MATRIX_SIZE);

  // Reference CPU computation
  std::vector<unsigned char> h_seqs_cpu(h_seqs);
  {
    const double repeatProb        = 0.005;
    const double repeatEndProb     = 0.05;
    const double repeatOffsetDecay = 0.9;
    const double b2b    = 1.0 - repeatProb;
    const double f2f0   = 1.0 - repeatEndProb;
    const double f2b    = repeatEndProb;
    const double b2fGrowth = 1.0 / repeatOffsetDecay;
    const double b2fLast = repeatProb * firstRepeatOffsetProb(b2fGrowth, MAX_REPEAT_OFFSET);
    const double b2fLast_inv = 1.0 / b2fLast;
    std::vector<double> pow_lkp(MAX_REPEAT_OFFSET);
    double p = b2fLast;
    for (int i = 0; i < MAX_REPEAT_OFFSET; i++) { pow_lkp[i] = p; p *= b2fGrowth; }

    for (int gid = 0; gid < numSeq; gid++) {
      unsigned char* seq = &h_seqs_cpu[gid * SEQ_LEN];
      float probs[SEQ_LEN];
      double fgProbs[MAX_REPEAT_OFFSET] = {0};
      double scaleFactors[SEQ_LEN / SCALE_STEP + 1] = {0};
      calcRepeatProbs(probs, seq, SEQ_LEN, MAX_REPEAT_OFFSET,
          h_probMat.data(), b2b, f2f0, f2b, b2fLast_inv,
          pow_lkp.data(), fgProbs, SCALE_STEP, scaleFactors);
      maskProbableLetters(SEQ_LEN, seq, probs, h_maskTable.data());
    }
  }

  Kokkos::initialize(argc, argv);
  {
    // Device Views
    Kokkos::View<unsigned char*> d_seqs("seqs", total);
    Kokkos::View<double*>        d_probMat("probMat", MATRIX_SIZE * MATRIX_SIZE);
    Kokkos::View<unsigned char*> d_maskTable("maskTable", MATRIX_SIZE);

    {
      auto h_s = Kokkos::create_mirror_view(d_seqs);
      for (int i = 0; i < total; i++) h_s(i) = h_seqs[i];
      Kokkos::deep_copy(d_seqs, h_s);
    }
    {
      auto h_m = Kokkos::create_mirror_view(d_probMat);
      for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) h_m(i) = h_probMat[i];
      Kokkos::deep_copy(d_probMat, h_m);
    }
    {
      auto h_t = Kokkos::create_mirror_view(d_maskTable);
      for (int i = 0; i < MATRIX_SIZE; i++) h_t(i) = h_maskTable[i];
      Kokkos::deep_copy(d_maskTable, h_t);
    }

    const double repeatProb        = 0.005;
    const double repeatEndProb     = 0.05;
    const double repeatOffsetDecay = 0.9;

    auto t_start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      // Reset sequences to original before each repeat
      {
        auto h_s = Kokkos::create_mirror_view(d_seqs);
        for (int i = 0; i < total; i++) h_s(i) = h_seqs[i];
        Kokkos::deep_copy(d_seqs, h_s);
      }

      Kokkos::parallel_for("masking", numSeq, KOKKOS_LAMBDA(int gid) {
        unsigned char* seqBeg = &d_seqs(gid * SEQ_LEN);

        const double b2b    = 1.0 - repeatProb;
        const double f2f0   = 1.0 - repeatEndProb;
        const double f2b_   = repeatEndProb;
        const double b2fGrowth = 1.0 / repeatOffsetDecay;
        const double b2fLast = repeatProb * firstRepeatOffsetProb(b2fGrowth, MAX_REPEAT_OFFSET);
        const double b2fLast_inv = 1.0 / b2fLast;

        double pow_lkp[MAX_REPEAT_OFFSET];
        double pp = b2fLast;
        for (int i = 0; i < MAX_REPEAT_OFFSET; i++) { pow_lkp[i] = pp; pp *= b2fGrowth; }

        double foregroundProbs[MAX_REPEAT_OFFSET] = {0};
        double scaleFactors[SEQ_LEN / SCALE_STEP + 2] = {0};
        float  probabilities[SEQ_LEN];

        calcRepeatProbs(probabilities, seqBeg, SEQ_LEN, MAX_REPEAT_OFFSET,
            d_probMat.data(), b2b, f2f0, f2b_, b2fLast_inv,
            pow_lkp, foregroundProbs, SCALE_STEP, scaleFactors);

        maskProbableLetters(SEQ_LEN, seqBeg, probabilities, d_maskTable.data());
      });
      Kokkos::fence();
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average masking kernel time: %f (s)\n", elapsed * 1e-9 / repeat);

    // Verify last iteration result
    {
      auto h_s = Kokkos::create_mirror_view(d_seqs);
      Kokkos::deep_copy(h_s, d_seqs);

      int errors = 0;
      for (int i = 0; i < numSeq; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
          if (h_s(i*SEQ_LEN+j) != h_seqs_cpu[i*SEQ_LEN+j]) errors++;
        }
      }
      printf("Verification: %s (errors=%d)\n", errors == 0 ? "PASS" : "FAIL", errors);
    }
  }
  Kokkos::finalize();
  return 0;
}
