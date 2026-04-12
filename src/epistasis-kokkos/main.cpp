#include <math.h>
#include <float.h>
#include <iostream>
#include <chrono>
#include <Kokkos_Core.hpp>

using namespace std::chrono;
typedef high_resolution_clock myclock;
typedef duration<float> myduration;

// ---- CPU reference implementation ----
static unsigned pc(unsigned x) {
  unsigned count;
  for (count = 0; x; count++) x &= x - 1;
  return count;
}

static float gamma_fn(unsigned int n) {
  if (n == 0) return 0.0f;
  return ((float)n + 0.5f) * logf((float)n) - ((float)n - 1.0f) * logf(expf(1.0f));
}

static void reference(const unsigned int* data_zeros, const unsigned int* data_ones,
    float* scores, int num_snp, int PP_zeros, int PP_ones,
    unsigned int mask_zeros, unsigned int mask_ones) {
  int p, k;
  for (int i = 0; i < num_snp; i++) {
    for (int j = 0; j < num_snp; j++) {
      if (j > i) {
        unsigned int ft[2*9] = {0};
        const unsigned int* SNPi = &data_zeros[i*2];
        const unsigned int* SNPj = &data_zeros[j*2];
        for (p = 0; p < 2*PP_zeros*num_snp - 2*num_snp; p += 2*num_snp) {
          unsigned int di2 = ~(SNPi[p]|SNPi[p+1]), dj2 = ~(SNPj[p]|SNPj[p+1]);
          ft[0]+=pc(SNPi[p]&SNPj[p]); ft[1]+=pc(SNPi[p]&SNPj[p+1]); ft[2]+=pc(SNPi[p]&dj2);
          ft[3]+=pc(SNPi[p+1]&SNPj[p]); ft[4]+=pc(SNPi[p+1]&SNPj[p+1]); ft[5]+=pc(SNPi[p+1]&dj2);
          ft[6]+=pc(di2&SNPj[p]); ft[7]+=pc(di2&SNPj[p+1]); ft[8]+=pc(di2&dj2);
        }
        p = 2*PP_zeros*num_snp - 2*num_snp;
        unsigned int di2 = ~(SNPi[p]|SNPi[p+1])&mask_zeros, dj2 = ~(SNPj[p]|SNPj[p+1])&mask_zeros;
        ft[0]+=pc(SNPi[p]&SNPj[p]); ft[1]+=pc(SNPi[p]&SNPj[p+1]); ft[2]+=pc(SNPi[p]&dj2);
        ft[3]+=pc(SNPi[p+1]&SNPj[p]); ft[4]+=pc(SNPi[p+1]&SNPj[p+1]); ft[5]+=pc(SNPi[p+1]&dj2);
        ft[6]+=pc(di2&SNPj[p]); ft[7]+=pc(di2&SNPj[p+1]); ft[8]+=pc(di2&dj2);
        SNPi = &data_ones[i*2]; SNPj = &data_ones[j*2];
        for (p = 0; p < 2*PP_ones*num_snp - 2*num_snp; p += 2*num_snp) {
          unsigned int di2 = ~(SNPi[p]|SNPi[p+1]), dj2 = ~(SNPj[p]|SNPj[p+1]);
          ft[9]+=pc(SNPi[p]&SNPj[p]); ft[10]+=pc(SNPi[p]&SNPj[p+1]); ft[11]+=pc(SNPi[p]&dj2);
          ft[12]+=pc(SNPi[p+1]&SNPj[p]); ft[13]+=pc(SNPi[p+1]&SNPj[p+1]); ft[14]+=pc(SNPi[p+1]&dj2);
          ft[15]+=pc(di2&SNPj[p]); ft[16]+=pc(di2&SNPj[p+1]); ft[17]+=pc(di2&dj2);
        }
        p = 2*PP_ones*num_snp - 2*num_snp;
        di2 = ~(SNPi[p]|SNPi[p+1])&mask_ones; dj2 = ~(SNPj[p]|SNPj[p+1])&mask_ones;
        ft[9]+=pc(SNPi[p]&SNPj[p]); ft[10]+=pc(SNPi[p]&SNPj[p+1]); ft[11]+=pc(SNPi[p]&dj2);
        ft[12]+=pc(SNPi[p+1]&SNPj[p]); ft[13]+=pc(SNPi[p+1]&SNPj[p+1]); ft[14]+=pc(SNPi[p+1]&dj2);
        ft[15]+=pc(di2&SNPj[p]); ft[16]+=pc(di2&SNPj[p+1]); ft[17]+=pc(di2&dj2);
        float score = 0.0f;
        for (k = 0; k < 9; k++)
          score += gamma_fn(ft[k]+ft[9+k]+1) - gamma_fn(ft[k]) - gamma_fn(ft[9+k]);
        score = fabsf(score);
        if (score == 0.0f) score = FLT_MAX;
        scores[i*num_snp+j] = score;
      }
    }
  }
}

static int min_score(const float* scores, int nrows, int ncols) {
  float s = scores[0]; int sol = 0;
  for (int i = 1; i < nrows*ncols; i++)
    if (s > scores[i]) { s = scores[i]; sol = i; }
  return sol;
}

// ---- Device helper ----
KOKKOS_INLINE_FUNCTION float gammafunction(unsigned int n) {
  if (n == 0) return 0.0f;
  return ((float)n + 0.5f) * Kokkos::log((float)n) - ((float)n - 1.0f);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <num_pac> <num_snp> <iteration>\n";
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    int num_pac   = atoi(argv[1]);
    int num_snp   = atoi(argv[2]);
    int iteration = atoi(argv[3]);

    srand(100);

    unsigned char *SNP_Data = (unsigned char*)aligned_alloc(64, (size_t)num_pac * num_snp);
    unsigned char *Ph_Data  = (unsigned char*)aligned_alloc(64, (size_t)num_pac);

    for (int i = 0; i < num_pac; i++)
      for (int j = 0; j < num_snp; j++)
        SNP_Data[i * num_snp + j] = rand() % 3;
    for (int i = 0; i < num_pac; i++) Ph_Data[i] = rand() % 2;

    unsigned char *SNP_Data_trans = (unsigned char*)aligned_alloc(64, (size_t)num_pac * num_snp);
    for (int i = 0; i < num_pac; i++)
      for (int j = 0; j < num_snp; j++)
        SNP_Data_trans[j * num_pac + i] = SNP_Data[i * num_snp + j];

    int phen_ones = 0;
    for (int i = 0; i < num_pac; i++)
      if (Ph_Data[i] == 1) phen_ones++;

    int PP_zeros = (int)ceil((1.0*(num_pac - phen_ones)) / 32.0);
    int PP_ones  = (int)ceil((1.0*phen_ones) / 32.0);

    unsigned int *bin_data_zeros = (unsigned int*)aligned_alloc(64, (size_t)num_snp * PP_zeros * 2 * sizeof(unsigned int));
    unsigned int *bin_data_ones  = (unsigned int*)aligned_alloc(64, (size_t)num_snp * PP_ones  * 2 * sizeof(unsigned int));
    memset(bin_data_zeros, 0, (size_t)num_snp * PP_zeros * 2 * sizeof(unsigned int));
    memset(bin_data_ones,  0, (size_t)num_snp * PP_ones  * 2 * sizeof(unsigned int));

    for (int i = 0; i < num_snp; i++) {
      int x_zeros = -1, x_ones = -1;
      int n_zeros = 0,  n_ones = 0;
      for (int j = 0; j < num_pac; j++) {
        unsigned int temp = (unsigned int)SNP_Data_trans[i * num_pac + j];
        if (Ph_Data[j] == 1) {
          if (n_ones % 32 == 0) x_ones++;
          bin_data_ones[i * PP_ones * 2 + x_ones*2 + 0] <<= 1;
          bin_data_ones[i * PP_ones * 2 + x_ones*2 + 1] <<= 1;
          if (temp == 0 || temp == 1)
            bin_data_ones[i * PP_ones * 2 + x_ones*2 + temp] |= 1;
          n_ones++;
        } else {
          if (n_zeros % 32 == 0) x_zeros++;
          bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 0] <<= 1;
          bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 1] <<= 1;
          if (temp == 0 || temp == 1)
            bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + temp] |= 1;
          n_zeros++;
        }
      }
    }

    unsigned int mask_zeros = 0xFFFFFFFF;
    for (int x = num_pac - phen_ones; x < PP_zeros * 32; x++)
      mask_zeros >>= 1;

    unsigned int mask_ones = 0xFFFFFFFF;
    for (int x = phen_ones; x < PP_ones * 32; x++)
      mask_ones >>= 1;

    // Transpose binary data structures
    unsigned int *bin_data_ones_trans  = (unsigned int*)aligned_alloc(64, (size_t)num_snp * PP_ones  * 2 * sizeof(unsigned int));
    unsigned int *bin_data_zeros_trans = (unsigned int*)aligned_alloc(64, (size_t)num_snp * PP_zeros * 2 * sizeof(unsigned int));

    for (int i = 0; i < num_snp; i++)
      for (int j = 0; j < PP_ones; j++) {
        bin_data_ones_trans[(j * num_snp + i) * 2 + 0] = bin_data_ones[(i * PP_ones + j) * 2 + 0];
        bin_data_ones_trans[(j * num_snp + i) * 2 + 1] = bin_data_ones[(i * PP_ones + j) * 2 + 1];
      }

    for (int i = 0; i < num_snp; i++)
      for (int j = 0; j < PP_zeros; j++) {
        bin_data_zeros_trans[(j * num_snp + i) * 2 + 0] = bin_data_zeros[(i * PP_zeros + j) * 2 + 0];
        bin_data_zeros_trans[(j * num_snp + i) * 2 + 1] = bin_data_zeros[(i * PP_zeros + j) * 2 + 1];
      }

    float *scores     = (float*)aligned_alloc(64, (size_t)num_snp * num_snp * sizeof(float));
    float *scores_ref = (float*)aligned_alloc(64, (size_t)num_snp * num_snp * sizeof(float));
    for (int x = 0; x < num_snp * num_snp; x++)
      scores[x] = scores_ref[x] = FLT_MAX;

    // Kokkos device views
    Kokkos::View<unsigned int*> d_data_zeros("d_data_zeros", (size_t)num_snp * PP_zeros * 2);
    Kokkos::View<unsigned int*> d_data_ones ("d_data_ones",  (size_t)num_snp * PP_ones  * 2);
    Kokkos::View<float*>        d_scores    ("d_scores",     (size_t)num_snp * num_snp);

    {
      auto h = Kokkos::create_mirror_view(d_data_zeros);
      for (int k = 0; k < num_snp * PP_zeros * 2; k++) h(k) = bin_data_zeros_trans[k];
      Kokkos::deep_copy(d_data_zeros, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_data_ones);
      for (int k = 0; k < num_snp * PP_ones * 2; k++) h(k) = bin_data_ones_trans[k];
      Kokkos::deep_copy(d_data_ones, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_scores);
      for (int k = 0; k < num_snp * num_snp; k++) h(k) = scores[k];
      Kokkos::deep_copy(d_scores, h);
    }

    auto kstart = myclock::now();

    for (int iter = 0; iter < iteration; iter++) {
      Kokkos::parallel_for("epistasis",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_snp, num_snp}),
        KOKKOS_LAMBDA(int ii, int jj) {
          if (jj <= ii) return;

          unsigned int ft[18];
          for (int kk = 0; kk < 18; kk++) ft[kk] = 0;

          // ---- phenotype 0 ----
          {
            const unsigned int* SNPi = d_data_zeros.data() + ii * 2;
            const unsigned int* SNPj = d_data_zeros.data() + jj * 2;
            int p;
            for (p = 0; p < 2 * PP_zeros * num_snp - 2 * num_snp; p += 2 * num_snp) {
              unsigned int a0 = SNPi[p], a1 = SNPi[p+1];
              unsigned int b0 = SNPj[p], b1 = SNPj[p+1];
              unsigned int di2 = ~(a0 | a1);
              unsigned int dj2 = ~(b0 | b1);
              ft[0] += __builtin_popcount(a0 & b0);
              ft[1] += __builtin_popcount(a0 & b1);
              ft[2] += __builtin_popcount(a0 & dj2);
              ft[3] += __builtin_popcount(a1 & b0);
              ft[4] += __builtin_popcount(a1 & b1);
              ft[5] += __builtin_popcount(a1 & dj2);
              ft[6] += __builtin_popcount(di2 & b0);
              ft[7] += __builtin_popcount(di2 & b1);
              ft[8] += __builtin_popcount(di2 & dj2);
            }
            // remainder block
            p = 2 * PP_zeros * num_snp - 2 * num_snp;
            unsigned int a0 = SNPi[p], a1 = SNPi[p+1];
            unsigned int b0 = SNPj[p], b1 = SNPj[p+1];
            unsigned int di2 = (~(a0 | a1)) & mask_zeros;
            unsigned int dj2 = (~(b0 | b1)) & mask_zeros;
            ft[0] += __builtin_popcount(a0 & b0);
            ft[1] += __builtin_popcount(a0 & b1);
            ft[2] += __builtin_popcount(a0 & dj2);
            ft[3] += __builtin_popcount(a1 & b0);
            ft[4] += __builtin_popcount(a1 & b1);
            ft[5] += __builtin_popcount(a1 & dj2);
            ft[6] += __builtin_popcount(di2 & b0);
            ft[7] += __builtin_popcount(di2 & b1);
            ft[8] += __builtin_popcount(di2 & dj2);
          }

          // ---- phenotype 1 ----
          {
            const unsigned int* SNPi = d_data_ones.data() + ii * 2;
            const unsigned int* SNPj = d_data_ones.data() + jj * 2;
            int p;
            for (p = 0; p < 2 * PP_ones * num_snp - 2 * num_snp; p += 2 * num_snp) {
              unsigned int a0 = SNPi[p], a1 = SNPi[p+1];
              unsigned int b0 = SNPj[p], b1 = SNPj[p+1];
              unsigned int di2 = ~(a0 | a1);
              unsigned int dj2 = ~(b0 | b1);
              ft[9]  += __builtin_popcount(a0 & b0);
              ft[10] += __builtin_popcount(a0 & b1);
              ft[11] += __builtin_popcount(a0 & dj2);
              ft[12] += __builtin_popcount(a1 & b0);
              ft[13] += __builtin_popcount(a1 & b1);
              ft[14] += __builtin_popcount(a1 & dj2);
              ft[15] += __builtin_popcount(di2 & b0);
              ft[16] += __builtin_popcount(di2 & b1);
              ft[17] += __builtin_popcount(di2 & dj2);
            }
            // remainder block
            p = 2 * PP_ones * num_snp - 2 * num_snp;
            unsigned int a0 = SNPi[p], a1 = SNPi[p+1];
            unsigned int b0 = SNPj[p], b1 = SNPj[p+1];
            unsigned int di2 = (~(a0 | a1)) & mask_ones;
            unsigned int dj2 = (~(b0 | b1)) & mask_ones;
            ft[9]  += __builtin_popcount(a0 & b0);
            ft[10] += __builtin_popcount(a0 & b1);
            ft[11] += __builtin_popcount(a0 & dj2);
            ft[12] += __builtin_popcount(a1 & b0);
            ft[13] += __builtin_popcount(a1 & b1);
            ft[14] += __builtin_popcount(a1 & dj2);
            ft[15] += __builtin_popcount(di2 & b0);
            ft[16] += __builtin_popcount(di2 & b1);
            ft[17] += __builtin_popcount(di2 & dj2);
          }

          float score = 0.0f;
          for (int kk = 0; kk < 9; kk++)
            score += gammafunction(ft[kk] + ft[9+kk] + 1)
                   - gammafunction(ft[kk])
                   - gammafunction(ft[9+kk]);
          score = Kokkos::fabs(score);
          if (score == 0.0f) score = FLT_MAX;
          d_scores(ii * num_snp + jj) = score;
        });
      Kokkos::fence();
    }

    myduration ktime = myclock::now() - kstart;
    std::cout << "Average kernel execution time: "
              << ktime.count() / iteration << " (s)" << std::endl;

    {
      auto h = Kokkos::create_mirror_view(d_scores);
      Kokkos::deep_copy(h, d_scores);
      for (int k = 0; k < num_snp * num_snp; k++) scores[k] = h(k);
    }

    int p1 = min_score(scores, num_snp, num_snp);

    reference(bin_data_zeros_trans, bin_data_ones_trans, scores_ref, num_snp,
              PP_zeros, PP_ones, mask_zeros, mask_ones);

    int p2 = min_score(scores_ref, num_snp, num_snp);

    bool ok = (p1 == p2) && (fabsf(scores[p1] - scores_ref[p2]) < 1e-3f);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;

    free(bin_data_zeros);
    free(bin_data_ones);
    free(bin_data_zeros_trans);
    free(bin_data_ones_trans);
    free(scores);
    free(scores_ref);
    free(SNP_Data);
    free(SNP_Data_trans);
    free(Ph_Data);
  }
  Kokkos::finalize();
  return 0;
}
