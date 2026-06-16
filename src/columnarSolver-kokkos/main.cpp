/*
Copyright 1998–2018 Bernhard Esslinger and the CrypTool Team.
Kokkos port of columnarSolver-omp/main.cpp.
All OMP target teams are replaced with Kokkos::TeamPolicy.
*/

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

// ---- Constants -------------------------------------------------------------
#define B ((int)32)
#define T ((int)32)
#define THREADS ((int)B*T)
#define CLIMBINGS 150000
#define ALPHABET 26
#define totalBigrams ((int)ALPHABET*ALPHABET)
#define CAP ((float)999999.0)
#define KEY_LENGTH 30
#define SECTION_CONSTANT (ENCRYPTEDLEN/KEY_LENGTH)
#define HEUR_THRESHOLD_OP1 50
#define HEUR_THRESHOLD_OP2 70
#define OP1_HOP 4
#define OP2_HOP 2

#define ENCRYPTED_T \
  "tteohtedanisroudesereguwocubsoitoabbofeiaiutsdheeisatsarsturesuaastniersrotnesctrctxdiwmhcusyenorndasmhaipnnptmaeecspegdeislwoheoiymreeotbsspiatoanihrelhwctftrhpuunhoianunreetrioettatlsnehtbaecpvgtltcirottonesnobeeeireaymrtohaawnwtesssvassirsrhabapnsynntitsittchitoosbtelmlaouitrehhwfeiaandeitciegfreoridhdcsheucrnoihdeoswobaceeaorgndlstigeearsotoetduedininttpedststntefoeaheoesuetvmmiorftuuhsurof"
#define ENCRYPTEDLEN ((int)sizeof(ENCRYPTED_T)-1)

#define DECRYPTED_T \
  "thedistinctionbetweentherouteciphertranspositionandthesubstitutioncipherwherewholewordsaresubstitutedforlettersoftheoriginaltextmustbemadeonthebasisofthewordsactuallyuseditisbettertoconsidersuchamessageasaroutecipherwhenthewordsusedappeartohavesomeconsecutivemeaningbearingonthesituationathandasubstitutioncipherofthisvarietywouldonlybeusedfortransmissionofashortmessageofgreatimportanceandsecrecy"

// ---- Device helper functions -----------------------------------------------

KOKKOS_INLINE_FUNCTION
float LCG_random_float(unsigned int *seed) {
    const unsigned int m = 2147483648u;
    const unsigned int a = 26757677u;
    const unsigned int c = 1u;
    *seed = (a * (*seed) + c) % m;
    return (float)(*seed) / (float)m;
}

KOKKOS_INLINE_FUNCTION
void LCG_random_init(unsigned int *seed) {
    const unsigned int m = 2147483648u;
    const unsigned int a = 26757677u;
    const unsigned int c = 1u;
    *seed = (a * (*seed) + c) % m;
}

KOKKOS_INLINE_FUNCTION
void decrypt(const int *encrypted, const int *key, int *decrypted) {
    int columns[KEY_LENGTH][SECTION_CONSTANT + 1];
    int offset = 0;
    int colLength[KEY_LENGTH];

    for (int j = 0; j < KEY_LENGTH; ++j) {
        colLength[j] = ENCRYPTEDLEN / KEY_LENGTH;
        if (j < ENCRYPTEDLEN % KEY_LENGTH) colLength[j]++;
    }

    for (int keyPos = 0; keyPos < KEY_LENGTH; ++keyPos) {
        offset = 0;
        for (int i = 0; i < KEY_LENGTH; ++i)
            if (key[i] < key[keyPos]) offset += colLength[i];
        for (int j = 0; j < colLength[keyPos]; ++j)
            columns[key[keyPos]][j] = encrypted[offset + j];
    }

    for (int j = 0; j < ENCRYPTEDLEN; ++j)
        decrypted[j] = columns[key[j % KEY_LENGTH]][j / KEY_LENGTH];
}

KOKKOS_INLINE_FUNCTION
void swapElements(int *key, int posLeft, int posRight) {
    if (posLeft != posRight) {
        key[posLeft]  -= key[posRight];
        key[posRight] += key[posLeft];
        key[posLeft]   = key[posRight] - key[posLeft];
    }
}

KOKKOS_INLINE_FUNCTION
void swapBlock(int *key, int posLeft, int posRight, int length) {
    for (int i = 0; i < length; i++)
        swapElements(key, (posLeft + i) % KEY_LENGTH, (posRight + i) % KEY_LENGTH);
}

// ---- Main kernel ----------------------------------------------------------
// ---- Host utility functions ------------------------------------------------

bool extractBigrams(float *scores, const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Failed to open %s\n", filename); return true; }
    while (1) {
        char  tempBigram[4];
        float score = 0.0f;
        if (fscanf(f, "%s %f", tempBigram, &score) < 2) break;
        scores[(tempBigram[0] - 'a') * ALPHABET + (tempBigram[1] - 'a')] = score;
    }
    fclose(f);
    return false;
}

bool verify(int *encrMap) {
    const char *expect = DECRYPTED_T;
    for (int j = 0; j < ENCRYPTEDLEN; ++j)
        if (encrMap[j] + 'a' != expect[j]) return false;
    return true;
}

float candidateScore(int *decrMsg, float *scores) {
    float total = 0.0f;
    for (int j = 0; j < ENCRYPTEDLEN - 1; ++j)
        total += scores[ALPHABET * decrMsg[j] + decrMsg[j + 1]];
    return total;
}

// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 2) { printf("Usage: %s <bigrams_file>\n", argv[0]); return 1; }
    const char *filename = argv[1];

    Kokkos::initialize(argc, argv);
    {
        int encryptedMap[ENCRYPTEDLEN];
        for (int j = 0; j < ENCRYPTEDLEN; ++j)
            encryptedMap[j] = ENCRYPTED_T[j] - 'a';

        float scores[totalBigrams];
        memset(scores, 0, sizeof(scores));
        if (extractBigrams(scores, filename)) { Kokkos::finalize(); return 1; }

        // ---- Allocate device Views ----------------------------------------
        Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_scores_wrap(scores, totalBigrams);
        Kokkos::View<float*> d_scores("d_scores", totalBigrams);
        Kokkos::deep_copy(d_scores, h_scores_wrap);

        Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_enc_wrap(encryptedMap, ENCRYPTEDLEN);
        Kokkos::View<int*> d_encryptedMap("d_enc", ENCRYPTEDLEN);
        Kokkos::deep_copy(d_encryptedMap, h_enc_wrap);

        Kokkos::View<unsigned int*> d_state     ("d_state",     THREADS);
        Kokkos::View<int*>          d_decrypted ("d_decrypted", (size_t)ENCRYPTEDLEN * THREADS);

        // ---- Init kernel: initialize LCG states ---------------------------
        Kokkos::parallel_for("init_state", THREADS,
            KOKKOS_LAMBDA(int idx) {
                unsigned int s = (unsigned int)idx;
                for (int i = 0; i < idx; i++) LCG_random_init(&s);
                d_state(idx) = s;
            });
        Kokkos::fence();

        auto start = std::chrono::steady_clock::now();

        // ---- Decode kernel: THREADS independent workers ----------------------
        // Each worker runs its own hill-climbing search.
        // (The original OMP version cached d_scores in per-team shared memory
        //  for performance; here we read d_scores directly per thread.)
        Kokkos::parallel_for("decodeKernel", THREADS,
            KOKKOS_LAMBDA(int idx) {
                // Each thread reads scores from global memory directly
                unsigned int localState = d_state(idx);

                int key[KEY_LENGTH];
                int localDecrypted[ENCRYPTEDLEN];
                int bestLocalDecrypted[ENCRYPTEDLEN];
                int backupKey[KEY_LENGTH];
                int shiftHelper[KEY_LENGTH];

                float bestScore = CAP;

                for (int j = 0; j < KEY_LENGTH; ++j) key[j] = j;
                for (int j = 0; j < KEY_LENGTH; ++j)
                    swapElements(key, j, (int)(LCG_random_float(&localState) * KEY_LENGTH));

                for (int cycle = 0; cycle < CLIMBINGS; ++cycle) {
                    for (int j = 0; j < KEY_LENGTH; j++) backupKey[j] = key[j];

                    float tempScore = 0.0f;
                    int branch = (int)(LCG_random_float(&localState) * 100);

                    if (branch < HEUR_THRESHOLD_OP1) {
                        int hops = 1 + (int)(LCG_random_float(&localState) * OP1_HOP);
                        for (int j = 0; j < hops; j++) {
                            int l = (int)(LCG_random_float(&localState) * KEY_LENGTH);
                            int r = (int)(LCG_random_float(&localState) * KEY_LENGTH);
                            swapElements(key, l, r);
                        }
                    } else if (branch < HEUR_THRESHOLD_OP2) {
                        int hops = 1 + (int)(LCG_random_float(&localState) * OP2_HOP);
                        for (int j = 0; j < hops; j++) {
                            int bs = (int)(LCG_random_float(&localState) * KEY_LENGTH);
                            int be = (int)(LCG_random_float(&localState) * KEY_LENGTH);
                            int bl = 1 + (int)(LCG_random_float(&localState) *
                                              (abs(bs - be) - 1));
                            swapBlock(key, bs, be, bl);
                        }
                    } else {
                        int l  = 1 + (int)(LCG_random_float(&localState) * (KEY_LENGTH - 2));
                        int f  = (int)(LCG_random_float(&localState) * (KEY_LENGTH - 1));
                        int t  = (f + 1 + (int)(LCG_random_float(&localState) *
                                               (KEY_LENGTH - 2))) % KEY_LENGTH;
                        int t0 = (t - f + KEY_LENGTH) % KEY_LENGTH;
                        int n  = (t0 + l) % KEY_LENGTH;
                        for (int j = 0; j < KEY_LENGTH; j++) shiftHelper[j] = key[j];
                        for (int j = 0; j < n; j++) {
                            int ff = (f + j) % KEY_LENGTH;
                            int tt = (((t0 + j) % n) + f) % KEY_LENGTH;
                            key[tt] = shiftHelper[ff];
                        }
                    }

                    decrypt(d_encryptedMap.data(), key, localDecrypted);

                    for (int j = 0; j < ENCRYPTEDLEN - 1; ++j)
                        tempScore += d_scores[ALPHABET * localDecrypted[j] +
                                              localDecrypted[j + 1]];

                    if (tempScore < bestScore) {
                        bestScore = tempScore;
                        for (int j = 0; j < ENCRYPTEDLEN; ++j)
                            bestLocalDecrypted[j] = localDecrypted[j];
                    } else {
                        for (int j = 0; j < KEY_LENGTH; j++) key[j] = backupKey[j];
                    }
                }

                for (int j = 0; j < ENCRYPTEDLEN; ++j)
                    d_decrypted[idx * ENCRYPTEDLEN + j] = bestLocalDecrypted[j];
            });
        Kokkos::fence();

        auto end  = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Kernel execution time %f (s)\n", time * 1e-9f);

        // ---- Copy result back to host ------------------------------------
        auto h_dec = Kokkos::create_mirror_view(d_decrypted);
        Kokkos::deep_copy(h_dec, d_decrypted);

        // ---- Find best candidate ----------------------------------------
        int   bestCandidate = 0;
        float bestScore     = CAP;
        for (int j = 0; j < THREADS; ++j) {
            float s = candidateScore(&h_dec(ENCRYPTEDLEN * j), scores);
            if (s < bestScore) { bestScore = s; bestCandidate = j; }
        }

        bool pass = verify(&h_dec(ENCRYPTEDLEN * bestCandidate));
        printf("%s\n", pass ? "PASS" : "FAIL");
    }
    Kokkos::finalize();
    return 0;
}
