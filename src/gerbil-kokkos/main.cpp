/*
Gerbil simplified k-mer counting benchmark.
Kokkos port (OpenMP backend).

Algorithm:
  1. Generate random DNA sequences (A/C/G/T)
  2. Extract all k-mers via sliding window (length k)
  3. Encode each k-mer as a 64-bit integer (2 bits per base: A=0, C=1, G=2, T=3)
  4. Count occurrences using a hash table with atomic_add

Hash table:
  - Size: HT_SIZE = 1 << 24 (16M buckets)
  - Index: (kmer_hash ^ (kmer_hash >> 17)) & (HT_SIZE - 1)
  - Each bucket stores a count (int); collisions are accepted (approximate)
  - Unique k-mers = number of non-zero buckets

Usage: ./main [numSequences [seqLen [k [repeat]]]]
  numSequences = number of sequences      (default 1000)
  seqLen       = length of each sequence  (default 1000)
  k            = k-mer length             (default 32)
  repeat       = benchmark repetitions    (default 1)
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>

static const int HT_SIZE = 1 << 24; // 16M

// ---------------------------------------------------------------------------
// 2-bit encoding: A=0, C=1, G=2, T=3
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int base_encode(char c)
{
    if (c == 'A' || c == 'a') return 0;
    if (c == 'C' || c == 'c') return 1;
    if (c == 'G' || c == 'g') return 2;
    return 3; // T/t or anything else
}

// ---------------------------------------------------------------------------
// Simple hash for a 64-bit k-mer encoding
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
unsigned int kmer_hash(uint64_t kmer)
{
    kmer = (~kmer) + (kmer << 21);
    kmer =   kmer  ^ (kmer >> 24);
    kmer = ( kmer  + (kmer << 3)) + (kmer << 8);
    kmer =   kmer  ^ (kmer >> 14);
    kmer = ( kmer  + (kmer << 2)) + (kmer << 4);
    kmer =   kmer  ^ (kmer >> 28);
    kmer =   kmer  + (kmer << 31);
    return (unsigned int)kmer;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int numSeq  = 1000;
    int seqLen  = 1000;
    int k       = 32;
    int repeat  = 1;

    if (argc > 1) numSeq  = atoi(argv[1]);
    if (argc > 2) seqLen  = atoi(argv[2]);
    if (argc > 3) k       = atoi(argv[3]);
    if (argc > 4) repeat  = atoi(argv[4]);

    if (k > 32) { fprintf(stderr, "k must be <= 32\n"); return 1; }
    if (seqLen < k) { fprintf(stderr, "seqLen must be >= k\n"); return 1; }

    printf("K-mer length: %d\n", k);

    const long long kmers_per_seq = seqLen - k + 1;
    const long long total_kmers   = (long long)numSeq * kmers_per_seq;
    printf("Total k-mers: %lld\n", total_kmers);

    Kokkos::initialize(argc, argv);
    {
        using ViewI   = Kokkos::View<int*>;
        using ViewC   = Kokkos::View<char*>;
        using ViewULL = Kokkos::View<uint64_t*>;

        // Generate all sequences in a flat character array: seqs[s * seqLen + p]
        const int totalChars = numSeq * seqLen;
        ViewC d_seqs("seqs", totalChars);

        // Hash table
        ViewI d_ht("ht", HT_SIZE);
        Kokkos::deep_copy(d_ht, 0);

        // Generate random sequences on device
        const int sl = seqLen;
        Kokkos::parallel_for("genSeqs", numSeq, KOKKOS_LAMBDA(int s) {
            uint64_t state = (uint64_t)(s + 1) * 6364136223846793005ULL;
            const char bases[4] = {'A', 'C', 'G', 'T'};
            for (int p = 0; p < sl; p++) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                d_seqs(s * sl + p) = bases[(state >> 33) & 3];
            }
        });
        Kokkos::fence();

        double total_ms = 0.0;

        for (int rep = 0; rep < repeat; rep++) {

            Kokkos::deep_copy(d_ht, 0);

            auto t1 = std::chrono::steady_clock::now();

            // Count k-mers: parallel over (sequence, position) pairs
            const int kk = k;
            const int ht_mask = HT_SIZE - 1;

            Kokkos::parallel_for("countKmers", numSeq, KOKKOS_LAMBDA(int s) {
                const int base_off = s * sl;
                // Build first k-mer
                uint64_t kmer    = 0;
                const uint64_t mask = (kk < 32) ? ((1ULL << (2*kk)) - 1ULL) : ~0ULL;
                for (int p = 0; p < kk; p++) {
                    kmer = ((kmer << 2) | (uint64_t)base_encode(d_seqs(base_off + p))) & mask;
                }
                // Hash and count first k-mer
                unsigned int idx = kmer_hash(kmer) & ht_mask;
                Kokkos::atomic_fetch_add(&d_ht(idx), 1);

                // Slide window
                for (int p = kk; p < sl; p++) {
                    kmer = ((kmer << 2) | (uint64_t)base_encode(d_seqs(base_off + p))) & mask;
                    idx  = kmer_hash(kmer) & ht_mask;
                    Kokkos::atomic_fetch_add(&d_ht(idx), 1);
                }
            });

            Kokkos::fence();
            auto t2 = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }

        // Count unique k-mers (non-zero hash buckets)
        long long unique_kmers = 0;
        Kokkos::parallel_reduce("countUnique", HT_SIZE,
            KOKKOS_LAMBDA(int i, long long& cnt) {
                if (d_ht(i) > 0) cnt++;
            }, unique_kmers);

        printf("Unique k-mers: %lld\n", unique_kmers);
        printf("Kernel time: %.3f (ms)\n", total_ms / repeat);
    }
    Kokkos::finalize();
    return 0;
}
