/*
 * DNA De-redundancy – Kokkos port
 * Reads a FASTA file and removes redundant sequences using
 * an index-based word-matching and alignment approach.
 */

#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#undef max
#undef min
#define kmax(a,b) ((a)<(b)?(b):(a))
#define kmin(a,b) ((a)<(b)?(a):(b))

//---------------------------------------------------------------------
// Data structures
//---------------------------------------------------------------------
struct Option {
  std::string inputFile, outputFile;
  float threshold;
  int   wordLength;
};

struct Read { std::string data, name; };

void checkOption(int argc, char** argv, Option& opt) {
  opt.inputFile  = "input.fasta";
  opt.outputFile = "output.fasta";
  opt.threshold  = 0.95f;
  for (int i = 1; i < argc; i += 2) {
    switch (argv[i][0]) {
      case 'i': opt.inputFile  = argv[i+1]; break;
      case 'o': opt.outputFile = argv[i+1]; break;
      case 't': opt.threshold  = std::atof(argv[i+1]); break;
    }
  }
  if (opt.threshold < 0.8f || opt.threshold >= 1.f) {
    std::cout << "Threshold out of range.\n"; exit(0);
  }
  int tmp = (int)((opt.threshold*100 - 80) / 5);
  switch (tmp) {
    case 0: opt.wordLength = 4; break;
    case 1: opt.wordLength = 5; break;
    case 2: opt.wordLength = 6; break;
    default: opt.wordLength = 7; break;
  }
  std::cout << "input:\t"      << opt.inputFile  << "\n"
            << "output:\t"     << opt.outputFile << "\n"
            << "threshold:\t"  << opt.threshold  << "\n"
            << "word length:\t"<< opt.wordLength  << "\n";
}

bool readFile(std::vector<Read>& reads, Option& opt) {
  std::ifstream f(opt.inputFile);
  if (!f.is_open()) { std::cout << "Failed to open " << opt.inputFile << "\n"; return true; }
  Read rd;
  std::string line;
  std::getline(f, line); rd.name = line;
  while (std::getline(f, line)) {
    if (line[0] == '>') { reads.push_back(rd); rd.name = line; rd.data = ""; continue; }
    rd.data += line;
  }
  reads.push_back(rd);
  std::sort(reads.begin(), reads.end(),
            [](const Read& a, const Read& b){ return a.data.size() > b.data.size(); });
  return false;
}

//---------------------------------------------------------------------
// Device kernels
//---------------------------------------------------------------------
void kernel_baseToNumber(Kokkos::View<char*> d_reads, long length) {
  Kokkos::parallel_for("baseToNumber", length, KOKKOS_LAMBDA(long idx) {
    char c = d_reads(idx);
    if      (c=='A'||c=='a') d_reads(idx) = 0;
    else if (c=='C'||c=='c') d_reads(idx) = 1;
    else if (c=='G'||c=='g') d_reads(idx) = 2;
    else if (c=='T'||c=='t'||c=='U'||c=='u') d_reads(idx) = 3;
    else                     d_reads(idx) = 4;
  });
}

void kernel_compressData(Kokkos::View<const int*> d_lengths,
    Kokkos::View<const long*> d_offsets, Kokkos::View<const char*> d_reads,
    Kokkos::View<unsigned int*> d_compressed, Kokkos::View<int*> d_gaps, int readsCount)
{
  Kokkos::parallel_for("compressData", readsCount, KOKKOS_LAMBDA(int idx) {
    long mark = d_offsets(idx) / 16;
    int round = 0, gapCount = 0;
    unsigned int cTemp = 0;
    long start = d_offsets(idx), end = start + d_lengths(idx);
    for (long i = start; i < end; i++) {
      unsigned char base = (unsigned char)d_reads(i);
      if (base < 4) {
        cTemp += base << (15 - round) * 2;
        if (++round == 16) { d_compressed(mark++) = cTemp; cTemp = 0; round = 0; }
      } else gapCount++;
    }
    d_compressed(mark) = cTemp;
    d_gaps(idx) = gapCount;
  });
}

template<int WL>
void kernel_createIndex(Kokkos::View<const char*> d_reads,
    Kokkos::View<const int*> d_lengths, Kokkos::View<const long*> d_offsets,
    Kokkos::View<unsigned short*> d_indexs, Kokkos::View<long*> d_words,
    Kokkos::View<int*> d_magicBase, int readsCount)
{
  Kokkos::parallel_for("createIndex", readsCount, KOKKOS_LAMBDA(int idx) {
    int start = (int)d_offsets(idx), end = start + d_lengths(idx);
    int magic[4] = {0,0,0,0};
    char bases[WL+1];
    for (int i = 0; i <= WL; i++) bases[i] = 4;
    int wordCount = 0;
    for (int i = start; i < end; i++) {
      for (int j = 0; j < WL; j++) bases[j] = bases[j+1];
      bases[WL] = d_reads(i);
      switch ((int)bases[WL]) { case 0: magic[0]++; break; case 1: magic[1]++; break;
                                 case 2: magic[2]++; break; case 3: magic[3]++; break; }
      unsigned short iv = 0; int flag = 0;
      for (int j = 0; j < WL; j++) {
        iv += (bases[j] & 3) << (WL-1-j)*2;
        flag += kmax((int)(bases[j] - 3), 0);
      }
      d_indexs(i) = flag ? 65535 : iv;
      wordCount += flag ? 0 : 1;
    }
    d_words(idx) = wordCount;
    for (int m = 0; m < 4; m++) d_magicBase(idx*4+m) = magic[m];
  });
}

void kernel_createCutoff(float threshold, int wordLength,
    Kokkos::View<const int*> d_lengths, Kokkos::View<const long*> d_words,
    Kokkos::View<int*> d_wordCutoff, int readsCount)
{
  Kokkos::parallel_for("createCutoff", readsCount, KOKKOS_LAMBDA(long idx) {
    int length   = d_lengths(idx);
    int required = length - wordLength + 1;
    int cutoff   = (int)ceilf((float)length * (1.f - threshold) * (float)wordLength);
    d_wordCutoff(idx) = required - cutoff;
  });
}

void kernel_mergeIndex(Kokkos::View<const long*> d_offsets,
    Kokkos::View<const unsigned short*> d_indexs,
    Kokkos::View<unsigned short*> d_orders,
    Kokkos::View<const long*> d_words, int readsCount)
{
  Kokkos::parallel_for("mergeIndex", readsCount, KOKKOS_LAMBDA(long idx) {
    int start = (int)d_offsets(idx), end = (int)(start + d_words(idx));
    if (start >= end) return;
    unsigned short prev = d_indexs(start), cur; int cnt = 1;
    for (int i = start+1; i < end; i++) {
      cur = d_indexs(i);
      if (cur == prev) { cnt++; d_orders(i-1) = 0; }
      else { d_orders(i-1) = (unsigned short)cnt; prev = cur; cnt = 1; }
    }
    d_orders(end-1) = (unsigned short)cnt;
  });
}

void kernel_makeTable(Kokkos::View<const long*> d_offsets,
    Kokkos::View<const unsigned short*> d_indexs,
    Kokkos::View<const unsigned short*> d_orders,
    Kokkos::View<const long*> d_words,
    Kokkos::View<unsigned short*> d_table, int representative)
{
  int start = (int)d_offsets(representative);
  int end   = (int)(start + d_words(representative));
  int n     = end - start;
  Kokkos::parallel_for("makeTable", n, KOKKOS_LAMBDA(int i) {
    unsigned short ord = d_orders(start + i);
    if (ord != 0)
      Kokkos::atomic_assign(&d_table(d_indexs(start + i)), ord);
  });
}

void kernel_cleanTable(Kokkos::View<const long*> d_offsets,
    Kokkos::View<const unsigned short*> d_indexs,
    Kokkos::View<const unsigned short*> d_orders,
    Kokkos::View<const long*> d_words,
    Kokkos::View<unsigned short*> d_table, int representative)
{
  int start = (int)d_offsets(representative);
  int end   = (int)(start + d_words(representative));
  int n     = end - start;
  Kokkos::parallel_for("cleanTable", n, KOKKOS_LAMBDA(int i) {
    if (d_orders(start + i) != 0)
      d_table(d_indexs(start + i)) = 0;
  });
}

void kernel_magic(float threshold, Kokkos::View<const int*> d_lengths,
    Kokkos::View<const int*> d_magicBase, Kokkos::View<int*> d_cluster,
    int representative, int readsCount)
{
  int offsetOne = representative * 4;
  Kokkos::parallel_for("magic", readsCount, KOKKOS_LAMBDA(int idx) {
    if (d_cluster(idx) < 0) {
      int offsetTwo = idx * 4;
      int magic = kmin(d_magicBase(offsetOne+0), d_magicBase(offsetTwo+0))
                + kmin(d_magicBase(offsetOne+1), d_magicBase(offsetTwo+1))
                + kmin(d_magicBase(offsetOne+2), d_magicBase(offsetTwo+2))
                + kmin(d_magicBase(offsetOne+3), d_magicBase(offsetTwo+3));
      int minLen = (int)ceilf((float)d_lengths(idx) * threshold);
      if (magic > minLen) d_cluster(idx) = -2;
    }
  });
}

void kernel_filter(float threshold, int wordLength,
    Kokkos::View<const int*> d_lengths,
    Kokkos::View<const long*> d_offsets,
    Kokkos::View<const unsigned short*> d_indexs,
    Kokkos::View<const unsigned short*> d_orders,
    Kokkos::View<const long*> d_words,
    Kokkos::View<const int*> d_wordCutoff,
    Kokkos::View<int*> d_cluster,
    Kokkos::View<const unsigned short*> d_table, int readsCount)
{
  // TeamPolicy: one team per read, 128 threads per team for reduction
  Kokkos::TeamPolicy<> policy(readsCount, 128);
  Kokkos::parallel_for("filter", policy,
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      int gid = team.league_rank();
      if (d_cluster(gid) != -2) return;
      int result = 0;
      long start = d_offsets(gid), end = start + d_words(gid);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, (int)(end - start)),
        [&](int i, int& lsum) {
          int pos = (int)start + i;
          lsum += kmin((int)d_table(d_indexs(pos)), (int)d_orders(pos));
        }, result);
      if (team.team_rank() == 0) {
        d_cluster(gid) = (result > d_wordCutoff(gid)) ? -3 : -1;
      }
    });
}

void kernel_align(float threshold,
    Kokkos::View<const int*> d_lengths, Kokkos::View<const long*> d_offsets,
    Kokkos::View<const unsigned int*> d_compressed, Kokkos::View<const int*> d_gaps,
    int representative, Kokkos::View<int*> d_cluster, int readsCount)
{
  Kokkos::parallel_for("align", readsCount, KOKKOS_LAMBDA(int idx) {
    if (d_cluster(idx) != -3) return;
    int target = representative, query = idx;
    int minLength    = (int)ceilf((float)d_lengths(idx) * threshold);
    int targetLength = d_lengths(target) - d_gaps(target);
    int queryLength  = d_lengths(query)  - d_gaps(query);
    int target32     = targetLength / 16 + 1;
    int query32      = queryLength  / 16 + 1;
    long tOff = d_offsets(target) / 16;
    long qOff = d_offsets(query)  / 16;
    int shift = (int)ceilf((float)(targetLength - (float)queryLength*threshold) / 16.f);
    short rowNow[3000]      = {0};
    short rowPrevious[3000] = {0};
    int colPrev[17] = {0}, colNow[17] = {0};
    int complete = 0;

    for (int i = 0; i < query32 && !complete; i++) {
      for (int j = 0; j < 17; j++) { colPrev[j] = colNow[j] = 0; }
      int tIdx = 0;
      unsigned int qPack = d_compressed(qOff + i);
      for (int j = 0; j < target32 && !complete; j++) {
        colPrev[0] = rowPrevious[tIdx];
        unsigned int tPack = d_compressed(tOff + j);
        for (int k = 30; k >= 0; k -= 2) {
          int tBase = (tPack >> k) & 3;
          int m = 0; colNow[m] = rowPrevious[tIdx+1];
          for (int l = 30; l >= 0; l -= 2) {
            m++;
            int qBase = (qPack >> l) & 3;
            colNow[m] = colPrev[m-1] + (qBase == tBase ? 1 : 0);
            colNow[m] = kmax(colNow[m], colNow[m-1]);
            colNow[m] = kmax(colNow[m], colPrev[m]);
          }
          tIdx++; rowNow[tIdx] = colNow[16];
          if (tIdx == targetLength) {
            if (i == query32-1) {
              d_cluster(idx) = (colNow[queryLength%16] >= minLength) ? target : -1;
              complete = 1;
            }
            break;
          }
          k -= 2; tBase = (tPack >> k) & 3;
          m = 0; colPrev[m] = rowPrevious[tIdx+1];
          for (int l = 30; l >= 0; l -= 2) {
            m++;
            int qBase = (qPack >> l) & 3;
            colPrev[m] = colNow[m-1] + (qBase == tBase ? 1 : 0);
            colPrev[m] = kmax(colPrev[m], colPrev[m-1]);
            colPrev[m] = kmax(colPrev[m], colNow[m]);
          }
          tIdx++; rowNow[tIdx] = colPrev[16];
          if (tIdx == targetLength) {
            if (i == query32-1) {
              d_cluster(idx) = (colPrev[queryLength%16] >= minLength) ? target : -1;
              complete = 1;
            }
            break;
          }
        }
        if (complete) break;
      }
      if (complete) break;
      // Exchange rows
      i++;
      for (int j = 0; j < 17; j++) { colPrev[j] = colNow[j] = 0; }
      tIdx = 0; qPack = d_compressed(qOff + i);
      for (int j = 0; j < target32 && !complete; j++) {
        unsigned int tPack = d_compressed(tOff + j);
        for (int k = 30; k >= 0; k -= 2) {
          int tBase = (tPack >> k) & 3;
          int m = 0; colNow[m] = rowNow[tIdx+1];
          for (int l = 30; l >= 0; l -= 2) {
            m++;
            int qBase = (qPack >> l) & 3;
            colNow[m] = colPrev[m-1] + (qBase == tBase ? 1 : 0);
            colNow[m] = kmax(colNow[m], colNow[m-1]);
            colNow[m] = kmax(colNow[m], colPrev[m]);
          }
          tIdx++; rowPrevious[tIdx] = colNow[16];
          if (tIdx == targetLength) {
            if (i == query32-1) {
              d_cluster(idx) = (colNow[queryLength%16] >= minLength) ? target : -1;
              complete = 1;
            }
            break;
          }
          if (complete) break;
          k -= 2; tBase = (tPack >> k) & 3;
          m = 0; colPrev[m] = rowNow[tIdx+1];
          for (int l = 30; l >= 0; l -= 2) {
            m++;
            int qBase = (qPack >> l) & 3;
            colPrev[m] = colNow[m-1] + (qBase == tBase ? 1 : 0);
            colPrev[m] = kmax(colPrev[m], colPrev[m-1]);
            colPrev[m] = kmax(colPrev[m], colNow[m]);
          }
          tIdx++; rowPrevious[tIdx] = colPrev[16];
          if (tIdx == targetLength) {
            if (i == query32-1) {
              d_cluster(idx) = (colPrev[queryLength%16] >= minLength) ? target : -1;
              complete = 1;
            }
            break;
          }
          if (complete) break;
        }
        if (complete) break;
      }
    }
  });
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv) {
  Option option;
  checkOption(argc, argv, option);
  std::vector<Read> reads;
  if (readFile(reads, option)) return 1;

  int readsCount = (int)reads.size();
  std::vector<int>  h_lengths(readsCount);
  std::vector<long> h_offsets(readsCount + 1);
  h_offsets[0] = 0;
  for (int i = 0; i < readsCount; i++) {
    int len = (int)reads[i].data.size();
    h_lengths[i] = len;
    h_offsets[i+1] = h_offsets[i] + (long)(len/16)*16 + 16;
  }
  long total_length = h_offsets[readsCount];
  std::vector<char> h_reads(total_length, 0);
  for (int i = 0; i < readsCount; i++)
    memcpy(&h_reads[h_offsets[i]], reads[i].data.c_str(), h_lengths[i]);

  std::vector<unsigned int> h_compressed(total_length / 16, 0);
  std::vector<int>           h_gaps(readsCount, 0);
  std::vector<unsigned short>h_indexs(total_length, 0);
  std::vector<long>          h_words(readsCount, 0);
  std::vector<unsigned short>h_orders(total_length, 0);
  std::vector<int>           h_magicBase(readsCount * 4, 0);
  std::vector<int>           h_cluster(readsCount, -1);
  std::vector<unsigned short>h_table(65536, 0);
  std::vector<int>           h_wordCutoff(readsCount, 0);

  auto t1 = std::chrono::high_resolution_clock::now();

  Kokkos::initialize(argc, argv);
  {
    // Device Views
    Kokkos::View<char*>           d_reads("reads",    total_length);
    Kokkos::View<int*>            d_lengths("lengths", readsCount);
    Kokkos::View<long*>           d_offsets("offsets", readsCount+1);
    Kokkos::View<unsigned int*>   d_compressed("comp", total_length/16);
    Kokkos::View<int*>            d_gaps("gaps",       readsCount);
    Kokkos::View<unsigned short*> d_indexs("indexs",   total_length);
    Kokkos::View<long*>           d_words("words",     readsCount);
    Kokkos::View<unsigned short*> d_orders("orders",   total_length);
    Kokkos::View<int*>            d_magicBase("magic",  readsCount*4);
    Kokkos::View<int*>            d_cluster("cluster", readsCount);
    Kokkos::View<unsigned short*> d_table("table",     65536);
    Kokkos::View<int*>            d_wordCutoff("wco",  readsCount);

    // Upload
    {
      auto hv = Kokkos::create_mirror_view(d_reads);
      for (long i = 0; i < total_length; i++) hv(i) = h_reads[i];
      Kokkos::deep_copy(d_reads, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_lengths);
      for (int i = 0; i < readsCount; i++) hv(i) = h_lengths[i];
      Kokkos::deep_copy(d_lengths, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_offsets);
      for (int i = 0; i <= readsCount; i++) hv(i) = h_offsets[i];
      Kokkos::deep_copy(d_offsets, hv);
    }
    Kokkos::deep_copy(d_cluster, -1);
    Kokkos::deep_copy(d_table, (unsigned short)0);

    // Step 1: convert bases
    kernel_baseToNumber(d_reads, total_length);
    Kokkos::fence();

    // Step 2: compress
    kernel_compressData(d_lengths, d_offsets, d_reads, d_compressed, d_gaps, readsCount);
    Kokkos::fence();

    // Step 3: create index
    int wordLength = option.wordLength;
    switch (wordLength) {
      case 4: kernel_createIndex<4>(d_reads,d_lengths,d_offsets,d_indexs,d_words,d_magicBase,readsCount); break;
      case 5: kernel_createIndex<5>(d_reads,d_lengths,d_offsets,d_indexs,d_words,d_magicBase,readsCount); break;
      case 6: kernel_createIndex<6>(d_reads,d_lengths,d_offsets,d_indexs,d_words,d_magicBase,readsCount); break;
      default: kernel_createIndex<7>(d_reads,d_lengths,d_offsets,d_indexs,d_words,d_magicBase,readsCount); break;
    }
    Kokkos::fence();

    // Step 4: cutoff
    kernel_createCutoff(option.threshold, wordLength, d_lengths, d_words, d_wordCutoff, readsCount);
    Kokkos::fence();

    // Step 5: sort index on host
    {
      auto hv_idx  = Kokkos::create_mirror_view(d_indexs);
      auto hv_off  = Kokkos::create_mirror_view(d_offsets);
      auto hv_wrd  = Kokkos::create_mirror_view(d_words);
      Kokkos::deep_copy(hv_idx,  d_indexs);
      Kokkos::deep_copy(hv_off,  d_offsets);
      Kokkos::deep_copy(hv_wrd,  d_words);
      for (int i = 0; i < readsCount; i++) {
        int start = (int)hv_off(i);
        int len   = (int)hv_wrd(i);
        std::sort(&hv_idx(start), &hv_idx(start) + len);
      }
      Kokkos::deep_copy(d_indexs, hv_idx);
    }

    // Step 6: merge index
    kernel_mergeIndex(d_offsets, d_indexs, d_orders, d_words, readsCount);
    Kokkos::fence();

    // Host mirror for cluster (needed for updateRepresentative)
    auto hv_cluster = Kokkos::create_mirror_view(d_cluster);
    Kokkos::deep_copy(hv_cluster, d_cluster);

    int r = -1;
    while (r < readsCount) {
      // Update representative on host
      Kokkos::deep_copy(hv_cluster, d_cluster);
      r++;
      while (r < readsCount) {
        if (hv_cluster(r) < 0) { hv_cluster(r) = r; break; }
        r++;
      }
      if (r >= readsCount) break;
      Kokkos::deep_copy(d_cluster, hv_cluster);

      kernel_makeTable(d_offsets, d_indexs, d_orders, d_words, d_table, r);
      Kokkos::fence();
      kernel_magic(option.threshold, d_lengths, d_magicBase, d_cluster, r, readsCount);
      Kokkos::fence();
      kernel_filter(option.threshold, wordLength, d_lengths, d_offsets,
                    d_indexs, d_orders, d_words, d_wordCutoff, d_cluster, d_table, readsCount);
      Kokkos::fence();
      kernel_align(option.threshold, d_lengths, d_offsets, d_compressed, d_gaps,
                   r, d_cluster, readsCount);
      Kokkos::fence();
      kernel_cleanTable(d_offsets, d_indexs, d_orders, d_words, d_table, r);
      Kokkos::fence();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("Device offload time %lf secs\n", total_time / 1e6);

    // Copy cluster back
    Kokkos::deep_copy(hv_cluster, d_cluster);
    for (int i = 0; i < readsCount; i++) h_cluster[i] = hv_cluster(i);
  }
  Kokkos::finalize();

  std::ofstream outf(option.outputFile);
  int sum = 0;
  for (int i = 0; i < readsCount; i++)
    if (h_cluster[i] == i) { outf << reads[i].name << "\n" << reads[i].data << "\n"; sum++; }
  outf.close();
  std::cout << "cluster count: " << sum << "\n";
  return 0;
}
