// Kokkos port of MD5 Hash benchmark
// Original OMP target source: src/md5hash-omp/MD5Hash.cpp
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <chrono>
#include <Kokkos_Core.hpp>

#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
#define F(x,y,z) ((x & y) | ((~x) & z))
#define G(x,y,z) ((x & z) | ((~z) & y))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | (~z)))
#define ROUND(w, r, k, v, x, y, z, func) \
{ \
  a = a + func(b,c,d) + k + w; \
  unsigned int temp = d; \
  d = c; \
  c = b; \
  b = b + LEFTROTATE(a, r); \
  a = temp; \
}

KOKKOS_INLINE_FUNCTION
void md5_2words(unsigned int *words, unsigned int len, unsigned int *digest)
{
  unsigned int h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
  unsigned int a = h0, b = h1, c = h2, d = h3;
  unsigned int WL = len * 8, W0 = words[0], W1 = words[1];
  switch (len) {
    case 0: W0 |= 0x00000080; break; case 1: W0 |= 0x00008000; break;
    case 2: W0 |= 0x00800000; break; case 3: W0 |= 0x80000000; break;
    case 4: W1 |= 0x00000080; break; case 5: W1 |= 0x00008000; break;
    case 6: W1 |= 0x00800000; break; case 7: W1 |= 0x80000000; break;
  }
  ROUND(W0,7,0xd76aa478,a,b,c,d,F); ROUND(W1,12,0xe8c7b756,d,a,b,c,F);
  ROUND(0,17,0x242070db,c,d,a,b,F); ROUND(0,22,0xc1bdceee,b,c,d,a,F);
  ROUND(0,7,0xf57c0faf,a,b,c,d,F);  ROUND(0,12,0x4787c62a,d,a,b,c,F);
  ROUND(0,17,0xa8304613,c,d,a,b,F); ROUND(0,22,0xfd469501,b,c,d,a,F);
  ROUND(0,7,0x698098d8,a,b,c,d,F);  ROUND(0,12,0x8b44f7af,d,a,b,c,F);
  ROUND(0,17,0xffff5bb1,c,d,a,b,F); ROUND(0,22,0x895cd7be,b,c,d,a,F);
  ROUND(0,7,0x6b901122,a,b,c,d,F);  ROUND(0,12,0xfd987193,d,a,b,c,F);
  ROUND(WL,17,0xa679438e,c,d,a,b,F);ROUND(0,22,0x49b40821,b,c,d,a,F);
  ROUND(W1,5,0xf61e2562,a,b,c,d,G); ROUND(0,9,0xc040b340,d,a,b,c,G);
  ROUND(0,14,0x265e5a51,c,d,a,b,G); ROUND(W0,20,0xe9b6c7aa,b,c,d,a,G);
  ROUND(0,5,0xd62f105d,a,b,c,d,G);  ROUND(0,9,0x02441453,d,a,b,c,G);
  ROUND(0,14,0xd8a1e681,c,d,a,b,G); ROUND(0,20,0xe7d3fbc8,b,c,d,a,G);
  ROUND(0,5,0x21e1cde6,a,b,c,d,G);  ROUND(WL,9,0xc33707d6,d,a,b,c,G);
  ROUND(0,14,0xf4d50d87,c,d,a,b,G); ROUND(0,20,0x455a14ed,b,c,d,a,G);
  ROUND(0,5,0xa9e3e905,a,b,c,d,G);  ROUND(0,9,0xfcefa3f8,d,a,b,c,G);
  ROUND(0,14,0x676f02d9,c,d,a,b,G); ROUND(0,20,0x8d2a4c8a,b,c,d,a,G);
  ROUND(0,4,0xfffa3942,a,b,c,d,H);  ROUND(0,11,0x8771f681,d,a,b,c,H);
  ROUND(0,16,0x6d9d6122,c,d,a,b,H); ROUND(WL,23,0xfde5380c,b,c,d,a,H);
  ROUND(W1,4,0xa4beea44,a,b,c,d,H); ROUND(0,11,0x4bdecfa9,d,a,b,c,H);
  ROUND(0,16,0xf6bb4b60,c,d,a,b,H); ROUND(0,23,0xbebfbc70,b,c,d,a,H);
  ROUND(0,4,0x289b7ec6,a,b,c,d,H);  ROUND(W0,11,0xeaa127fa,d,a,b,c,H);
  ROUND(0,16,0xd4ef3085,c,d,a,b,H); ROUND(0,23,0x04881d05,b,c,d,a,H);
  ROUND(0,4,0xd9d4d039,a,b,c,d,H);  ROUND(0,11,0xe6db99e5,d,a,b,c,H);
  ROUND(0,16,0x1fa27cf8,c,d,a,b,H); ROUND(0,23,0xc4ac5665,b,c,d,a,H);
  ROUND(W0,6,0xf4292244,a,b,c,d,I); ROUND(0,10,0x432aff97,d,a,b,c,I);
  ROUND(WL,15,0xab9423a7,c,d,a,b,I);ROUND(0,21,0xfc93a039,b,c,d,a,I);
  ROUND(0,6,0x655b59c3,a,b,c,d,I);  ROUND(0,10,0x8f0ccc92,d,a,b,c,I);
  ROUND(0,15,0xffeff47d,c,d,a,b,I); ROUND(W1,21,0x85845dd1,b,c,d,a,I);
  ROUND(0,6,0x6fa87e4f,a,b,c,d,I);  ROUND(0,10,0xfe2ce6e0,d,a,b,c,I);
  ROUND(0,15,0xa3014314,c,d,a,b,I); ROUND(0,21,0x4e0811a1,b,c,d,a,I);
  ROUND(0,6,0xf7537e82,a,b,c,d,I);  ROUND(0,10,0xbd3af235,d,a,b,c,I);
  ROUND(0,15,0x2ad7d2bb,c,d,a,b,I); ROUND(0,21,0xeb86d391,b,c,d,a,I);
  digest[0] = h0 + a; digest[1] = h1 + b; digest[2] = h2 + c; digest[3] = h3 + d;
}

KOKKOS_INLINE_FUNCTION
int FindKeyspaceSize(int byteLength, int valsPerByte) {
  int keyspace = 1;
  for (int i = 0; i < byteLength; ++i) {
    if (keyspace >= 0x7fffffff / valsPerByte) return -1;
    keyspace *= valsPerByte;
  }
  return keyspace;
}

KOKKOS_INLINE_FUNCTION
void IndexToKey(unsigned int index, int byteLength, int valsPerByte, unsigned char vals[8]) {
  vals[0] = index % valsPerByte; index /= valsPerByte;
  vals[1] = index % valsPerByte; index /= valsPerByte;
  vals[2] = index % valsPerByte; index /= valsPerByte;
  vals[3] = index % valsPerByte; index /= valsPerByte;
  vals[4] = index % valsPerByte; index /= valsPerByte;
  vals[5] = index % valsPerByte; index /= valsPerByte;
  vals[6] = index % valsPerByte; index /= valsPerByte;
  vals[7] = index % valsPerByte;
}

std::string AsHex(unsigned char *vals, int len) {
  std::ostringstream out;
  char tmp[256];
  for (int i = 0; i < len; ++i) { sprintf(tmp, "%2.2X", vals[i]); out << tmp; }
  return out.str();
}

void FindKeyWithDigest_GPU(
    const unsigned int searchDigest[4],
    const int byteLength,
    const int valsPerByte,
    int *foundIndex,
    unsigned char foundKey[8],
    unsigned int foundDigest[4])
{
  int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
  int nblocks = (int)ceil((double)keyspace / (double)valsPerByte);

  unsigned int sd0 = searchDigest[0], sd1 = searchDigest[1];
  unsigned int sd2 = searchDigest[2], sd3 = searchDigest[3];

  Kokkos::View<int[1]>           d_foundIndex("d_foundIndex");
  Kokkos::View<unsigned char[8]> d_foundKey("d_foundKey");
  Kokkos::View<unsigned int[4]>  d_foundDigest("d_foundDigest");

  Kokkos::deep_copy(d_foundIndex, -1);

  Kokkos::parallel_for("md5hash",
    Kokkos::RangePolicy<>(0, nblocks),
    KOKKOS_LAMBDA(const int threadid) {
      int startindex = threadid * valsPerByte;
      unsigned char key[8] = {0,0,0,0,0,0,0,0};
      IndexToKey(startindex, byteLength, valsPerByte, key);
      for (int j = 0; j < valsPerByte && startindex + j < keyspace; ++j) {
        unsigned int digest[4];
        md5_2words((unsigned int*)key, byteLength, digest);
        if (digest[0] == sd0 && digest[1] == sd1 && digest[2] == sd2 && digest[3] == sd3) {
          d_foundIndex(0) = startindex + j;
          for (int k = 0; k < 8; k++) d_foundKey(k) = key[k];
          for (int k = 0; k < 4; k++) d_foundDigest(k) = digest[k];
        }
        ++key[0];
      }
    });
  Kokkos::fence();

  auto h_idx = Kokkos::create_mirror_view(d_foundIndex);
  auto h_key = Kokkos::create_mirror_view(d_foundKey);
  auto h_dig = Kokkos::create_mirror_view(d_foundDigest);
  Kokkos::deep_copy(h_idx, d_foundIndex);
  Kokkos::deep_copy(h_key, d_foundKey);
  Kokkos::deep_copy(h_dig, d_foundDigest);

  *foundIndex = h_idx(0);
  for (int k = 0; k < 8; k++) foundKey[k] = h_key(k);
  for (int k = 0; k < 4; k++) foundDigest[k] = h_dig(k);
}

void FindKeyWithDigest_CPU(
    const unsigned int searchDigest[4],
    const int byteLength,
    const int valsPerByte,
    int *foundIndex,
    unsigned char foundKey[8],
    unsigned int foundDigest[4])
{
  int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
  for (int i = 0; i < keyspace; i += valsPerByte) {
    unsigned char key[8] = {0,0,0,0,0,0,0,0};
    IndexToKey(i, byteLength, valsPerByte, key);
    for (int j = 0; j < valsPerByte; ++j) {
      unsigned int digest[4];
      md5_2words((unsigned int*)key, byteLength, digest);
      if (digest[0]==searchDigest[0] && digest[1]==searchDigest[1] &&
          digest[2]==searchDigest[2] && digest[3]==searchDigest[3]) {
        *foundIndex = i + j;
        for (int k = 0; k < 8; k++) foundKey[k] = key[k];
        for (int k = 0; k < 4; k++) foundDigest[k] = digest[k];
      }
      ++key[0];
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <offload 0|1> <passes>\n";
    return 1;
  }
  int offload = atoi(argv[1]);
  int passes  = atoi(argv[2]);
  bool verbose = true;

  Kokkos::initialize(argc, argv);
  {
    for (int size = 1; size <= 4; size++) {
      const int sizes_byteLength[]  = { 7,  5,  6,  5};
      const int sizes_valsPerByte[] = {10, 35, 25, 70};
      const int byteLength   = sizes_byteLength[size-1];
      const int valsPerByte  = sizes_valsPerByte[size-1];

      if (verbose)
        std::cout << "Searching keys of length " << byteLength
                  << " bytes and " << valsPerByte << " values per byte\n";

      const int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
      if (keyspace < 0) { std::cerr << "Error: keyspace overflow\n"; return -1; }
      if (byteLength > 7) { std::cerr << "Error: byteLength > 7\n"; return -1; }
      if (verbose) std::cout << "|keyspace| = " << keyspace << "\n";

      srandom(12345);
      for (int pass = 0; pass < passes; ++pass) {
        int randomIndex = random() % keyspace;
        unsigned char randomKey[8] = {0,0,0,0,0,0,0,0};
        unsigned int  randomDigest[4];
        IndexToKey(randomIndex, byteLength, valsPerByte, randomKey);
        md5_2words((unsigned int*)randomKey, byteLength, randomDigest);

        if (verbose) {
          std::cout << "\n--- pass " << pass << " ---\n";
          std::cout << " randomIndex  = " << randomIndex << "\n";
          std::cout << " randomKey    = 0x" << AsHex(randomKey, 8) << "\n";
          std::cout << " randomDigest = " << AsHex((unsigned char*)randomDigest, 16) << "\n";
        }

        unsigned int foundDigest[4] = {0,0,0,0};
        int foundIndex = -1;
        unsigned char foundKey[8] = {0,0,0,0,0,0,0,0};

        auto start = std::chrono::steady_clock::now();
        if (offload == 0)
          FindKeyWithDigest_CPU(randomDigest, byteLength, valsPerByte, &foundIndex, foundKey, foundDigest);
        else
          FindKeyWithDigest_GPU(randomDigest, byteLength, valsPerByte, &foundIndex, foundKey, foundDigest);
        auto end = std::chrono::steady_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        double rate = (double)keyspace / ((double)t / 1000) / 1.e9;
        if (verbose) std::cout << "time = " << t << " ms, rate = " << rate << " GHash/sec\n";

        bool ok = (foundIndex >= 0) && (foundIndex == randomIndex);
        for (int k = 0; k < 8 && ok; k++) ok = (foundKey[k] == randomKey[k]);
        for (int k = 0; k < 4 && ok; k++) ok = (foundDigest[k] == randomDigest[k]);
        if (!ok) std::cerr << "\nERROR: mismatch\n";
        else if (verbose) {
          std::cout << " foundIndex  = " << foundIndex << "\n";
          std::cout << " foundKey    = 0x" << AsHex(foundKey, 8) << "\n";
          std::cout << " foundDigest = " << AsHex((unsigned char*)foundDigest, 16) << "\n";
        }
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
