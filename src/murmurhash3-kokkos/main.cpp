//-------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
//-------------------------------------------------------------------------
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <Kokkos_Core.hpp>

#define KOKKOS_INLINE_FUNCTION_ROTL KOKKOS_INLINE_FUNCTION

KOKKOS_INLINE_FUNCTION
uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define ROTL64(x,y)     rotl64(x,y)
#define BIG_CONSTANT(x) (x##LLU)

KOKKOS_INLINE_FUNCTION
uint64_t getblock64(const uint8_t* p, uint32_t i) {
  uint64_t s = 0;
  for (uint32_t n = 0; n < 8; n++)
    s |= ((uint64_t)p[8*i+n] << (n*8));
  return s;
}

KOKKOS_INLINE_FUNCTION
uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

KOKKOS_INLINE_FUNCTION
void MurmurHash3_x64_128(const uint8_t* data, const uint32_t len,
                         const uint32_t seed, uint64_t* out)
{
  const uint32_t nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  for (uint32_t i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock64(data, i*2+0);
    uint64_t k2 = getblock64(data, i*2+1);

    k1 *= c1; k1 = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
    h1 = ROTL64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

    k2 *= c2; k2 = ROTL64(k2,33); k2 *= c1; h2 ^= k2;
    h2 = ROTL64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
  }

  const uint8_t* tail = data + nblocks*16;
  uint64_t k1 = 0, k2 = 0;

  switch (len & 15) {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48; // fall through
    case 14: k2 ^= ((uint64_t)tail[13]) << 40; // fall through
    case 13: k2 ^= ((uint64_t)tail[12]) << 32; // fall through
    case 12: k2 ^= ((uint64_t)tail[11]) << 24; // fall through
    case 11: k2 ^= ((uint64_t)tail[10]) << 16; // fall through
    case 10: k2 ^= ((uint64_t)tail[ 9]) <<  8; // fall through
    case  9: k2 ^= ((uint64_t)tail[ 8]) <<  0;
      k2 *= c2; k2 = ROTL64(k2,33); k2 *= c1; h2 ^= k2;
      // fall through
    case  8: k1 ^= ((uint64_t)tail[ 7]) << 56; // fall through
    case  7: k1 ^= ((uint64_t)tail[ 6]) << 48; // fall through
    case  6: k1 ^= ((uint64_t)tail[ 5]) << 40; // fall through
    case  5: k1 ^= ((uint64_t)tail[ 4]) << 32; // fall through
    case  4: k1 ^= ((uint64_t)tail[ 3]) << 24; // fall through
    case  3: k1 ^= ((uint64_t)tail[ 2]) << 16; // fall through
    case  2: k1 ^= ((uint64_t)tail[ 1]) <<  8; // fall through
    case  1: k1 ^= ((uint64_t)tail[ 0]) <<  0;
      k1 *= c1; k1 = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
  }

  h1 ^= len; h2 ^= len;
  h1 += h2;  h2 += h1;
  h1 = fmix64(h1); h2 = fmix64(h2);
  h1 += h2;  h2 += h1;

  out[0] = h1;
  out[1] = h2;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  uint32_t numKeys = atoi(argv[1]);
  uint32_t repeat  = atoi(argv[2]);

  srand(3);

  // Build host-side per-key data
  uint32_t* length = (uint32_t*)malloc(sizeof(uint32_t) * numKeys);
  uint8_t** keys   = (uint8_t**)malloc(sizeof(uint8_t*) * numKeys);
  uint64_t** out   = (uint64_t**)malloc(sizeof(uint64_t*) * numKeys);

  for (uint32_t i = 0; i < numKeys; i++) {
    length[i] = rand() % 10000;
    keys[i]   = (uint8_t*)malloc(length[i]);
    out[i]    = (uint64_t*)malloc(2 * sizeof(uint64_t));
    for (uint32_t c = 0; c < length[i]; c++) keys[i][c] = c % 256;
    // reference hash on host
    MurmurHash3_x64_128(keys[i], length[i], i, out[i]);
  }

  // Build flattened key array
  uint32_t* d_length = (uint32_t*)malloc(sizeof(uint32_t) * (numKeys + 1));
  uint32_t total_length = 0;
  d_length[0] = 0;
  for (uint32_t i = 0; i < numKeys; i++) {
    total_length += length[i];
    d_length[i+1] = total_length;
  }
  uint8_t* d_keys_h = (uint8_t*)malloc(sizeof(uint8_t) * total_length);
  for (uint32_t i = 0; i < numKeys; i++)
    memcpy(d_keys_h + d_length[i], keys[i], length[i]);

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<uint8_t*>  keys_d("keys",   total_length);
    Kokkos::View<uint32_t*> len_d ("lengths", numKeys + 1);
    Kokkos::View<uint32_t*> klen_d("klens",   numKeys);
    Kokkos::View<uint64_t*> out_d ("out",     2 * numKeys);

    // Copy to device
    {
      auto km = Kokkos::create_mirror_view(keys_d);
      memcpy(km.data(), d_keys_h, total_length);
      Kokkos::deep_copy(keys_d, km);
    }
    {
      auto lm = Kokkos::create_mirror_view(len_d);
      for (uint32_t i = 0; i <= numKeys; i++) lm(i) = d_length[i];
      Kokkos::deep_copy(len_d, lm);
    }
    {
      auto lm = Kokkos::create_mirror_view(klen_d);
      for (uint32_t i = 0; i < numKeys; i++) lm(i) = length[i];
      Kokkos::deep_copy(klen_d, lm);
    }

    auto start = std::chrono::steady_clock::now();

    for (uint32_t n = 0; n < repeat; n++) {
      Kokkos::parallel_for("murmurhash3", numKeys,
        KOKKOS_LAMBDA(const uint32_t i) {
          const uint8_t*  key_ptr = keys_d.data() + len_d(i);
          uint64_t*       out_ptr = out_d.data()  + i * 2;
          MurmurHash3_x64_128(key_ptr, klen_d(i), i, out_ptr);
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

    // Copy back and verify
    auto out_h = Kokkos::create_mirror_view(out_d);
    Kokkos::deep_copy(out_h, out_d);

    bool error = false;
    for (uint32_t i = 0; i < numKeys; i++) {
      if (out_h(2*i) != out[i][0] || out_h(2*i+1) != out[i][1]) {
        error = true;
        break;
      }
    }
    printf("%s\n", error ? "FAIL" : "SUCCESS");
  }
  Kokkos::finalize();

  for (uint32_t i = 0; i < numKeys; i++) { free(out[i]); free(keys[i]); }
  free(keys); free(out); free(length);
  free(d_keys_h); free(d_length);
  return 0;
}
