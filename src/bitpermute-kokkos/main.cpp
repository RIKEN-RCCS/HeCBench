/*
 * Kokkos port of bitpermute (NTT bit-reversal permutation).
 * Replaces CUDA intrinsics __brev/__brevll with portable bit-reversal.
 */

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>

#define MAX_LG_DOMAIN_SIZE 28
typedef long fr_t;

#if MAX_LG_DOMAIN_SIZE <= 32
typedef unsigned int index_t;
#else
typedef size_t index_t;
#endif

// Portable bit reversal for 32-bit values
KOKKOS_INLINE_FUNCTION
unsigned int brev32(unsigned int x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  x = (x >> 16) | (x << 16);
  return x;
}

KOKKOS_INLINE_FUNCTION
unsigned long long brev64(unsigned long long x) {
  return ((unsigned long long)brev32((unsigned int)x) << 32)
       | (unsigned long long)brev32((unsigned int)(x >> 32));
}

KOKKOS_INLINE_FUNCTION
index_t bit_rev(index_t i, unsigned int nbits) {
  if (sizeof(i) == 4 || nbits <= 32)
    return (index_t)(brev32((unsigned int)i) >> (32 - nbits));
  else
    return (index_t)(brev64((unsigned long long)i) >> (64 - nbits));
}

// CPU reference implementation
template<typename T>
T reverseBits(T num) {
  unsigned int NO_OF_BITS = sizeof(T) * 8;
  T reverse_num = 0;
  for (unsigned int i = 0; i < NO_OF_BITS; i++)
    if (num & ((T)1 << i))
      reverse_num |= (T)1 << ((NO_OF_BITS - 1) - i);
  return reverse_num;
}

template<typename T>
T rev_cpu(T i, unsigned int nbits) {
  if (sizeof(T) == 4 || nbits <= 32)
    return reverseBits(i) >> (8*sizeof(unsigned int) - nbits);
  else
    return reverseBits(i) >> (8*sizeof(unsigned long long) - nbits);
}

void bit_rev_cpu(fr_t* out, const fr_t* in, uint32_t lg_domain_size) {
  uint32_t domain_size = 1 << lg_domain_size;
  for (uint32_t i = 0; i < domain_size; i++) {
    index_t r = rev_cpu(i, lg_domain_size);
    if (i < r) {
      out[r] = in[i];
      out[i] = in[r];
    }
  }
}

void bit_permute(const int lg_domain_size, const int repeat) {
  assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
  size_t domain_size = (size_t)1 << lg_domain_size;
  printf("Domain size is %zu\n", domain_size);

  Kokkos::View<fr_t*> d_inout("d_inout", domain_size);
  std::vector<fr_t> h_inout(domain_size), h_out(domain_size);
  for (size_t i = 0; i < domain_size; i++) h_inout[i] = h_out[i] = (fr_t)i;

  // Warmup + verify
  auto hv = Kokkos::create_mirror_view(d_inout);
  for (size_t i = 0; i < domain_size; i++) hv(i) = h_inout[i];
  Kokkos::deep_copy(d_inout, hv);

  // GPU bit reversal
  Kokkos::parallel_for("bit_rev", domain_size, KOKKOS_LAMBDA(size_t i) {
    index_t r = bit_rev((index_t)i, lg_domain_size);
    if (i < (size_t)r) {
      fr_t t0 = d_inout(i);
      fr_t t1 = d_inout(r);
      d_inout(i) = t1;
      d_inout(r) = t0;
    }
  });
  Kokkos::fence();
  Kokkos::deep_copy(hv, d_inout);

  // CPU reference
  bit_rev_cpu(h_out.data(), h_inout.data(), lg_domain_size);

  int error = memcmp(h_out.data(), hv.data(), domain_size * sizeof(fr_t));
  printf("%s\n", error ? "FAIL" : "PASS");

  // Reset for timing
  for (size_t i = 0; i < domain_size; i++) hv(i) = (fr_t)i;
  Kokkos::deep_copy(d_inout, hv);

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("bit_rev_timed", domain_size, KOKKOS_LAMBDA(size_t i) {
      index_t rev = bit_rev((index_t)i, lg_domain_size);
      if (i < (size_t)rev) {
        fr_t t0 = d_inout(i);
        fr_t t1 = d_inout(rev);
        d_inout(i) = t1;
        d_inout(rev) = t0;
      }
    });
  }
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3f) / repeat);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);
    bit_permute(10, repeat);
    bit_permute(11, repeat);
    bit_permute(15, repeat);
    bit_permute(27, repeat);
    bit_permute(28, repeat);
  }
  Kokkos::finalize();
  return 0;
}
