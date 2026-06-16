// Jenkins hash benchmark – Kokkos port from OpenMP target version.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <Kokkos_Core.hpp>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c) \
{ \
  a -= c; a ^= rot(c, 4); c += b; \
  b -= a; b ^= rot(a, 6); a += c; \
  c -= b; c ^= rot(b, 8); b += a; \
  a -= c; a ^= rot(c,16); c += b; \
  b -= a; b ^= rot(a,19); a += c; \
  c -= b; c ^= rot(b, 4); b += a; \
}

#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c, 4); \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

KOKKOS_INLINE_FUNCTION
unsigned int mixRemainder(unsigned int a, unsigned int b, unsigned int c,
                          unsigned int k0, unsigned int k1, unsigned int k2,
                          unsigned int length)
{
  switch (length) {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff;   b+=k1; a+=k0; break;
    case  9: c+=k2&0xff;     b+=k1; a+=k0; break;
    case  8: b+=k1; a+=k0; break;
    case  7: b+=k1&0xffffff; a+=k0; break;
    case  6: b+=k1&0xffff;   a+=k0; break;
    case  5: b+=k1&0xff;     a+=k0; break;
    case  4: a+=k0; break;
    case  3: a+=k0&0xffffff; break;
    case  2: a+=k0&0xffff;   break;
    case  1: a+=k0&0xff;     break;
    case  0: return c;
  }
  final(a, b, c);
  return c;
}

// CPU reference – identical logic to hashlittle
static unsigned int hashlittle(const void* key, size_t length, unsigned int initval)
{
  unsigned int a, b, c;
  a = b = c = 0xdeadbeef + (unsigned int)length + initval;

  const unsigned int* k = (const unsigned int*)key;
  while (length > 12) {
    a += k[0]; b += k[1]; c += k[2];
    mix(a, b, c);
    length -= 12; k += 3;
  }
  unsigned int r0=k[0], r1=k[1], r2=k[2];
  return mixRemainder(a, b, c, r0, r1, r2, (unsigned int)length);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <block_size> <N> <repeat>\n", argv[0]);
    return 1;
  }

  int           block_size = atoi(argv[1]);
  unsigned long N          = atol(argv[2]);
  int           repeat     = atoi(argv[3]);

  // Sample gold check
  const char* str = "Four score and seven years ago";
  unsigned int gold = hashlittle(str, 30, 1);
  printf("input string: %s  hash = %.8x\n", str, gold);  // expect cd628161

  // Allocate host arrays – each key is padded to 64 bytes (16 words)
  unsigned int* keys     = (unsigned int*)malloc(sizeof(unsigned int) * N * 16);
  unsigned int* lens     = (unsigned int*)malloc(sizeof(unsigned int) * N);
  unsigned int* initvals = (unsigned int*)malloc(sizeof(unsigned int) * N);
  unsigned int* out_h    = (unsigned int*)malloc(sizeof(unsigned int) * N);

  char src[64];
  memcpy(src, str, 30);
  srand(2);
  for (unsigned long i = 0; i < N; i++) {
    memcpy((unsigned char*)keys + i*16*sizeof(unsigned int), src, 64);
    lens[i]     = (unsigned int)(rand() % 61);
    initvals[i] = (unsigned int)(i % 2);
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<unsigned int*> d_keys    ("keys",     N * 16);
    Kokkos::View<unsigned int*> d_lens    ("lens",     N);
    Kokkos::View<unsigned int*> d_initvals("initvals", N);
    Kokkos::View<unsigned int*> d_out     ("out",      N);

    {
      auto hk = Kokkos::create_mirror_view(d_keys);
      auto hl = Kokkos::create_mirror_view(d_lens);
      auto hi = Kokkos::create_mirror_view(d_initvals);
      for (unsigned long i = 0; i < N*16; i++) hk(i) = keys[i];
      for (unsigned long i = 0; i < N;    i++) { hl(i) = lens[i]; hi(i) = initvals[i]; }
      Kokkos::deep_copy(d_keys,     hk);
      Kokkos::deep_copy(d_lens,     hl);
      Kokkos::deep_copy(d_initvals, hi);
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("jenkins_hash",
        Kokkos::RangePolicy<>(0, (long)N),
        KOKKOS_LAMBDA(long id) {
          unsigned int length  = d_lens(id);
          unsigned int initval = d_initvals(id);
          const unsigned int* k = &d_keys(id * 16);

          unsigned int a, b, c;
          a = b = c = 0xdeadbeef + length + initval;

          unsigned int r0, r1, r2;
          while (length > 12) {
            a += k[0]; b += k[1]; c += k[2];
            mix(a, b, c);
            length -= 12; k += 3;
          }
          r0 = k[0]; r1 = k[1]; r2 = k[2];
          d_out(id) = mixRemainder(a, b, c, r0, r1, r2, length);
        });
    }
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel execution time: %f s\n", (ns * 1e-9) / repeat);

    auto h_out = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_out, d_out);
    for (unsigned long i = 0; i < N; i++) out_h[i] = h_out(i);
  }
  Kokkos::finalize();

  // Verify
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    unsigned int ref = hashlittle(&keys[i*16], lens[i], initvals[i]);
    if (out_h[i] != ref) {
      printf("Error at %lu: gpu=%.8x cpu=%.8x\n", i, out_h[i], ref);
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(keys); free(lens); free(initvals); free(out_h);
  return error ? 1 : 0;
}
