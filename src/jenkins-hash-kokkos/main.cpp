#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

/*
   -------------------------------------------------------------------------------
   mix -- mix 3 32-bit values reversibly.
   -------------------------------------------------------------------------------
*/
#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

/*
   -------------------------------------------------------------------------------
   final -- final mixing of 3 32-bit values (a,b,c) into c
   -------------------------------------------------------------------------------
*/
#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

KOKKOS_INLINE_FUNCTION
unsigned int mixRemainder(unsigned int a,
    unsigned int b,
    unsigned int c,
    unsigned int k0,
    unsigned int k1,
    unsigned int k2,
    unsigned int length)
{
  switch(length)
  {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff; b+=k1; a+=k0; break;
    case 9 : c+=k2&0xff; b+=k1; a+=k0; break;
    case 8 : b+=k1; a+=k0; break;
    case 7 : b+=k1&0xffffff; a+=k0; break;
    case 6 : b+=k1&0xffff; a+=k0; break;
    case 5 : b+=k1&0xff; a+=k0; break;
    case 4 : a+=k0; break;
    case 3 : a+=k0&0xffffff; break;
    case 2 : a+=k0&0xffff; break;
    case 1 : a+=k0&0xff; break;
    case 0 : return c;              /* zero length strings require no mixing */
  }

  final(a,b,c);
  return c;
}

unsigned int hashlittle( const void *key, size_t length, unsigned int initval)
{
  unsigned int a,b,c;                                          /* internal state */

  /* Set up the internal state */
  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  const unsigned int *k = (const unsigned int *)key;         /* read 32-bit chunks */

  /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
  while (length > 12)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a,b,c);
    length -= 12;
    k += 3;
  }

  switch(length)
  {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=k[2]&0xffffff; b+=k[1]; a+=k[0]; break;
    case 10: c+=k[2]&0xffff; b+=k[1]; a+=k[0]; break;
    case 9 : c+=k[2]&0xff; b+=k[1]; a+=k[0]; break;
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=k[1]&0xffffff; a+=k[0]; break;
    case 6 : b+=k[1]&0xffff; a+=k[0]; break;
    case 5 : b+=k[1]&0xff; a+=k[0]; break;
    case 4 : a+=k[0]; break;
    case 3 : a+=k[0]&0xffffff; break;
    case 2 : a+=k[0]&0xffff; break;
    case 1 : a+=k[0]&0xff; break;
    case 0 : return c;              /* zero length strings require no mixing */
  }

  final(a,b,c);
  return c;
}


int main(int argc, char** argv) {

  if (argc != 4) {
    printf("Usage: %s <block size> <number of strings> <repeat>\n", argv[0]);
    return 1;
  }

  int block_size = atoi(argv[1]);  // work group size (unused in Kokkos CPU path)
  unsigned long N = atol(argv[2]); // total number of strings
  int repeat = atoi(argv[3]);

  // sample gold result
  const char* str = "Four score and seven years ago";
  unsigned int c = hashlittle(str, 30, 1);
  printf("input string: %s hash is %.8x\n", str, c);   /* cd628161 */

  unsigned int *keys_h = NULL;
  unsigned int *lens_h = NULL;
  unsigned int *initvals_h = NULL;
  unsigned int *out_h = NULL;

  // padded to 64 bytes (16 words)
  posix_memalign((void**)&keys_h,    1024, sizeof(unsigned int)*N*16);
  posix_memalign((void**)&lens_h,    1024, sizeof(unsigned int)*N);
  posix_memalign((void**)&initvals_h,1024, sizeof(unsigned int)*N);
  posix_memalign((void**)&out_h,     1024, sizeof(unsigned int)*N);

  // the kernel supports up to 60 bytes
  srand(2);
  char src[64];
  memcpy(src, str, 64);
  for (unsigned long i = 0; i < N; i++) {
    memcpy((unsigned char*)keys_h + i*16*sizeof(unsigned int), src, 64);
    lens_h[i]    = rand()%61;
    initvals_h[i] = i%2;
  }

  Kokkos::initialize(argc, argv);
  {
    // Device Views
    Kokkos::View<unsigned int*> d_keys    ("keys",     N*16);
    Kokkos::View<unsigned int*> d_lens    ("lens",     N);
    Kokkos::View<unsigned int*> d_initvals("initvals", N);
    Kokkos::View<unsigned int*> d_out     ("out",      N);

    // Host mirrors for data transfer
    auto hm_keys     = Kokkos::create_mirror_view(d_keys);
    auto hm_lens     = Kokkos::create_mirror_view(d_lens);
    auto hm_initvals = Kokkos::create_mirror_view(d_initvals);
    auto hm_out      = Kokkos::create_mirror_view(d_out);

    for (unsigned long i = 0; i < N*16; i++) hm_keys(i)     = keys_h[i];
    for (unsigned long i = 0; i < N;    i++) hm_lens(i)     = lens_h[i];
    for (unsigned long i = 0; i < N;    i++) hm_initvals(i) = initvals_h[i];

    Kokkos::deep_copy(d_keys,     hm_keys);
    Kokkos::deep_copy(d_lens,     hm_lens);
    Kokkos::deep_copy(d_initvals, hm_initvals);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("jenkins_hash",
        Kokkos::RangePolicy<Kokkos::IndexType<unsigned long>>(0, N),
        KOKKOS_LAMBDA(const unsigned long id) {
          unsigned int length   = d_lens(id);
          unsigned int initval  = d_initvals(id);
          const unsigned int *k = d_keys.data() + id*16;  // each key slot is 16 words (64 bytes), data up to 60 bytes

          unsigned int a, b, c;
          unsigned int r0, r1, r2;
          a = b = c = 0xdeadbeef + length + initval;

          /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
          while (length > 12) {
            a += k[0];
            b += k[1];
            c += k[2];
            mix(a,b,c);
            length -= 12;
            k += 3;
          }
          r0 = k[0];
          r1 = k[1];
          r2 = k[2];

          d_out(id) = mixRemainder(a, b, c, r0, r1, r2, length);
        });
    }
    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

    Kokkos::deep_copy(hm_out, d_out);
    for (unsigned long i = 0; i < N; i++) out_h[i] = hm_out(i);
  }
  Kokkos::finalize();

  printf("Verify the results computed on the device..\n");
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    c = hashlittle(&keys_h[i*16], lens_h[i], initvals_h[i]);
    if (out_h[i] != c) {
      printf("Error: at %lu gpu hash is %.8x  cpu hash is %.8x\n", i, out_h[i], c);
      error = true;
      break;
    }
  }

  printf("%s\n", error ? "FAIL" : "PASS");

  free(keys_h);
  free(lens_h);
  free(initvals_h);
  free(out_h);

  return 0;
}
