#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdint>

#include "complex.h"

// LCG random number generator – callable from device
KOKKOS_INLINE_FUNCTION
double LCG_random_double(uint64_t* seed) {
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

KOKKOS_INLINE_FUNCTION
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
  const uint64_t m = 9223372036854775808ULL;
  uint64_t a = 2806196910506780709ULL;
  uint64_t c = 1ULL;
  n = n % m;
  uint64_t a_new = 1, c_new = 0;
  while (n > 0) {
    if (n & 1) { a_new *= a; c_new = c_new * a + c; }
    c *= (a + 1);
    a *= a;
    n >>= 1;
  }
  return (a_new * seed + c_new) % m;
}

static bool check(const char* cs, int n) {
  for (int i = 0; i < n; i++)
    if (cs[i] != 5) return false;
  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<char*> cs("cs", n);
    auto h_cs = Kokkos::create_mirror_view(cs);

    // ---- single-precision warmup ----
    Kokkos::parallel_for("complex_float_warmup", n, KOKKOS_LAMBDA(int i) {
      uint64_t seed = fast_forward_LCG(1ULL, (uint64_t)i);
      float r1 = (float)LCG_random_double(&seed);
      float r2 = (float)LCG_random_double(&seed);
      float r3 = (float)LCG_random_double(&seed);
      float r4 = (float)LCG_random_double(&seed);
      FloatComplex z1 = make_FloatComplex(r1, r2);
      FloatComplex z2 = make_FloatComplex(r3, r4);
      char s  = fabsf(Cabsf(Cmulf(z1,z2)) - Cabsf(z1)*Cabsf(z2)) < 1e-3f;
      s += fabsf(Cabsf(Caddf(z1,z2))*Cabsf(Caddf(z1,z2)) -
                 Crealf(Cmulf(Caddf(z1,z2), Caddf(Conjf(z1),Conjf(z2))))) < 1e-3f;
      s += fabsf(Cabsf(Csubf(z1,z2))*Cabsf(Csubf(z1,z2)) -
                 Crealf(Cmulf(Csubf(z1,z2), Csubf(Conjf(z1),Conjf(z2))))) < 1e-3f;
      s += fabsf(Crealf(Caddf(Cmulf(z1,Conjf(z2)), Cmulf(z2,Conjf(z1)))) -
                 2.0f*(Crealf(z1)*Crealf(z2) + Cimagf(z1)*Cimagf(z2))) < 1e-3f;
      s += fabsf(Cabsf(Cdivf(Conjf(z1),z2)) - Cabsf(Cdivf(Conjf(z1),Conjf(z2)))) < 1e-3f;
      cs(i) = s;
    });

    // ---- double-precision warmup ----
    Kokkos::parallel_for("complex_double_warmup", n, KOKKOS_LAMBDA(int i) {
      uint64_t seed = fast_forward_LCG(1ULL, (uint64_t)i);
      double r1 = LCG_random_double(&seed);
      double r2 = LCG_random_double(&seed);
      double r3 = LCG_random_double(&seed);
      double r4 = LCG_random_double(&seed);
      DoubleComplex z1 = make_DoubleComplex(r1, r2);
      DoubleComplex z2 = make_DoubleComplex(r3, r4);
      char s  = fabs(Cabs(Cmul(z1,z2)) - Cabs(z1)*Cabs(z2)) < 1e-3;
      s += fabs(Cabs(Cadd(z1,z2))*Cabs(Cadd(z1,z2)) -
                Creal(Cmul(Cadd(z1,z2), Cadd(Conj(z1),Conj(z2))))) < 1e-3;
      s += fabs(Cabs(Csub(z1,z2))*Cabs(Csub(z1,z2)) -
                Creal(Cmul(Csub(z1,z2), Csub(Conj(z1),Conj(z2))))) < 1e-3;
      s += fabs(Creal(Cadd(Cmul(z1,Conj(z2)), Cmul(z2,Conj(z1)))) -
                2.0*(Creal(z1)*Creal(z2) + Cimag(z1)*Cimag(z2))) < 1e-3;
      s += fabs(Cabs(Cdiv(Conj(z1),z2)) - Cabs(Cdiv(Conj(z1),Conj(z2)))) < 1e-3;
      cs(i) = s;
    });
    Kokkos::fence();

    // ---- timed single-precision ----
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("complex_float", n, KOKKOS_LAMBDA(int i) {
        uint64_t seed = fast_forward_LCG(1ULL, (uint64_t)i);
        float r1 = (float)LCG_random_double(&seed);
        float r2 = (float)LCG_random_double(&seed);
        float r3 = (float)LCG_random_double(&seed);
        float r4 = (float)LCG_random_double(&seed);
        FloatComplex z1 = make_FloatComplex(r1, r2);
        FloatComplex z2 = make_FloatComplex(r3, r4);
        char s  = fabsf(Cabsf(Cmulf(z1,z2)) - Cabsf(z1)*Cabsf(z2)) < 1e-3f;
        s += fabsf(Cabsf(Caddf(z1,z2))*Cabsf(Caddf(z1,z2)) -
                   Crealf(Cmulf(Caddf(z1,z2), Caddf(Conjf(z1),Conjf(z2))))) < 1e-3f;
        s += fabsf(Cabsf(Csubf(z1,z2))*Cabsf(Csubf(z1,z2)) -
                   Crealf(Cmulf(Csubf(z1,z2), Csubf(Conjf(z1),Conjf(z2))))) < 1e-3f;
        s += fabsf(Crealf(Caddf(Cmulf(z1,Conjf(z2)), Cmulf(z2,Conjf(z1)))) -
                   2.0f*(Crealf(z1)*Crealf(z2) + Cimagf(z1)*Cimagf(z2))) < 1e-3f;
        s += fabsf(Cabsf(Cdivf(Conjf(z1),z2)) - Cabsf(Cdivf(Conjf(z1),Conjf(z2)))) < 1e-3f;
        cs(i) = s;
      });
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    printf("Average kernel execution time (float) %f (s)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-9f / repeat);

    Kokkos::deep_copy(h_cs, cs);
    bool float_ok = check(h_cs.data(), n);

    // ---- timed double-precision ----
    Kokkos::fence();
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("complex_double", n, KOKKOS_LAMBDA(int i) {
        uint64_t seed = fast_forward_LCG(1ULL, (uint64_t)i);
        double r1 = LCG_random_double(&seed);
        double r2 = LCG_random_double(&seed);
        double r3 = LCG_random_double(&seed);
        double r4 = LCG_random_double(&seed);
        DoubleComplex z1 = make_DoubleComplex(r1, r2);
        DoubleComplex z2 = make_DoubleComplex(r3, r4);
        char s  = fabs(Cabs(Cmul(z1,z2)) - Cabs(z1)*Cabs(z2)) < 1e-3;
        s += fabs(Cabs(Cadd(z1,z2))*Cabs(Cadd(z1,z2)) -
                  Creal(Cmul(Cadd(z1,z2), Cadd(Conj(z1),Conj(z2))))) < 1e-3;
        s += fabs(Cabs(Csub(z1,z2))*Cabs(Csub(z1,z2)) -
                  Creal(Cmul(Csub(z1,z2), Csub(Conj(z1),Conj(z2))))) < 1e-3;
        s += fabs(Creal(Cadd(Cmul(z1,Conj(z2)), Cmul(z2,Conj(z1)))) -
                  2.0*(Creal(z1)*Creal(z2) + Cimag(z1)*Cimag(z2))) < 1e-3;
        s += fabs(Cabs(Cdiv(Conj(z1),z2)) - Cabs(Cdiv(Conj(z1),Conj(z2)))) < 1e-3;
        cs(i) = s;
      });
    }
    Kokkos::fence();
    t1 = std::chrono::steady_clock::now();
    printf("Average kernel execution time (double) %f (s)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-9f / repeat);

    Kokkos::deep_copy(h_cs, cs);
    bool double_ok = check(h_cs.data(), n);

    printf("%s\n", (float_ok && double_ok) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
