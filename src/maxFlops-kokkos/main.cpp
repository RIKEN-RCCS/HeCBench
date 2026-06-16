#include <chrono>
#include <iostream>
#include <cstdlib>
#include <Kokkos_Core.hpp>

// ─── Arithmetic operation macros (identical to OMP kernels.h) ────────────────

#define ADD1_OP   s=v-s;
#define ADD2_OP   ADD1_OP s2=v-s2;
#define ADD4_OP   ADD2_OP s3=v-s3; s4=v-s4;
#define ADD8_OP   ADD4_OP s5=v-s5; s6=v-s6; s7=v-s7; s8=v-s8;

#define MUL1_OP   s=s*s*v;
#define MUL2_OP   MUL1_OP s2=s2*s2*v;
#define MUL4_OP   MUL2_OP s3=s3*s3*v; s4=s4*s4*v;
#define MUL8_OP   MUL4_OP s5=s5*s5*v; s6=s6*s6*v; s7=s7*s7*v; s8=s8*s8*v;

#define MADD1_OP  s=v1-s*v2;
#define MADD2_OP  MADD1_OP s2=v1-s2*v2;
#define MADD4_OP  MADD2_OP s3=v1-s3*v2; s4=v1-s4*v2;
#define MADD8_OP  MADD4_OP s5=v1-s5*v2; s6=v1-s6*v2; s7=v1-s7*v2; s8=v1-s8*v2;

#define MULMADD1_OP  s=(v1-v2*s)*s;
#define MULMADD2_OP  MULMADD1_OP s2=(v1-v2*s2)*s2;
#define MULMADD4_OP  MULMADD2_OP s3=(v1-v2*s3)*s3; s4=(v1-v2*s4)*s4;
#define MULMADD8_OP  MULMADD4_OP s5=(v1-v2*s5)*s5; s6=(v1-v2*s6)*s6; s7=(v1-v2*s7)*s7; s8=(v1-v2*s8)*s8;

#define ADD1_MOP20 \
  ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP \
  ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP
#define ADD2_MOP20 \
  ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP \
  ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP
#define ADD4_MOP10 \
  ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP \
  ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP
#define ADD8_MOP5 \
  ADD8_OP ADD8_OP ADD8_OP ADD8_OP ADD8_OP

#define MUL1_MOP20 \
  MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP \
  MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP
#define MUL2_MOP20 \
  MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP \
  MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP
#define MUL4_MOP10 \
  MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP \
  MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP
#define MUL8_MOP5 \
  MUL8_OP MUL8_OP MUL8_OP MUL8_OP MUL8_OP

#define MADD1_MOP20 \
  MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
  MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP
#define MADD2_MOP20 \
  MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP \
  MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP
#define MADD4_MOP10 \
  MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP \
  MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP
#define MADD8_MOP5 \
  MADD8_OP MADD8_OP MADD8_OP MADD8_OP MADD8_OP

#define MULMADD1_MOP20 \
  MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
  MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
  MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
  MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP
#define MULMADD2_MOP20 \
  MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
  MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
  MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
  MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP
#define MULMADD4_MOP10 \
  MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP \
  MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP
#define MULMADD8_MOP5 \
  MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP

// ─── Kokkos kernel wrappers ───────────────────────────────────────────────────

template <class T>
void Add1(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Add1", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid);
    for (int j = 0; j < nIters; ++j) {
      ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20
      ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20
    }
    data(gid) = s;
  });
}

template <class T>
void Add2(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Add2", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid), s2 = (T)10.0 - s;
    for (int j = 0; j < nIters; ++j) {
      ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
      ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
    }
    data(gid) = s + s2;
  });
}

template <class T>
void Add4(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Add4", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid), s2 = (T)10.0 - s, s3 = (T)9.0 - s, s4 = (T)9.0 - s2;
    for (int j = 0; j < nIters; ++j) {
      ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
      ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
    }
    data(gid) = (s + s2) + (s3 + s4);
  });
}

template <class T>
void Add8(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Add8", nFloats, KOKKOS_LAMBDA(int gid) {
    T s  = data(gid), s2 = (T)10.0 - s,  s3 = (T)9.0 - s,  s4 = (T)9.0 - s2,
      s5 = (T)8.0 - s, s6 = (T)8.0 - s2, s7 = (T)7.0 - s, s8 = (T)7.0 - s2;
    for (int j = 0; j < nIters; ++j) {
      ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
      ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
    }
    data(gid) = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
  });
}

template <class T>
void Mul1(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Mul1", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid) - data(gid) + (T)0.999;
    for (int j = 0; j < nIters; ++j) {
      MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
      MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
    }
    data(gid) = s;
  });
}

template <class T>
void Mul2(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Mul2", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid) - data(gid) + (T)0.999, s2 = s - (T)0.0001;
    for (int j = 0; j < nIters; ++j) {
      MUL2_MOP20 MUL2_MOP20 MUL2_MOP20
      MUL2_MOP20 MUL2_MOP20
    }
    data(gid) = s + s2;
  });
}

template <class T>
void Mul4(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Mul4", nFloats, KOKKOS_LAMBDA(int gid) {
    T s  = data(gid) - data(gid) + (T)0.999,
      s2 = s - (T)0.0001, s3 = s - (T)0.0002, s4 = s - (T)0.0003;
    for (int j = 0; j < nIters; ++j) {
      MUL4_MOP10 MUL4_MOP10 MUL4_MOP10
      MUL4_MOP10 MUL4_MOP10
    }
    data(gid) = (s + s2) + (s3 + s4);
  });
}

template <class T>
void Mul8(Kokkos::View<T*> data, int nFloats, int nIters, T v) {
  Kokkos::parallel_for("Mul8", nFloats, KOKKOS_LAMBDA(int gid) {
    T s  = data(gid) - data(gid) + (T)0.999,
      s2 = s - (T)0.0001, s3 = s - (T)0.0002, s4 = s - (T)0.0003,
      s5 = s - (T)0.0004, s6 = s - (T)0.0005, s7 = s - (T)0.0006, s8 = s - (T)0.0007;
    for (int j = 0; j < nIters; ++j) {
      MUL8_MOP5 MUL8_MOP5 MUL8_MOP5
      MUL8_MOP5 MUL8_MOP5
    }
    data(gid) = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
  });
}

template <class T>
void MAdd1(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MAdd1", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid);
    for (int j = 0; j < nIters; ++j) {
      MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
      MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
    }
    data(gid) = s;
  });
}

template <class T>
void MAdd2(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MAdd2", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid), s2 = (T)10.0 - s;
    for (int j = 0; j < nIters; ++j) {
      MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
      MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
    }
    data(gid) = s + s2;
  });
}

template <class T>
void MAdd4(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MAdd4", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid), s2 = (T)10.0 - s, s3 = (T)9.0 - s, s4 = (T)9.0 - s2;
    for (int j = 0; j < nIters; ++j) {
      MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
      MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
    }
    data(gid) = (s + s2) + (s3 + s4);
  });
}

template <class T>
void MAdd8(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MAdd8", nFloats, KOKKOS_LAMBDA(int gid) {
    T s  = data(gid), s2 = (T)10.0 - s, s3 = (T)9.0 - s, s4 = (T)9.0 - s2,
      s5 = (T)8.0 - s, s6 = (T)8.0 - s2, s7 = (T)7.0 - s, s8 = (T)7.0 - s2;
    for (int j = 0; j < nIters; ++j) {
      MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
      MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
    }
    data(gid) = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
  });
}

template <class T>
void MulMAdd1(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MulMAdd1", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid);
    for (int j = 0; j < nIters; ++j) {
      MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
      MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
    }
    data(gid) = s;
  });
}

template <class T>
void MulMAdd2(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MulMAdd2", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid), s2 = (T)10.0 - s;
    for (int j = 0; j < nIters; ++j) {
      MULMADD2_MOP20 MULMADD2_MOP20
      MULMADD2_MOP20 MULMADD2_MOP20
    }
    data(gid) = s + s2;
  });
}

template <class T>
void MulMAdd4(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MulMAdd4", nFloats, KOKKOS_LAMBDA(int gid) {
    T s = data(gid), s2 = (T)10.0 - s, s3 = (T)9.0 - s, s4 = (T)9.0 - s2;
    for (int j = 0; j < nIters; ++j) {
      MULMADD4_MOP10 MULMADD4_MOP10
      MULMADD4_MOP10 MULMADD4_MOP10
    }
    data(gid) = (s + s2) + (s3 + s4);
  });
}

template <class T>
void MulMAdd8(Kokkos::View<T*> data, int nFloats, int nIters, T v1, T v2) {
  Kokkos::parallel_for("MulMAdd8", nFloats, KOKKOS_LAMBDA(int gid) {
    T s  = data(gid), s2 = (T)10.0 - s, s3 = (T)9.0 - s, s4 = (T)9.0 - s2,
      s5 = (T)8.0 - s, s6 = (T)8.0 - s2, s7 = (T)7.0 - s, s8 = (T)7.0 - s2;
    for (int j = 0; j < nIters; ++j) {
      MULMADD8_MOP5 MULMADD8_MOP5
      MULMADD8_MOP5 MULMADD8_MOP5
    }
    data(gid) = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
  });
}

// ─── Test driver ─────────────────────────────────────────────────────────────

template <typename T>
void test(const int repeat, const int numFloats) {
  Kokkos::View<T*> d_mem("d_mem", numFloats);

  // Initialise host data
  T* hostMem = (T*) malloc(sizeof(T) * numFloats);
  srand48(123);
  for (int j = 0; j < numFloats / 2; ++j)
    hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
  {
    auto h = Kokkos::create_mirror_view(d_mem);
    memcpy(h.data(), hostMem, sizeof(T) * numFloats);
    Kokkos::deep_copy(d_mem, h);
  }
  free(hostMem);

  // Helper lambda: reload data + time a kernel
  auto time_kernel = [&](const char* name, auto&& kernel_fn) {
    // reload
    {
      T* hm = (T*) malloc(sizeof(T) * numFloats);
      srand48(123);
      for (int j = 0; j < numFloats / 2; ++j)
        hm[j] = hm[numFloats - j - 1] = (T)(drand48() * 10.0);
      auto h = Kokkos::create_mirror_view(d_mem);
      memcpy(h.data(), hm, sizeof(T) * numFloats);
      Kokkos::deep_copy(d_mem, h);
      free(hm);
    }
    Kokkos::fence();
    auto t0 = std::chrono::high_resolution_clock::now();
    kernel_fn();
    Kokkos::fence();
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("kernel execution time (%s): %f (s)\n", name,
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9f);
  };

  // Warmup
  for (int i = 0; i < 4; i++) {
    Add1<T>(d_mem, numFloats, repeat, (T)10.0);
    Add2<T>(d_mem, numFloats, repeat, (T)10.0);
    Add4<T>(d_mem, numFloats, repeat, (T)10.0);
    Add8<T>(d_mem, numFloats, repeat, (T)10.0);
  }
  Kokkos::fence();

  time_kernel("Add1", [&]{ Add1<T>(d_mem, numFloats, repeat, (T)10.0); });
  time_kernel("Add2", [&]{ Add2<T>(d_mem, numFloats, repeat, (T)10.0); });
  time_kernel("Add4", [&]{ Add4<T>(d_mem, numFloats, repeat, (T)10.0); });
  time_kernel("Add8", [&]{ Add8<T>(d_mem, numFloats, repeat, (T)10.0); });

  for (int i = 0; i < 4; i++) {
    Mul1<T>(d_mem, numFloats, repeat, (T)1.01);
    Mul2<T>(d_mem, numFloats, repeat, (T)1.01);
    Mul4<T>(d_mem, numFloats, repeat, (T)1.01);
    Mul8<T>(d_mem, numFloats, repeat, (T)1.01);
  }
  Kokkos::fence();

  time_kernel("Mul1", [&]{ Mul1<T>(d_mem, numFloats, repeat, (T)1.01); });
  time_kernel("Mul2", [&]{ Mul2<T>(d_mem, numFloats, repeat, (T)1.01); });
  time_kernel("Mul4", [&]{ Mul4<T>(d_mem, numFloats, repeat, (T)1.01); });
  time_kernel("Mul8", [&]{ Mul8<T>(d_mem, numFloats, repeat, (T)1.01); });

  for (int i = 0; i < 4; i++) {
    MAdd1<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899);
    MAdd2<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899);
    MAdd4<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899);
    MAdd8<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899);
  }
  Kokkos::fence();

  time_kernel("MAdd1", [&]{ MAdd1<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899); });
  time_kernel("MAdd2", [&]{ MAdd2<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899); });
  time_kernel("MAdd4", [&]{ MAdd4<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899); });
  time_kernel("MAdd8", [&]{ MAdd8<T>(d_mem, numFloats, repeat, (T)10.0, (T)0.9899); });

  for (int i = 0; i < 4; i++) {
    MulMAdd1<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355);
    MulMAdd2<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355);
    MulMAdd4<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355);
    MulMAdd8<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355);
  }
  Kokkos::fence();

  time_kernel("MulMAdd1", [&]{ MulMAdd1<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355); });
  time_kernel("MulMAdd2", [&]{ MulMAdd2<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355); });
  time_kernel("MulMAdd4", [&]{ MulMAdd4<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355); });
  time_kernel("MulMAdd8", [&]{ MulMAdd8<T>(d_mem, numFloats, repeat, (T)3.75, (T)0.355); });
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat    = atoi(argv[1]);
  const int numFloats = 2 * 1024 * 1024;

  Kokkos::initialize(argc, argv);

  printf("=== Single-precision floating-point kernels ===\n");
  test<float>(repeat, numFloats);

  printf("=== Double-precision floating-point kernels ===\n");
  test<double>(repeat, numFloats);

  Kokkos::finalize();
  return 0;
}
