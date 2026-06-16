#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// Inline reference from atomicIntrinsics-cuda/reference.h (7 operations only)
template <class T>
void computeGold(T *gpuData, const int len)
{
  T val = 0;
  bool ok = true;

  for (int i = 0; i < len; ++i)
    val += (T)10;
  if (val != gpuData[0]) {
    printf("Add failed: %d != %d\n", (int)val, (int)gpuData[0]);
    ok = false;
  }

  val = 0;
  for (int i = 0; i < len; ++i)
    val -= (T)10;
  if (val != gpuData[1]) {
    printf("Sub failed: %d != %d\n", (int)val, (int)gpuData[1]);
    ok = false;
  }

  val = (T)(-256);
  for (int i = 0; i < len; ++i)
    val = val > (T)i ? val : (T)i;
  if (val != gpuData[2]) {
    printf("Max failed: %d != %d\n", (int)val, (int)gpuData[2]);
    ok = false;
  }

  val = (T)256;
  for (int i = 0; i < len; ++i)
    val = val < (T)i ? val : (T)i;
  if (val != gpuData[3]) {
    printf("Min failed: %d != %d\n", (int)val, (int)gpuData[3]);
    ok = false;
  }

  val = (T)0xff;
  for (int i = 0; i < len; ++i)
    val &= (T)(2 * i + 7);
  if (val != gpuData[4]) {
    printf("And failed: %d != %d\n", (int)val, (int)gpuData[4]);
    ok = false;
  }

  val = 0;
  for (int i = 0; i < len; ++i)
    val |= (T)(1 << i);
  if (val != gpuData[5]) {
    printf("Or failed: %d != %d\n", (int)val, (int)gpuData[5]);
    ok = false;
  }

  val = (T)0xff;
  for (int i = 0; i < len; ++i)
    val ^= (T)i;
  if (val != gpuData[6]) {
    printf("Xor failed: %d != %d\n", (int)val, (int)gpuData[6]);
    ok = false;
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
}

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  const int numData = 7;
  const T data[] = {0, 0, (T)-256, (T)256, (T)255, 0, (T)255};

  Kokkos::View<T*> d_gpuData("gpuData", numData);
  auto h_gpuData = Kokkos::create_mirror_view(d_gpuData);

  // Verification run
  for (int n = 0; n < repeat; n++) {
    for (int i = 0; i < numData; i++) h_gpuData(i) = data[i];
    Kokkos::deep_copy(d_gpuData, h_gpuData);

    Kokkos::parallel_for("atomics", len, KOKKOS_LAMBDA(int i) {
      Kokkos::atomic_add(&d_gpuData(0), (T)10);
      Kokkos::atomic_add(&d_gpuData(1), -(T)10);
      Kokkos::atomic_fetch_and(&d_gpuData(4), (T)(2 * i + 7));
      Kokkos::atomic_fetch_or(&d_gpuData(5), (T)(1 << i));
      Kokkos::atomic_fetch_xor(&d_gpuData(6), (T)i);
    });
    Kokkos::fence();

    Kokkos::parallel_for("max_reduce", len, KOKKOS_LAMBDA(int i) {
      Kokkos::atomic_max(&d_gpuData(2), (T)i);
    });
    Kokkos::fence();

    Kokkos::parallel_for("min_reduce", len, KOKKOS_LAMBDA(int i) {
      Kokkos::atomic_min(&d_gpuData(3), (T)i);
    });
    Kokkos::fence();
  }

  Kokkos::deep_copy(h_gpuData, d_gpuData);
  T gpuData_host[7];
  for (int i = 0; i < numData; i++) gpuData_host[i] = h_gpuData(i);
  computeGold<T>(gpuData_host, len);

  // Timing run
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("atomics_t", len, KOKKOS_LAMBDA(int i) {
      Kokkos::atomic_add(&d_gpuData(0), (T)10);
      Kokkos::atomic_add(&d_gpuData(1), -(T)10);
      Kokkos::atomic_fetch_and(&d_gpuData(4), (T)(2 * i + 7));
      Kokkos::atomic_fetch_or(&d_gpuData(5), (T)(1 << i));
      Kokkos::atomic_fetch_xor(&d_gpuData(6), (T)i);
    });
    Kokkos::fence();

    Kokkos::parallel_for("max_t", len, KOKKOS_LAMBDA(int i) {
      Kokkos::atomic_max(&d_gpuData(2), (T)i);
    });
    Kokkos::fence();

    Kokkos::parallel_for("min_t", len, KOKKOS_LAMBDA(int i) {
      Kokkos::atomic_min(&d_gpuData(3), (T)i);
    });
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    testcase<int>(repeat);
    testcase<unsigned int>(repeat);
  }
  Kokkos::finalize();
  return 0;
}
