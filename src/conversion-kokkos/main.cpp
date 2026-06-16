#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>

typedef unsigned char uchar;

template <typename Td, typename Ts>
void convert(int nelems, int niters) {
  Kokkos::View<Ts*> d_src("d_src", nelems);
  Kokkos::View<Td*> d_dst("d_dst", nelems);

  // Warm-up run
  Kokkos::parallel_for("convert_warmup", nelems,
    KOKKOS_LAMBDA(int i) { d_dst(i) = static_cast<Td>(d_src(i)); });
  Kokkos::fence();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    Kokkos::parallel_for("convert", nelems,
      KOKKOS_LAMBDA(int idx) { d_dst(idx) = static_cast<Td>(d_src(idx)); });
    Kokkos::fence();
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                / (double)niters / 1.0e6;
  double size = (sizeof(Td) + sizeof(Ts)) * (double)nelems / 1e9;
  printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size, time, size / time);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int niters = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    printf("float -> float\n");  convert<float, float>(nelems, niters);
    printf("float -> int\n");    convert<int,   float>(nelems, niters);
    printf("float -> char\n");   convert<char,  float>(nelems, niters);
    printf("float -> uchar\n");  convert<uchar, float>(nelems, niters);

    printf("int -> int\n");    convert<int,   int>(nelems, niters);
    printf("int -> float\n");  convert<float, int>(nelems, niters);
    printf("int -> char\n");   convert<char,  int>(nelems, niters);
    printf("int -> uchar\n");  convert<uchar, int>(nelems, niters);

    printf("char -> int\n");    convert<int,   char>(nelems, niters);
    printf("char -> float\n");  convert<float, char>(nelems, niters);
    printf("char -> char\n");   convert<char,  char>(nelems, niters);
    printf("char -> uchar\n");  convert<uchar, char>(nelems, niters);

    printf("uchar -> int\n");    convert<int,   uchar>(nelems, niters);
    printf("uchar -> float\n");  convert<float, uchar>(nelems, niters);
    printf("uchar -> char\n");   convert<char,  uchar>(nelems, niters);
    printf("uchar -> uchar\n");  convert<uchar, uchar>(nelems, niters);
  }
  Kokkos::finalize();
  return 0;
}
