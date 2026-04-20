// Kokkos port of Gaussian Elimination benchmark
// Original OMP target source: src/gaussian-omp/gaussianElim.cpp
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void init_matrix(float *m, int size) {
  float lamda = -0.01f;
  std::vector<float> coe(2 * size - 1);
  for (int i = 0; i < size; i++) {
    float coe_i = 10 * expf(lamda * i);
    coe[size - 1 + i] = coe_i;
    coe[size - 1 - i] = coe_i;
  }
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      m[i * size + j] = coe[size - 1 - i + j];
}

void gaussian_reference(float *a, float *b, float *m, float *finalVec, int size) {
  for (int t = 0; t < size - 1; t++) {
    for (int i = 0; i < size - 1 - t; i++)
      m[size * (i + t + 1) + t] = a[size * (i + t + 1) + t] / a[size * t + t];
    for (int x = 0; x < size - 1 - t; x++) {
      for (int y = 0; y < size - t; y++) {
        a[size * (x + 1 + t) + y + t] -= m[size * (x + 1 + t) + t] * a[size * t + y + t];
        if (y == 0) b[x + 1 + t] -= m[size * (x + 1 + t) + (y + t)] * b[t];
      }
    }
  }
  // BackSub
  for (int i = 0; i < size; i++) {
    finalVec[size - i - 1] = b[size - i - 1];
    for (int j = 0; j < i; j++)
      finalVec[size - i - 1] -= a[size * (size - i - 1) + (size - j - 1)] * finalVec[size - j - 1];
    finalVec[size - i - 1] /= a[size * (size - i - 1) + (size - i - 1)];
  }
}

void BackSub(float *a, float *b, float *finalVec, int size) {
  for (int i = 0; i < size; i++) {
    finalVec[size - i - 1] = b[size - i - 1];
    for (int j = 0; j < i; j++)
      finalVec[size - i - 1] -= a[size * (size - i - 1) + (size - j - 1)] * finalVec[size - j - 1];
    finalVec[size - i - 1] /= a[size * (size - i - 1) + (size - i - 1)];
  }
}

void ForwardSub(float *h_a, float *h_b, float *h_m, int size, int timing) {
  Kokkos::View<float*> d_a("d_a", size * size);
  Kokkos::View<float*> d_b("d_b", size);
  Kokkos::View<float*> d_m("d_m", size * size);

  auto ha = Kokkos::create_mirror_view(d_a);
  auto hb = Kokkos::create_mirror_view(d_b);
  auto hm = Kokkos::create_mirror_view(d_m);
  for (int i = 0; i < size * size; i++) { ha(i) = h_a[i]; hm(i) = h_m[i]; }
  for (int i = 0; i < size; i++) hb(i) = h_b[i];
  Kokkos::deep_copy(d_a, ha);
  Kokkos::deep_copy(d_b, hb);
  Kokkos::deep_copy(d_m, hm);

  auto start = std::chrono::steady_clock::now();

  for (int t = 0; t < size - 1; t++) {
    const int rows = size - 1 - t;
    const int cols = size - t;

    // Fan1: compute multipliers
    Kokkos::parallel_for("gaussian_fan1",
      Kokkos::RangePolicy<>(0, rows),
      KOKKOS_LAMBDA(const int i) {
        d_m(size * (i + t + 1) + t) = d_a(size * (i + t + 1) + t) / d_a(size * t + t);
      });
    Kokkos::fence();

    // Fan2: eliminate column
    Kokkos::parallel_for("gaussian_fan2",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {rows, cols}),
      KOKKOS_LAMBDA(const int x, const int y) {
        d_a(size * (x + 1 + t) + (y + t)) -=
          d_m(size * (x + 1 + t) + t) * d_a(size * t + (y + t));
        if (y == 0)
          d_b(x + 1 + t) -= d_m(size * (x + 1 + t) + (y + t)) * d_b(t);
      });
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if (timing) printf("Total kernel execution time %lld (us)\n", (long long)elapsed);

  Kokkos::deep_copy(ha, d_a);
  Kokkos::deep_copy(hb, d_b);
  Kokkos::deep_copy(hm, d_m);
  for (int i = 0; i < size * size; i++) { h_a[i] = ha(i); h_m[i] = hm(i); }
  for (int i = 0; i < size; i++) h_b[i] = hb(i);
}

int parseCommandline(int argc, char *argv[], char *filename, int *q, int *t, int *size) {
  if (argc < 2) return 1;
  char flag;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      flag = argv[i][1];
      switch (flag) {
        case 's': i++; *size = atoi(argv[i]);
          printf("Create a square matrix (%d x %d) internally\n", *size, *size); break;
        case 'f': i++; strncpy(filename, argv[i], 100);
          printf("Read file from %s\n", filename); break;
        case 'h': return 1;
        case 'q': *q = 1; break;
        case 't': *t = 1; break;
      }
    }
  }
  return 0;
}

void PrintMat(float *ary, int size, int nrow, int ncol) {
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) printf("%8.2e ", ary[size * i + j]);
    printf("\n");
  }
  printf("\n");
}

void PrintAry(float *ary, int ary_size) {
  for (int i = 0; i < ary_size; i++) printf("%.2e ", ary[i]);
  printf("\n\n");
}

void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol) {
  for (int i = 0; i < nrow; i++)
    for (int j = 0; j < ncol; j++) fscanf(fp, "%f", ary + size * i + j);
}

void InitAry(FILE *fp, float *ary, int ary_size) {
  for (int i = 0; i < ary_size; i++) fscanf(fp, "%f", &ary[i]);
}

int main(int argc, char *argv[]) {
  float *a = NULL, *b = NULL, *m = NULL, *finalVec = NULL;
  int size = -1, quiet = 0, timing = 0;
  char filename[200] = "";

  if (parseCommandline(argc, argv, filename, &quiet, &timing, &size)) {
    printf("Usage: %s -s <size> [-qt]\n", argv[0]);
    return 0;
  }

  if (size < 1) {
    FILE *fp = fopen(filename, "r");
    fscanf(fp, "%d", &size);
    a = (float *)malloc(size * size * sizeof(float));
    InitMat(fp, size, a, size, size);
    b = (float *)malloc(size * sizeof(float));
    InitAry(fp, b, size);
    fclose(fp);
  } else {
    a = (float *)malloc(size * size * sizeof(float));
    init_matrix(a, size);
    b = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) b[i] = 1.0f;
  }

  m = (float *)malloc(size * size * sizeof(float));
  memset(m, 0, size * size * sizeof(float));
  finalVec = (float *)malloc(size * sizeof(float));

  // Reference on host
  float *a_host = (float *)malloc(size * size * sizeof(float));
  float *b_host = (float *)malloc(size * sizeof(float));
  float *m_host = (float *)malloc(size * size * sizeof(float));
  float *finalVec_host = (float *)malloc(size * sizeof(float));
  memcpy(a_host, a, size * size * sizeof(float));
  memcpy(b_host, b, size * sizeof(float));
  memcpy(m_host, m, size * size * sizeof(float));
  gaussian_reference(a_host, b_host, m_host, finalVec_host, size);

  if (!quiet) {
    printf("The input matrix a is:\n"); PrintMat(a, size, size, size);
    printf("The input array b is:\n");  PrintAry(b, size);
  }

  long long offload_start = get_time();
  Kokkos::initialize(argc, argv);
  {
    ForwardSub(a, b, m, size, timing);
  }
  Kokkos::finalize();
  long long offload_end = get_time();

  if (timing) printf("Device offloading time %lld (us)\n\n", offload_end - offload_start);

  BackSub(a, b, finalVec, size);

  if (!quiet) {
    printf("The result of array a after forwardsub:\n"); PrintMat(a, size, size, size);
    printf("The result of array b after forwardsub:\n"); PrintAry(b, size);
    printf("The solution is:\n"); PrintAry(finalVec, size);
  }

  printf("Checking the results..\n");
  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (fabsf(finalVec[i] - finalVec_host[i]) > 1e-3f) {
      ok = false;
      printf("Result mismatch at index %d: %f(kokkos)  %f(host)\n",
             i, finalVec[i], finalVec_host[i]);
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(m); free(a); free(b); free(finalVec);
  free(a_host); free(m_host); free(b_host); free(finalVec_host);
  return 0;
}
