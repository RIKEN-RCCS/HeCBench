#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "reference.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width  = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_bytes  = width * height * sizeof(char);
  const int output_bytes = width * height * sizeof(float);

  char  *input      = (char *)  malloc(input_bytes);
  float *output     = (float *) malloc(output_bytes);
  float *output_ref = (float *) malloc(output_bytes);

  srand(123);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      input[i * width + j] = rand() % 16;

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<char*>  d_input ("d_input",  width * height);
    Kokkos::View<float*> d_output("d_output", width * height);
    Kokkos::View<float*> d_logTable("d_logTable", 26);

    // Mirror views for host↔device copies
    auto h_input = Kokkos::create_mirror_view(d_input);
    for (int i = 0; i < width * height; i++) h_input(i) = input[i];
    Kokkos::deep_copy(d_input, h_input);

    // Build logTable: logTable[i] = i <= 1 ? 0 : i*log2f(i)
    auto h_logTable = Kokkos::create_mirror_view(d_logTable);
    for (int i = 0; i <= 25; i++)
      h_logTable(i) = (i <= 1) ? 0.0f : i * log2f((float)i);
    Kokkos::deep_copy(d_logTable, h_logTable);

    Kokkos::fence();

    // -------------------------------------------------------------------------
    // Baseline kernel: MDRangePolicy over (y, x)
    // -------------------------------------------------------------------------
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("entropy_baseline",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {height, width}),
        KOKKOS_LAMBDA(const int y, const int x) {
          int count[16];
          for (int i = 0; i < 16; i++) count[i] = 0;
          int total = 0;

          for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
              int xx = x + dx;
              int yy = y + dy;
              if (xx >= 0 && yy >= 0 && yy < height && xx < width) {
                count[(int)(unsigned char)d_input(yy * width + xx)]++;
                total++;
              }
            }
          }

          float ent = 0.0f;
          if (total >= 1) {
            for (int k = 0; k < 16; k++) {
              float p = (float)count[k] / (float)total;
              if (p > 0.0f) ent -= p * log2f(p);
            }
          }
          d_output(y * width + x) = ent;
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel (baseline) execution time %f (s)\n", (time * 1e-9) / repeat);

    // -------------------------------------------------------------------------
    // Optimised kernel: uses the precomputed log table
    // Formula: entropy = -sum(p*log2(p)) = (1/total)*(-sum(count[k]*log2(count[k]))) + log2(total)
    //   = (1/total)*(sum(logTable[count[k]]) /*negated*/ ) + log2(total)
    //   CUDA version: entropy -= logTable[count[k]]; entropy = entropy/total + log2(total)
    //   where logTable[i] = i*log2(i), so -sum(count[k]*log2(count[k]))/total + log2(total)
    //   = sum(count[k]/total * log2(total/count[k])) = -sum(p*log2(p))  ✓
    // -------------------------------------------------------------------------
    start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("entropy_opt",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {height, width}),
        KOKKOS_LAMBDA(const int y, const int x) {
          int count[16];
          for (int i = 0; i < 16; i++) count[i] = 0;
          int total = 0;

          for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
              int xx = x + dx;
              int yy = y + dy;
              if (xx >= 0 && yy >= 0 && yy < height && xx < width) {
                count[(int)(unsigned char)d_input(yy * width + xx)]++;
                total++;
              }
            }
          }

          float ent = 0.0f;
          if (total >= 1) {
            for (int k = 0; k < 16; k++)
              ent -= d_logTable(count[k]);
            ent = ent / total + log2f((float)total);
          }
          d_output(y * width + x) = ent;
        });
      Kokkos::fence();
    }

    end  = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel (optimized) execution time %f (s)\n", (time * 1e-9) / repeat);

    // Copy result back
    auto h_output = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_output, d_output);
    for (int i = 0; i < width * height; i++) output[i] = h_output(i);
  }
  Kokkos::finalize();

  // Verify against CPU reference
  reference(output_ref, input, height, width);

  bool ok = true;
  for (int i = 0; i < height && ok; i++) {
    for (int j = 0; j < width && ok; j++) {
      if (fabsf(output[i * width + j] - output_ref[i * width + j]) > 1e-3f) {
        printf("Mismatch at (%d,%d): gpu=%f ref=%f\n",
               i, j, output[i * width + j], output_ref[i * width + j]);
        ok = false;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output);
  free(output_ref);
  return 0;
}
