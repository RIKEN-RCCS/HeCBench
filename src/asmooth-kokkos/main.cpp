/*
 * Adaptive image smoothing.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// CPU reference
void reference(int Lx, int Ly, int threshold, int maxRad,
               float *img, int *box, float *norm, float *out)
{
  float q, sum, s;
  int ksum;

  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      sum = 0.f; s = q = 1; ksum = 0;
      while (sum < threshold && q < maxRad) {
        s = q; sum = 0.f; ksum = 0;
        for (int i = -s; i < s+1; i++)
          for (int j = -s; j < s+1; j++)
            if (x-s>=0 && x+s<Lx && y-s>=0 && y+s<Ly) { sum += img[(x+i)*Ly+y+j]; ksum++; }
        q++;
      }
      box[x*Ly+y] = (int)s;
      for (int i = -s; i < s+1; i++)
        for (int j = -s; j < s+1; j++)
          if (x-s>=0 && x+s<Lx && y-s>=0 && y+s<Ly)
            if (ksum != 0) norm[(x+i)*Ly+y+j] += 1.f / (float)ksum;
    }
  }
  for (int x = 0; x < Lx; x++)
    for (int y = 0; y < Ly; y++)
      if (norm[x*Ly+y] != 0) img[x*Ly+y] /= norm[x*Ly+y];
  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      s = (float)box[x*Ly+y]; sum = 0.f; ksum = 0;
      for (int i = -(int)s; i < (int)s+1; i++)
        for (int j = -(int)s; j < (int)s+1; j++)
          if (x-(int)s>=0 && x+(int)s<Lx && y-(int)s>=0 && y+(int)s<Ly) { sum += img[(x+i)*Ly+y+j]; ksum++; }
      if (ksum != 0) out[x*Ly+y] = sum / (float)ksum;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
    return 1;
  }

  const int Lx        = atoi(argv[1]);
  const int Ly        = Lx;
  const int Threshold = atoi(argv[2]);
  const int MaxRad    = atoi(argv[3]);
  const int repeat    = atoi(argv[4]);

  const int size = Lx * Ly;

  float *img   = (float*) malloc(size * sizeof(float));
  float *norm  = (float*) malloc(size * sizeof(float));
  float *h_norm = (float*) malloc(size * sizeof(float));
  int   *box   = (int*)   malloc(size * sizeof(int));
  int   *h_box = (int*)   malloc(size * sizeof(int));
  float *out   = (float*) malloc(size * sizeof(float));
  float *h_out = (float*) malloc(size * sizeof(float));

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = (float)(rand() % 256);
    norm[i] = 0.f; box[i] = 0; out[i] = 0.f;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_img("d_img", size);
    Kokkos::View<float*> d_norm("d_norm", size);
    Kokkos::View<int*>   d_box("d_box", size);
    Kokkos::View<float*> d_out("d_out", size);

    auto h_img_v  = Kokkos::create_mirror_view(d_img);
    auto h_norm_v = Kokkos::create_mirror_view(d_norm);
    auto h_out_v  = Kokkos::create_mirror_view(d_out);

    double total_time = 0.0;

    for (int iter = 0; iter < repeat; iter++) {
      // Restore input
      for (int i = 0; i < size; i++) { h_img_v(i) = img[i]; h_norm_v(i) = 0.f; }
      Kokkos::deep_copy(d_img,  h_img_v);
      Kokkos::deep_copy(d_norm, h_norm_v);

      auto start = std::chrono::steady_clock::now();

      // Kernel 1: compute box sizes and accumulate norm
      Kokkos::parallel_for("asmooth_k1",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Lx,Ly}),
          KOKKOS_LAMBDA(int x, int y) {
            float sum = 0.f;
            float s = 1.f, q = 1.f;
            int ksum = 0;
            while (sum < (float)Threshold && q < (float)MaxRad) {
              s = q; sum = 0.f; ksum = 0;
              for (int i = -(int)s; i <= (int)s; i++)
                for (int j = -(int)s; j <= (int)s; j++)
                  if (x-(int)s>=0 && x+(int)s<Lx && y-(int)s>=0 && y+(int)s<Ly) {
                    sum += d_img(((x+i)*Ly)+(y+j));
                    ksum++;
                  }
              q++;
            }
            d_box(x*Ly+y) = (int)s;
            for (int i = -(int)s; i <= (int)s; i++)
              for (int j = -(int)s; j <= (int)s; j++)
                if (x-(int)s>=0 && x+(int)s<Lx && y-(int)s>=0 && y+(int)s<Ly)
                  if (ksum != 0)
                    Kokkos::atomic_add(&d_norm((x+i)*Ly+(y+j)), 1.f / (float)ksum);
          });
      Kokkos::fence();

      // Kernel 2: normalize
      Kokkos::parallel_for("asmooth_k2",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Lx,Ly}),
          KOKKOS_LAMBDA(int x, int y) {
            if (d_norm(x*Ly+y) != 0.f)
              d_img(x*Ly+y) /= d_norm(x*Ly+y);
          });
      Kokkos::fence();

      // Kernel 3: resmooth with normalized image
      Kokkos::parallel_for("asmooth_k3",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Lx,Ly}),
          KOKKOS_LAMBDA(int x, int y) {
            int s = d_box(x*Ly+y);
            float sum = 0.f;
            int ksum = 0;
            for (int i = -s; i <= s; i++)
              for (int j = -s; j <= s; j++)
                if (x-s>=0 && x+s<Lx && y-s>=0 && y+s<Ly) {
                  sum += d_img((x+i)*Ly+(y+j));
                  ksum++;
                }
            if (ksum != 0) d_out(x*Ly+y) = sum / (float)ksum;
          });
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    printf("Average filtering time %lf (s)\n", (total_time * 1e-9) / repeat);

    auto h_out_v2  = Kokkos::create_mirror_view(d_out);
    auto h_box_v   = Kokkos::create_mirror_view(d_box);
    auto h_norm_v2 = Kokkos::create_mirror_view(d_norm);
    Kokkos::deep_copy(h_out_v2,  d_out);
    Kokkos::deep_copy(h_box_v,   d_box);
    Kokkos::deep_copy(h_norm_v2, d_norm);
    for (int i = 0; i < size; i++) {
      out[i]  = h_out_v2(i);
      box[i]  = h_box_v(i);
      norm[i] = h_norm_v2(i);
    }
  }
  Kokkos::finalize();

  // Verify against CPU reference
  reference(Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (fabsf(norm[i] - h_norm[i]) > 1e-3f) { printf("norm: %d %f %f\n", i, norm[i], h_norm[i]); ok = false; break; }
    if (fabsf(out[i]  - h_out[i])  > 1e-3f) { printf("out: %d %f %f\n",  i, out[i],  h_out[i]);  ok = false; break; }
    if (box[i] != h_box[i])                  { printf("box: %d %d %d\n",  i, box[i],  h_box[i]);  ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(img); free(norm); free(h_norm); free(box); free(h_box); free(out); free(h_out);
  return 0;
}
