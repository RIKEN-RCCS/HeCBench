/*
 * Background subtraction from video frames.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

typedef unsigned char uchar;

void findMovingPixels(size_t imgSize,
                      Kokkos::View<const uchar*> Img,
                      Kokkos::View<const uchar*> Img1,
                      Kokkos::View<const uchar*> Img2,
                      Kokkos::View<const uchar*> Tn,
                      Kokkos::View<uchar*>       Mp)
{
  Kokkos::parallel_for("findMovingPixels", imgSize, KOKKOS_LAMBDA(size_t i) {
    if (abs((int)Img(i) - Img1(i)) > Tn(i) || abs((int)Img(i) - Img2(i)) > Tn(i))
      Mp(i) = 255;
    else
      Mp(i) = 0;
  });
  Kokkos::fence();
}

void updateBackground(size_t imgSize,
                      Kokkos::View<const uchar*> Img,
                      Kokkos::View<const uchar*> Mp,
                      Kokkos::View<uchar*>       Bn)
{
  Kokkos::parallel_for("updateBackground", imgSize, KOKKOS_LAMBDA(size_t i) {
    if (Mp(i) == 0)
      Bn(i) = (uchar)(0.92f * Bn(i) + 0.08f * Img(i));
  });
  Kokkos::fence();
}

void updateThreshold(size_t imgSize,
                     Kokkos::View<const uchar*> Img,
                     Kokkos::View<const uchar*> Mp,
                     Kokkos::View<const uchar*> Bn,
                     Kokkos::View<uchar*>       Tn)
{
  Kokkos::parallel_for("updateThreshold", imgSize, KOKKOS_LAMBDA(size_t i) {
    if (Mp(i) == 0) {
      float th = 0.92f * Tn(i) + 0.24f * ((int)Img(i) - Bn(i));
      Tn(i) = (uchar)Kokkos::fmax(th, 20.f);
    }
  });
  Kokkos::fence();
}

void merge(size_t imgSize,
           Kokkos::View<const uchar*> Img,
           Kokkos::View<const uchar*> Img1,
           Kokkos::View<const uchar*> Img2,
           Kokkos::View<uchar*>       Tn,
           Kokkos::View<uchar*>       Bn)
{
  Kokkos::parallel_for("merge", imgSize, KOKKOS_LAMBDA(size_t i) {
    if (abs((int)Img(i) - Img1(i)) <= Tn(i) && abs((int)Img(i) - Img2(i)) <= Tn(i)) {
      Bn(i) = (uchar)(0.92f * Bn(i) + 0.08f * Img(i));
      float th = 0.92f * Tn(i) + 0.24f * ((int)Img(i) - Bn(i));
      Tn(i) = (uchar)Kokkos::fmax(th, 20.f);
    }
  });
  Kokkos::fence();
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <merge> <repeat>\n", argv[0]);
    return 1;
  }
  const int width  = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int merged = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int imgSize = width * height;

  uchar *Img  = (uchar*) malloc(imgSize);
  uchar *Img1 = (uchar*) malloc(imgSize);
  uchar *Img2 = (uchar*) malloc(imgSize);
  uchar *Bn   = (uchar*) malloc(imgSize);
  uchar *Mp   = (uchar*) malloc(imgSize);
  uchar *Tn   = (uchar*) malloc(imgSize);

  std::default_random_engine g(19937);
  std::uniform_int_distribution<int> distr(0, 255);
  for (int i = 0; i < imgSize; i++) {
    Img[i] = Img1[i] = Img2[i] = Bn[i] = distr(g);
    Tn[i] = 128;
    Mp[i] = 0;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<uchar*> d_Img("d_Img", imgSize);
    Kokkos::View<uchar*> d_Img1("d_Img1", imgSize);
    Kokkos::View<uchar*> d_Img2("d_Img2", imgSize);
    Kokkos::View<uchar*> d_Bn("d_Bn", imgSize);
    Kokkos::View<uchar*> d_Mp("d_Mp", imgSize);
    Kokkos::View<uchar*> d_Tn("d_Tn", imgSize);

    auto h_Img  = Kokkos::create_mirror_view(d_Img);
    auto h_Img1 = Kokkos::create_mirror_view(d_Img1);
    auto h_Img2 = Kokkos::create_mirror_view(d_Img2);
    auto h_Bn   = Kokkos::create_mirror_view(d_Bn);
    auto h_Mp   = Kokkos::create_mirror_view(d_Mp);
    auto h_Tn   = Kokkos::create_mirror_view(d_Tn);

    for (int i = 0; i < imgSize; i++) {
      h_Img(i) = Img[i]; h_Img1(i) = Img1[i]; h_Img2(i) = Img2[i];
      h_Bn(i) = Bn[i]; h_Mp(i) = Mp[i]; h_Tn(i) = Tn[i];
    }
    Kokkos::deep_copy(d_Img, h_Img); Kokkos::deep_copy(d_Img1, h_Img1);
    Kokkos::deep_copy(d_Img2, h_Img2); Kokkos::deep_copy(d_Bn, h_Bn);
    Kokkos::deep_copy(d_Mp, h_Mp); Kokkos::deep_copy(d_Tn, h_Tn);

    // Rotate image views for triple buffering
    Kokkos::View<uchar*> views[3] = {d_Img, d_Img1, d_Img2};
    int cur = 0;

    long long time = 0;
    for (int i = 0; i < repeat; i++) {
      // Rotate
      Kokkos::View<uchar*> tmp = views[2];
      views[2] = views[1];
      views[1] = views[0];
      views[0] = tmp;

      if (i >= 2) {
        auto start = std::chrono::steady_clock::now();
        if (merged) {
          merge(imgSize, views[0], views[1], views[2], d_Tn, d_Bn);
        } else {
          findMovingPixels(imgSize, views[0], views[1], views[2], d_Tn, d_Mp);
          updateBackground(imgSize, views[0], d_Mp, d_Bn);
          updateThreshold(imgSize, views[0], d_Mp, d_Bn, d_Tn);
        }
        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
    }

    float kernel_time = (repeat <= 2) ? 0.f : (time * 1e-3f) / (repeat - 2);
    printf("Average kernel execution time: %f (us)\n", kernel_time);

    Kokkos::deep_copy(h_Tn, d_Tn);
    for (int i = 0; i < imgSize; i++) Tn[i] = h_Tn(i);
  }
  Kokkos::finalize();

  int sum = 0;
  int bin[4] = {0, 0, 0, 0};
  for (int j = 0; j < imgSize; j++) {
    sum += abs(Tn[j] - 128);
    if (Tn[j] < 64) bin[0]++;
    else if (Tn[j] < 128) bin[1]++;
    else if (Tn[j] < 192) bin[2]++;
    else bin[3]++;
  }
  sum = sum / imgSize;
  printf("Average threshold change is %d\n", sum);
  printf("Bin counts are %d %d %d %d\n", bin[0], bin[1], bin[2], bin[3]);

  free(Img); free(Img1); free(Img2); free(Bn); free(Mp); free(Tn);
  return 0;
}
