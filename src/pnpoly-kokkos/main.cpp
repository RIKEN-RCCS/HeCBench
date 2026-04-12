/*
 * Point-in-polygon problem using the crossing number algorithm.
 * Ported to Kokkos from the OMP target version.
 *
 * Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <Kokkos_Core.hpp>

#define VERTICES 600
#define BLOCK_SIZE_X 256

struct float2 {
  float x, y;
};

KOKKOS_INLINE_FUNCTION
int is_between(float a, float b, float c) {
  return (b > a) != (c > a);
}

void pnpoly_base(
    Kokkos::View<int*> bitmap,
    Kokkos::View<const float2*> point,
    Kokkos::View<const float2*> vertex,
    int n)
{
  Kokkos::parallel_for("pnpoly_base", n, KOKKOS_LAMBDA(int i) {
    int c = 0;
    float2 p = point(i);
    int k = VERTICES - 1;
    for (int j = 0; j < VERTICES; k = j++) {
      float2 vj = vertex(j);
      float2 vk = vertex(k);
      float slope = (vk.x - vj.x) / (vk.y - vj.y);
      if (((vj.y > p.y) != (vk.y > p.y)) &&
          (p.x < slope * (p.y - vj.y) + vj.x)) {
        c = !c;
      }
    }
    bitmap(i) = c;
  });
  Kokkos::fence();
}

template <int tile_size>
void pnpoly_opt(
    Kokkos::View<int*> bitmap,
    Kokkos::View<const float2*> point,
    Kokkos::View<const float2*> vertex,
    int n)
{
  Kokkos::parallel_for("pnpoly_opt", n, KOKKOS_LAMBDA(int i) {
    int c[tile_size];
    float2 lpoint[tile_size];
    for (int ti = 0; ti < tile_size; ti++) {
      c[ti] = 0;
      if (i + BLOCK_SIZE_X * ti < n)
        lpoint[ti] = point(i + BLOCK_SIZE_X * ti);
    }

    int k = VERTICES - 1;
    for (int j = 0; j < VERTICES; k = j++) {
      float2 vj = vertex(j);
      float2 vk = vertex(k);
      float slope = (vk.x - vj.x) / (vk.y - vj.y);

      for (int ti = 0; ti < tile_size; ti++) {
        float2 p = lpoint[ti];
        if (is_between(p.y, vj.y, vk.y) &&
            (p.x < slope * (p.y - vj.y) + vj.x)) {
          c[ti] = !c[ti];
        }
      }
    }

    for (int ti = 0; ti < tile_size; ti++) {
      if (i + BLOCK_SIZE_X * ti < n)
        bitmap(i + BLOCK_SIZE_X * ti) = c[ti];
    }
  });
  Kokkos::fence();
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = VERTICES;

  std::default_random_engine rng(123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*) malloc(sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex = (float2*) malloc(vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int *bitmap_ref = (int*) malloc(nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc(nPoints * sizeof(int));

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float2*> d_point("d_point", nPoints);
    Kokkos::View<float2*> d_vertex("d_vertex", vertices);
    Kokkos::View<int*> d_bitmap_ref("d_bitmap_ref", nPoints);
    Kokkos::View<int*> d_bitmap_opt("d_bitmap_opt", nPoints);

    auto h_point  = Kokkos::create_mirror_view(d_point);
    auto h_vertex = Kokkos::create_mirror_view(d_vertex);
    for (int i = 0; i < nPoints;  i++) { h_point(i).x  = point[i].x;  h_point(i).y  = point[i].y; }
    for (int i = 0; i < vertices; i++) { h_vertex(i).x = vertex[i].x; h_vertex(i).y = vertex[i].y; }
    Kokkos::deep_copy(d_point,  h_point);
    Kokkos::deep_copy(d_vertex, h_vertex);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_base(d_bitmap_ref,
                  Kokkos::View<const float2*>(d_point),
                  Kokkos::View<const float2*>(d_vertex),
                  nPoints);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_opt<1>(d_bitmap_opt,
                    Kokkos::View<const float2*>(d_point),
                    Kokkos::View<const float2*>(d_vertex),
                    nPoints);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_opt<2>(d_bitmap_opt,
                    Kokkos::View<const float2*>(d_point),
                    Kokkos::View<const float2*>(d_vertex),
                    nPoints);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_opt<4>(d_bitmap_opt,
                    Kokkos::View<const float2*>(d_point),
                    Kokkos::View<const float2*>(d_vertex),
                    nPoints);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_opt<8>(d_bitmap_opt,
                    Kokkos::View<const float2*>(d_point),
                    Kokkos::View<const float2*>(d_vertex),
                    nPoints);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_opt<16>(d_bitmap_opt,
                     Kokkos::View<const float2*>(d_point),
                     Kokkos::View<const float2*>(d_vertex),
                     nPoints);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_opt<32>(d_bitmap_opt,
                     Kokkos::View<const float2*>(d_point),
                     Kokkos::View<const float2*>(d_vertex),
                     nPoints);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      pnpoly_opt<64>(d_bitmap_opt,
                     Kokkos::View<const float2*>(d_point),
                     Kokkos::View<const float2*>(d_vertex),
                     nPoints);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

    auto h_bitmap_ref = Kokkos::create_mirror_view(d_bitmap_ref);
    auto h_bitmap_opt = Kokkos::create_mirror_view(d_bitmap_opt);
    Kokkos::deep_copy(h_bitmap_ref, d_bitmap_ref);
    Kokkos::deep_copy(h_bitmap_opt, d_bitmap_opt);
    for (int i = 0; i < nPoints; i++) {
      bitmap_ref[i] = h_bitmap_ref(i);
      bitmap_opt[i] = h_bitmap_opt(i);
    }
  }
  Kokkos::finalize();

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints * sizeof(int));

  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
