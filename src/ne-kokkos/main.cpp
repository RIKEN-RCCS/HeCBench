/*
 * Normal Estimation from a point cloud.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

struct float3 {
  float x, y, z;
};

struct float4 {
  float x, y, z, w;
};

KOKKOS_INLINE_FUNCTION
float3 operator*(const float3 &a, float b) {
  return {a.x * b, a.y * b, a.z * b};
}

KOKKOS_INLINE_FUNCTION
float3 operator-(const float3 &a, const float3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

KOKKOS_INLINE_FUNCTION
float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

KOKKOS_INLINE_FUNCTION
float3 normalize(const float3 &v) {
  float invLen = 1.f / Kokkos::sqrt(dot(v, v));
  return v * invLen;
}

KOKKOS_INLINE_FUNCTION
float3 cross(const float3 &a, const float3 &b) {
  return {a.y * b.z - a.z * b.y,
          a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

KOKKOS_INLINE_FUNCTION
float length(const float3 &v) {
  return Kokkos::sqrt(dot(v, v));
}

KOKKOS_INLINE_FUNCTION
float4 normalEstimate(const float3 *points, int idx, int width, int height)
{
  float3 query_pt = points[idx];
  if (Kokkos::isnan(query_pt.z))
    return {0.f, 0.f, 0.f, 0.f};

  int xIdx = idx % width;
  int yIdx = idx / width;

  bool west_valid  = (xIdx > 1)         && !Kokkos::isnan(points[idx-1].z)
                                         && Kokkos::fabs(points[idx-1].z - query_pt.z) < 200.f;
  bool east_valid  = (xIdx < width-1)   && !Kokkos::isnan(points[idx+1].z)
                                         && Kokkos::fabs(points[idx+1].z - query_pt.z) < 200.f;
  bool north_valid = (yIdx > 1)         && !Kokkos::isnan(points[idx-width].z)
                                         && Kokkos::fabs(points[idx-width].z - query_pt.z) < 200.f;
  bool south_valid = (yIdx < height-1)  && !Kokkos::isnan(points[idx+width].z)
                                         && Kokkos::fabs(points[idx+width].z - query_pt.z) < 200.f;

  float3 horiz = {0.f, 0.f, 0.f}, vert = {0.f, 0.f, 0.f};

  if ( west_valid &&  east_valid)  horiz = points[idx+1] - points[idx-1];
  if ( west_valid && !east_valid)  horiz = points[idx]   - points[idx-1];
  if (!west_valid &&  east_valid)  horiz = points[idx+1] - points[idx];
  if (!west_valid && !east_valid)  return {0.f, 0.f, 0.f, 1.f};

  if ( south_valid &&  north_valid) vert = points[idx-width] - points[idx+width];
  if ( south_valid && !north_valid) vert = points[idx]       - points[idx+width];
  if (!south_valid &&  north_valid) vert = points[idx-width] - points[idx];
  if (!south_valid && !north_valid) return {0.f, 0.f, 0.f, 1.f};

  float3 normal = cross(horiz, vert);

  float curvature = (float)(Kokkos::fabs(horiz.z) > 0.04f || Kokkos::fabs(vert.z) > 0.04f ||
                             !west_valid || !east_valid || !north_valid || !south_valid);

  float3 mc = normalize(normal);
  if (dot(query_pt, mc) > 0.f)
    mc = mc * -1.f;

  return {mc.x, mc.y, mc.z, curvature};
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width  = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int numPts = width * height;

  float3 *points        = (float3*) malloc(numPts * sizeof(float3));
  float4 *normal_points = (float4*) malloc(numPts * sizeof(float4));

  srand(123);
  for (int i = 0; i < numPts; i++) {
    points[i].x = (float)(rand() % width);
    points[i].y = (float)(rand() % height);
    points[i].z = (float)(rand() % 256);
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float3*> d_points("d_points", numPts);
    Kokkos::View<float4*> d_normals("d_normals", numPts);

    auto h_points = Kokkos::create_mirror_view(d_points);
    for (int i = 0; i < numPts; i++) {
      h_points(i).x = points[i].x;
      h_points(i).y = points[i].y;
      h_points(i).z = points[i].z;
    }
    Kokkos::deep_copy(d_points, h_points);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("ne", numPts, KOKKOS_LAMBDA(int idx) {
        d_normals(idx) = normalEstimate(d_points.data(), idx, width, height);
      });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    auto h_normals = Kokkos::create_mirror_view(d_normals);
    Kokkos::deep_copy(h_normals, d_normals);
    for (int i = 0; i < numPts; i++) {
      normal_points[i].x = h_normals(i).x;
      normal_points[i].y = h_normals(i).y;
      normal_points[i].z = h_normals(i).z;
      normal_points[i].w = h_normals(i).w;
    }
  }
  Kokkos::finalize();

  float sx = 0.f, sy = 0.f, sz = 0.f, sw = 0.f;
  for (int i = 0; i < numPts; i++) {
    sx += normal_points[i].x;
    sy += normal_points[i].y;
    sz += normal_points[i].z;
    sw += normal_points[i].w;
  }
  printf("Checksum: x=%f y=%f z=%f w=%f\n", sx, sy, sz, sw);

  free(normal_points);
  free(points);
  return 0;
}
