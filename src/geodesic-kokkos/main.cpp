#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

struct float4 {
  float x, y, z, w;
};

// Host reference implementation
float distance_host(float lat1, float lon1, float lat2, float lon2)
{
  const float GDC_DEG_TO_RAD  = 3.141592654f / 180.0f;
  const float GDC_FLATTENING  = 1.0f - (6356752.31424518f / 6378137.0f);
  const float GDC_ECCENTRICITY= (6356752.31424518f / 6378137.0f);
  const float GDC_ELLIPSOIDAL = 1.0f/(6356752.31414f/6378137.0f)/(6356752.31414f/6378137.0f) - 1.0f;
  const float GDC_SEMI_MINOR  = 6356752.31424518f;
  const float EPS             = 0.5e-5f;

  float rad_lat1 = lat1 * GDC_DEG_TO_RAD;
  float rad_lon1 = lon1 * GDC_DEG_TO_RAD;
  float rad_lat2 = lat2 * GDC_DEG_TO_RAD;
  float rad_lon2 = lon2 * GDC_DEG_TO_RAD;

  float TU1 = GDC_ECCENTRICITY * sinf(rad_lat1) / cosf(rad_lat1);
  float TU2 = GDC_ECCENTRICITY * sinf(rad_lat2) / cosf(rad_lat2);
  float CU1 = 1.0f / sqrtf(TU1*TU1 + 1.0f);
  float SU1 = CU1 * TU1;
  float CU2 = 1.0f / sqrtf(TU2*TU2 + 1.0f);
  float dist = CU1 * CU2;
  float BAZ = dist * TU2;
  float FAZ = BAZ * TU1;
  float X = rad_lon2 - rad_lon1;

  float SX, CX, SY, CY, Y, SA, C2A, CZ, E, C, D, B;
  (void)B;
  do {
    SX = sinf(X); CX = cosf(X);
    TU1 = CU2 * SX;
    TU2 = BAZ - SU1 * CU2 * CX;
    SY = sqrtf(TU1*TU1 + TU2*TU2);
    CY = dist * CX + FAZ;
    Y = atan2f(SY, CY);
    SA = dist * SX / SY;
    C2A = -SA*SA + 1.0f;
    CZ = FAZ + FAZ;
    if (C2A > 0.0f) CZ = -CZ / C2A + CY;
    E = CZ*CZ*2.0f - 1.0f;
    C = ((-3.0f*C2A + 4.0f)*GDC_FLATTENING + 4.0f)*C2A*GDC_FLATTENING / 16.0f;
    D = X;
    X = ((E*CY*C + CZ)*SY*C + Y)*SA;
    X = (1.0f - C)*X*GDC_FLATTENING + rad_lon2 - rad_lon1;
  } while (fabsf(D - X) > EPS);

  X = sqrtf(GDC_ELLIPSOIDAL * C2A + 1.0f) + 1.0f;
  X = (X - 2.0f) / X;
  C = 1.0f - X;
  C = (X*X/4.0f + 1.0f) / C;
  D = (0.375f*X*X - 1.0f) * X;
  X = E * CY;
  dist = 1.0f - E - E;
  dist = ((((SY*SY*4.0f - 3.0f)*dist*CZ*D/6.0f - X)*D/4.0f + CZ)*SY*D + Y)*C*GDC_SEMI_MINOR;
  return dist;
}

void distance_device(const float4* VA, float* VC, int N, int iteration)
{
  Kokkos::View<float4*> va_d("VA", N);
  Kokkos::View<float*>  vc_d("VC", N);

  {
    auto m = Kokkos::create_mirror_view(va_d);
    for (int i = 0; i < N; i++) m(i) = VA[i];
    Kokkos::deep_copy(va_d, m);
  }

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < iteration; n++) {
    Kokkos::parallel_for("geodesic", N,
      KOKKOS_LAMBDA(const int wiID) {
        const float GDC_DEG_TO_RAD  = 3.141592654f / 180.0f;
        const float GDC_FLATTENING  = 1.0f - (6356752.31424518f / 6378137.0f);
        const float GDC_ECCENTRICITY= (6356752.31424518f / 6378137.0f);
        const float GDC_ELLIPSOIDAL = 1.0f/(6356752.31414f/6378137.0f)/(6356752.31414f/6378137.0f) - 1.0f;
        const float GDC_SEMI_MINOR  = 6356752.31424518f;
        const float EPS             = 0.5e-5f;

        const float rad_lat1 = va_d(wiID).x * GDC_DEG_TO_RAD;
        const float rad_lon1 = va_d(wiID).y * GDC_DEG_TO_RAD;
        const float rad_lat2 = va_d(wiID).z * GDC_DEG_TO_RAD;
        const float rad_lon2 = va_d(wiID).w * GDC_DEG_TO_RAD;

        float TU1 = GDC_ECCENTRICITY * sinf(rad_lat1) / cosf(rad_lat1);
        float TU2 = GDC_ECCENTRICITY * sinf(rad_lat2) / cosf(rad_lat2);
        float CU1 = 1.0f / sqrtf(TU1*TU1 + 1.0f);
        float SU1 = CU1 * TU1;
        float CU2 = 1.0f / sqrtf(TU2*TU2 + 1.0f);
        float dist = CU1 * CU2;
        float BAZ = dist * TU2;
        float FAZ = BAZ * TU1;
        float X = rad_lon2 - rad_lon1;
        float SX, CX, SY, CY, Y, SA, C2A, CZ, E, C, D;

        do {
          SX = sinf(X); CX = cosf(X);
          TU1 = CU2 * SX;
          TU2 = BAZ - SU1 * CU2 * CX;
          SY = sqrtf(TU1*TU1 + TU2*TU2);
          CY = dist * CX + FAZ;
          Y = atan2f(SY, CY);
          SA = dist * SX / SY;
          C2A = -SA*SA + 1.0f;
          CZ = FAZ + FAZ;
          if (C2A > 0.0f) CZ = -CZ / C2A + CY;
          E = CZ*CZ*2.0f - 1.0f;
          C = ((-3.0f*C2A + 4.0f)*GDC_FLATTENING + 4.0f)*C2A*GDC_FLATTENING / 16.0f;
          D = X;
          X = ((E*CY*C + CZ)*SY*C + Y)*SA;
          X = (1.0f - C)*X*GDC_FLATTENING + rad_lon2 - rad_lon1;
        } while (fabsf(D - X) > EPS);

        X = sqrtf(GDC_ELLIPSOIDAL * C2A + 1.0f) + 1.0f;
        X = (X - 2.0f) / X;
        C = 1.0f - X;
        C = (X*X/4.0f + 1.0f) / C;
        D = (0.375f*X*X - 1.0f) * X;
        X = E * CY;
        dist = 1.0f - E - E;
        dist = ((((SY*SY*4.0f - 3.0f)*dist*CZ*D/6.0f - X)*D/4.0f + CZ)*SY*D + Y)*C*GDC_SEMI_MINOR;
        vc_d(wiID) = dist;
      });
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);

  auto vc_m = Kokkos::create_mirror_view(vc_d);
  Kokkos::deep_copy(vc_m, vc_d);
  for (int i = 0; i < N; i++) VC[i] = vc_m(i);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage %s <repeat>\n", argv[0]);
    return 1;
  }
  int iteration = atoi(argv[1]);

  int num_cities     = 2097152;
  int num_ref_cities = 6;
  int index_map[]    = {436483, 1952407, 627919, 377884, 442703, 1863423};
  int N              = num_cities * num_ref_cities;

  const char* filename = "locations.txt";
  printf("Reading city locations from file %s...\n", filename);
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    // try sibling directory
    filename = "../geodesic-omp/locations.txt";
    fp = fopen(filename, "r");
    if (!fp) { perror("Error opening locations.txt"); return 1; }
    printf("Opened %s\n", filename);
  }

  float4* input           = (float4*)aligned_alloc(4096, N * sizeof(float4));
  float*  output          = (float*)aligned_alloc(4096, N * sizeof(float));
  float*  expected_output = (float*)malloc(N * sizeof(float));

  int city = 0;
  float lat, lon;
  while (fscanf(fp, "%f %f\n", &lat, &lon) != EOF) {
    input[city].x = lat;
    input[city].y = lon;
    city++;
    if (city == num_cities) break;
  }
  fclose(fp);

  for (int c = 1; c < num_ref_cities; c++)
    for (int i = 0; i < num_cities; i++)
      input[c*num_cities + i] = input[i];

  for (int c = 0; c < num_ref_cities; c++) {
    int index = index_map[c] - 1;
    for (int j = c*num_cities; j < (c+1)*num_cities; ++j) {
      input[j].z = input[index].x;
      input[j].w = input[index].y;
    }
  }

  for (int i = 0; i < N; i++)
    expected_output[i] = distance_host(input[i].x, input[i].y, input[i].z, input[i].w);

  Kokkos::initialize(argc, argv);
  {
    distance_device(input, output, N, iteration);
  }
  Kokkos::finalize();

  float error_rate = 0.f;
  for (int i = 0; i < N; i++) {
    float diff = fabsf(output[i] - expected_output[i]);
    if (diff > error_rate) error_rate = diff;
  }
  printf("The maximum error in distance for single precision is %f\n", error_rate);

  free(input); free(output); free(expected_output);
  return 0;
}
