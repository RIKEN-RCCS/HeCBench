#include <limits>
#include <algorithm>
#include <cstdlib>

struct hec_float2 {
  float x, y;
};

inline float hd(const hec_float2 ap, const hec_float2 bp);

int cmpfunc(const void *a, const void *b) {
  return (*(float *)a - *(float *)b) > 0.f ? 1 : 0;
}

float computeDirDistance(const hec_float2 Apoints[], const hec_float2 Bpoints[],
                         int numA, int numB) {
  float *disA = (float *)malloc(sizeof(float) * numA);

  for (int i = 0; i < numA; i++) {
    float d = std::numeric_limits<float>::max();
    for (int j = 0; j < numB; j++) {
      float t = hd(Apoints[i], Bpoints[j]);
      d = std::min(t, d);
    }
    disA[i] = d;
  }
  qsort(disA, numA, sizeof(float), cmpfunc);
  float dis = disA[numA - 1];

  free(disA);
  return dis;
}

float hausdorff_distance(const hec_float2 Apoints[], const hec_float2 Bpoints[],
                         int numA, int numB) {
  float hAB = computeDirDistance(Apoints, Bpoints, numA, numB);
  float hBA = computeDirDistance(Bpoints, Apoints, numB, numA);
  return std::max(hAB, hBA);
}
