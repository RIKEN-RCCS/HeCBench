// Kokkos port of Nearest Neighbor benchmark
// Original OMP target source: src/nn-omp/nearestNeighbor.cpp
#include <iostream>
#include <vector>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>

#define REC_LENGTH 49

typedef struct latLong {
  float lat;
  float lng;
} LatLong;

typedef struct record {
  char recString[REC_LENGTH];
  float distance;
} Record;

void printUsage() {
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt]\n");
  printf("  -r  Number of results to show\n");
  printf("  -lat Query latitude\n");
  printf("  -lng Query longitude\n");
  printf("  -t  Show timing info\n");
  printf("  -q  Quiet (less output)\n");
}

int parseCommandline(int argc, char *argv[], char *filename,
                     int *r, float *lat, float *lng,
                     int *repeat, int *q, int *t) {
  int i;
  if (argc < 2) return 1;
  strncpy(filename, argv[1], 99);
  char flag;
  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      flag = argv[i][1];
      switch (flag) {
        case 'r': i++; *r = atoi(argv[i]); break;
        case 'l': flag = argv[i][2];
          switch (flag) {
            case 'a': i++; *lat = atof(argv[i]); break;
            case 'n': i++; *lng = atof(argv[i]); break;
          } break;
        case 'q': *q = 1; break;
        case 't': *t = 1; break;
        case 'n': i++; *repeat = atoi(argv[i]); break;
      }
    }
  }
  if ((*lat == 0.0) && (*lng == 0.0)) return 1;
  return 0;
}

int loadData(char *filename, std::vector<Record> &records,
             std::vector<LatLong> &locations) {
  FILE *flist, *fp;
  int i = 0;
  char dbname[64];
  int recNum = 0;

  flist = fopen(filename, "r");
  while (!feof(flist)) {
    if (fscanf(flist, "%s\n", dbname) != 1) {
      fprintf(stderr, "error reading filelist\n");
      exit(0);
    }
    fp = fopen(dbname, "r");
    if (!fp) {
      printf("error opening a db input file\n");
      exit(0);
    }
    while (!feof(fp)) {
      records.push_back(Record());
      locations.push_back(LatLong());
      Record &record = records[i];
      LatLong &latLong = locations[i];
      if (fscanf(fp, "%s %f %f\n", record.recString, &latLong.lat, &latLong.lng) != 3) break;
      recNum++;
      i++;
    }
    fclose(fp);
  }
  fclose(flist);
  return i;
}

void findLowest(std::vector<Record> &records, float *distances,
                int numRecords, int topN) {
  for (int i = 0; i < topN; i++) {
    int minLoc = i;
    for (int j = i + 1; j < numRecords; j++) {
      if (distances[j] < distances[minLoc]) minLoc = j;
    }
    Record tmp = records[i];
    records[i] = records[minLoc];
    records[minLoc] = tmp;
    float tmpd = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tmpd;
  }
}

void FindNearestNeighbors(
    int numRecords,
    std::vector<LatLong> &locations,
    float lat, float lng,
    float *distances,
    int repeat, int timing)
{
  Kokkos::View<LatLong*> d_locations("d_locations", numRecords);
  Kokkos::View<float*>   d_distances("d_distances", numRecords);

  auto h_locations = Kokkos::create_mirror_view(d_locations);
  for (int i = 0; i < numRecords; i++) h_locations(i) = locations[i];
  Kokkos::deep_copy(d_locations, h_locations);

  auto start = std::chrono::steady_clock::now();

  for (int k = 0; k < repeat; k++) {
    Kokkos::parallel_for("nn_distances",
      Kokkos::RangePolicy<>(0, numRecords),
      KOKKOS_LAMBDA(const int gid) {
        LatLong ll = d_locations(gid);
        float dlat = lat - ll.lat;
        float dlng = lng - ll.lng;
        d_distances(gid) = sqrtf(dlat * dlat + dlng * dlng);
      });
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  auto h_distances = Kokkos::create_mirror_view(d_distances);
  Kokkos::deep_copy(h_distances, d_distances);
  for (int i = 0; i < numRecords; i++) distances[i] = h_distances(i);
}

int main(int argc, char *argv[]) {
  std::vector<Record> records;
  float *recordDistances;
  std::vector<LatLong> locations;
  char filename[100];
  int resultsCount = 10, quiet = 0, timing = 0, repeat = 1;
  float lat = 0.0f, lng = 0.0f;

  if (parseCommandline(argc, argv, filename, &resultsCount,
                       &lat, &lng, &repeat, &quiet, &timing)) {
    printUsage();
    return 0;
  }

  int numRecords = loadData(filename, records, locations);
  if (!quiet) {
    printf("Number of records: %d\n", numRecords);
    printf("Finding the %d closest neighbors.\n", resultsCount);
  }
  if (resultsCount > numRecords) resultsCount = numRecords;

  recordDistances = (float *)malloc(sizeof(float) * numRecords);

  Kokkos::initialize(argc, argv);
  {
    auto start = std::chrono::steady_clock::now();
    FindNearestNeighbors(numRecords, locations, lat, lng, recordDistances, repeat, timing);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (timing) printf("Device offloading time %f (s)\n", time * 1e-9);
  }
  Kokkos::finalize();

  findLowest(records, recordDistances, numRecords, resultsCount);
  if (!quiet) {
    for (int i = 0; i < resultsCount; i++)
      printf("%s --> Distance=%f\n", records[i].recString, records[i].distance);
  }
  free(recordDistances);
  return 0;
}
