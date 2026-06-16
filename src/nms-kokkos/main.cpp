/*
 * NMS Benchmarking Framework - Kokkos port
 * "Work-Efficient Parallel Non-Maximum Suppression Kernels"
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt> <repeat>\n\n");
  printf("               detections.txt -> Input file with coordinates, width, scores\n");
  printf("               output.txt     -> Output file after NMS\n");
  printf("               repeat         -> Kernel execution count\n\n");
}

int get_optimal_dim(int val)
{
  int div, neg, cntneg, cntpos;
  neg = 1;
  div = 16;
  cntneg = div;
  cntpos = div;
  for(int i=0; i<5; i++)
  {
    if(val % div == 0)
      return div;
    if(neg) { cntneg--; div = cntneg; neg = 0; }
    else    { cntpos++; div = cntpos; neg = 1; }
  }
  return 16;
}

int get_upper_limit(int val, int mul)
{
  int cnt = mul;
  while(cnt < val) cnt += mul;
  if(cnt > MAX_DETECTIONS) cnt = MAX_DETECTIONS;
  return cnt;
}

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {
    int x, y, w;
    float score;

    if(argc != 4)
    {
      print_help();
      Kokkos::finalize();
      return 0;
    }

    int ndetections = 0;

    FILE *fp = fopen(argv[1], "r");
    if (!fp)
    {
      printf("Error: Unable to open file %s\n", argv[1]);
      Kokkos::finalize();
      return -1;
    }

    // Host storage: 4 floats per detection (x, y, z, w)
    std::vector<float> cpu_points(4 * MAX_DETECTIONS, 0.0f);

    while(!feof(fp))
    {
      int cnt = fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);
      if (cnt != 4)
      {
        printf("Error: Invalid file format at line %d\n", ndetections);
        fclose(fp);
        Kokkos::finalize();
        return -1;
      }
      cpu_points[ndetections*4+0] = (float) x;
      cpu_points[ndetections*4+1] = (float) y;
      cpu_points[ndetections*4+2] = (float) w;
      cpu_points[ndetections*4+3] = score;
      ndetections++;
    }
    fclose(fp);
    printf("Number of detections read: %d\n", ndetections);

    // Device views
    Kokkos::View<float*> rects("rects", 4 * MAX_DETECTIONS);
    Kokkos::View<unsigned char*> nmsbitmap("nmsbitmap", (size_t)MAX_DETECTIONS * MAX_DETECTIONS);
    Kokkos::View<unsigned char*> pointsbitmap("pointsbitmap", MAX_DETECTIONS);

    auto h_rects = Kokkos::create_mirror_view(rects);
    for (int i = 0; i < 4 * MAX_DETECTIONS; i++)
      h_rects(i) = cpu_points[i];
    Kokkos::deep_copy(rects, h_rects);

    // Initialize nmsbitmap to 1, pointsbitmap to 0
    Kokkos::deep_copy(nmsbitmap, (unsigned char)1);
    Kokkos::deep_copy(pointsbitmap, (unsigned char)0);

    int repeat = atoi(argv[3]);
    int limit = get_upper_limit(ndetections, 16);

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("generate_nms_bitmap",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{limit,limit}),
        KOKKOS_LAMBDA(int j, int i) {
          if (rects(i*4+3) < rects(j*4+3)) {
            float area = (rects(j*4+2) + 1.0f) * (rects(j*4+2) + 1.0f);
            float ww = Kokkos::fmax(0.0f,
                         Kokkos::fmin(rects(i*4+0) + rects(i*4+2),
                                      rects(j*4+0) + rects(j*4+2))
                         - Kokkos::fmax(rects(i*4+0), rects(j*4+0)) + 1.0f);
            float hh = Kokkos::fmax(0.0f,
                         Kokkos::fmin(rects(i*4+1) + rects(i*4+2),
                                      rects(j*4+1) + rects(j*4+2))
                         - Kokkos::fmax(rects(i*4+1), rects(j*4+1)) + 1.0f);
            nmsbitmap(i * MAX_DETECTIONS + j) =
              (unsigned char)((((ww * hh) / area) < 0.3f) && (rects(j*4+2) != 0));
          }
        });
    }
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (generate_nms_bitmap): %f (s)\n",
           (time * 1e-9f) / repeat);

    // Reduce: for each detection i, check if all entries in row i of nmsbitmap are 1
    start = std::chrono::steady_clock::now();
    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("reduce_nms_bitmap", ndetections,
        KOKKOS_LAMBDA(int i) {
          bool all_one = true;
          for (int jj = 0; jj < MAX_DETECTIONS; jj++) {
            if (!nmsbitmap(i * MAX_DETECTIONS + jj)) {
              all_one = false;
              break;
            }
          }
          pointsbitmap(i) = all_one ? (unsigned char)1 : (unsigned char)0;
        });
    }
    Kokkos::fence();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (reduce_nms_bitmap): %f (s)\n",
           (time * 1e-9f) / repeat);

    // Copy results back
    auto h_pointsbitmap = Kokkos::create_mirror_view(pointsbitmap);
    Kokkos::deep_copy(h_pointsbitmap, pointsbitmap);

    fp = fopen(argv[2], "w");
    if (!fp)
    {
      printf("Error: Unable to open output file %s\n", argv[2]);
      Kokkos::finalize();
      return -1;
    }

    int totaldets = 0;
    for(int i = 0; i < ndetections; i++)
    {
      if(h_pointsbitmap(i))
      {
        x     = (int) cpu_points[i*4+0];
        y     = (int) cpu_points[i*4+1];
        w     = (int) cpu_points[i*4+2];
        score = cpu_points[i*4+3];
        fprintf(fp, "%d,%d,%d,%f\n", x, y, w, score);
        totaldets++;
      }
    }
    fclose(fp);
    printf("Detections after NMS: %d\n", totaldets);
  }
  Kokkos::finalize();
  return 0;
}
