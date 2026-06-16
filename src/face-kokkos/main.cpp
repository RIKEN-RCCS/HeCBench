/*
 * Face detection main - Kokkos port.
 * Original by Francesco Comaschi (TU Eindhoven).
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "image.h"
#include "stdio-wrapper.h"
#include "haar.h"

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <input image file> <classifier information> "
           "<class information> <output image file>\n", argv[0]);
    return -1;
  }

  Kokkos::initialize(argc, argv);
  {
    float scaleFactor  = 1.2f;
    int minNeighbours  = 1;

    printf("-- entering main function --\r\n");
    printf("-- loading image --\r\n");

    MyImage imageObj;
    MyImage *image = &imageObj;

    int flag = readPgm(argv[1], image);
    if (flag == -1) {
      printf("Unable to open input image\n");
      Kokkos::finalize();
      return 1;
    }

    int total_nodes = readTextClassifier(argv[2], argv[3]);
    if (total_nodes == -1) {
      Kokkos::finalize();
      return 1;
    }

    printf("-- loading cascade classifier --\r\n");

    myCascade cascadeObj;
    myCascade *cascade = &cascadeObj;
    MySize minSize = {20, 20};
    MySize maxSize = {0, 0};

    cascade->n_stages = 25;
    cascade->total_nodes = 2913;
    cascade->orig_window_size.height = 24;
    cascade->orig_window_size.width  = 24;

    printf("-- detecting faces --\r\n");
    std::vector<MyRect> result;

    auto start = std::chrono::steady_clock::now();
    result = detectObjects(image, minSize, maxSize, cascade, scaleFactor,
                           minNeighbours, total_nodes);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Object detection time %f (s)\n", time * 1e-9f);

    for (unsigned int i = 0; i < result.size(); i++)
      drawRectangle(image, result[i]);

    printf("-- saving output --\r\n");
    writePgm(argv[4], image);
    printf("-- image saved --\r\n");

    releaseTextClassifier();
    freeImage(image);
  }
  Kokkos::finalize();
  return 0;
}
