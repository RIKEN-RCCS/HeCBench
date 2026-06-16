/*
 * Detection overlay box drawing.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct Box {
  int width, height, left, top;
};

// CPU reference
template<typename T>
void reference(const T *input, T *output,
               int imgWidth, int imgHeight,
               const Box *detections, int numDetections,
               float4 colors)
{
  for (int n = 0; n < numDetections; n++) {
    int boxWidth  = detections[n].width;
    int boxHeight = detections[n].height;
    int x0 = detections[n].left;
    int y0 = detections[n].top;

    for (int box_y = 0; box_y < boxHeight; box_y++) {
      for (int box_x = 0; box_x < boxWidth; box_x++) {
        int x = box_x + x0;
        int y = box_y + y0;
        if (x < imgWidth && y < imgHeight) {
          T px = input[y * imgWidth + x];
          float alpha = colors.w / 255.f;
          float ialph = 1.f - alpha;
          px.x = alpha * colors.x + ialph * px.x;
          px.y = alpha * colors.y + ialph * px.y;
          px.z = alpha * colors.z + ialph * px.z;
          output[y * imgWidth + x] = px;
        }
      }
    }
  }
}

template<typename T>
void DetectionOverlayBox(
    Kokkos::View<const T*> input,
    Kokkos::View<T*>       output,
    int imgWidth, int imgHeight,
    int x0, int y0, int boxWidth, int boxHeight,
    float4 color)
{
  Kokkos::parallel_for("overlay",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{boxHeight,boxWidth}),
      KOKKOS_LAMBDA(int box_y, int box_x) {
        int x = box_x + x0;
        int y = box_y + y0;
        if (x < imgWidth && y < imgHeight) {
          T px = input(y * imgWidth + x);
          float alpha = color.w / 255.f;
          float ialph = 1.f - alpha;
          px.x = alpha * color.x + ialph * px.x;
          px.y = alpha * color.y + ialph * px.y;
          px.z = alpha * color.z + ialph * px.z;
          output(y * imgWidth + x) = px;
        }
      });
  Kokkos::fence();
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <width> <height>\n", argv[0]);
    return 1;
  }

  const int width  = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int img_size = width * height;

  srand(123);
  float3 *input      = (float3*) malloc(img_size * sizeof(float3));
  float3 *output     = (float3*) malloc(img_size * sizeof(float3));
  float3 *ref_output = (float3*) malloc(img_size * sizeof(float3));

  for (int i = 0; i < img_size; i++) {
    output[i].x = ref_output[i].x = input[i].x = (float)(rand() % 256);
    output[i].y = ref_output[i].y = input[i].y = (float)(rand() % 256);
    output[i].z = ref_output[i].z = input[i].z = (float)(rand() % 256);
  }

  float4 colors = {255.f, 204.f, 203.f, 1.f};

  const int numDetections = (int)(img_size * 0.8f);
  Box *detections = (Box*) malloc(numDetections * sizeof(Box));
  for (int i = 0; i < numDetections; i++) {
    detections[i].width  = 64 + rand() % 128;
    detections[i].height = 64 + rand() % 128;
    detections[i].left   = rand() % (width  - 64);
    detections[i].top    = rand() % (height - 64);
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float3*> d_input("d_input", img_size);
    Kokkos::View<float3*> d_output("d_output", img_size);

    auto h_input  = Kokkos::create_mirror_view(d_input);
    auto h_output = Kokkos::create_mirror_view(d_output);
    for (int i = 0; i < img_size; i++) {
      h_input(i)  = input[i];
      h_output(i) = output[i];
    }
    Kokkos::deep_copy(d_input,  h_input);
    Kokkos::deep_copy(d_output, h_output);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < numDetections; n++) {
      DetectionOverlayBox<float3>(
          d_input, d_output, width, height,
          detections[n].left, detections[n].top,
          detections[n].width, detections[n].height, colors);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

    Kokkos::deep_copy(h_output, d_output);
    for (int i = 0; i < img_size; i++) output[i] = h_output(i);
  }
  Kokkos::finalize();

  reference<float3>(input, ref_output, width, height, detections, numDetections, colors);

  bool ok = true;
  for (int i = 0; i < img_size; i++) {
    if (fabsf(ref_output[i].x - output[i].x) > 1e-3f ||
        fabsf(ref_output[i].y - output[i].y) > 1e-3f ||
        fabsf(ref_output[i].z - output[i].z) > 1e-3f) {
      printf("Error at index %d\n", i);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output);
  free(ref_output);
  free(detections);
  return 0;
}
