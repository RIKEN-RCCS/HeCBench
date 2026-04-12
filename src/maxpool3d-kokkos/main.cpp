#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

typedef float DTYPE;

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <image count> <repeat>\n", argv[0]);
    return 1;
  }
  const int i_img_width  = atoi(argv[1]);
  const int i_img_height = atoi(argv[2]);
  const int i_img_count  = atoi(argv[3]);
  const int repeat       = atoi(argv[4]);

  if (i_img_width % 16 != 0 || i_img_height % 16 != 0) {
    printf("image dimension is a multiple of 16\n");
    return 1;
  }

  const int Hstride = 2, Vstride = 2;
  const int o_img_width  = i_img_width  / Hstride;
  const int o_img_height = i_img_height / Vstride;
  const int pool_width   = Hstride;
  const int pool_height  = Vstride;

  printf("input image width %d Hstride %d\n",  i_img_width,  Hstride);
  printf("input image height %d Vstride %d\n", i_img_height, Vstride);
  printf("output image width %d\n",  o_img_width);
  printf("output image height %d\n", o_img_height);

  const int size_image  = i_img_width * i_img_height;
  const int size_output = o_img_width * o_img_height;

  DTYPE *h_image  = (DTYPE*)malloc(sizeof(DTYPE) * size_image  * i_img_count);
  DTYPE *h_output = (DTYPE*)malloc(sizeof(DTYPE) * size_output * i_img_count);
  DTYPE *d_output = (DTYPE*)malloc(sizeof(DTYPE) * size_output * i_img_count);

  srand(2);
  for (int j = 0; j < i_img_count; j++)
    for (int i = 0; i < size_image; i++)
      h_image[j * size_image + i] = rand() % 256 / (DTYPE)255;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<DTYPE*> d_img("d_img", size_image * i_img_count);
    Kokkos::View<DTYPE*> d_out("d_out", size_output * i_img_count);

    {
      auto h = Kokkos::create_mirror_view(d_img);
      for (int i = 0; i < size_image * i_img_count; i++) h(i) = h_image[i];
      Kokkos::deep_copy(d_img, h);
    }

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("maxpool3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
          {0, 0, 0}, {i_img_count, o_img_height, o_img_width}),
        KOKKOS_LAMBDA(int z, int y, int x) {
          const int xidx = Hstride * x;
          const int yidx = Vstride * y;
          DTYPE maxval = (DTYPE)0;
          for (int r = 0; r < pool_height; r++) {
            const int base = (z * i_img_height + yidx + r) * i_img_width + xidx;
            for (int c = 0; c < pool_width; c++) {
              DTYPE v = d_img(base + c);
              if (v > maxval) maxval = v;
            }
          }
          d_out(((z * o_img_height) + y) * o_img_width + x) = maxval;
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    auto hout = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(hout, d_out);
    for (int i = 0; i < size_output * i_img_count; i++) d_output[i] = hout(i);
  }
  Kokkos::finalize();

  // CPU reference
  for (int z = 0; z < i_img_count; z++) {
    for (int y = 0; y < o_img_height; y++) {
      for (int x = 0; x < o_img_width; x++) {
        const int xidx = Hstride * x;
        const int yidx = Vstride * y;
        DTYPE maxval = (DTYPE)0;
        for (int r = 0; r < pool_height; r++) {
          const int base = ((z * i_img_height + yidx + r) * i_img_width) + xidx;
          for (int c = 0; c < pool_width; c++) {
            DTYPE v = h_image[base + c];
            if (v > maxval) maxval = v;
          }
        }
        h_output[((z * o_img_height) + y) * o_img_width + x] = maxval;
      }
    }
  }

  int status = memcmp(h_output, d_output, sizeof(DTYPE) * i_img_count * o_img_height * o_img_width);
  printf("%s\n", (status == 0) ? "PASS" : "FAIL");

  free(h_image); free(h_output); free(d_output);
  return status;
}
