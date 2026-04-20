/*
 * Debayer (Malvar-He-Cutler demosaicing) – Kokkos port
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// Types
typedef unsigned char  uchar;
typedef unsigned int   uint;

struct uchar4 { uchar x, y, z, w; };
struct uchar3 { uchar x, y, z; };

// Bayer patterns
enum { RGGB=0, GRBG=1, GBRG=2, BGGR=3 };

#define OUTPUT_CHANNELS 4
#define ALPHA_VALUE     255

// Clamp a value to [0,255]
KOKKOS_INLINE_FUNCTION uchar clamp_uchar(int v) {
  return (uchar)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// Reflect-border-exclusive sampling (returns flat index or border value)
KOKKOS_INLINE_FUNCTION int tex2D_reflect(const uchar* input_p, int height, int width,
    int pitch, int r, int c)
{
  // Reflect at borders (exclusive: doesn't duplicate boundary)
  if (c < 0) c = -c;
  if (c >= width)  c = width  - (c - width)  - 2;
  if (r < 0) r = -r;
  if (r >= height) r = height - (r - height) - 2;
  // Clamp as safety
  c = c < 0 ? 0 : (c >= width  ? width -1 : c);
  r = r < 0 ? 0 : (r >= height ? height-1 : r);
  return input_p[r * pitch + c];
}

// Demosaic kernel using flat parallel_for (one thread per output pixel)
KOKKOS_INLINE_FUNCTION int reflect_coord(int x, int size) {
  if (x < 0) x = -x;
  if (x >= size) x = size - (x - size) - 2;
  if (x < 0) x = 0;
  if (x >= size) x = size - 1;
  return x;
}

void malvar_he_cutler_demosaic_kokkos(
    uint /*teamX*/, uint /*teamY*/,
    uint height, uint width,
    const uchar* /*input_p*/,  int input_pitch,
    uchar* /*output_p*/,       int output_pitch,
    int bayer_pattern,
    Kokkos::View<uchar*>  d_input,
    Kokkos::View<uchar*>  d_output)
{
  const int h = (int)height, w = (int)width;
  const int ipitch = input_pitch, opitch = output_pitch;
  const int bp = bayer_pattern;

  Kokkos::parallel_for("demosaic", h * w,
    KOKKOS_LAMBDA(int idx) {
      int g_r = idx / w;
      int g_c = idx % w;

      // Inline reflect-border sample
      auto F = [&](int dc, int dr) -> int {
        int rc = reflect_coord(g_c + dc, w);
        int rr = reflect_coord(g_r + dr, h);
        return d_input[rr * ipitch + rc];
      };

      int Fij = F(0, 0);
      int R1 = (4*Fij + 2*(F(-1,0)+F(0,-1)+F(1,0)+F(0,1))
               - F(-2,0) - F(2,0) - F(0,-2) - F(0,2)) / 8;
      int R2 = (8*(F(-1,0)+F(1,0)) + 10*Fij + F(0,-2) + F(0,2)
               - 2*(F(-1,-1)+F(1,-1)+F(-1,1)+F(1,1))
               - 2*(F(-2,0)+F(2,0))) / 16;
      int R3 = (8*(F(0,-1)+F(0,1)) + 10*Fij + F(-2,0) + F(2,0)
               - 2*(F(-1,-1)+F(1,-1)+F(-1,1)+F(1,1))
               - 2*(F(0,-2)+F(0,2))) / 16;
      int R4 = (12*Fij - 3*(F(-2,0)+F(2,0)+F(0,-2)+F(0,2))
               + 4*(F(-1,-1)+F(1,-1)+F(-1,1)+F(1,1))) / 16;

      int red_col  = (bp == GRBG || bp == BGGR) ? 1 : 0;
      int red_row  = (bp == GBRG || bp == BGGR) ? 1 : 0;
      int blue_col = 1 - red_col;
      int blue_row = 1 - red_row;

      int r_mod_2 = g_r & 1, c_mod_2 = g_c & 1;
      int in_red_row     = (r_mod_2 == red_row);
      int in_blue_row    = (r_mod_2 == blue_row);
      int is_red_pixel   = (r_mod_2 == red_row)  & (c_mod_2 == red_col);
      int is_blue_pixel  = (r_mod_2 == blue_row) & (c_mod_2 == blue_col);
      int is_green_pixel = !(is_red_pixel | is_blue_pixel);

      uchar R_out = clamp_uchar(
          Fij * is_red_pixel + R4 * is_blue_pixel
          + R2 * (is_green_pixel & in_red_row)
          + R3 * (is_green_pixel & in_blue_row));
      uchar B_out = clamp_uchar(
          Fij * is_blue_pixel + R4 * is_red_pixel
          + R3 * (is_green_pixel & in_red_row)
          + R2 * (is_green_pixel & in_blue_row));
      uchar G_out = clamp_uchar(Fij*is_green_pixel + R1*(!is_green_pixel));

      int out_base = g_r * opitch + g_c * OUTPUT_CHANNELS;
      d_output[out_base + 0] = R_out;
      d_output[out_base + 1] = G_out;
      d_output[out_base + 2] = B_out;
      d_output[out_base + 3] = (uchar)ALPHA_VALUE;
    });
}

int main(int argc, char* argv[]) {
  if (argc != 4) { printf("Usage: %s <width> <height> <repeat>\n", argv[0]); return 1; }
  const int width  = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_pitch  = width;
  const int output_pitch = width * OUTPUT_CHANNELS;
  const int numPix       = width * height;

  uchar* input  = (uchar*)malloc(numPix);
  uchar* output = (uchar*)malloc(numPix * OUTPUT_CHANNELS);

  const int bayer_pattern = RGGB;
  srand(123);
  for (int i = 0; i < numPix; i++) input[i] = rand() % 256;

  const uint teamX = (width  + tile_cols - 1) / tile_cols;
  const uint teamY = (height + tile_rows - 1) / tile_rows;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<uchar*> d_input;
    Kokkos::View<uchar*> d_output("output", numPix * OUTPUT_CHANNELS);

    {
      Kokkos::View<uchar*> d_in_tmp("in_tmp", numPix);
      auto h_in = Kokkos::create_mirror_view(d_in_tmp);
      for (int i = 0; i < numPix; i++) h_in(i) = input[i];
      Kokkos::deep_copy(d_in_tmp, h_in);
      d_input = d_in_tmp;
    }

    auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      Kokkos::deep_copy(d_output, (uchar)0);
      malvar_he_cutler_demosaic_kokkos(teamX, teamY, height, width,
          input, input_pitch, output, output_pitch, bayer_pattern,
          d_input, d_output);
      Kokkos::fence();
    }
    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average kernel execution time %f (s)\n", elapsed * 1e-9 / repeat);

    auto h_out = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_out, d_output);
    for (int i = 0; i < numPix * OUTPUT_CHANNELS; i++) output[i] = h_out(i);
  }
  Kokkos::finalize();

  long sum = 0;
  for (int i = 0; i < numPix * OUTPUT_CHANNELS; i++) sum += output[i];
  printf("Checksum: %ld\n", sum);

  free(input); free(output);
  return 0;
}
