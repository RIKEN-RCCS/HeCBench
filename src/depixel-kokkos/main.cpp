#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

typedef unsigned int uint;

struct float3 {
  float x, y, z;
};

KOKKOS_INLINE_FUNCTION
float saturatef(float v) {
  return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

KOKKOS_INLINE_FUNCTION
uint rgbToyuv(float3 rgba) {
  float3 yuv;
  yuv.x = 0.299f*rgba.x + 0.587f*rgba.y + 0.114f*rgba.z;
  yuv.y = 0.713f*(rgba.x - yuv.x) + 0.5f;
  yuv.z = 0.564f*(rgba.z - yuv.x) + 0.5f;
  yuv.x = saturatef(yuv.x);
  yuv.y = saturatef(yuv.y);
  yuv.z = saturatef(yuv.z);
  return (uint(255)<<24) | (uint(yuv.z*255.f)<<16) | (uint(yuv.y*255.f)<<8) | uint(yuv.x*255.f);
}

KOKKOS_INLINE_FUNCTION
bool isConnected(uint lnode, uint rnode) {
  int ly = lnode & 0xff, lu = (lnode>>8)&0xff, lv = (lnode>>16)&0xff;
  int ry = rnode & 0xff, ru = (rnode>>8)&0xff, rv = (rnode>>16)&0xff;
  return !((Kokkos::abs(ly-ry) > 48) || (Kokkos::abs(lu-ru) > 7) || (Kokkos::abs(lv-rv) > 6));
}

KOKKOS_INLINE_FUNCTION
uint bitCount(uint v) {
  uint c;
  for (c = 0; v; ++c) v &= v - 1;
  return c;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]);
    return 1;
  }
  int width  = atoi(argv[1]);
  int height = atoi(argv[2]);
  int repeat = atoi(argv[3]);
  int size   = width * height;

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.f, 0.4f);

  float3* img = (float3*)malloc(size * sizeof(float3));
  uint*   out = (uint*)  malloc(size * sizeof(uint));

  float sum = 0.f, total_time = 0.f;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float3*> d_img("img", size);
    Kokkos::View<uint*>   d_tmp("tmp", size);
    Kokkos::View<uint*>   d_out("out", size);
    auto h_img = Kokkos::create_mirror_view(d_img);
    auto h_out = Kokkos::create_mirror_view(d_out);

    for (int n = 0; n < repeat; n++) {
      for (int i = 0; i < size; i++) {
        img[i].x = dis(gen); img[i].y = dis(gen); img[i].z = dis(gen);
        h_img(i) = img[i];
      }
      Kokkos::deep_copy(d_img, h_img);

      auto start = std::chrono::steady_clock::now();

      // check_connect kernel
      Kokkos::parallel_for("check_connect", size, KOKKOS_LAMBDA(int center) {
        int row = center / width, col = center % width;
        unsigned char con = 0;
        uint yuv_c = rgbToyuv(d_img(center));

        auto neighbor = [&](int dr, int dc) -> uint {
          int nr = (row + dr < 0 || row + dr >= height) ? row : row + dr;
          int nc = (col + dc < 0 || col + dc >= width)  ? col : col + dc;
          return rgbToyuv(d_img(nr * width + nc));
        };

        // upper-left
        { int nr = (row>0&&col>0)?row-1:row, nc = (col>0&&row>0)?col-1:col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y)); }
        // upper
        { int nr = row>0?row-1:row, nc = col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y))<<1; }
        // upper-right
        { int nr = (row>0&&col<width-1)?row-1:row, nc = (col<width-1&&row>0)?col+1:col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y))<<2; }
        // right
        { int nr = row, nc = col<width-1?col+1:col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y))<<3; }
        // lower-right
        { int nr = (row<height-1&&col<width-1)?row+1:row, nc = (col<width-1&&row<height-1)?col+1:col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y))<<4; }
        // lower
        { int nr = row<height-1?row+1:row, nc = col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y))<<5; }
        // lower-left
        { int nr = (row<height-1&&col>0)?row+1:row, nc = (col>0&&row<height-1)?col-1:col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y))<<6; }
        // left
        { int nr = row, nc = col>0?col-1:col;
          uint y = rgbToyuv(d_img(nr*width+nc));
          con += (uint)(!(row==nr&&col==nc) && isConnected(yuv_c,y))<<7; }

        d_tmp(center) = (yuv_c>>16&0xFF)<<24 | (yuv_c>>8&0xFF)<<16 | (yuv_c&0xFF)<<8 | con;
      });
      Kokkos::fence();

      // eliminate_crosses kernel
      Kokkos::parallel_for("eliminate_crosses", size, KOKKOS_LAMBDA(int center) {
        int row = center / width, col = center % width;
        int start_row    = row > 2      ? row - 3 : 0;
        int start_col    = col > 2      ? col - 3 : 0;
        int end_row      = row < width - 4  ? row + 4 : width - 1;
        int end_col      = col < height - 4 ? col + 4 : height - 1;
        int weight_l = 0, weight_r = 0;
        d_out(center) = 0;
        bool remove_cross = false;

        if (row < height-1 && col < width-1) {
          const uint* id = d_tmp.data();
          uint* od = d_out.data();
          od[center] = (id[center]&0x08)>>3 |
                       ((id[center+width+1]&0x02)>>1)<<1 |
                       ((id[center+width+1]&0x80)>>7)<<2 |
                       ((id[center]&0x20)>>5)<<3;

          if (id[center]&0x10 && id[center+1]&0x40) {
            if (id[center]&0x28 && id[center+1]&0xA0) {
              od[center] = ((id[center]>>8)&0xFFFFFF)<<8 | od[center];
              remove_cross = true;
            } else {
              if (id[center] == 0x10) weight_l += 5;
              if (id[center+1] == 0x40) weight_r += 5;

              int sum_l = 0, sum_r = 0;
              for (int i = start_row; i <= end_row; ++i)
                for (int j = start_col; j <= end_col; ++j)
                  if (i*width+j != center && i*width+j != center+1) {
                    sum_l += (int)isConnected(id[center]>>8, id[i*width+j]>>8);
                    sum_r += (int)isConnected(id[center+1]>>8, id[i*width+j]>>8);
                  }

              weight_r += sum_l > sum_r ? sum_l - sum_r : 0;
              weight_l += sum_l < sum_r ? sum_r - sum_l : 0;

              // curve tracing for left diagonal
              int c_row = row, c_col = col;
              uint curve_l = id[c_row*width+c_col]&0xFF;
              uint edge_l = 16;
              int sml = 1;
              while (bitCount(curve_l)==2 && sml < width*height) {
                edge_l = curve_l - edge_l;
                switch(edge_l){
                  case 1:   c_row--; c_col--; break;
                  case 2:   c_row--;          break;
                  case 4:   c_row--; c_col++; break;
                  case 8:           c_col++; break;
                  case 16:  c_row++; c_col++; break;
                  case 32:  c_row++;          break;
                  case 64:  c_row++; c_col--; break;
                  case 128:          c_col--; break;
                }
                edge_l = (edge_l > 8) ? edge_l>>4 : edge_l<<4;
                curve_l = id[c_row*width+c_col]&0xFF;
                ++sml;
              }
              c_row = row+1; c_col = col+1;
              curve_l = id[c_row*width+c_col]&0xFF; edge_l = 1;
              while (bitCount(curve_l)==2 && sml < width*height) {
                edge_l = curve_l - edge_l;
                switch(edge_l){
                  case 1:   c_row--; c_col--; break;
                  case 16:  c_row++; c_col++; break;
                  case 2:   c_row--;          break;
                  case 4:   c_row--; c_col++; break;
                  case 8:           c_col++; break;
                  case 32:  c_row++;          break;
                  case 64:  c_row++; c_col--; break;
                  case 128:          c_col--; break;
                }
                edge_l = (edge_l > 8) ? edge_l>>4 : edge_l<<4;
                curve_l = id[c_row*width+c_col]&0xFF;
                ++sml;
              }
              // curve tracing for right diagonal
              c_row = row; c_col = col+1;
              uint curve_r = id[c_row*width+c_col]&0xFF;
              uint edge_r = 64;
              int smr = 1;
              while (bitCount(curve_r)==2 && smr < width*height) {
                edge_r = curve_r - edge_r;
                switch(edge_r){
                  case 64:  c_row++; c_col--; break;
                  case 1:   c_row--; c_col--; break;
                  case 2:   c_row--;          break;
                  case 4:   c_row--; c_col++; break;
                  case 8:           c_col++; break;
                  case 32:  c_row++;          break;
                  case 16:  c_row++; c_col++; break;
                  case 128:          c_col--; break;
                }
                edge_r = (edge_r > 8) ? edge_r>>4 : edge_r<<4;
                curve_r = id[c_row*width+c_col]&0xFF;
                ++smr;
              }
              c_row = row+1; c_col = col;
              curve_r = id[c_row*width+c_col]&0xFF; edge_r = 4;
              while (bitCount(curve_r)==2 && smr < width*height) {
                edge_r = curve_r - edge_r;
                switch(edge_r){
                  case 4:   c_row--; c_col++; break;
                  case 16:  c_row++; c_col++; break;
                  case 2:   c_row--;          break;
                  case 1:   c_row--; c_col--; break;
                  case 8:           c_col++; break;
                  case 32:  c_row++;          break;
                  case 64:  c_row++; c_col--; break;
                  case 128:          c_col--; break;
                }
                edge_r = (edge_r > 8) ? edge_r>>4 : edge_r<<4;
                curve_r = id[c_row*width+c_col]&0xFF;
                ++smr;
              }

              weight_l += sml > smr ? sml - smr : 0;
              weight_r += sml < smr ? smr - sml : 0;

              if (weight_l > weight_r) {
                od[center] |= 0x10;
                od[center] = ((id[center]>>8)&0xFFFFFF)<<8 | od[center];
                remove_cross = true;
              } else if (weight_r > weight_l) {
                od[center] |= 0x20;
                od[center] = ((id[center]>>8)&0xFFFFFF)<<8 | od[center];
                remove_cross = true;
              }
            }
          }
          if (!remove_cross)
            od[center] = od[center] | (((id[center]&0x10)>>4)<<4) | (((id[center+1]&0x40)>>6)<<5);
        }
        if (!remove_cross)
          d_out(center) = ((d_tmp(center)>>8)&0xFFFFFF)<<8 | d_out(center);
      });
      Kokkos::fence();

      auto end2 = std::chrono::steady_clock::now();
      total_time += std::chrono::duration<float>(end2 - start).count();

      Kokkos::deep_copy(h_out, d_out);
      float lsum = 0;
      for (int i = 0; i < size; i++) {
        uint v = h_out(i);
        lsum += (v & 0xff)/256.f + ((v>>8)&0xff)/256.f + ((v>>16)&0xff)/256.f + ((v>>24)&0xff)/256.f;
      }
      sum += lsum / size;
    }
  }
  Kokkos::finalize();

  printf("Image size: %d (width) x %d (height)\ncheckSum: %f\n", width, height, sum);
  printf("Average kernel time over %d iterations: %f (s)\n", repeat, total_time / repeat);

  free(img); free(out);
  return 0;
}
