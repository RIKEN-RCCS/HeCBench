// Port of zoom CUDA benchmark to Kokkos
// Implements image zoom-in and zoom-out via area averaging

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

static void zoom_in_kokkos(
    Kokkos::View<float*> input, Kokkos::View<float*> output,
    int input_h, int input_w, int output_h, int output_w,
    int pitch, int out_h_start, int out_h_end,
    int out_w_start, int out_w_end, int NC)
{
  Kokkos::parallel_for("zoom_in",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{NC, output_h, output_w}),
    KOKKOS_LAMBDA(const int nc, const int oh, const int ow) {
      if (oh < out_h_start || oh >= out_h_end ||
          ow < out_w_start || ow >= out_w_end) return;

      int start_h = (int)floorf((oh * input_h) / (float)output_h);
      int end_h   = (int)ceilf(((oh + 1) * input_h) / (float)output_h);
      int start_w = (int)floorf((ow * input_w) / (float)output_w);
      int end_w   = (int)ceilf(((ow + 1) * input_w) / (float)output_w);
      int del_h = end_h - start_h;
      int del_w = end_w - start_w;

      float sum = 0.f;
      for (int i = 0; i < del_h; i++)
        for (int j = 0; j < del_w; j++)
          sum += input(nc * pitch + (start_h + i) * input_w + (start_w + j));
      sum /= (float)del_h;
      sum /= (float)del_w;

      output(nc * pitch + (oh - out_h_start) * input_w + (ow - out_w_start)) = sum;
    });
}

static void zoom_out_kokkos(
    Kokkos::View<float*> input, Kokkos::View<float*> output,
    int input_h, int input_w, int output_h, int output_w,
    int pitch, int out_h_start, int out_h_end,
    int out_w_start, int out_w_end, int NC)
{
  Kokkos::parallel_for("zoom_out",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{NC, output_h, output_w}),
    KOKKOS_LAMBDA(const int nc, const int oh, const int ow) {
      int start_h = (int)floorf((oh * input_h) / (float)output_h);
      int end_h   = (int)ceilf(((oh + 1) * input_h) / (float)output_h);
      int start_w = (int)floorf((ow * input_w) / (float)output_w);
      int end_w   = (int)ceilf(((ow + 1) * input_w) / (float)output_w);
      int del_h = end_h - start_h;
      int del_w = end_w - start_w;

      float sum = 0.f;
      for (int i = 0; i < del_h; i++)
        for (int j = 0; j < del_w; j++)
          sum += input(nc * pitch + (start_h + i) * input_w + (start_w + j));
      sum /= (float)del_h;
      sum /= (float)del_w;

      output(nc * pitch + (oh + out_h_start) * input_w + (ow + out_w_start)) = sum;
    });
}

static void zoom_out_edge_pad_kokkos(
    Kokkos::View<float*> output,
    int height, int width, int pitch,
    int h_start, int w_start, int h_end, int w_end, int NC)
{
  Kokkos::parallel_for("zoom_pad",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{NC, height, width}),
    KOKKOS_LAMBDA(const int nc, const int h, const int w) {
      int base = nc * pitch;
      bool in_h = h >= h_start && h < h_end;
      bool in_w = w >= w_start && w < w_end;
      if (in_h && in_w) return; // already filled

      int src_h = h, src_w = w;
      if (h < h_start)  src_h = h_start;
      if (h >= h_end)   src_h = h_end - 1;
      if (w < w_start)  src_w = w_start;
      if (w >= w_end)   src_w = w_end - 1;
      output(base + h * width + w) = output(base + src_h * width + src_w);
    });
}

static void zoom(int repeat, int input_sizes[4], float zoom_factor[2]) {
  int N = input_sizes[0], C = input_sizes[1];
  int H = input_sizes[2], W = input_sizes[3];
  int out_H = (int)floorf(H * zoom_factor[0]);
  int out_W = (int)floorf(W * zoom_factor[1]);
  int NC = N * C;

  bool is_zoom_in  = out_H > H && out_W > W;
  bool is_zoom_out = out_H < H && out_W < W;
  if (!is_zoom_in && !is_zoom_out) {
    printf("Zoom factors only handle simultaneous expansion(or shrinkage). Exit\n");
    return;
  }

  int pitch = H * W;
  size_t img_size = (size_t)NC * H * W;

  Kokkos::View<float*> d_in ("in",  img_size);
  Kokkos::View<float*> d_out("out", img_size);

  {
    auto h = Kokkos::create_mirror_view(d_in);
    std::default_random_engine rng(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t i = 0; i < img_size; i++) h(i) = nd(rng);
    Kokkos::deep_copy(d_in, h);
  }

  int pad_dims[2][2] = {{0,0},{0,0}};
  int slice_dims[2][2] = {{0,0},{0,0}};

  {
    int diff = H - out_H, half = abs(diff) / 2;
    if (diff > 0) { pad_dims[0][0] = half; pad_dims[0][1] = diff - half; }
    else { slice_dims[0][0] = half; slice_dims[0][1] = H + half; }
    diff = W - out_W; half = abs(diff) / 2;
    if (diff > 0) { pad_dims[1][0] = half; pad_dims[1][1] = diff - half; }
    else { slice_dims[1][0] = half; slice_dims[1][1] = W + half; }
  }

  long total_time = 0;
  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    Kokkos::deep_copy(d_out, 0.f);
    Kokkos::fence();

    if (is_zoom_in) {
      zoom_in_kokkos(d_in, d_out, H, W, out_H, out_W, pitch,
                     slice_dims[0][0], slice_dims[0][1],
                     slice_dims[1][0], slice_dims[1][1], NC);
    } else {
      zoom_out_kokkos(d_in, d_out, H, W, out_H, out_W, pitch,
                      pad_dims[0][0], pad_dims[0][1],
                      pad_dims[1][0], pad_dims[1][1], NC);
      Kokkos::fence();
      zoom_out_edge_pad_kokkos(d_out, H, W, pitch,
                               pad_dims[0][0], pad_dims[1][0],
                               pad_dims[0][0] + out_H, pad_dims[1][0] + out_W, NC);
    }
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);
  double checksum = 0.0;
  for (size_t i = 0; i < img_size; i++) checksum += h_out(i);

  printf("Average execution time of the %s kernel: %f (us)\n",
         is_zoom_in ? "zoom-in" : "zoom-out", total_time * 1e-3 / repeat);
  printf("Kernel checksum: %lf\n", checksum);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Usage: %s <batch> <channel> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }
  int input_sizes[4] = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]) };
  int repeat = atoi(argv[5]);

  Kokkos::initialize(argc, argv);
  {
    float zf[2];
    zf[0] = 1.5f; zf[1] = 2.5f;
    zoom(repeat, input_sizes, zf);

    zf[0] = 0.6f; zf[1] = 0.9f;
    zoom(repeat, input_sizes, zf);
  }
  Kokkos::finalize();
  return 0;
}
