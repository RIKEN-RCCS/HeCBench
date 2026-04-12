#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

template <class T, std::size_t CHANNELS_PER_ITER>
void resize(
    Kokkos::View<T*> d_output,
    size_t output_size, int out_height, int out_width,
    Kokkos::View<T*> d_input, int in_height, int in_width,
    float o2i_fy, float o2i_fx, bool round_val, bool half_pixel_centers)
{
    int iters_required = (int)(output_size / CHANNELS_PER_ITER);

    Kokkos::parallel_for("resize", iters_required, KOKKOS_LAMBDA(int iter) {
        int in_image_size  = in_height  * in_width;
        int out_image_size = out_height * out_width;

        const int c_start = (iter / out_image_size) * (int)CHANNELS_PER_ITER;
        const int y = (iter % out_image_size) / out_width;
        const int x = iter % out_width;

        float in_yf = half_pixel_centers ? (y + 0.5f) * o2i_fy : y * o2i_fy;
        int in_y    = round_val ? (int)lroundf(in_yf) : (int)in_yf;

        float in_xf = half_pixel_centers ? (x + 0.5f) * o2i_fx : x * o2i_fx;
        int in_x    = round_val ? (int)lroundf(in_xf) : (int)in_xf;

        in_x = (in_x < in_width  - 1) ? in_x : in_width  - 1;
        in_y = (in_y < in_height - 1) ? in_y : in_height - 1;

        int in_idx  = c_start * in_image_size  + in_y * in_width  + in_x;
        int out_idx = c_start * out_image_size + y    * out_width  + x;

        for (int i = 0; i < (int)CHANNELS_PER_ITER; i++) {
            d_output[out_idx] = d_input[in_idx];
            in_idx  += in_image_size;
            out_idx += out_image_size;
        }
    });
    Kokkos::fence();
}

template <class T, std::size_t CHANNELS_PER_ITER>
void resize_bilinear(
    Kokkos::View<T*> d_output,
    size_t output_size, int out_height, int out_width,
    Kokkos::View<T*> d_input, int in_height, int in_width,
    float o2i_fy, float o2i_fx, bool half_pixel_centers)
{
    int iters_required = (int)(output_size / CHANNELS_PER_ITER);

    Kokkos::parallel_for("resize_bilinear", iters_required, KOKKOS_LAMBDA(int iter) {
        int in_image_size  = in_height  * in_width;
        int out_image_size = out_height * out_width;

        const int c_start = (iter / out_image_size) * (int)CHANNELS_PER_ITER;
        const int c_end   = c_start + (int)CHANNELS_PER_ITER;

        const int y = (iter % out_image_size) / out_width;
        const int x = iter % out_width;

        float in_x = half_pixel_centers ? fmaxf((x + 0.5f) * o2i_fx - 0.5f, 0.0f) : x * o2i_fx;
        float in_y = half_pixel_centers ? fmaxf((y + 0.5f) * o2i_fy - 0.5f, 0.0f) : y * o2i_fy;

        int in_x0 = (int)in_x;
        int in_x1 = (in_x0 + 1 < in_width)  ? in_x0 + 1 : in_width  - 1;

        int in_y0 = (int)in_y;
        int in_y1 = (in_y0     < in_height) ? in_y0     : in_height - 1;
        int in_y2 = (in_y0 + 1 < in_height) ? in_y0 + 1 : in_height - 1;

        int in_offset_r0 = c_start * in_image_size + in_y1 * in_width;
        int in_offset_r1 = c_start * in_image_size + in_y2 * in_width;
        int out_idx      = c_start * out_image_size + y * out_width + x;

        for (int c = c_start; c < c_end; c++) {
            T v_00 = d_input[in_offset_r0 + in_x0];
            T v_01 = d_input[in_offset_r0 + in_x1];
            T v_10 = d_input[in_offset_r1 + in_x0];
            T v_11 = d_input[in_offset_r1 + in_x1];

            d_output[out_idx] =
                v_00 +
                T(in_y - in_y0) * T(v_10 - v_00) +
                T(in_x - in_x0) * T(v_01 - v_00) +
                T(in_y - in_y0) * T(in_x - in_x0) * T(v_11 - v_01 - v_10 + v_00);

            in_offset_r0 += in_image_size;
            in_offset_r1 += in_image_size;
            out_idx       += out_image_size;
        }
    });
    Kokkos::fence();
}

template <class T>
void resize_image(
    const int in_width, const int in_height,
    const int out_width, const int out_height,
    const int num_channels, const int repeat,
    const bool bilinear = false)
{
    size_t in_image_size  = (size_t)in_height * in_width;
    size_t in_size        = num_channels * in_image_size;
    size_t in_size_bytes  = sizeof(T) * in_size;

    size_t out_image_size = (size_t)out_height * out_width;
    size_t out_size       = num_channels * out_image_size;
    size_t out_size_bytes = sizeof(T) * out_size;

    const float fx = (float)in_width  / out_width;
    const float fy = (float)in_height / out_height;

    Kokkos::View<T*> d_input ("input",  in_size);
    Kokkos::View<T*> d_output("output", out_size);

    auto h_input = Kokkos::create_mirror_view(d_input);
    for (size_t i = 0; i < in_size; i++)
        h_input(i) = static_cast<T>((i + 1) % 13);
    Kokkos::deep_copy(d_input, h_input);

    auto start = std::chrono::steady_clock::now();

    if (bilinear) {
        for (int i = 0; i < repeat; i++) {
            resize_bilinear<T, 8>(
                d_output, out_size, out_height, out_width,
                d_input,  in_height,  in_width, fy, fx, true);
        }
    } else {
        for (int i = 0; i < repeat; i++) {
            resize<T, 8>(
                d_output, out_size, out_height, out_width,
                d_input,  in_height,  in_width, fy, fx, true, true);
        }
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %lf (us)    Perf: %lf (GB/s)\n",
           time * 1e-3 / repeat,
           (in_size_bytes + out_size_bytes) * (double)repeat / time);
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printf("Usage: %s <input image width> <input image height>\n", argv[0]);
        printf("          <output image width> <output image height>\n");
        printf("          <image channels> <repeat>\n");
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        const int in_width    = atoi(argv[1]);
        const int in_height   = atoi(argv[2]);
        const int out_width   = atoi(argv[3]);
        const int out_height  = atoi(argv[4]);
        const int num_channels = atoi(argv[5]);
        const int repeat      = atoi(argv[6]);

        printf("Resize %d images from (%d x %d) to (%d x %d)\n",
               num_channels, in_width, in_height, out_width, out_height);

        printf("\nThe size of each pixel is 1 byte\n");
        resize_image<unsigned char>(in_width, in_height, out_width, out_height, num_channels, repeat);
        printf("\nBilinear resizing\n");
        resize_image<unsigned char>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

        printf("\nThe size of each pixel is 2 bytes\n");
        resize_image<unsigned short>(in_width, in_height, out_width, out_height, num_channels, repeat);
        printf("\nBilinear resizing\n");
        resize_image<unsigned short>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

        printf("\nThe size of each pixel is 4 bytes\n");
        resize_image<unsigned int>(in_width, in_height, out_width, out_height, num_channels, repeat);
        printf("\nBilinear resizing\n");
        resize_image<unsigned int>(in_width, in_height, out_width, out_height, num_channels, repeat, true);
    }
    Kokkos::finalize();
    return 0;
}
