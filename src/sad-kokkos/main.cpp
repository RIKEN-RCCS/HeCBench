#include <iostream>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "bitmap_image.hpp"

#define THRESHOLD 20

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: ./main <image> <template image> <repeat>\n";
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    bitmap_image main_image(argv[1]);
    bitmap_image template_image(argv[2]);
    const int repeat = atoi(argv[3]);

    const int main_width  = main_image.width();
    const int main_height = main_image.height();
    const int main_size   = main_width * main_height;

    const int template_width  = template_image.width();
    const int template_height = template_image.height();
    const int template_size   = template_width * template_height;

    const int height_difference = main_height - template_height;
    const int width_difference  = main_width  - template_width;
    const int sad_array_size    = (height_difference + 1) * (width_difference + 1);

    unsigned char* h_main_image     = new unsigned char[3 * main_size];
    unsigned char* h_template_image = new unsigned char[3 * template_size];

    for (int row = 0; row < main_height; row++) {
      for (int col = 0; col < main_width; col++) {
        rgb_t colors;
        main_image.get_pixel(col, row, colors);
        h_main_image[(row * main_width + col) * 3 + 0] = colors.red;
        h_main_image[(row * main_width + col) * 3 + 1] = colors.green;
        h_main_image[(row * main_width + col) * 3 + 2] = colors.blue;
      }
    }
    for (int row = 0; row < template_height; row++) {
      for (int col = 0; col < template_width; col++) {
        rgb_t colors;
        template_image.get_pixel(col, row, colors);
        h_template_image[(row * template_width + col) * 3 + 0] = colors.red;
        h_template_image[(row * template_width + col) * 3 + 1] = colors.green;
        h_template_image[(row * template_width + col) * 3 + 2] = colors.blue;
      }
    }

    // Device views
    Kokkos::View<unsigned char*> d_main("main_img", 3 * main_size);
    Kokkos::View<unsigned char*> d_tmpl("tmpl_img", 3 * template_size);
    Kokkos::View<int*>           d_sad("sad_array", sad_array_size);

    {
      auto hm = Kokkos::View<unsigned char*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          h_main_image, 3 * main_size);
      auto ht = Kokkos::View<unsigned char*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          h_template_image, 3 * template_size);
      Kokkos::deep_copy(d_main, hm);
      Kokkos::deep_copy(d_tmpl, ht);
    }

    double kernel_time = 0.0;
    int h_min_mse = THRESHOLD;
    int h_num_occurrences = 0;

    auto begin = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      auto kbegin = std::chrono::steady_clock::now();

      // Compute SAD for every valid position
      Kokkos::parallel_for("sad_compute",
        Kokkos::RangePolicy<>(0, main_height * main_width),
        KOKKOS_LAMBDA(int idx) {
          int row = idx / main_width;
          int col = idx % main_width;

          int overlap_width  = main_width  - col < template_width  ? main_width  - col : template_width;
          int overlap_height = main_height - row < template_height ? main_height - row : template_height;

          int sad_result = 0;
          for (int kr = 0; kr < overlap_height; kr++) {
            for (int kc = 0; kc < overlap_width; kc++) {
              const int image_addr  = ((row + kr) * main_width + (col + kc)) * 3;
              const int kernel_addr = (kr * template_width + kc) * 3;
              const int m_r = (int)d_main(image_addr + 0);
              const int m_g = (int)d_main(image_addr + 1);
              const int m_b = (int)d_main(image_addr + 2);
              const int t_r = (int)d_tmpl(kernel_addr + 0);
              const int t_g = (int)d_tmpl(kernel_addr + 1);
              const int t_b = (int)d_tmpl(kernel_addr + 2);
              sad_result += abs(m_r - t_r) + abs(m_g - t_g) + abs(m_b - t_b);
            }
          }
          int norm_sad = (int)(sad_result / (float)template_size);
          int my_index = row * main_width + col;
          if (my_index < sad_array_size)
            d_sad(my_index) = norm_sad;
        });

      // Reduce to find minimum SAD
      int m = THRESHOLD;
      Kokkos::parallel_reduce("sad_min",
        Kokkos::RangePolicy<>(0, sad_array_size),
        KOKKOS_LAMBDA(int i, int& lmin) {
          if (d_sad(i) < lmin) lmin = d_sad(i);
        },
        Kokkos::Min<int>(m));

      // Count occurrences of minimum
      int n = 0;
      Kokkos::parallel_reduce("sad_count",
        Kokkos::RangePolicy<>(0, sad_array_size),
        KOKKOS_LAMBDA(int i, int& cnt) {
          if (d_sad(i) == m) cnt++;
        }, n);

      auto kend = std::chrono::steady_clock::now();
      kernel_time += std::chrono::duration_cast<std::chrono::milliseconds>(kend - kbegin).count();

      h_min_mse = m;
      h_num_occurrences = n;
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << "Parallel Computation Results: " << std::endl;
    std::cout << "Kernel time in msec: " << kernel_time << std::endl;
    std::cout << "Elapsed time in msec = " << elapsed_time << std::endl;
    std::cout << "Main Image Dimensions: " << main_width << "*" << main_height << std::endl;
    std::cout << "Template Image Dimensions: " << template_width << "*" << template_height << std::endl;
    std::cout << "Found Minimum:  " << h_min_mse << std::endl;
    std::cout << "Number of Occurances: " << h_num_occurrences << std::endl;

    delete[] h_main_image;
    delete[] h_template_image;
  }
  Kokkos::finalize();
  return 0;
}
