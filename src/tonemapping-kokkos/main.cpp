#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
float luminance(float r, float g, float b) {
  return (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <input_file> <iterations>\n";
    return 1;
  }
  const char* inputImageName = argv[1];
  const int   iterations     = atoi(argv[2]);
  const float cPattanaik     = 0.25f;
  const float gammaPattanaik = 0.4f;
  const float deltaPattanaik = 0.000002f;
  const unsigned int numChannels = 4;

  // Read image file
  std::ifstream inputFile(inputImageName, std::ifstream::binary);
  if (!inputFile.is_open()) {
    std::cout << "Cannot open " << inputImageName << "\n";
    return 1;
  }

  unsigned int width, height;
  inputFile >> width >> height;

  std::cout << "Input file name " << inputImageName << "\n";
  std::cout << "Width of the image " << width << "\n";
  std::cout << "Height of the image " << height << "\n";

  const int total = width * height * numChannels;
  float* input  = (float*)malloc(total * sizeof(float));
  float* output = (float*)malloc(total * sizeof(float));

  for (unsigned int y = 0; y < height; y++)
    for (unsigned int x = 0; x < width; x++)
      for (unsigned int c = 0; c < numChannels; c++)
        inputFile >> input[(y * width * numChannels) + (x * numChannels + c)];
  inputFile.close();

  // Compute average luminance on host
  float averageLuminance = 0.0f;
  for (unsigned int y = 0; y < height; y++)
    for (unsigned int x = 0; x < width; x++) {
      float r = input[(y * width * numChannels) + (x * numChannels + 0)];
      float g = input[(y * width * numChannels) + (x * numChannels + 1)];
      float b = input[(y * width * numChannels) + (x * numChannels + 2)];
      averageLuminance += luminance(r, g, b);
    }
  averageLuminance /= (float)(width * height);
  std::cout << "Average luminance value in the image " << averageLuminance << "\n";

  Kokkos::initialize(argc, argv);
  {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using mem_space  = typename exec_space::memory_space;

    Kokkos::View<float*, mem_space> d_input ("input",  total);
    Kokkos::View<float*, mem_space> d_output("output", total);

    {
      auto h_input = Kokkos::create_mirror_view(d_input);
      for (int i = 0; i < total; i++) h_input(i) = input[i];
      Kokkos::deep_copy(d_input, h_input);
    }

    const float avgLum  = averageLuminance;
    const float gamma   = gammaPattanaik;
    const float c       = cPattanaik;
    const float delta   = deltaPattanaik;
    const int   w       = (int)width;
    const int   h       = (int)height;
    const int   nc      = (int)numChannels;

    // Warm up
    for (int i = 0; i < 2 && iterations != 1; i++) {
      Kokkos::parallel_for(
        "tonemapping_warmup",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0,0},{(int)height,(int)width}),
        KOKKOS_LAMBDA(int y, int x) {
          float r1 = d_input[w*nc*y + x*nc+0];
          float g1 = d_input[w*nc*y + x*nc+1];
          float b1 = d_input[w*nc*y + x*nc+2];
          float yLum = luminance(r1,g1,b1);
          float gcP  = c * avgLum;
          float yLP  = yLum;
          if (x>0 && y>0 && x<w-1 && y<h-1) {
            float leftUp    = luminance(d_input[w*nc*(y-1)+(x-1)*nc+0],d_input[w*nc*(y-1)+(x-1)*nc+1],d_input[w*nc*(y-1)+(x-1)*nc+2]);
            float up        = luminance(d_input[w*nc*(y-1)+(x  )*nc+0],d_input[w*nc*(y-1)+(x  )*nc+1],d_input[w*nc*(y-1)+(x  )*nc+2]);
            float rightUp   = luminance(d_input[w*nc*(y-1)+(x+1)*nc+0],d_input[w*nc*(y-1)+(x+1)*nc+1],d_input[w*nc*(y-1)+(x+1)*nc+2]);
            float left      = luminance(d_input[w*nc*(y  )+(x-1)*nc+0],d_input[w*nc*(y  )+(x-1)*nc+1],d_input[w*nc*(y  )+(x-1)*nc+2]);
            float right     = luminance(d_input[w*nc*(y  )+(x+1)*nc+0],d_input[w*nc*(y  )+(x+1)*nc+1],d_input[w*nc*(y  )+(x+1)*nc+2]);
            float leftDown  = luminance(d_input[w*nc*(y+1)+(x-1)*nc+0],d_input[w*nc*(y+1)+(x-1)*nc+1],d_input[w*nc*(y+1)+(x-1)*nc+2]);
            float down      = luminance(d_input[w*nc*(y+1)+(x  )*nc+0],d_input[w*nc*(y+1)+(x  )*nc+1],d_input[w*nc*(y+1)+(x  )*nc+2]);
            float rightDown = luminance(d_input[w*nc*(y+1)+(x+1)*nc+0],d_input[w*nc*(y+1)+(x+1)*nc+1],d_input[w*nc*(y+1)+(x+1)*nc+2]);
            yLP = (leftUp+up+rightUp+left+right+leftDown+down+rightDown)/8.0f;
          }
          float cLP = yLP * Kokkos::log(delta + yLP/yLum) + gcP;
          float yDP = yLum / (yLum + cLP);
          d_output[w*nc*y+x*nc+0] = Kokkos::pow(r1/yLum, gamma) * yDP;
          d_output[w*nc*y+x*nc+1] = Kokkos::pow(g1/yLum, gamma) * yDP;
          d_output[w*nc*y+x*nc+2] = Kokkos::pow(b1/yLum, gamma) * yDP;
          d_output[w*nc*y+x*nc+3] = d_input[w*nc*y+x*nc+3];
        });
      Kokkos::fence();
    }

    std::cout << "Executing kernel for " << iterations << " iterations\n";
    std::cout << "-------------------------------------------\n";

    double total_time = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
      Kokkos::fence();
      auto t0 = std::chrono::steady_clock::now();

      Kokkos::parallel_for(
        "tonemapping",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0,0},{(int)height,(int)width}),
        KOKKOS_LAMBDA(int y, int x) {
          float r1 = d_input[w*nc*y + x*nc+0];
          float g1 = d_input[w*nc*y + x*nc+1];
          float b1 = d_input[w*nc*y + x*nc+2];
          float yLum = luminance(r1,g1,b1);
          float gcP  = c * avgLum;
          float yLP  = yLum;
          if (x>0 && y>0 && x<w-1 && y<h-1) {
            float leftUp    = luminance(d_input[w*nc*(y-1)+(x-1)*nc+0],d_input[w*nc*(y-1)+(x-1)*nc+1],d_input[w*nc*(y-1)+(x-1)*nc+2]);
            float up        = luminance(d_input[w*nc*(y-1)+(x  )*nc+0],d_input[w*nc*(y-1)+(x  )*nc+1],d_input[w*nc*(y-1)+(x  )*nc+2]);
            float rightUp   = luminance(d_input[w*nc*(y-1)+(x+1)*nc+0],d_input[w*nc*(y-1)+(x+1)*nc+1],d_input[w*nc*(y-1)+(x+1)*nc+2]);
            float left      = luminance(d_input[w*nc*(y  )+(x-1)*nc+0],d_input[w*nc*(y  )+(x-1)*nc+1],d_input[w*nc*(y  )+(x-1)*nc+2]);
            float right     = luminance(d_input[w*nc*(y  )+(x+1)*nc+0],d_input[w*nc*(y  )+(x+1)*nc+1],d_input[w*nc*(y  )+(x+1)*nc+2]);
            float leftDown  = luminance(d_input[w*nc*(y+1)+(x-1)*nc+0],d_input[w*nc*(y+1)+(x-1)*nc+1],d_input[w*nc*(y+1)+(x-1)*nc+2]);
            float down      = luminance(d_input[w*nc*(y+1)+(x  )*nc+0],d_input[w*nc*(y+1)+(x  )*nc+1],d_input[w*nc*(y+1)+(x  )*nc+2]);
            float rightDown = luminance(d_input[w*nc*(y+1)+(x+1)*nc+0],d_input[w*nc*(y+1)+(x+1)*nc+1],d_input[w*nc*(y+1)+(x+1)*nc+2]);
            yLP = (leftUp+up+rightUp+left+right+leftDown+down+rightDown)/8.0f;
          }
          float cLP = yLP * Kokkos::log(delta + yLP/yLum) + gcP;
          float yDP = yLum / (yLum + cLP);
          d_output[w*nc*y+x*nc+0] = Kokkos::pow(r1/yLum, gamma) * yDP;
          d_output[w*nc*y+x*nc+1] = Kokkos::pow(g1/yLum, gamma) * yDP;
          d_output[w*nc*y+x*nc+2] = Kokkos::pow(b1/yLum, gamma) * yDP;
          d_output[w*nc*y+x*nc+3] = d_input[w*nc*y+x*nc+3];
        });

      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    }
    printf("Average kernel execution time: %f (us)\n", (total_time * 1e-3f) / iterations);

    // Copy result back for verification
    {
      auto h_out = Kokkos::create_mirror_view(d_output);
      Kokkos::deep_copy(h_out, d_output);
      for (int i = 0; i < total; i++) output[i] = h_out(i);
    }
  }
  Kokkos::finalize();

  // Verify vs reference
  float* refOutput = (float*)malloc(total * sizeof(float));
  float gcP = cPattanaik * averageLuminance;

  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      float r1 = input[y*width*numChannels + x*numChannels+0];
      float g1 = input[y*width*numChannels + x*numChannels+1];
      float b1 = input[y*width*numChannels + x*numChannels+2];
      float yLum = luminance(r1,g1,b1);
      float yLP = yLum;
      if (x>0 && y>0 && x<width-1 && y<height-1) {
        float lU = luminance(input[width*numChannels*(y-1)+(x-1)*numChannels+0],input[width*numChannels*(y-1)+(x-1)*numChannels+1],input[width*numChannels*(y-1)+(x-1)*numChannels+2]);
        float u  = luminance(input[width*numChannels*(y-1)+(x  )*numChannels+0],input[width*numChannels*(y-1)+(x  )*numChannels+1],input[width*numChannels*(y-1)+(x  )*numChannels+2]);
        float rU = luminance(input[width*numChannels*(y-1)+(x+1)*numChannels+0],input[width*numChannels*(y-1)+(x+1)*numChannels+1],input[width*numChannels*(y-1)+(x+1)*numChannels+2]);
        float l  = luminance(input[width*numChannels*(y  )+(x-1)*numChannels+0],input[width*numChannels*(y  )+(x-1)*numChannels+1],input[width*numChannels*(y  )+(x-1)*numChannels+2]);
        float r  = luminance(input[width*numChannels*(y  )+(x+1)*numChannels+0],input[width*numChannels*(y  )+(x+1)*numChannels+1],input[width*numChannels*(y  )+(x+1)*numChannels+2]);
        float lD = luminance(input[width*numChannels*(y+1)+(x-1)*numChannels+0],input[width*numChannels*(y+1)+(x-1)*numChannels+1],input[width*numChannels*(y+1)+(x-1)*numChannels+2]);
        float d  = luminance(input[width*numChannels*(y+1)+(x  )*numChannels+0],input[width*numChannels*(y+1)+(x  )*numChannels+1],input[width*numChannels*(y+1)+(x  )*numChannels+2]);
        float rD = luminance(input[width*numChannels*(y+1)+(x+1)*numChannels+0],input[width*numChannels*(y+1)+(x+1)*numChannels+1],input[width*numChannels*(y+1)+(x+1)*numChannels+2]);
        yLP = (lU+u+rU+l+r+lD+d+rD)/8.0f;
      }
      float cLP = yLP * logf(deltaPattanaik + yLP/yLum) + gcP;
      float yDP = yLum / (yLum + cLP);
      refOutput[y*width*numChannels+x*numChannels+0] = powf(r1/yLum, gammaPattanaik) * yDP;
      refOutput[y*width*numChannels+x*numChannels+1] = powf(g1/yLum, gammaPattanaik) * yDP;
      refOutput[y*width*numChannels+x*numChannels+2] = powf(b1/yLum, gammaPattanaik) * yDP;
      refOutput[y*width*numChannels+x*numChannels+3] = input[y*width*numChannels+x*numChannels+3];
    }
  }

  float error = 0.0f;
  for (int i = 0; i < total; i++) error += refOutput[i] - output[i];
  error /= (float)(height * width);

  if (error > 0.000001f)
    std::cout << "FAIL with normalized error: " << error << "\n";
  else
    std::cout << "PASS\n";

  free(input); free(output); free(refOutput);
  return 0;
}
