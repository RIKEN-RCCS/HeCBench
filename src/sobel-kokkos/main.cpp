#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

struct hec_uchar4 { unsigned char x, y, z, w; };
struct hec_float4 { float x, y, z, w; };

KOKKOS_INLINE_FUNCTION
hec_float4 convert_float4(hec_uchar4 data)
{
  return {(float)data.x, (float)data.y, (float)data.z, (float)data.w};
}

KOKKOS_INLINE_FUNCTION
hec_uchar4 convert_uchar4(hec_float4 v)
{
  auto clamp = [](float c) -> unsigned char {
    return (unsigned char)((c > 255.f) ? 255.f : (c < 0.f ? 0.f : c));
  };
  return {clamp(v.x), clamp(v.y), clamp(v.z), clamp(v.w)};
}

KOKKOS_INLINE_FUNCTION
hec_float4 operator+(hec_float4 a, hec_float4 b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

KOKKOS_INLINE_FUNCTION
hec_float4 operator-(hec_float4 a, hec_float4 b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

KOKKOS_INLINE_FUNCTION
hec_float4 operator*(hec_float4 a, hec_float4 b)
{
  return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

static int read_u16(const std::vector<unsigned char>& b, size_t off)
{
  return (int)b[off] | ((int)b[off + 1] << 8);
}

static int read_i32(const std::vector<unsigned char>& b, size_t off)
{
  return (int)b[off] | ((int)b[off + 1] << 8) |
         ((int)b[off + 2] << 16) | ((int)b[off + 3] << 24);
}

static bool load_bmp(const char* path, std::vector<hec_uchar4>& pixels,
                     int& width, int& height)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < 54 || bytes[0] != 'B' || bytes[1] != 'M') return false;

  const int data_offset = read_i32(bytes, 10);
  width = read_i32(bytes, 18);
  int raw_height = read_i32(bytes, 22);
  height = std::abs(raw_height);
  const int bpp = read_u16(bytes, 28);
  const int compression = read_i32(bytes, 30);
  if (width <= 0 || height <= 0 || compression != 0 || (bpp != 24 && bpp != 32)) return false;

  const int bytes_per_pixel = bpp / 8;
  const int row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;
  pixels.assign((size_t)width * height, {0, 0, 0, 255});

  for (int y = 0; y < height; ++y) {
    const int src_y = (raw_height > 0) ? (height - 1 - y) : y;
    const size_t row = (size_t)data_offset + (size_t)src_y * row_stride;
    for (int x = 0; x < width; ++x) {
      const size_t p = row + (size_t)x * bytes_per_pixel;
      if (p + bytes_per_pixel > bytes.size()) return false;
      pixels[(size_t)y * width + x] = {
          bytes[p + 2],
          bytes[p + 1],
          bytes[p + 0],
          (bpp == 32) ? bytes[p + 3] : (unsigned char)255};
    }
  }
  return true;
}

static void reference(hec_uchar4* output, const hec_uchar4* input,
                      int width, int height)
{
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      int c = x + y * width;
      hec_float4 i00 = convert_float4(input[c - 1 - width]);
      hec_float4 i01 = convert_float4(input[c - width]);
      hec_float4 i02 = convert_float4(input[c + 1 - width]);
      hec_float4 i10 = convert_float4(input[c - 1]);
      hec_float4 i12 = convert_float4(input[c + 1]);
      hec_float4 i20 = convert_float4(input[c - 1 + width]);
      hec_float4 i21 = convert_float4(input[c + width]);
      hec_float4 i22 = convert_float4(input[c + 1 + width]);
      const hec_float4 two = {2.f, 2.f, 2.f, 2.f};
      hec_float4 gx = i00 + two * i10 + i20 - i02 - two * i12 - i22;
      hec_float4 gy = i00 - i20 + two * i01 - two * i21 + i02 - i22;
      output[c] = convert_uchar4({sqrtf(gx.x * gx.x + gy.x * gy.x) / 2.f,
                                  sqrtf(gx.y * gx.y + gy.y * gy.y) / 2.f,
                                  sqrtf(gx.z * gx.z + gy.z * gy.z) / 2.f,
                                  sqrtf(gx.w * gx.w + gy.w * gy.w) / 2.f});
    }
  }
}

static bool compare(const hec_uchar4* refData, const hec_uchar4* data,
                    int length, float epsilon = 1e-6f)
{
  float error = 0.0f;
  float ref = 0.0f;
  for (int i = 1; i < length; ++i) {
    const float r[4] = {(float)refData[i].x, (float)refData[i].y,
                        (float)refData[i].z, (float)refData[i].w};
    const float d[4] = {(float)data[i].x, (float)data[i].y,
                        (float)data[i].z, (float)data[i].w};
    for (int c = 0; c < 4; ++c) {
      const float diff = r[c] - d[c];
      error += diff * diff;
      ref += r[c] * r[c];
    }
  }
  if (fabsf(ref) < 1e-7f) return false;
  return sqrtf(error) / sqrtf(ref) < epsilon;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <path to file> <repeat>\n", argv[0]);
    return 1;
  }

  const char* filePath = argv[1];
  const int iterations = std::atoi(argv[2]);

  int width = 0;
  int height = 0;
  std::vector<hec_uchar4> inputImageData;
  if (!load_bmp(filePath, inputImageData, width, height)) {
    std::printf("Failed to load input image!\n");
    return 1;
  }

  const int imageSize = width * height;
  std::vector<hec_uchar4> outputImageData(imageSize, {0, 0, 0, 0});
  std::vector<hec_uchar4> verificationOutput(imageSize, {0, 0, 0, 0});

  std::printf("Image height = %d and width = %d\n", height, width);
  std::printf("Executing kernel for %d iterations", iterations);
  std::printf("-------------------------------------------\n");

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<hec_uchar4*> d_input("input", imageSize);
    Kokkos::View<hec_uchar4*> d_output("output", imageSize);

    auto h_input = Kokkos::View<hec_uchar4*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        inputImageData.data(), imageSize);
    Kokkos::deep_copy(d_input, h_input);
    Kokkos::deep_copy(d_output, hec_uchar4{0, 0, 0, 0});

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
      Kokkos::parallel_for("sobel",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {height - 1, width - 1}),
        KOKKOS_LAMBDA(const int y, const int x) {
          int c = x + y * width;
          hec_float4 i00 = convert_float4(d_input(c - 1 - width));
          hec_float4 i01 = convert_float4(d_input(c - width));
          hec_float4 i02 = convert_float4(d_input(c + 1 - width));
          hec_float4 i10 = convert_float4(d_input(c - 1));
          hec_float4 i12 = convert_float4(d_input(c + 1));
          hec_float4 i20 = convert_float4(d_input(c - 1 + width));
          hec_float4 i21 = convert_float4(d_input(c + width));
          hec_float4 i22 = convert_float4(d_input(c + 1 + width));
          const hec_float4 two = {2.f, 2.f, 2.f, 2.f};
          hec_float4 gx = i00 + two * i10 + i20 - i02 - two * i12 - i22;
          hec_float4 gy = i00 - i20 + two * i01 - two * i21 + i02 - i22;
          d_output(c) = convert_uchar4({sqrtf(gx.x * gx.x + gy.x * gy.x) / 2.f,
                                        sqrtf(gx.y * gx.y + gy.y * gy.y) / 2.f,
                                        sqrtf(gx.z * gx.z + gy.z * gy.z) / 2.f,
                                        sqrtf(gx.w * gx.w + gy.w * gy.w) / 2.f});
        });
    }
    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);

    auto h_output = Kokkos::View<hec_uchar4*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        outputImageData.data(), imageSize);
    Kokkos::deep_copy(h_output, d_output);
  }
  Kokkos::finalize();

  reference(verificationOutput.data(), inputImageData.data(), width, height);
  std::printf("%s\n", compare(verificationOutput.data(), outputImageData.data(), imageSize) ? "PASS" : "FAIL");

  return 0;
}
