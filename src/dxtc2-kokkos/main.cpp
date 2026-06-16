#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <chrono>
#include <assert.h>
#include "dds.h"
#include "block.h"
#include "shrUtils.h"
#include "permutations.h"

// float4/uint4 structs (no OMP needed)
struct float4 { float x, y, z, w; };
struct uint4  { unsigned int x, y, z, w; };

#define ERROR_THRESHOLD 0.02f
#define MIN(a,b) ((a)<(b)?(a):(b))

//=============================================================================
// Device helper functions  (sequential per-block implementation)
//=============================================================================

KOKKOS_INLINE_FUNCTION
float4 make_float4(float x, float y, float z, float w) {
  float4 r; r.x=x; r.y=y; r.z=z; r.w=w; return r;
}

KOKKOS_INLINE_FUNCTION
float4 operator+(float4 a, float4 b) { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
KOKKOS_INLINE_FUNCTION
void   operator+=(float4 &a, float4 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; }
KOKKOS_INLINE_FUNCTION
float4 operator-(float4 a, float4 b) { return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
KOKKOS_INLINE_FUNCTION
float4 operator*(float4 a, float4 b) { return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w}; }
KOKKOS_INLINE_FUNCTION
float4 operator*(float4 a, float s)  { return {a.x*s,   a.y*s,   a.z*s,   a.w*s  }; }
KOKKOS_INLINE_FUNCTION
float4 operator*(float s, float4 a)  { return {s*a.x,   s*a.y,   s*a.z,   s*a.w  }; }

KOKKOS_INLINE_FUNCTION
float dev_clamp(float x, float upper, float lower) {
  return fminf(upper, fmaxf(x, lower));
}

KOKKOS_INLINE_FUNCTION
float4 firstEigenVector(float *matrix) {
  float4 v = {1.0f, 1.0f, 1.0f, 0.0f};
  for(int i=0; i<8; i++) {
    float x = v.x*matrix[0] + v.y*matrix[1] + v.z*matrix[2];
    float y = v.x*matrix[1] + v.y*matrix[3] + v.z*matrix[4];
    float z = v.x*matrix[2] + v.y*matrix[4] + v.z*matrix[5];
    float m = fmaxf(fmaxf(x,y),z);
    float iv = 1.0f / m;
    v.x = x*iv; v.y = y*iv; v.z = z*iv;
  }
  return v;
}

KOKKOS_INLINE_FUNCTION
float4 roundAndExpand(float4 v, unsigned short *w) {
  unsigned short x = (unsigned short)rintf(dev_clamp(v.x, 1.0f, 0.0f) * 31.0f);
  unsigned short y = (unsigned short)rintf(dev_clamp(v.y, 1.0f, 0.0f) * 63.0f);
  unsigned short z = (unsigned short)rintf(dev_clamp(v.z, 1.0f, 0.0f) * 31.0f);
  *w = (unsigned short)((x << 11) | (y << 5) | z);
  v.x = x * 0.03227752766457f;
  v.y = y * 0.01583151765563f;
  v.z = z * 0.03227752766457f;
  return v;
}

KOKKOS_INLINE_FUNCTION
float evalPermutation(const float4 *colors, unsigned int permutation,
                      unsigned short *start, unsigned short *end,
                      float4 color_sum,
                      const float *alphaTable, const int *prods, float weight)
{
  float4 alphax_sum = {0,0,0,0};
  int akku = 0;
  for(int i=0; i<16; i++) {
    const unsigned int bits = permutation >> (2*i);
    alphax_sum += alphaTable[bits&3] * colors[i];
    akku       += prods[bits&3];
  }
  float alpha2_sum   = (float)(akku >> 16);
  float beta2_sum    = (float)((akku >> 8) & 0xff);
  float alphabeta_sum= (float)((akku >> 0) & 0xff);
  float4 betax_sum = weight * color_sum - alphax_sum;
  const float factor = 1.0f / (alpha2_sum*beta2_sum - alphabeta_sum*alphabeta_sum);
  float4 a = (alphax_sum*beta2_sum  - betax_sum*alphabeta_sum) * factor;
  float4 b = (betax_sum*alpha2_sum  - alphax_sum*alphabeta_sum) * factor;
  a = roundAndExpand(a, start);
  b = roundAndExpand(b, end);
  float4 e = a*a*alpha2_sum + b*b*beta2_sum
             + 2.0f*(a*b*alphabeta_sum - a*alphax_sum - b*betax_sum);
  return (1.0f/weight) * (e.x + e.y + e.z);
}

// Process one block of 16 pixels sequentially → writes two uint32s into result[2*bid]
KOKKOS_INLINE_FUNCTION
void processDXTBlock(
    int bid,
    const unsigned int *image,
    const unsigned int *permutations,
    const float *alphaTable4, const int *prods4,
    const float *alphaTable3, const int *prods3,
    unsigned int *result)
{
  // Load 16 colors
  float4 colors[16];
  for(int i=0; i<16; i++) {
    unsigned int c = image[bid*16 + i];
    colors[i].x = ((c >>  0) & 0xFF) * (1.0f/255.0f);
    colors[i].y = ((c >>  8) & 0xFF) * (1.0f/255.0f);
    colors[i].z = ((c >> 16) & 0xFF) * (1.0f/255.0f);
    colors[i].w = 0.0f;
  }

  // Compute color sum
  float4 color_sum = {0,0,0,0};
  for(int i=0; i<16; i++) color_sum += colors[i];

  // Compute covariance matrix (6 unique entries of 3x3 symmetric matrix)
  float cov[6] = {0,0,0,0,0,0};
  float4 s = {0.0625f, 0.0625f, 0.0625f, 0.0625f};
  for(int i=0; i<16; i++) {
    float4 d = colors[i] - color_sum * s;
    cov[0] += d.x*d.x;
    cov[1] += d.x*d.y;
    cov[2] += d.x*d.z;
    cov[3] += d.y*d.y;
    cov[4] += d.y*d.z;
    cov[5] += d.z*d.z;
  }

  // First eigenvector for best fit line
  float4 axis = firstEigenVector(cov);

  // Project colors onto axis and sort
  float proj[16];
  for(int i=0; i<16; i++)
    proj[i] = colors[i].x*axis.x + colors[i].y*axis.y + colors[i].z*axis.z;

  // Compute sort ranks
  int xrefs[16];
  for(int i=0; i<16; i++) {
    int rank = 0;
    for(int j=0; j<16; j++) rank += (proj[j] < proj[i]) ? 1 : 0;
    xrefs[i] = rank;
  }
  // Resolve ties (stable sort)
  for(int i=0; i<16; i++)
    for(int j=0; j<i; j++)
      if(xrefs[i] == xrefs[j]) ++xrefs[i];

  // Scatter: sorted_colors[xrefs[i]] = colors[i]
  float4 sorted[16];
  for(int i=0; i<16; i++) sorted[xrefs[i]] = colors[i];

  // Evaluate all 992 permutations with weight 9
  float  bestError = FLT_MAX;
  unsigned short bestStart = 0, bestEnd = 0;
  unsigned int   bestPermutation = 0;
  unsigned int   s_perms[160];

  for(int pidx=0; pidx<992; pidx++) {
    unsigned int perm = permutations[pidx];
    if(pidx < 160) s_perms[pidx] = perm;
    unsigned short st, en;
    float err = evalPermutation(sorted, perm, &st, &en, color_sum,
                                alphaTable4, prods4, 9.0f);
    if(err < bestError) {
      bestError = err;
      bestPermutation = perm;
      bestStart = st;
      bestEnd   = en;
    }
  }
  // Normalise ordering
  if(bestStart < bestEnd) {
    unsigned short tmp = bestEnd; bestEnd = bestStart; bestStart = tmp;
    bestPermutation ^= 0x55555555u;
  }

  // Evaluate 160 permutations with weight 4
  for(int pidx=0; pidx<160; pidx++) {
    unsigned int perm = s_perms[pidx];
    unsigned short st, en;
    float err = evalPermutation(sorted, perm, &st, &en, color_sum,
                                alphaTable3, prods3, 4.0f);
    if(err < bestError) {
      bestError = err;
      bestPermutation = perm;
      bestStart = st;
      bestEnd   = en;
      if(bestStart > bestEnd) {
        unsigned short tmp = bestEnd; bestEnd = bestStart; bestStart = tmp;
        bestPermutation ^= (~bestPermutation >> 1) & 0x55555555u;
      }
    }
  }

  // Reorder permutation using xrefs (inverse scatter)
  if(bestStart == bestEnd) bestPermutation = 0;
  unsigned int indices = 0;
  for(int i=0; i<16; i++) {
    int ref = xrefs[i];
    indices |= ((bestPermutation >> (2*ref)) & 3) << (2*i);
  }

  result[2*bid]   = ((unsigned int)bestEnd << 16) | bestStart;
  result[2*bid+1] = indices;
}

//=============================================================================
// main
//=============================================================================
int main(int argc, char **argv) {
  if(argc != 4) {
    printf("Usage: %s <ppm> <ref.dds> <repeat>\n", argv[0]);
    return 1;
  }
  const char *image_path     = argv[1];
  const char *ref_image_path = argv[2];
  const int   numIterations  = atoi(argv[3]);

  unsigned int width, height;
  unsigned int *h_img = NULL;

  shrLoadPPM4ub(image_path, (unsigned char**)&h_img, &width, &height);
  assert(h_img != NULL);
  printf("Loaded '%s', %d x %d pixels\n\n", image_path, width, height);

  // Convert linear image to block-linear
  const unsigned int memSize = width * height;
  unsigned int *block_image  = (unsigned int*)malloc(memSize * sizeof(unsigned int));
  for(unsigned int by=0; by<height/4; by++)
    for(unsigned int bx=0; bx<width/4; bx++)
      for(int i=0; i<16; i++) {
        int x = i & 3, y = i / 4;
        block_image[(by*(width/4)+bx)*16+i] =
          ((unsigned int*)h_img)[(by*4+y)*4*(width/4)+bx*4+x];
      }

  // Compute permutations on CPU
  unsigned int permutations[1024];
  computePermutations(permutations);

  const float alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
  const float alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
  const int   prods4[4]      = {0x090000, 0x000900, 0x040102, 0x010402};
  const int   prods3[4]      = {0x040000, 0x000400, 0x040101, 0x010401};

  const unsigned int numBlocks = (width/4) * (height/4);
  const unsigned int compressedSize = numBlocks * 8; // 8 bytes per DXT1 block
  unsigned int *h_result = (unsigned int*)malloc(compressedSize);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<unsigned int*> d_image("image",        memSize);
    Kokkos::View<unsigned int*> d_perms("permutations", 1024);
    Kokkos::View<float*>        d_a4("alphaTable4",     4);
    Kokkos::View<float*>        d_a3("alphaTable3",     4);
    Kokkos::View<int*>          d_p4("prods4",          4);
    Kokkos::View<int*>          d_p3("prods3",          4);
    Kokkos::View<unsigned int*> d_result("result",      compressedSize/4);

    // Upload
    {
      auto hi = Kokkos::create_mirror_view(d_image);
      auto hp = Kokkos::create_mirror_view(d_perms);
      auto ha4= Kokkos::create_mirror_view(d_a4);
      auto ha3= Kokkos::create_mirror_view(d_a3);
      auto hp4= Kokkos::create_mirror_view(d_p4);
      auto hp3= Kokkos::create_mirror_view(d_p3);
      for(unsigned int i=0; i<memSize; i++)  hi(i) = block_image[i];
      for(int i=0; i<1024; i++) hp(i) = permutations[i];
      for(int i=0; i<4; i++) { ha4(i)=alphaTable4[i]; ha3(i)=alphaTable3[i]; hp4(i)=prods4[i]; hp3(i)=prods3[i]; }
      Kokkos::deep_copy(d_image, hi);
      Kokkos::deep_copy(d_perms, hp);
      Kokkos::deep_copy(d_a4, ha4); Kokkos::deep_copy(d_a3, ha3);
      Kokkos::deep_copy(d_p4, hp4); Kokkos::deep_copy(d_p3, hp3);
    }

    printf("\nRunning DXT Compression on %u x %u image...\n", width, height);
    printf("%u blocks\n\n", numBlocks);

    auto start = std::chrono::steady_clock::now();

    for(int iter=0; iter<numIterations; iter++) {
      Kokkos::parallel_for("dxt1_compress", (int)numBlocks,
        KOKKOS_LAMBDA(int bid) {
          processDXTBlock(bid,
                          d_image.data(), d_perms.data(),
                          d_a4.data(), d_p4.data(),
                          d_a3.data(), d_p3.data(),
                          d_result.data());
        });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
    printf("Average kernel execution time %f (s)\n", (time_ns * 1e-9f) / numIterations);

    // Download result
    {
      auto hr = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(hr, d_result);
      for(unsigned int i=0; i<compressedSize/4; i++) h_result[i] = hr(i);
    }
  }
  Kokkos::finalize();

  // Write DDS file
  char output_filename[1024];
  strcpy(output_filename, image_path);
  strcpy(output_filename + strlen(image_path) - 3, "dds");
  FILE *fp = fopen(output_filename, "wb");
  assert(fp != NULL);
  DDSHeader header;
  memset(&header, 0, sizeof(header));
  header.fourcc     = FOURCC_DDS;
  header.size       = 124;
  header.flags      = (DDSD_WIDTH|DDSD_HEIGHT|DDSD_CAPS|DDSD_PIXELFORMAT|DDSD_LINEARSIZE);
  header.height     = height;
  header.width      = width;
  header.pitch      = compressedSize;
  header.pf.size    = 32;
  header.pf.flags   = DDPF_FOURCC;
  header.pf.fourcc  = FOURCC_DXT1;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  fwrite(&header,   sizeof(DDSHeader), 1, fp);
  fwrite(h_result,  compressedSize,    1, fp);
  fclose(fp);

  // Compare against reference
  printf("\nComparing against reference...\n");
  fp = fopen(ref_image_path, "rb");
  assert(fp != NULL);
  fseek(fp, sizeof(DDSHeader), SEEK_SET);
  unsigned int refSize = (width/4)*(height/4)*8;
  unsigned int *reference = (unsigned int*)malloc(refSize);
  fread(reference, refSize, 1, fp);
  fclose(fp);

  float rms = 0;
  for(unsigned int y=0; y<height; y+=4)
    for(unsigned int x=0; x<width; x+=4) {
      unsigned int idx = (y/4)*(width/4)+(x/4);
      rms += compareBlock(((BlockDXT1*)h_result)+idx, ((BlockDXT1*)reference)+idx);
    }
  rms /= width * height * 3;
  printf("RMS(reference, result) = %f\n\n", rms);
  if(rms <= ERROR_THRESHOLD) printf("PASS\n"); else printf("FAIL\n");

  free(block_image);
  free(h_result);
  free(h_img);
  free(reference);
  return 0;
}
