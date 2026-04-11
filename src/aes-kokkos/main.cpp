#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <Kokkos_Core.hpp>

#include "SDKBitMap.h"
#include "aes.h"
#include "kernels.h"

// Forward declarations (defined in reference.cpp and utils.cpp)
extern void reference(uchar * output, uchar * input, uchar * rKey,
               unsigned int explandedKeySize, unsigned int width,
               unsigned int height, bool inverse, unsigned int rounds,
               unsigned int keySize);

extern void convertColorToGray(const uchar4 *pixels, uchar *gray,
                               const int height, const int width);
extern void convertGrayToGray(const uchar4 *pixels, uchar *gray,
                              const int height, const int width);
extern void createRoundKey(uchar * eKey, uchar * rKey);
extern void keyExpansion(uchar * key, uchar * expandedKey,
                         unsigned int keySize, unsigned int explandedKeySize);

template<typename T>
extern int fillRandom(T * arrayPtr, const int width, const int height,
                      const T rangeMin, const T rangeMax, unsigned int seed);

int main(int argc, char * argv[])
{
  if (argc != 4) {
    printf("Usage: %s <iterations> <0 or 1> <path to bitmap image file>\n", argv[0]);
    printf("0=encrypt, 1=decrypt\n");
    return 1;
  }

  const unsigned int keySizeBits = 128;
  const unsigned int rounds = 10;
  const unsigned int seed = 123;

  const int iterations = atoi(argv[1]);
  const bool decrypt = atoi(argv[2]);
  const char* filePath = argv[3];

  SDKBitMap image;
  image.load(filePath);
  const int width  = image.getWidth();
  const int height = image.getHeight();

  /* check condition for the bitmap to be initialized */
  if (width <= 0 || height <= 0) return 1;

  std::cout << "Image width and height: " 
            << width << " " << height << std::endl;

  uchar4 *pixels = image.getPixels();

  unsigned int sizeBytes = width*height*sizeof(uchar);
  uchar *input = (uchar*)malloc(sizeBytes); 

  /* initialize the input array, do NOTHING but assignment when decrypt*/
  if (decrypt)
    convertGrayToGray(pixels, input, height, width);
  else
    convertColorToGray(pixels, input, height, width);

  unsigned int keySize = keySizeBits/8; // 1 Byte = 8 bits

  unsigned int keySizeBytes = keySize*sizeof(uchar);

  uchar *key = (uchar*)malloc(keySizeBytes);

  fillRandom<uchar>(key, keySize, 1, 0, 255, seed); 

  // expand the key
  unsigned int explandedKeySize = (rounds+1)*keySize;

  unsigned int explandedKeySizeBytes = explandedKeySize*sizeof(uchar);

  uchar *expandedKey = (uchar*)malloc(explandedKeySizeBytes);
  uchar *roundKey    = (uchar*)malloc(explandedKeySizeBytes);

  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for(unsigned int i = 0; i < rounds+1; ++i)
  {
    createRoundKey(expandedKey + keySize*i, roundKey + keySize*i);
  }

  // save device result
  uchar* output = (uchar*)malloc(sizeBytes);

  Kokkos::initialize(argc, argv);
  {
    // Create device views
    Kokkos::View<uchar*> d_input("input", sizeBytes);
    Kokkos::View<uchar*> d_output("output", sizeBytes);
    Kokkos::View<uchar*> d_roundKey("roundKey", explandedKeySizeBytes);
    Kokkos::View<uchar*> d_sbox("sbox", 256);
    Kokkos::View<uchar*> d_rsbox("rsbox", 256);

    // Copy data to device
    auto h_input = Kokkos::View<uchar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(input, sizeBytes);
    auto h_roundKey = Kokkos::View<uchar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(roundKey, explandedKeySizeBytes);
    auto h_sbox = Kokkos::View<uchar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(sbox, 256);
    auto h_rsbox = Kokkos::View<uchar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(rsbox, 256);

    Kokkos::deep_copy(d_input, h_input);
    Kokkos::deep_copy(d_roundKey, h_roundKey);
    Kokkos::deep_copy(d_sbox, h_sbox);
    Kokkos::deep_copy(d_rsbox, h_rsbox);

    std::cout << "Executing kernel for " << iterations 
              << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // The CUDA kernel uses a 2D grid (width/4, height/4) with block (1, 4)
    // Each block processes a 4x4 block of the image using 4 threads (one per row)
    // In Kokkos, we use a TeamPolicy to replicate this pattern
    const int numBlocks = (width/4) * (height/4);
    const int teamSize = 4;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchViewUchar4 = Kokkos::View<uchar4*, scratch_space, Kokkos::MemoryUnmanaged>;

    size_t scratch_bytes = ScratchViewUchar4::shmem_size(4) * 2; // block0[4] and block1[4]

    auto policy = team_policy(numBlocks, teamSize)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++)
    {
      if (decrypt) {
        Kokkos::parallel_for("AESDecrypt", policy,
          KOKKOS_LAMBDA(const member_type &team) {
            ScratchViewUchar4 block0(team.team_scratch(0), 4);
            ScratchViewUchar4 block1(team.team_scratch(0), 4);

            int leagueRank = team.league_rank();
            unsigned int bx = leagueRank % (width/4);
            unsigned int by = leagueRank / (width/4);
            unsigned int localIndex = team.team_rank();

            unsigned int globalIndex = (((by * width/4) + bx) * 4) + localIndex;

            uchar4 galiosCoeff[4];
            galiosCoeff[0] = uchar4(14, 0, 0, 0);
            galiosCoeff[1] = uchar4(11, 0, 0, 0);
            galiosCoeff[2] = uchar4(13, 0, 0, 0);
            galiosCoeff[3] = uchar4( 9, 0, 0, 0);

            // Load input as uchar4
            block0(localIndex) = uchar4(
              d_input(globalIndex*4),
              d_input(globalIndex*4+1),
              d_input(globalIndex*4+2),
              d_input(globalIndex*4+3));

            // Load round key as uchar4
            uchar4 rk = uchar4(
              d_roundKey((4*rounds + localIndex)*4),
              d_roundKey((4*rounds + localIndex)*4+1),
              d_roundKey((4*rounds + localIndex)*4+2),
              d_roundKey((4*rounds + localIndex)*4+3));

            block0(localIndex) ^= rk;

            for(unsigned int r=rounds-1; r > 0; --r)
            {
              block0(localIndex) = shiftRowsInvDevice(block0(localIndex), localIndex);
              block0(localIndex) = sboxRead(d_rsbox.data(), block0(localIndex));

              team.team_barrier();
              uchar4 rkr = uchar4(
                d_roundKey((r*4 + localIndex)*4),
                d_roundKey((r*4 + localIndex)*4+1),
                d_roundKey((r*4 + localIndex)*4+2),
                d_roundKey((r*4 + localIndex)*4+3));
              block1(localIndex) = block0(localIndex) ^ rkr;

              team.team_barrier();
              block0(localIndex) = mixColumnsDevice(block1.data(), galiosCoeff, localIndex);
            }

            block0(localIndex) = shiftRowsInvDevice(block0(localIndex), localIndex);
            block0(localIndex) = sboxRead(d_rsbox.data(), block0(localIndex));

            uchar4 rk0 = uchar4(
              d_roundKey(localIndex*4),
              d_roundKey(localIndex*4+1),
              d_roundKey(localIndex*4+2),
              d_roundKey(localIndex*4+3));
            uchar4 result = block0(localIndex) ^ rk0;

            d_output(globalIndex*4)   = result.x;
            d_output(globalIndex*4+1) = result.y;
            d_output(globalIndex*4+2) = result.z;
            d_output(globalIndex*4+3) = result.w;
          }
        );
      }
      else {
        Kokkos::parallel_for("AESEncrypt", policy,
          KOKKOS_LAMBDA(const member_type &team) {
            ScratchViewUchar4 block0(team.team_scratch(0), 4);
            ScratchViewUchar4 block1(team.team_scratch(0), 4);

            int leagueRank = team.league_rank();
            unsigned int bx = leagueRank % (width/4);
            unsigned int by = leagueRank / (width/4);
            unsigned int localIndex = team.team_rank();

            unsigned int globalIndex = (((by * width/4) + bx) * 4) + localIndex;

            uchar4 galiosCoeff[4];
            galiosCoeff[0] = uchar4(2, 0, 0, 0);
            galiosCoeff[1] = uchar4(3, 0, 0, 0);
            galiosCoeff[2] = uchar4(1, 0, 0, 0);
            galiosCoeff[3] = uchar4(1, 0, 0, 0);

            // Load input as uchar4
            block0(localIndex) = uchar4(
              d_input(globalIndex*4),
              d_input(globalIndex*4+1),
              d_input(globalIndex*4+2),
              d_input(globalIndex*4+3));

            // Load round key as uchar4
            uchar4 rk = uchar4(
              d_roundKey(localIndex*4),
              d_roundKey(localIndex*4+1),
              d_roundKey(localIndex*4+2),
              d_roundKey(localIndex*4+3));

            block0(localIndex) ^= rk;

            for(unsigned int r=1; r < rounds; ++r)
            {
              block0(localIndex) = sboxRead(d_sbox.data(), block0(localIndex));
              block0(localIndex) = shiftRowsDevice(block0(localIndex), localIndex);

              team.team_barrier();
              block1(localIndex) = mixColumnsDevice(block0.data(), galiosCoeff, localIndex);

              team.team_barrier();
              uchar4 rkr = uchar4(
                d_roundKey((r*4 + localIndex)*4),
                d_roundKey((r*4 + localIndex)*4+1),
                d_roundKey((r*4 + localIndex)*4+2),
                d_roundKey((r*4 + localIndex)*4+3));
              block0(localIndex) = block1(localIndex) ^ rkr;
            }

            block0(localIndex) = sboxRead(d_sbox.data(), block0(localIndex));
            block0(localIndex) = shiftRowsDevice(block0(localIndex), localIndex);

            uchar4 rkFinal = uchar4(
              d_roundKey((rounds*4 + localIndex)*4),
              d_roundKey((rounds*4 + localIndex)*4+1),
              d_roundKey((rounds*4 + localIndex)*4+2),
              d_roundKey((rounds*4 + localIndex)*4+3));
            uchar4 result = block0(localIndex) ^ rkFinal;

            d_output(globalIndex*4)   = result.x;
            d_output(globalIndex*4+1) = result.y;
            d_output(globalIndex*4+2) = result.z;
            d_output(globalIndex*4+3) = result.w;
          }
        );
      }
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

    // Copy result back
    auto h_output = Kokkos::View<uchar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(output, sizeBytes);
    Kokkos::deep_copy(h_output, d_output);
  }
  Kokkos::finalize();

  // Verify
  uchar *verificationOutput = (uchar *) malloc(sizeBytes);

  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);

  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, sizeBytes) == 0)
    std::cout<<"Pass\n";
  else
    std::cout<<"Fail\n";

  /* release program resources (input memory etc.) */
  if(input) free(input);
  if(key) free(key);
  if(expandedKey) free(expandedKey);
  if(roundKey) free(roundKey);
  if(output) free(output);
  if(verificationOutput) free(verificationOutput);

  return 0;
}
