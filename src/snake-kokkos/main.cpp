/*
 * Copyright (c) <2017 - 2020>, ETH Zurich and Bilkent University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or other
 *   materials provided with the distribution.
 * - Neither the names of the ETH Zurich, Bilkent University,
 *   nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without specific
 *   prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <Kokkos_Core.hpp>

using namespace std::chrono;

#define warp_size 32
#define NBytes 8

typedef unsigned int uint;

KOKKOS_INLINE_FUNCTION
uint lsr(uint x, int sa) {
  if (sa > 0 && sa < 32) return (x >> sa);
  return x;
}

KOKKOS_INLINE_FUNCTION
uint lsl(uint x, int sa) {
  if (sa > 0 && sa < 32) return (x << sa);
  return x;
}

KOKKOS_INLINE_FUNCTION
uint set_bit(uint &data, int y) {
  data |= lsl(1u, y);
  return data;
}

KOKKOS_INLINE_FUNCTION
uint popcnt(uint x) {
  x -= ((x >> 1) & 0x55555555);
  x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
  x = (((x >> 4) + x) & 0x0f0f0f0f);
  x += (x >> 8);
  x += (x >> 16);
  return x & 0x0000003f;
}

KOKKOS_INLINE_FUNCTION
int __clz(int x) {
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return 32 - popcnt(x);
}

// CPU reference — helpers above are in scope (plain inline for host)
#include "reference.h"

int main(int argc, const char * const argv[])
{
  if (argc != 5) {
    printf("Usage: ./%s [ReadLength] [ReadandRefFile] [#reads] [repeat]\n", argv[0]);
    exit(-1);
  }

  Kokkos::initialize(argc, const_cast<char**>(argv));
  {
    int ReadLength = atoi(argv[1]);
    int NumReads   = atoi(argv[3]);
    int repeat     = atoi(argv[4]);
    int Size_of_uint_in_Bit = 32;

    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t rd;
    char *p;

    int Number_of_warps_inside_each_block = 8;
    int Concurrent_threads_In_Block = warp_size * Number_of_warps_inside_each_block;
    (void)Concurrent_threads_In_Block; // sizing hint not needed for RangePolicy

    int F_ErrorThreshold = 0;

    uint* ReadSeq      = (uint *) calloc(NumReads * 8, sizeof(uint));
    uint* RefSeq       = (uint *) calloc(NumReads * 8, sizeof(uint));
    int*  DFinal_Results = (int *) calloc(NumReads, sizeof(int));
    int*  HFinal_Results = (int *) calloc(NumReads, sizeof(int));

    int tokenIndex = 1;
    fp = fopen(argv[2], "r");
    if (!fp) {
      printf("The file %s does not exist or you do not have access permission\n", argv[2]);
      Kokkos::finalize();
      return 0;
    }

    for (int this_read = 0; this_read < NumReads; this_read++) {
      rd = getline(&line, &len, fp);
      (void)rd;
      tokenIndex = 1;
      for (p = strtok(line, "\t"); p != NULL; p = strtok(NULL, "\t"))
      {
        if (tokenIndex == 1)
        {
          for (int j = 0; j < ReadLength; j++)
          {
            if (p[j] == 'C')
            {
              ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
            }
            else if (p[j] == 'G')
            {
              ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
            }
            else if (p[j] == 'T')
            {
              ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
              ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
            }
          }
        }
        else if (tokenIndex == 2)
        {
          for (int j = 0; j < ReadLength; j++)
          {
            if (p[j] == 'C')
            {
              RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
            }
            else if (p[j] == 'G')
            {
              RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
            }
            else if (p[j] == 'T')
            {
              RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
              RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] =
                set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)],
                        31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
            }
          }
        }
        tokenIndex = tokenIndex + 1;
      }
    }
    fclose(fp);
    free(line);

    // Allocate device views
    Kokkos::View<uint*> d_ReadSeq("ReadSeq", NumReads * 8);
    Kokkos::View<uint*> d_RefSeq("RefSeq",   NumReads * 8);
    Kokkos::View<int*>  d_Results("Results",  NumReads);

    // Copy host data to device (once, outside the loopPar)
    {
      auto h_ReadSeq = Kokkos::View<uint*, Kokkos::HostSpace,
                                    Kokkos::MemoryUnmanaged>(ReadSeq, NumReads * 8);
      auto h_RefSeq  = Kokkos::View<uint*, Kokkos::HostSpace,
                                    Kokkos::MemoryUnmanaged>(RefSeq,  NumReads * 8);
      Kokkos::deep_copy(d_ReadSeq, h_ReadSeq);
      Kokkos::deep_copy(d_RefSeq,  h_RefSeq);
    }

    // Host mirror for results copy-back
    auto h_Results = Kokkos::View<int*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(DFinal_Results, NumReads);

    bool error = false;
    for (int loopPar = 0; loopPar <= 25; loopPar++) {

      F_ErrorThreshold = (loopPar * ReadLength) / 100;

      auto t1 = high_resolution_clock::now();

      for (int n = 0; n < repeat; n++) {
        Kokkos::parallel_for("sneaky_snake",
          Kokkos::RangePolicy<>(0, NumReads),
          KOKKOS_LAMBDA(int tid) {

            uint ReadsPerThread[NBytes];
            uint RefsPerThread[NBytes];

            #pragma unroll
            for (int i = 0; i < NBytes; i++) {
              ReadsPerThread[i] = d_ReadSeq(tid*8 + i);
              RefsPerThread[i]  = d_RefSeq(tid*8 + i);
            }

            d_Results(tid) = 1;

            uint ReadCompTmp = 0;
            uint RefCompTmp = 0;
            uint DiagonalResult = 0;

            uint ReadTmp1 = 0;
            uint ReadTmp2 = 0;

            uint RefTmp1 = 0;
            uint RefTmp2 = 0;

            uint CornerCase = 0;

            int localCounter    = 0;
            int localCounterMax = 0;
            int globalCounter   = 0;
            int Max_leading_zeros = 0;
            int AccumulatedErrs = 0;

            int Diagonal   = 0;
            int ShiftValue = 0;

            int j = 0;

            while ( (j < 7) && (globalCounter < 200) )
            {
              Diagonal = 0;
              RefTmp1 = lsl(RefsPerThread[j], ShiftValue);
              RefTmp2 = lsr(RefsPerThread[j + 1], 32 - ShiftValue);
              ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue);
              ReadTmp2 = lsr(ReadsPerThread[j + 1], 32 - ShiftValue);

              ReadCompTmp = ReadTmp1 | ReadTmp2;
              RefCompTmp  = RefTmp1  | RefTmp2;
              DiagonalResult  = ReadCompTmp ^ RefCompTmp;
              localCounterMax = __clz(DiagonalResult);

              // Upper diagonals
              for (int e = 1; e <= F_ErrorThreshold; e++)
              {
                Diagonal += 1;
                CornerCase = 0;
                if ( (j == 0) && ( (ShiftValue - (2*e)) < 0 ) )
                {
                  ReadTmp1 = lsr(ReadsPerThread[j], 2*e - ShiftValue);
                  ReadTmp2 = 0;

                  ReadCompTmp = ReadTmp1 | ReadTmp2;
                  RefCompTmp  = RefTmp1  | RefTmp2;

                  DiagonalResult = ReadCompTmp ^ RefCompTmp;

                  CornerCase = 0;
                  for (int Ci = 0; Ci < (2*e) - ShiftValue; Ci++)
                  {
                    set_bit(CornerCase, 31 - Ci);
                  }

                  DiagonalResult = DiagonalResult | CornerCase;
                  localCounter = __clz(DiagonalResult);
                }
                else if ( (ShiftValue - (2*e)) < 0 )
                {
                  ReadTmp1 = lsl(ReadsPerThread[j-1], 32 - (2*e - ShiftValue));
                  ReadTmp2 = lsr(ReadsPerThread[j],   2*e - ShiftValue);

                  ReadCompTmp = ReadTmp1 | ReadTmp2;
                  RefCompTmp  = RefTmp1  | RefTmp2;

                  DiagonalResult = ReadCompTmp ^ RefCompTmp;

                  localCounter = __clz(DiagonalResult);
                }
                else
                {
                  ReadTmp1 = lsl(ReadsPerThread[j],   ShiftValue - 2*e);
                  ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue - 2*e));

                  ReadCompTmp = ReadTmp1 | ReadTmp2;
                  RefCompTmp  = RefTmp1  | RefTmp2;

                  DiagonalResult = ReadCompTmp ^ RefCompTmp;

                  localCounter = __clz(DiagonalResult);
                }
                if (localCounter > localCounterMax)
                  localCounterMax = localCounter;
              }

              // Lower diagonals
              for (int e = 1; e <= F_ErrorThreshold; e++)
              {
                Diagonal += 1;
                CornerCase = 0;
                if (j < 5)
                {
                  if ((ShiftValue + 2*e) < 32)
                  {
                    ReadTmp1 = lsl(ReadsPerThread[j],   ShiftValue + 2*e);
                    ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue + 2*e));

                    ReadCompTmp = ReadTmp1 | ReadTmp2;
                    RefCompTmp  = RefTmp1  | RefTmp2;

                    DiagonalResult = ReadCompTmp ^ RefCompTmp;
                    localCounter = __clz(DiagonalResult);
                  }
                  else
                  {
                    ReadTmp1 = lsl(ReadsPerThread[j+1], (ShiftValue + 2*e) % 32);
                    ReadTmp2 = lsr(ReadsPerThread[j+2], 32 - (ShiftValue + 2*e) % 32);

                    ReadCompTmp = ReadTmp1 | ReadTmp2;
                    RefCompTmp  = RefTmp1  | RefTmp2;

                    DiagonalResult = ReadCompTmp ^ RefCompTmp;

                    localCounter = __clz(DiagonalResult);
                  }
                }
                else
                {
                  ReadTmp1 = lsl(ReadsPerThread[j],   ShiftValue + 2*e);
                  ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue + 2*e));

                  ReadCompTmp = ReadTmp1 | ReadTmp2;
                  RefCompTmp  = RefTmp1  | RefTmp2;
                  DiagonalResult = ReadCompTmp ^ RefCompTmp;

                  CornerCase = 0;
                  if ((globalCounter+32) > 200) {
                    for (int Ci = globalCounter+32-200;
                             Ci < globalCounter+32-200+2*e; Ci++)
                    {
                      set_bit(CornerCase, Ci);
                    }
                  }
                  else if ((globalCounter+32) >= (200 - (2*e))) {
                    for (int Ci = 0; Ci < (2*e); Ci++)
                    {
                      set_bit(CornerCase, Ci);
                    }
                  }
                  DiagonalResult = DiagonalResult | CornerCase;

                  localCounter = __clz(DiagonalResult);
                }

                if (localCounter > localCounterMax)
                  localCounterMax = localCounter;
              }

              Max_leading_zeros = 0;

              if ( (j == 6) && ( ((localCounterMax/2)*2) >= 8) )
              {
                Max_leading_zeros = 8;
                break;
              }
              else if ((localCounterMax/2*2) > Max_leading_zeros)
              {
                Max_leading_zeros = ((localCounterMax/2)*2);
              }

              if (((Max_leading_zeros/2) < 16) && (j < 5))
              {
                AccumulatedErrs += 1;
              }
              else if ((j == 6) && ((Max_leading_zeros/2) < 4))
              {
                AccumulatedErrs += 1;
              }

              if (AccumulatedErrs > F_ErrorThreshold)
              {
                d_Results(tid) = 0;
                break;
              }

              if (ShiftValue + Max_leading_zeros + 2 >= 32)
              {
                j += 1;
              }

              if (Max_leading_zeros == 32)
              {
                globalCounter += Max_leading_zeros;
              }
              else
              {
                ShiftValue    = ((ShiftValue + Max_leading_zeros + 2) % 32);
                globalCounter += (Max_leading_zeros + 2);
              }
            }
          }); // end parallel_for
      } // end repeat loop
      Kokkos::fence();

      auto t2 = high_resolution_clock::now();
      double elapsed_time = duration_cast<microseconds>(t2 - t1).count();

      // Copy results from device to host
      Kokkos::deep_copy(h_Results, d_Results);

      // Verify against CPU reference
      sneaky_snake_ref(ReadSeq, RefSeq, HFinal_Results, NumReads, F_ErrorThreshold);
      error = memcmp(DFinal_Results, HFinal_Results, NumReads * sizeof(int));
      if (error) break;

      int D_accepted = 0;
      for (int i = 0; i < NumReads; i++) if (DFinal_Results[i] == 1) D_accepted++;

      printf("Error threshold: %2d | Average kernel time (us): %5.4f | Accepted: %10d | Rejected: %10d\n",
             F_ErrorThreshold, elapsed_time / repeat, D_accepted, NumReads - D_accepted);
    }
    printf("%s\n", error ? "FAIL" : "PASS");

    free(ReadSeq);
    free(RefSeq);
    free(DFinal_Results);
    free(HFinal_Results);
  }
  Kokkos::finalize();
  return 0;
}
