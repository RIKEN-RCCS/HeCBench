/*
 * BitCracker: BitLocker password cracking tool, Kokkos version.
 * Copyright (C) 2013-2017  Elena Ago <elena dot ago at gmail dot com>
 *              Massimo Bernaschi <massimo dot bernaschi at gmail dot com>
 * 
 * This file is part of the BitCracker project: https://github.com/e-ago/bitcracker
 * 
 * BitCracker is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * BitCracker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with BitCracker. If not, see <http://www.gnu.org/licenses/>.
 */

#include <chrono>
#include <iostream>
#include "bitcracker.h"

#define ROR(x, i) (((x) << (32 - (i))) | ((x) >> (i)))

#define LOADSCHEDULE_WPRE(j, i)  \
  d_w_words_uint32[j] =                         \
  (uint32_t)block[i * 4 + 0] << 24  \
  | (uint32_t)block[i * 4 + 1] << 16  \
  | (uint32_t)block[i * 4 + 2] <<  8  \
  | (uint32_t)block[i * 4 + 3];

#define CALCSCHEDULE_WPRE(j)  \
  d_w_words_uint32[j] = d_w_words_uint32[j - 16] + d_w_words_uint32[j - 7]  \
  + (ROR(d_w_words_uint32[j - 15],  7) ^ ROR(d_w_words_uint32[j - 15], 18) ^ (d_w_words_uint32[j - 15] >>  3))  \
  + (ROR(d_w_words_uint32[j -  2], 17) ^ ROR(d_w_words_uint32[j -  2], 19) ^ (d_w_words_uint32[j -  2] >> 10));

int evaluate_w_block(
    unsigned char * salt,
    Kokkos::View<uint32_t*> d_w_words_uint32)
{
  if(salt == NULL)
    return BIT_FAILURE;

  // Build padding on host
  unsigned char padding[PADDING_SIZE];
  memset(padding, 0, PADDING_SIZE);
  padding[0] = 0x80;
  uint64_t msgLen = (FIXED_PART_INPUT_CHAIN_HASH << 3);
  for (int i = 0; i < 8; i++) {
    padding[PADDING_SIZE - 1 - i] = (uint8_t)(msgLen >> (i * 8));
  }

  // Copy salt and padding to device views
  Kokkos::View<unsigned char[SALT_SIZE]>    d_salt("d_salt");
  Kokkos::View<unsigned char[PADDING_SIZE]> d_padding("d_padding");

  auto h_salt    = Kokkos::create_mirror_view(d_salt);
  auto h_padding = Kokkos::create_mirror_view(d_padding);

  for (int i = 0; i < SALT_SIZE;    i++) h_salt(i)    = salt[i];
  for (int i = 0; i < PADDING_SIZE; i++) h_padding(i) = padding[i];

  Kokkos::deep_copy(d_salt,    h_salt);
  Kokkos::deep_copy(d_padding, h_padding);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  Kokkos::parallel_for("kernel_w_block",
      Kokkos::RangePolicy<>(0, 1024 * 16),
      KOKKOS_LAMBDA(const int tid_base) {
        for (uint64_t tid = (uint64_t)tid_base;
                      tid < NUM_HASH_BLOCKS;
                      tid += 1024 * 16) {

          uint64_t texBlockId;
          unsigned char block[HASH_BLOCK_NUM_UINT32];

          int i;
          // index 0-15
          for(i = 0; i < SALT_SIZE; i++){
            block[i] = d_salt(i);
          }
          // index 24-63
          for(i = 0; i < PADDING_SIZE; i++){
            block[i + 24] = d_padding(i);
          }

          // index 16-23
          block[16] = (unsigned char) (tid >> (0 * 8));
          block[17] = (unsigned char) (tid >> (1 * 8));
          block[18] = (unsigned char) (tid >> (2 * 8));
          block[19] = (unsigned char) (tid >> (3 * 8));
          block[20] = (unsigned char) (tid >> (4 * 8));
          block[21] = (unsigned char) (tid >> (5 * 8));
          block[22] = (unsigned char) (tid >> (6 * 8));
          block[23] = (unsigned char) (tid >> (7 * 8));

          texBlockId = HASH_BLOCK_NUM_UINT32 * tid;
          LOADSCHEDULE_WPRE(texBlockId +  0,  0)
          LOADSCHEDULE_WPRE(texBlockId +  1,  1)
          LOADSCHEDULE_WPRE(texBlockId +  2,  2)
          LOADSCHEDULE_WPRE(texBlockId +  3,  3)
          LOADSCHEDULE_WPRE(texBlockId +  4,  4)
          LOADSCHEDULE_WPRE(texBlockId +  5,  5)
          LOADSCHEDULE_WPRE(texBlockId +  6,  6)
          LOADSCHEDULE_WPRE(texBlockId +  7,  7)
          LOADSCHEDULE_WPRE(texBlockId +  8,  8)
          LOADSCHEDULE_WPRE(texBlockId +  9,  9)
          LOADSCHEDULE_WPRE(texBlockId + 10, 10)
          LOADSCHEDULE_WPRE(texBlockId + 11, 11)
          LOADSCHEDULE_WPRE(texBlockId + 12, 12)
          LOADSCHEDULE_WPRE(texBlockId + 13, 13)
          LOADSCHEDULE_WPRE(texBlockId + 14, 14)
          LOADSCHEDULE_WPRE(texBlockId + 15, 15)
          CALCSCHEDULE_WPRE(texBlockId + 16)
          CALCSCHEDULE_WPRE(texBlockId + 17)
          CALCSCHEDULE_WPRE(texBlockId + 18)
          CALCSCHEDULE_WPRE(texBlockId + 19)
          CALCSCHEDULE_WPRE(texBlockId + 20)
          CALCSCHEDULE_WPRE(texBlockId + 21)
          CALCSCHEDULE_WPRE(texBlockId + 22)
          CALCSCHEDULE_WPRE(texBlockId + 23)
          CALCSCHEDULE_WPRE(texBlockId + 24)
          CALCSCHEDULE_WPRE(texBlockId + 25)
          CALCSCHEDULE_WPRE(texBlockId + 26)
          CALCSCHEDULE_WPRE(texBlockId + 27)
          CALCSCHEDULE_WPRE(texBlockId + 28)
          CALCSCHEDULE_WPRE(texBlockId + 29)
          CALCSCHEDULE_WPRE(texBlockId + 30)
          CALCSCHEDULE_WPRE(texBlockId + 31)
          CALCSCHEDULE_WPRE(texBlockId + 32)
          CALCSCHEDULE_WPRE(texBlockId + 33)
          CALCSCHEDULE_WPRE(texBlockId + 34)
          CALCSCHEDULE_WPRE(texBlockId + 35)
          CALCSCHEDULE_WPRE(texBlockId + 36)
          CALCSCHEDULE_WPRE(texBlockId + 37)
          CALCSCHEDULE_WPRE(texBlockId + 38)
          CALCSCHEDULE_WPRE(texBlockId + 39)
          CALCSCHEDULE_WPRE(texBlockId + 40)
          CALCSCHEDULE_WPRE(texBlockId + 41)
          CALCSCHEDULE_WPRE(texBlockId + 42)
          CALCSCHEDULE_WPRE(texBlockId + 43)
          CALCSCHEDULE_WPRE(texBlockId + 44)
          CALCSCHEDULE_WPRE(texBlockId + 45)
          CALCSCHEDULE_WPRE(texBlockId + 46)
          CALCSCHEDULE_WPRE(texBlockId + 47)
          CALCSCHEDULE_WPRE(texBlockId + 48)
          CALCSCHEDULE_WPRE(texBlockId + 49)
          CALCSCHEDULE_WPRE(texBlockId + 50)
          CALCSCHEDULE_WPRE(texBlockId + 51)
          CALCSCHEDULE_WPRE(texBlockId + 52)
          CALCSCHEDULE_WPRE(texBlockId + 53)
          CALCSCHEDULE_WPRE(texBlockId + 54)
          CALCSCHEDULE_WPRE(texBlockId + 55)
          CALCSCHEDULE_WPRE(texBlockId + 56)
          CALCSCHEDULE_WPRE(texBlockId + 57)
          CALCSCHEDULE_WPRE(texBlockId + 58)
          CALCSCHEDULE_WPRE(texBlockId + 59)
          CALCSCHEDULE_WPRE(texBlockId + 60)
          CALCSCHEDULE_WPRE(texBlockId + 61)
          CALCSCHEDULE_WPRE(texBlockId + 62)
          CALCSCHEDULE_WPRE(texBlockId + 63)
        }
      });

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "kernel time (w_block) : " << time * 1e-3f << " us\n\n";

  return BIT_SUCCESS;
}
