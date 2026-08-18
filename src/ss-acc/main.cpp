/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  • Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  • Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <climits>
#include "StringSearch.h"

int verify(uint* resultCount, uint workGroupCount,
    uint* result, uint searchLenPerWG,
    std::vector<uint> &cpuResults)
{
  uint count = resultCount[0];
  for(uint i=1; i<workGroupCount; ++i)
  {
    uint found = resultCount[i];
    if(found > 0)
    {
      memcpy((result + count), (result + (i * searchLenPerWG)),
          found * sizeof(uint));
      count += found;
    }
  }
  std::sort(result, result+count);

  std::cout << "Device: found " << count << " times\n";

  // compare the results and see if they match
  bool pass = (count == cpuResults.size());
  pass = pass && std::equal (result, result+count, cpuResults.begin());
  if(pass)
  {
    std::cout << "Passed!\n" << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Failed\n" << std::endl;
    return -1;
  }
}

/**
* @brief Compare two strings with specified length
* @param text       start position on text string
* @param pattern    start position on pattern string
* @param length     Length to compare
* @return 0-failure, 1-success
*/
#pragma acc routine seq
int compare(const uchar* text, const uchar* pattern, uint length)
{
    for(uint l=0; l<length; ++l)
    {
        if (TOLOWER(text[l]) != pattern[l]) return 0;
    }
    return 1;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <path to file> <substring> <repeat>\n", argv[0]);
    return -1;
  }
  std::string file = std::string(argv[1]); // "StringSearch_Input.txt";
  std::string subStr = std::string(argv[2]);
  int iterations = atoi(argv[3]);

  if(iterations < 1)
  {
    std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
    exit(0);
  }

  // Check input text-file specified.
  if(file.length() == 0)
  {
    std::cout << "\n Error: Input File not specified..." << std::endl;
    return -1;
  }

  // Read the content of the file
  std::ifstream textFile(file.c_str(),
      std::ios::in|std::ios::binary|std::ios::ate);
  if(! textFile.is_open())
  {
    std::cout << "\n Unable to open file: " << file << std::endl;
    return -1;
  }

  uint textLength = (uint)(textFile.tellg());
  uchar* text = (uchar*)malloc(textLength+1);
  memset(text, 0, textLength+1);
  textFile.seekg(0, std::ios::beg);
  if (!textFile.read ((char*)text, textLength))
  {
    std::cout << "\n Reading file failed " << std::endl;
    textFile.close();
    return -1;
  }
  textFile.close();

  uint subStrLength = subStr.length();
  if(subStrLength == 0)
  {
    std::cout << "\nError: Sub-String not specified..." << std::endl;
    return -1;
  }

  if (textLength < subStrLength)
  {
    std::cout << "\nText size less than search pattern (" << textLength
      << " < " << subStrLength << ")" << std::endl;
    return -1;
  }

#ifdef ENABLE_2ND_LEVEL_FILTER
  if(subStrLength != 1 && subStrLength <= 16)
  {
    std::cout << "\nSearch pattern size should be longer than 16" << std::endl;
    return -1;
  }
#endif

  std::cout << "Search Pattern : " << subStr << std::endl;

  // Rreference implementation on host device
  std::vector<uint> cpuResults;

  uint last = subStrLength - 1;
  uint badCharSkip[UCHAR_MAX + 1];

  // Initialize the table with default values
  uint scan = 0;
  for(scan = 0; scan <= UCHAR_MAX; ++scan)
  {
    badCharSkip[scan] = subStrLength;
  }

  // populate the table with analysis on pattern
  for(scan = 0; scan < last; ++scan)
  {
    badCharSkip[toupper(subStr[scan])] = last - scan;
    badCharSkip[tolower(subStr[scan])] = last - scan;
  }

  // search the text
  uint curPos = 0;
  while((textLength - curPos) > last)
  {
    int p=last;
    for(scan=(last+curPos); COMPARE(text[scan], subStr[p--]); scan -= 1)
    {
      if (scan == curPos)
      {
        cpuResults.push_back(curPos);
        break;
      }
    }
    curPos += (scan == curPos) ? 1 : badCharSkip[text[last+curPos]];
  }

  std::cout << "CPU: found " << cpuResults.size() << " times\n";

  // Move subStr data host to device
  uchar* pattern = (uchar*)malloc(subStrLength);
  for (uint i = 0; i < subStrLength; i++) {
    pattern[i] = TOLOWER(subStr[i]);
  }

  uint totalSearchPos = textLength - subStrLength + 1;
  uint searchLenPerWG = SEARCH_BYTES_PER_WORKITEM * LOCAL_SIZE;
  uint workGroupCount = (totalSearchPos + searchLenPerWG - 1) / searchLenPerWG;

  uint* resultCount = (uint*) malloc(workGroupCount * sizeof(uint));
  uint* result = (uint*) malloc((textLength - subStrLength + 1) * sizeof(uint));

  // Initialize result arrays
  memset(resultCount, 0, workGroupCount * sizeof(uint));
  memset(result, 0, (textLength - subStrLength + 1) * sizeof(uint));

  const uint patternLength = subStrLength;
  const uint maxSearchLength = searchLenPerWG;

  double time = 0.0;

  #pragma acc data copyin(pattern[0:subStrLength], text[0:textLength]) \
                   copy(resultCount[0:workGroupCount], result[0:textLength - subStrLength + 1])
  {

  /**
  * @brief Naive kernel version of string search.
  *        Find all pattern positions in the given text
  * @param text               Input Text
  * @param textLength         Length of the text
  * @param pattern            Pattern string
  * @param patternLength      Pattern length
  * @param result             Result of all matched positions
  * @param resultCount        Result counts per Work-Group
  * @param maxSearchLength    Maximum search positions for each work-group
  */
  if(subStrLength == 1)
  {
    std::cout <<
      "\nRun only Naive-Kernel version of String Search for pattern size = 1" <<
      std::endl;
    std::cout << "\nExecuting String search naive for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int iter = 0; iter < iterations; iter++)
    {
      // Reset result counts for each iteration
      #pragma acc parallel loop present(resultCount)
      for (uint i = 0; i < workGroupCount; i++) {
        resultCount[i] = 0;
      }

      // Process each work group
      #pragma acc parallel loop gang \
                present(text, pattern, result, resultCount)
      for (uint groupIdx = 0; groupIdx < workGroupCount; groupIdx++)
      {
        // Last search idx for all work items
        uint lastSearchIdx = textLength - patternLength + 1;

        // global idx for all work items in a WorkGroup
        uint beginSearchIdx = groupIdx * maxSearchLength;
        uint endSearchIdx = beginSearchIdx + maxSearchLength;

        if(beginSearchIdx <= lastSearchIdx)
        {
          if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;

          // loop over positions in global buffer
          #pragma acc loop vector
          for(uint stringPos = beginSearchIdx; stringPos < endSearchIdx; stringPos++)
          {
            if (compare(text + stringPos, pattern, patternLength) == 1)
            {
              int count;
              #pragma acc atomic capture
              {
                count = resultCount[groupIdx];
                resultCount[groupIdx] = resultCount[groupIdx] + 1;
              }
              result[beginSearchIdx + count] = stringPos;
            }
          }
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Results are already available via copy clause
    #pragma acc update host(resultCount[0:workGroupCount])
    #pragma acc update host(result[0:textLength - subStrLength + 1])

    verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
  }

  /*
   @param text               Input Text
   @param textLength         Length of the text
   @param pattern            Pattern string
   @param patternLength      Pattern length
   @param result             Result of all matched positions
   @param resultCount        Result counts per Work-Group
   @param maxSearchLength    Maximum search positions for each work-group
   */
  if(subStrLength > 1) {
    std::cout << "\nExecuting String search with load balance for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int iter = 0; iter < iterations; iter++)
    {
      // Reset result counts for each iteration
      #pragma acc parallel loop present(resultCount)
      for (uint i = 0; i < workGroupCount; i++) {
        resultCount[i] = 0;
      }

      // Simplified approach: each gang processes one work group
      // Each vector element checks one position
      #pragma acc parallel loop gang \
                present(text, pattern, result, resultCount)
      for (uint groupIdx = 0; groupIdx < workGroupCount; groupIdx++)
      {
        // Last search idx for all work items
        uint lastSearchIdx = textLength - patternLength + 1;

        // global idx for all work items in a WorkGroup
        uint beginSearchIdx = groupIdx * maxSearchLength;
        uint endSearchIdx = beginSearchIdx + maxSearchLength;

        if(beginSearchIdx <= lastSearchIdx)
        {
          if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;

          uchar first = pattern[0];
          uchar second = pattern[1];

          // Level 1: Quick filter on 2-char match
          #pragma acc loop vector
          for(uint stringPos = beginSearchIdx; stringPos < endSearchIdx; stringPos++)
          {
            // Check first two characters
            if ((first == TOLOWER(text[stringPos])) &&
                (second == TOLOWER(text[stringPos + 1])))
            {
              #ifdef ENABLE_2ND_LEVEL_FILTER
              // Level 2: Check next 8 characters (positions 2-9)
              bool status = true;
              status = status && (pattern[2] == TOLOWER(text[stringPos + 2]));
              status = status && (pattern[3] == TOLOWER(text[stringPos + 3]));
              status = status && (pattern[4] == TOLOWER(text[stringPos + 4]));
              status = status && (pattern[5] == TOLOWER(text[stringPos + 5]));
              status = status && (pattern[6] == TOLOWER(text[stringPos + 6]));
              status = status && (pattern[7] == TOLOWER(text[stringPos + 7]));
              status = status && (pattern[8] == TOLOWER(text[stringPos + 8]));
              status = status && (pattern[9] == TOLOWER(text[stringPos + 9]));

              if (status)
              {
                // Level 3: Check remaining characters
                if (compare(text + stringPos + 10, pattern + 10, patternLength - 10) == 1)
                {
                  int count;
                  #pragma acc atomic capture
                  {
                    count = resultCount[groupIdx];
                    resultCount[groupIdx] = resultCount[groupIdx] + 1;
                  }
                  result[beginSearchIdx + count] = stringPos;
                }
              }
              #else
              // Level 2: Check remaining characters (from position 2)
              if (compare(text + stringPos + 2, pattern + 2, patternLength - 2) == 1)
              {
                int count;
                #pragma acc atomic capture
                {
                  count = resultCount[groupIdx];
                  resultCount[groupIdx] = resultCount[groupIdx] + 1;
                }
                result[beginSearchIdx + count] = stringPos;
              }
              #endif
            }
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Results are already available via copy clause
    #pragma acc update host(resultCount[0:workGroupCount])
    #pragma acc update host(result[0:textLength - subStrLength + 1])

    verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
  }

  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);

  } // end of acc data region

  free(text);
  free(pattern);
  free(result);
  free(resultCount);
  return 0;
}
