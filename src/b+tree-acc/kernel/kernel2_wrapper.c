#include <stdio.h>
#include <string.h>
#include "../common.h"                // (in directory provided here)
#include "../util/timer/timer.h"          // (in directory provided here)
#include "./kernel2_wrapper.h"      // (in directory provided here)

//========================================================================================================================================================================================================200
//  KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel2_wrapper(
    knode *knodes,
    long knodes_elem,
    long knodes_mem,  // not length in byte

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    long *lastKnode,
    long *offset_2,
    int *start,
    int *end,
    int *recstart,
    int *reclength)
{

  //======================================================================================================================================================150
  //  CPU VARIABLES
  //======================================================================================================================================================150

  // findRangeK kernel

  int threads = order < 256 ? order : 256;

#pragma acc data copyin(knodes[0: knodes_mem],\
                        start[0: count],\
                        end[0: count],\
                        currKnode[0: count],\
                        offset[0: count],\
                        lastKnode[0: count],\
                        offset_2[0: count])\
                 copy(recstart[0: count])\
                 copyout(reclength[0: count])
  {
    long long kernel_start = get_time();

    #pragma acc parallel loop gang num_gangs(count) vector_length(256) \
                         present(knodes[0: knodes_mem], start[0: count], \
                                 end[0: count], currKnode[0: count], \
                                 offset[0: count], lastKnode[0: count], \
                                 offset_2[0: count], recstart[0: count], \
                                 reclength[0: count])
    for(int bid = 0; bid < count; bid++){
      int i;
      for(i = 0; i < maxheight; i++){

        #pragma acc loop vector
        for(int thid = 0; thid < threads; thid++){
          if((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid+1] > start[bid])){
            // this conditional statement is inserted to avoid crush due to but in original code
            // "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
            if(knodes[currKnode[bid]].indices[thid] < knodes_elem) {
              offset[bid] = knodes[currKnode[bid]].indices[thid];
            }
          }
          if((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid+1] > end[bid])){
            // this conditional statement is inserted to avoid crush due to but in original code
            // "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
            if(knodes[lastKnode[bid]].indices[thid] < knodes_elem) {
              offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
            }
          }
        }

        // set for next tree level
        currKnode[bid] = offset[bid];
        lastKnode[bid] = offset_2[bid];
      }

      // Find the index of the starting record
      #pragma acc loop vector
      for(int thid = 0; thid < threads; thid++){
        if(knodes[currKnode[bid]].keys[thid] == start[bid]){
          recstart[bid] = knodes[currKnode[bid]].indices[thid];
        }
      }

      // Find the index of the ending record
      #pragma acc loop vector
      for(int thid = 0; thid < threads; thid++){
        if(knodes[lastKnode[bid]].keys[thid] == end[bid]){
          reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid]+1;
        }
      }
    }
    long long kernel_end = get_time();
    printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));
  }

#ifdef DEBUG
  for (int i = 0; i < count; i++)
	  printf("recstart[%d] = %d\n", i, recstart[i]);
  for (int i = 0; i < count; i++)
	  printf("reclength[%d] = %d\n", i, reclength[i]);
#endif

}
