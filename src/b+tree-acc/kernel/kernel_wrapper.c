#include <stdio.h>
#include <string.h>
#include "../common.h"                // (in directory provided here)
#include "../util/timer/timer.h"          // (in directory provided here)
#include "./kernel_wrapper.h"      // (in directory provided here)


void 
kernel_wrapper(  record *records,
    long records_mem, // not length in byte
    knode *knodes,
    long knodes_elem,
    long knodes_mem,  // not length in byte

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    int *keys,
    record *ans)
{

  //======================================================================================================================================================150
  //  CPU VARIABLES
  //======================================================================================================================================================150

  // findK kernel

  int threads = order < 256 ? order : 256;

  #pragma acc data copyin(knodes[0: knodes_mem],\
                          records[0: records_mem],\
                          keys[0: count], \
                          currKnode[0: count],\
                          offset[0: count])\
                   copyout(ans[0: count])
  {
    long long kernel_start = get_time();

    #pragma acc parallel loop gang num_gangs(count) vector_length(256) \
                         present(knodes[0: knodes_mem], records[0: records_mem], \
                                 keys[0: count], currKnode[0: count], \
                                 offset[0: count], ans[0: count])
    for(int bid = 0; bid < count; bid++){
      // processtree levels
      for(int i = 0; i < maxheight; i++){

        #pragma acc loop vector
        for(int thid = 0; thid < threads; thid++){
          // if value is between the two keys
          if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
            // this conditional statement is inserted to avoid crush due to but in original code
            // "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
            if(knodes[offset[bid]].indices[thid] < knodes_elem){
              offset[bid] = knodes[offset[bid]].indices[thid];
            }
          }
        }

        // set for next tree level
        currKnode[bid] = offset[bid];
      }

      //At this point, we have a candidate leaf node which may contain
      //the target record.  Check each key to hopefully find the record
      #pragma acc loop vector
      for(int thid = 0; thid < threads; thid++){
        if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
          ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
        }
      }
    }
    long long kernel_end = get_time();
    printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));
  } 

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("ans[%d] = %d\n", i, ans[i].value);
  printf("\n");
#endif

}
