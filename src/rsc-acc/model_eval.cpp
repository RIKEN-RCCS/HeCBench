/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <omp.h>
#include <math.h>
#include "support/common.h"

void call_RANSAC_kernel_block(int blocks, int threads, float *model_param_local,
    flowvector *flowvectors, int flowvector_count, int max_iter, int error_threshold,
    float convergence_threshold, int *g_out_id, int *model_candidate, int *outliers_candidate)
{
  #pragma acc parallel num_gangs(blocks) vector_length(threads) \
      present(model_param_local, flowvectors, g_out_id, model_candidate, outliers_candidate)
  {
    #pragma acc loop gang
		for (int loop_count = 0; loop_count < max_iter; ++loop_count) {

			const float *model_param = &model_param_local[4 * loop_count];

			if (model_param[0] == -2011) continue;

			int outlier_block_count = 0;

			#pragma acc loop vector reduction(+:outlier_block_count)
			for (int i = 0; i < flowvector_count; ++i) {
				flowvector fvreg = flowvectors[i];

				float vx_error =
						fvreg.x
					+ ((int)((fvreg.x - model_param[0]) * model_param[2])
						 - (int)((fvreg.y - model_param[1]) * model_param[3]))
					- fvreg.vx;

				float vy_error =
						fvreg.y
					+ ((int)((fvreg.y - model_param[1]) * model_param[2])
						 + (int)((fvreg.x - model_param[0]) * model_param[3]))
					- fvreg.vy;

				if ((fabsf(vx_error) >= error_threshold) || (fabsf(vy_error) >= error_threshold)) {
					outlier_block_count += 1;
				}
			}

			if (outlier_block_count < (int)(flowvector_count * convergence_threshold)) {
				int index;

				#pragma acc atomic capture
				{ index = g_out_id[0]++; }

				model_candidate[index]    = loop_count;
				outliers_candidate[index] = outlier_block_count;
			}
    }
  }
}
