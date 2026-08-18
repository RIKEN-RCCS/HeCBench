// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <openacc.h>
#include "backprop.h"
#include "reference.h"

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  setup(argc, argv);
  return 0;
}

int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  float *input_weights_one_dim;
  float *input_weights_one_dim_r;
  float *input_weights_prev_one_dim;
  float * partial_sum;
  float sum;
  unsigned int num_blocks = in / BLOCK_SIZE;

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_one_dim_r = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));


  // this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
  // todo: fix mem allocation
  int m = 0;
  for (int k = 0; k <= in; k++) {  
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_one_dim_r[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }

  printf("Performing device offload\n");

  double offload_start = get_time();

  float* input = net->input_units;
  float* input_weights = input_weights_one_dim;
  float* input_prev_weights = input_weights_prev_one_dim;
  float* hidden_delta = net->hidden_delta;

#include <math.h>

  // 型定義の確認（環境に合わせて float または double）
  typedef float dfloat;
 
#pragma acc data copyin(input[0:in+1],				\
                        hidden_delta[0:hid+1],			\
                        input_prev_weights[0:(in+1)*(hid+1)])	\
  copy(input_weights[0:(in+1)*(hid+1)])				\
  create(partial_sum[0:num_blocks*WIDTH])
  {

#pragma acc parallel loop gang num_gangs(num_blocks) vector_length(BLOCK_SIZE*BLOCK_SIZE)
    for (int by = 0; by < num_blocks; by++) {

      float input_node[HEIGHT];
      float weight_matrix[HEIGHT * WIDTH];
      
#pragma acc loop vector
      for (int v = 0; v < BLOCK_SIZE * BLOCK_SIZE; v++) {
	int tx = v % BLOCK_SIZE;
	int ty = v / BLOCK_SIZE;
	
	int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
	int index_in = HEIGHT * by + ty + 1;
	
	if (tx == 0) {
	  input_node[ty] = input[index_in];
	}
	weight_matrix[ty * WIDTH + tx] = input_weights[index];
      } 
     
#pragma acc loop vector
      for (int v = 0; v < BLOCK_SIZE * BLOCK_SIZE; v++) {
	int tx = v % BLOCK_SIZE;
	int ty = v / BLOCK_SIZE;
	weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];
      } 
     
      for (int i = 1; i <= HEIGHT; i = i * 2) {
	int power_two = i;
#pragma acc loop vector
	for (int v = 0; v < BLOCK_SIZE * BLOCK_SIZE; v++) {
	  int tx = v % BLOCK_SIZE;
	  int ty = v / BLOCK_SIZE;
	 
	  if (ty % (power_two * 2) == 0) {
	    int target_ty = ty + power_two;
	    if (target_ty < HEIGHT) {
	      weight_matrix[ty * WIDTH + tx] += weight_matrix[target_ty * WIDTH + tx];
	    }
	  }
	} 
      }
     
#pragma acc loop vector
      for (int v = 0; v < BLOCK_SIZE * BLOCK_SIZE; v++) {
	int tx = v % BLOCK_SIZE;
	int ty = v / BLOCK_SIZE;
	int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
       
	input_weights[index] = weight_matrix[ty * WIDTH + tx];
       
	if (tx == 0) {
	  partial_sum[by * hid + ty] = weight_matrix[ty];
	}
      }
    }
   
#pragma acc update self(partial_sum[0:num_blocks*WIDTH])
    for (int j = 1; j <= hid; j++) {
      float sum = 0.0;
      for (int k = 0; k < num_blocks; k++) {
	sum += partial_sum[k * hid + j - 1];
      }
     
#ifdef DEBUG
      printf("j=%d sum=%f\n", j, sum);
#endif
     
      sum += net->input_weights[0][j];
      net->hidden_units[j] = (float)(1.0 / (1.0 + exp(-sum)));
    }
   
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
   
#pragma acc update device(input_weights[0:(in+1)*(hid+1)])
   
#pragma acc parallel loop gang num_gangs(num_blocks) vector_length(BLOCK_SIZE*BLOCK_SIZE)
    for (int by = 0; by < num_blocks; by++) {
#pragma acc loop vector
      for (int v = 0; v < BLOCK_SIZE * BLOCK_SIZE; v++) {
	int tx = v % BLOCK_SIZE;
	int ty = v / BLOCK_SIZE;
       
	int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
	int index_y = HEIGHT * by + ty + 1;
	int index_x = tx + 1;
       
	float update_val = ((ETA * hidden_delta[index_x] * input[index_y]) + (MOMENTUM * input_prev_weights[index]));
	input_weights[index] += update_val;
	input_prev_weights[index] = update_val;
       
	if (ty == 0 && by == 0) {
	  float bias_update = ((ETA * hidden_delta[index_x]) + (MOMENTUM * input_prev_weights[index_x]));
	  input_weights[index_x] += bias_update;
	  input_prev_weights[index_x] = bias_update;
	}
      }
    }
  }

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);
  
  reference (in, hid, out, net, 
             input_weights_one_dim_r,
             input_weights_prev_one_dim,
             partial_sum); 

  bool ok = true;
  for (int i = 0; i < (in+1)*(hid+1); i++) {
    if (fabsf(input_weights_one_dim[i] - input_weights_one_dim_r[i]) >= 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++) 
    printf("i=%d input_units=%f\n", i, net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++) 
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_one_dim_r);

  return 0;
}
