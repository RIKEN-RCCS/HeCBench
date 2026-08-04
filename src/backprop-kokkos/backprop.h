#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#define BIGRND 0x7fffffff
#define THREADS 256
#define WIDTH 16
#define HEIGHT 16
#define BLOCK_SIZE 16

#define ETA 0.3f
#define MOMENTUM 0.3f
#define NUM_THREAD 4

typedef struct {
  int input_n;
  int hidden_n;
  int output_n;

  float *input_units;
  float *hidden_units;
  float *output_units;

  float *hidden_delta;
  float *output_delta;

  float *target;

  float **input_weights;
  float **hidden_weights;

  float **input_prev_weights;
  float **hidden_prev_weights;
} BPNN;

void bpnn_initialize(int seed);
BPNN *bpnn_create(int n_in, int n_hidden, int n_out);
void bpnn_free(BPNN *net);
void bpnn_train(BPNN *net, float *eo, float *eh);
void bpnn_feedforward(BPNN *net);
void bpnn_save(BPNN *net, char *filename);
BPNN *bpnn_read(char *filename);
void load(BPNN *net);
int bpnn_train_kernel(BPNN *net, float *eo, float *eh);
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);
void setup(int argc, char** argv);
float **alloc_2d_dbl(int m, int n);
float squash(float x);

#endif
