#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

// ─── Constants ──────────────────────────────────────────────────────────────
#define BIGRND    0x7fffffff
#define THREADS   256
#define WIDTH     16
#define HEIGHT    16
#define BLOCK_SIZE 16
#define ETA       0.3f
#define MOMENTUM  0.3f

// ─── BPNN data structure ─────────────────────────────────────────────────────
struct BPNN {
  int input_n, hidden_n, output_n;
  float *input_units, *hidden_units, *output_units;
  float *hidden_delta, *output_delta, *target;
  float **input_weights, **hidden_weights;
  float **input_prev_weights, **hidden_prev_weights;
};

// ─── Utility helpers ─────────────────────────────────────────────────────────
static float drnd()  { return (float)rand() / (float)BIGRND; }

static float squash(float x) { return 1.0f / (1.0f + expf(-x)); }

static float *alloc_1d(int n) {
  float *p = (float *)malloc((n) * sizeof(float));
  if (!p) { printf("alloc_1d failed\n"); exit(1); }
  return p;
}

static float **alloc_2d(int m, int n) {
  float **p = (float **)malloc(m * sizeof(float *));
  for (int i = 0; i < m; i++) p[i] = alloc_1d(n);
  return p;
}

static void randomize_weights(float **w, int m, int n) {
  for (int i = 0; i <= m; i++)
    for (int j = 0; j <= n; j++)
      w[i][j] = (float)rand() / RAND_MAX;
}

static void randomize_row(float *w, int m) {
  for (int i = 0; i <= m; i++) w[i] = 0.1f;
}

static void zero_weights(float **w, int m, int n) {
  for (int i = 0; i <= m; i++)
    for (int j = 0; j <= n; j++)
      w[i][j] = 0.0f;
}

static void bpnn_initialize(int seed) {
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}

static BPNN *bpnn_create(int n_in, int n_hidden, int n_out) {
  BPNN *net = (BPNN *)malloc(sizeof(BPNN));
  net->input_n  = n_in;
  net->hidden_n = n_hidden;
  net->output_n = n_out;
  net->input_units  = alloc_1d(n_in + 1);
  net->hidden_units = alloc_1d(n_hidden + 1);
  net->output_units = alloc_1d(n_out + 1);
  net->hidden_delta = alloc_1d(n_hidden + 1);
  net->output_delta = alloc_1d(n_out + 1);
  net->target       = alloc_1d(n_out + 1);
  net->input_weights       = alloc_2d(n_in + 1, n_hidden + 1);
  net->hidden_weights      = alloc_2d(n_hidden + 1, n_out + 1);
  net->input_prev_weights  = alloc_2d(n_in + 1, n_hidden + 1);
  net->hidden_prev_weights = alloc_2d(n_hidden + 1, n_out + 1);
  randomize_weights(net->input_weights, n_in, n_hidden);
  randomize_weights(net->hidden_weights, n_hidden, n_out);
  zero_weights(net->input_prev_weights, n_in, n_hidden);
  zero_weights(net->hidden_prev_weights, n_hidden, n_out);
  randomize_row(net->target, n_out);
  return net;
}

static void bpnn_free(BPNN *net) {
  int n1 = net->input_n, n2 = net->hidden_n;
  free(net->input_units); free(net->hidden_units); free(net->output_units);
  free(net->hidden_delta); free(net->output_delta); free(net->target);
  for (int i = 0; i <= n1; i++) { free(net->input_weights[i]); free(net->input_prev_weights[i]); }
  free(net->input_weights); free(net->input_prev_weights);
  for (int i = 0; i <= n2; i++) { free(net->hidden_weights[i]); free(net->hidden_prev_weights[i]); }
  free(net->hidden_weights); free(net->hidden_prev_weights);
  free(net);
}

// ─── CPU versions of the small-layer operations ──────────────────────────────
static void bpnn_layerforward_cpu(float *l1, float *l2, float **conn, int n1, int n2) {
  l1[0] = 1.0f;
  for (int j = 1; j <= n2; j++) {
    float sum = 0.0f;
    for (int k = 0; k <= n1; k++) sum += conn[k][j] * l1[k];
    l2[j] = squash(sum);
  }
}

static void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err) {
  float errsum = 0.0f;
  for (int j = 1; j <= nj; j++) {
    float o = output[j], t = target[j];
    delta[j] = o * (1.0f - o) * (t - o);
    errsum += fabsf(delta[j]);
  }
  *err = errsum;
}

static void bpnn_hidden_error(float *dh, int nh, float *do_, int no, float **who, float *hidden, float *err) {
  float errsum = 0.0f;
  for (int j = 1; j <= nh; j++) {
    float h = hidden[j], sum = 0.0f;
    for (int k = 1; k <= no; k++) sum += do_[k] * who[j][k];
    dh[j] = h * (1.0f - h) * sum;
    errsum += fabsf(dh[j]);
  }
  *err = errsum;
}

static void bpnn_adjust_weights_cpu(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw) {
  ly[0] = 1.0f;
  for (int j = 1; j <= ndelta; j++)
    for (int k = 0; k <= nly; k++) {
      float dw = ETA * delta[j] * ly[k] + MOMENTUM * oldw[k][j];
      w[k][j] += dw;
      oldw[k][j] = dw;
    }
}

// ─── Load input data ─────────────────────────────────────────────────────────
static int layer_size = 0;

static void load(BPNN *net) {
  float *units = net->input_units;
  int k = 1;
  for (int i = 0; i < layer_size; i++) { units[k] = (float)rand() / RAND_MAX; k++; }
}

// ─── Timing ──────────────────────────────────────────────────────────────────
static double get_time() {
  struct timeval t; gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

// ─── Kokkos-based training kernel ────────────────────────────────────────────
int bpnn_train_kernel(BPNN *net, float *eo, float *eh) {
  int in  = net->input_n;
  int hid = net->hidden_n;
  int out = net->output_n;

  int num_blocks = in / BLOCK_SIZE;   // in is always divisible by 16

  // Flatten 2-D weight arrays to 1-D for device transfer
  int wsize = (in + 1) * (hid + 1);
  std::vector<float> w1d(wsize), pw1d(wsize);
  int m = 0;
  for (int k = 0; k <= in; k++)
    for (int j = 0; j <= hid; j++) {
      w1d[m]  = net->input_weights[k][j];
      pw1d[m] = net->input_prev_weights[k][j];
      m++;
    }

  printf("Performing device offload\n");
  double t0 = get_time();

  // ── Create Kokkos Views ──────────────────────────────────────────────────
  Kokkos::View<float*> d_input("input",     in + 1);
  Kokkos::View<float*> d_w    ("w",         wsize);
  Kokkos::View<float*> d_pw   ("pw",        wsize);
  Kokkos::View<float*> d_psum ("psum",      num_blocks * hid);
  Kokkos::View<float*> d_delta("delta",     hid + 1);

  // Host mirrors for easy transfer
  auto h_input = Kokkos::create_mirror_view(d_input);
  auto h_w     = Kokkos::create_mirror_view(d_w);
  auto h_pw    = Kokkos::create_mirror_view(d_pw);
  auto h_psum  = Kokkos::create_mirror_view(d_psum);
  auto h_delta = Kokkos::create_mirror_view(d_delta);

  // Copy host data to mirrors
  for (int i = 0; i <= in;   i++) h_input(i) = net->input_units[i];
  for (int i = 0; i < wsize; i++) { h_w(i) = w1d[i]; h_pw(i) = pw1d[i]; }
  Kokkos::deep_copy(d_input, h_input);
  Kokkos::deep_copy(d_w,     h_w);
  Kokkos::deep_copy(d_pw,    h_pw);

  // ── Kernel 1: layerforward (input → hidden) ──────────────────────────────
  // For each block (by) and each hidden unit (j), compute partial dot product
  // partial_sum[by*hid + j] = sum_{ty=0}^{15} input[by*16+ty+1] * w[(by*16+ty+1)*(hid+1) + (j+1)]
  int hid_c = hid;
  Kokkos::parallel_for("layerforward",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_blocks, hid_c}),
    KOKKOS_LAMBDA(int by, int j) {
      int jj = j + 1; // 1-indexed hidden unit
      float sum = 0.0f;
      for (int ty = 0; ty < HEIGHT; ty++) {
        int ii = HEIGHT * by + ty + 1; // 1-indexed input unit
        sum += d_input(ii) * d_w(ii * (hid_c + 1) + jj);
      }
      d_psum(by * hid_c + j) = sum;
    });
  Kokkos::fence();

  Kokkos::deep_copy(h_psum, d_psum);

  // Accumulate partial sums and apply sigmoid (on host, small)
  for (int j = 1; j <= hid; j++) {
    float sum = w1d[0 * (hid + 1) + j]; // bias weight
    for (int k = 0; k < num_blocks; k++) sum += h_psum(k * hid + j - 1);
    net->hidden_units[j] = 1.0f / (1.0f + expf(-sum));
  }

  // ── CPU operations for small output layer ────────────────────────────────
  float out_err, hid_err;
  bpnn_layerforward_cpu(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;  *eh = hid_err;

  // Adjust hidden→output weights on CPU (small)
  bpnn_adjust_weights_cpu(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  // ── Kernel 2: adjust_weights (input → hidden) ────────────────────────────
  // Restore original weights (they weren't modified by kernel 1 in this port,
  // but we need delta on device)
  for (int i = 0; i <= hid; i++) h_delta(i) = net->hidden_delta[i];
  Kokkos::deep_copy(d_delta, h_delta);

  Kokkos::parallel_for("adjust_weights",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_blocks, hid_c}),
    KOKKOS_LAMBDA(int by, int tx) {
      int jj = tx + 1; // 1-indexed hidden unit
      // Regular weights: ly[ii] for ii = by*16+1 .. by*16+16
      for (int ty = 0; ty < HEIGHT; ty++) {
        int ii  = HEIGHT * by + ty + 1;
        int idx = ii * (hid_c + 1) + jj;
        float dw = ETA * d_delta(jj) * d_input(ii) + MOMENTUM * d_pw(idx);
        d_w(idx)  += dw;
        d_pw(idx)  = dw;
      }
      // Bias weight: w[0*(hid+1)+jj], only once per column (by==0)
      if (by == 0) {
        int idx = jj; // 0*(hid+1)+jj
        float dw = ETA * d_delta(jj) * 1.0f + MOMENTUM * d_pw(idx);
        d_w(idx)  += dw;
        d_pw(idx)  = dw;
      }
    });
  Kokkos::fence();

  Kokkos::deep_copy(h_w,  d_w);
  Kokkos::deep_copy(h_pw, d_pw);

  // Write updated weights back to 2-D host arrays
  m = 0;
  for (int k = 0; k <= in; k++)
    for (int j = 0; j <= hid; j++) {
      net->input_weights[k][j]      = h_w(m);
      net->input_prev_weights[k][j] = h_pw(m);
      m++;
    }

  double t1 = get_time();
  printf("Device offloading time = %lf(s)\n", t1 - t0);
  return 0;
}

// ─── Backprop face driver ─────────────────────────────────────────────────────
static void backprop_face() {
  BPNN *net = bpnn_create(layer_size, 16, 1);
  float out_err, hid_err;
  printf("Input layer size : %d\n", layer_size);
  load(net);
  printf("Starting training kernel\n");
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("\nFinish the training for one iteration\n");
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <number of input nodes>\n", argv[0]);
    return 1;
  }
  layer_size = atoi(argv[1]);
  if (layer_size % 16 != 0) {
    fprintf(stderr, "The number of input nodes must be divisible by 16\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    bpnn_initialize(7);
    backprop_face();
  }
  Kokkos::finalize();
  return 0;
}
