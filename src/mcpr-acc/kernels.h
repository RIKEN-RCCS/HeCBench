void compute_probs(
  const double* __restrict alphas,
  const double* __restrict rands,
        double* __restrict probs,
  int n, int K, int M,
  int threads, int blocks)
{
  #pragma acc parallel loop present(alphas, rands, probs)
  for (int i = 0; i < n; i++) {
    double maxval;    
    int m, k;
    int maxind;
    double M_d = (double) M; 
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){   // initialize probs (though already done on CPU)
      probs[i*K + k] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){   // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        w[k] = alphas[i*K + k] + rands[m*K + k];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        } 
      }
      probs[i*K + maxind] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      probs[i*K + k] /= M_d;
    }
  }
}

void compute_probs_unitStrides(
  const double* __restrict alphas,
  const double* __restrict rands,
        double* __restrict probs,
  int n, int K, int M,
  int threads, int blocks)
{
  #pragma acc parallel loop present(alphas, rands, probs)
  for (int i = 0; i < n; i++) {
    double maxval;    
    int m, k;
    int maxind;
    double M_d = (double) M; 
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){  // initialize probs (though already done on CPU)
      probs[k*n + i] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){    // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        // with +i we now have unit strides in inner loop
        w[k] = alphas[k*n + i] + rands[k*M + m];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        } 
      }
      probs[maxind*n + i] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      // unit strides
      probs[k*n + i] /= M_d;
    }
  }
}

void compute_probs_unitStrides_sharedMem(
  const double* __restrict alphas,
  const double* __restrict rands,
        double* __restrict probs,
  int n, int K, int M,
  int threads, int blocks)
{
  #pragma acc parallel loop present(alphas, rands, probs)
  for (int i = 0; i < n; i++) {

    // set up local memory: half for probs and half for w
    double probs_local[21]; // K
    double w[21]; // K

    double maxval;
    int m, k;
    int maxind;
    double M_d = (double) M;

    // initialize local probs
    for(k = 0; k < K; ++k) {
      probs_local[k] = 0.0;
    }

    // core computation
    for(m = 0; m < M; ++m){     // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){   // generate W ~ N(alpha, 1)
        w[k] = alphas[k*n + i] + rands[k*M + m];
      }
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        }
      }
      probs_local[maxind] += 1.0;
    }

    for(k = 0; k < K; ++k) {
      probs_local[k] /= M_d;
    }

    // copy to device memory so can be returned to CPU
    for(k = 0; k < K; ++k) {
      probs[k*n + i] = probs_local[k];
    }
  }
}
