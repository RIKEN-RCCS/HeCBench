#if 0
#define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
#define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })

#else
#pragma acc routine seq
inline int ppcg_min(int a, int b) { return a < b ? a : b; }

#pragma acc routine seq
inline int ppcg_max(int a, int b) { return a > b ? a : b; }

#endif

void chemv_kernel0(struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y,
                   float alpha_im, float alpha_re, float beta_im, float beta_re)
{
  #pragma acc parallel num_gangs(12) vector_length(32) \
              present(AT, X, Y)
  {
    float private_Re, private_Im; 

    #pragma acc loop gang
    for (int b0 = 0; b0 < 12; b0++) {
      for (int c1 = 0; c1 <= ppcg_min(368, 32 * b0 + 30); c1 += 32) {
        
        #pragma acc loop vector
        for (int t0 = 0; t0 < 32; t0++) {
          if (32 * b0 + t0 <= 369 && c1 == 0) {
            float p5_Re = ((Y[32 * b0 + t0].Re * beta_re) - (Y[32 * b0 + t0].Im * beta_im));
            float p5_Im = ((Y[32 * b0 + t0].Im * beta_re) + (Y[32 * b0 + t0].Re * beta_im));
            Y[32 * b0 + t0].Re = p5_Re;
            Y[32 * b0 + t0].Im = p5_Im;

            float p2_Re = (alpha_re * AT[11872 * b0 + 371 * t0].Re);
            float p2_Im = (alpha_im * AT[11872 * b0 + 371 * t0].Re);
            float p3_Re = ((p2_Re * X[32 * b0 + t0].Re) - (p2_Im * X[32 * b0 + t0].Im));
            float p3_Im = ((p2_Im * X[32 * b0 + t0].Re) + (p2_Re * X[32 * b0 + t0].Im));
            Y[32 * b0 + t0].Re += p3_Re;
            Y[32 * b0 + t0].Im += p3_Im;
          }
        }

        #pragma acc loop vector
        for (int t0 = 0; t0 < 32; t0++) {
          if (32 * b0 + t0 <= 369) {
            float sum_Re = 0.f;
            float sum_Im = 0.f;
            for (int c3 = 0; c3 <= ppcg_min(31, 32 * b0 + t0 - c1 - 1); c3 += 1) {
              float p97_Re = ((alpha_re * AT[32 * b0 + t0 + 370 * (c1 + c3)].Re) - (alpha_im * AT[32 * b0 + t0 + 370 * (c1 + c3)].Im));
              float p97_Im = ((alpha_im * AT[32 * b0 + t0 + 370 * (c1 + c3)].Re) + (alpha_re * AT[32 * b0 + t0 + 370 * (c1 + c3)].Im));
              sum_Re += ((p97_Re * X[c1 + c3].Re) - (p97_Im * X[c1 + c3].Im));
              sum_Im += ((p97_Im * X[c1 + c3].Re) + (p97_Re * X[c1 + c3].Im));
            }
            Y[32 * b0 + t0].Re += sum_Re;
            Y[32 * b0 + t0].Im += sum_Im;
          }
        }
      }
    }
  }
}

void chemv_kernel1(struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y,
                   float alpha_im, float alpha_re)
{
  #pragma acc parallel num_gangs(12) vector_length(32) \
              present(AT, X, Y)
  {
    #pragma acc loop gang
    for (int b0 = 0; b0 < 12; b0++) {
      for (int c1 = 5888 * b0; c1 <= ppcg_min(67712, 5856 * b0 + 6016); c1 += 32) {
        
        #pragma acc loop vector
        for (int t0 = 0; t0 < 32; t0++) {
          float local_sum_Re = 0.f;
          float local_sum_Im = 0.f;

          for (int c3 = ppcg_max(0, 5888 * b0 + 184 * t0 - c1); 
               c3 <= ppcg_min(31, 5856 * b0 + 183 * t0 - c1 + 368); c3 += 1) {
            
            float at_re = AT[5984 * b0 + 187 * t0 + c1 + c3 + 1].Re;
            float at_im = -AT[5984 * b0 + 187 * t0 + c1 + c3 + 1].Im;

            float p94_Re = (alpha_re * at_re) - (alpha_im * at_im);
            float p94_Im = (alpha_im * at_re) + (alpha_re * at_im);
            
            int x_idx = -5856 * b0 - 183 * t0 + c1 + c3 + 1;
            local_sum_Re += ((p94_Re * X[x_idx].Re) - (p94_Im * X[x_idx].Im));
            local_sum_Im += ((p94_Im * X[x_idx].Re) + (p94_Re * X[x_idx].Im));
          }
          Y[32 * b0 + t0].Re += local_sum_Re;
          Y[32 * b0 + t0].Im += local_sum_Im;
        }
      }
    }
  }
}
