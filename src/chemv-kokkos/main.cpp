#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define REPEAT 1000
#define N 370
#define LDAT N
#define INCX 1
#define INCY 1
#define AT_SIZE (N * LDAT)
#define X_SIZE  (N * INCX)
#define Y_SIZE  (N * INCY)

struct ComplexFloat { float Re, Im; };

void chemv_cpu(float alpha_re, float alpha_im, float beta_re, float beta_im,
               ComplexFloat *AT, ComplexFloat *X, ComplexFloat *Y)
{
  for (int i0 = 0; i0 < N; i0++) {
    float r = Y[i0].Re*beta_re - Y[i0].Im*beta_im;
    float im= Y[i0].Im*beta_re + Y[i0].Re*beta_im;
    Y[i0].Re = r; Y[i0].Im = im;
  }
  for (int i1 = 0; i1 < N; i1++) {
    float v2r = alpha_re*AT[i1*LDAT+i1].Re;
    float v2i = alpha_im*AT[i1*LDAT+i1].Re;
    float v3r = v2r*X[i1].Re - v2i*X[i1].Im;
    float v3i = v2i*X[i1].Re + v2r*X[i1].Im;
    Y[i1].Re += v3r; Y[i1].Im += v3i;
  }
  for (int i2 = 0; i2 < N-1; i2++) {
    for (int i3 = 0; i3 < N-1-i2; i3++) {
      int row = i2, col = i2+1+i3;
      float v94r = alpha_re*AT[i2*LDAT+col].Re - alpha_im*(-AT[i2*LDAT+col].Im);
      float v94i = alpha_im*AT[i2*LDAT+col].Re + alpha_re*(-AT[i2*LDAT+col].Im);
      float v95r = v94r*X[col].Re - v94i*X[col].Im;
      float v95i = v94i*X[col].Re + v94r*X[col].Im;
      Y[i2].Re += v95r; Y[i2].Im += v95i;
      float v97r = alpha_re*AT[i2*LDAT+col].Re - alpha_im*AT[i2*LDAT+col].Im;
      float v97i = alpha_im*AT[i2*LDAT+col].Re + alpha_re*AT[i2*LDAT+col].Im;
      float v98r = v97r*X[i2].Re - v97i*X[i2].Im;
      float v98i = v97i*X[i2].Re + v97r*X[i2].Im;
      Y[col].Re += v98r; Y[col].Im += v98i;
    }
  }
}

int main() {
  ComplexFloat AT[AT_SIZE], X[X_SIZE], Y_cpu[Y_SIZE], Y_gpu[Y_SIZE];

  for (int i = 0; i < N; i++) {
    X[i]     = {(float)(i+5), (float)(i*2)};
    Y_cpu[i] = {(float)(i*3), (float)(i+7)};
    Y_gpu[i] = {(float)(i*3), (float)(i+7)};
    for (int j = 0; j < LDAT; j++)
      AT[i*LDAT+j] = {(float)(i+j), (float)(i+3)};
  }
  const float alpha_re=3.14f, alpha_im=1.59f, beta_re=2.71f, beta_im=8.28f;

  chemv_cpu(alpha_re, alpha_im, beta_re, beta_im, AT, X, Y_cpu);

  Kokkos::initialize();
  {
    using ViewCF = Kokkos::View<ComplexFloat*>;
    ViewCF d_AT("AT", AT_SIZE), d_X("X", X_SIZE), d_Y("Y", Y_SIZE);

    auto h_AT = Kokkos::create_mirror_view(d_AT);
    auto h_X  = Kokkos::create_mirror_view(d_X);
    auto h_Y  = Kokkos::create_mirror_view(d_Y);
    for (int i=0;i<AT_SIZE;i++) h_AT(i)=AT[i];
    for (int i=0;i<X_SIZE;i++)  h_X(i)=X[i];
    for (int i=0;i<Y_SIZE;i++)  h_Y(i)=Y_gpu[i];
    Kokkos::deep_copy(d_AT,h_AT); Kokkos::deep_copy(d_X,h_X); Kokkos::deep_copy(d_Y,h_Y);

    auto t0 = std::chrono::steady_clock::now();

    for (int rep = 0; rep < REPEAT; rep++) {
      // kernel0: scale Y diagonal and lower triangle
      Kokkos::parallel_for("chemv_k0", Kokkos::RangePolicy<>(0,N),
        KOKKOS_LAMBDA(int row) {
          float r = d_Y(row).Re*beta_re - d_Y(row).Im*beta_im;
          float im= d_Y(row).Im*beta_re + d_Y(row).Re*beta_im;
          d_Y(row).Re = r; d_Y(row).Im = im;

          float v2r = alpha_re*d_AT(row*LDAT+row).Re;
          float v2i = alpha_im*d_AT(row*LDAT+row).Re;
          d_Y(row).Re += v2r*d_X(row).Re - v2i*d_X(row).Im;
          d_Y(row).Im += v2i*d_X(row).Re + v2r*d_X(row).Im;

          // lower triangle contribution to this row (row acts as column for upper)
          for (int c = row+1; c < N; c++) {
            float v97r = alpha_re*d_AT(row*LDAT+c).Re - alpha_im*d_AT(row*LDAT+c).Im;
            float v97i = alpha_im*d_AT(row*LDAT+c).Re + alpha_re*d_AT(row*LDAT+c).Im;
            Kokkos::atomic_add(&d_Y(c).Re, v97r*d_X(row).Re - v97i*d_X(row).Im);
            Kokkos::atomic_add(&d_Y(c).Im, v97i*d_X(row).Re + v97r*d_X(row).Im);
            float v94r = alpha_re*d_AT(row*LDAT+c).Re - alpha_im*(-d_AT(row*LDAT+c).Im);
            float v94i = alpha_im*d_AT(row*LDAT+c).Re + alpha_re*(-d_AT(row*LDAT+c).Im);
            Kokkos::atomic_add(&d_Y(row).Re, v94r*d_X(c).Re - v94i*d_X(c).Im);
            Kokkos::atomic_add(&d_Y(row).Im, v94i*d_X(c).Re + v94r*d_X(c).Im);
          }
        });
      Kokkos::fence();
    }
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average execution time of chemv kernels: %f (us)\n", (time * 1e-3f) / REPEAT);

    Kokkos::deep_copy(h_Y, d_Y);
    for (int i=0;i<N;i++) Y_gpu[i]=h_Y(i);
  }
  Kokkos::finalize();

  for (int i = 0; i < N; i++) {
    if (fabsf(Y_cpu[i].Re - Y_gpu[i].Re) > 1e-3f ||
        fabsf(Y_cpu[i].Im - Y_gpu[i].Im) > 1e-3f) {
      printf("%d %f %f\n", i, Y_cpu[i].Re, Y_gpu[i].Re);
      printf("FAILED\n");
      return EXIT_FAILURE;
    }
  }
  printf("PASSED\n");
  return EXIT_SUCCESS;
}
