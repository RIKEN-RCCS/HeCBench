#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

#include "main.h"

extern "C" {
#include "graphics.h"
#include "resize.h"
#include "timer.h"
}

int main(int argc, char* argv[])
{
  long long time0 = get_time();

  if (argc != 5) {
    printf("Usage: %s <repeat> <lambda> <number of rows> <number of columns>\n", argv[0]);
    return 1;
  }
  int  niter  = atoi(argv[1]);
  fp   lambda = (fp)atof(argv[2]);
  int  Nr     = atoi(argv[3]);
  int  Nc     = atoi(argv[4]);

  long long time1 = get_time();
  long long time2 = get_time();

  // Read original image
  int  image_ori_rows = 502;
  int  image_ori_cols = 458;
  long image_ori_elem = image_ori_rows * image_ori_cols;
  fp  *image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

  const char *input_image_path = "../data/srad/image.pgm";
  if (!read_graphics(input_image_path, image_ori, image_ori_rows, image_ori_cols, 1)) {
    printf("ERROR: failed to read input image at %s\n", input_image_path);
    if (image_ori) free(image_ori);
    return -1;
  }

  long long time3 = get_time();

  long Ne = (long)Nr * Nc;
  fp  *image = (fp*)malloc(sizeof(fp) * Ne);
  resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

  long long time4 = get_time();

  // Setup surrounding-pixel index arrays
  int r1 = 0, r2 = Nr - 1, c1 = 0, c2 = Nc - 1;
  long NeROI = (long)(r2 - r1 + 1) * (c2 - c1 + 1);

  int *iN = (int*)malloc(sizeof(int) * Nr);
  int *iS = (int*)malloc(sizeof(int) * Nr);
  int *jW = (int*)malloc(sizeof(int) * Nc);
  int *jE = (int*)malloc(sizeof(int) * Nc);

  for (int i = 0; i < Nr; i++) { iN[i] = i - 1; iS[i] = i + 1; }
  for (int j = 0; j < Nc; j++) { jW[j] = j - 1; jE[j] = j + 1; }
  iN[0]    = 0;     iS[Nr-1] = Nr - 1;
  jW[0]    = 0;     jE[Nc-1] = Nc - 1;

  long long time5 = get_time();

  Kokkos::initialize(argc, argv);
  {
    // Allocate device views
    Kokkos::View<fp*>  image_d("image", Ne);
    Kokkos::View<int*> iN_d("iN", Nr), iS_d("iS", Nr);
    Kokkos::View<int*> jW_d("jW", Nc), jE_d("jE", Nc);
    Kokkos::View<fp*>  dN_d("dN", Ne), dS_d("dS", Ne);
    Kokkos::View<fp*>  dW_d("dW", Ne), dE_d("dE", Ne);
    Kokkos::View<fp*>  c_d("c", Ne);

    // Copy host data to device
    {
      auto img_hv = Kokkos::View<fp*,  Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(image, Ne);
      auto iN_hv  = Kokkos::View<int*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(iN, Nr);
      auto iS_hv  = Kokkos::View<int*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(iS, Nr);
      auto jW_hv  = Kokkos::View<int*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(jW, Nc);
      auto jE_hv  = Kokkos::View<int*, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(jE, Nc);
      Kokkos::deep_copy(image_d, img_hv);
      Kokkos::deep_copy(iN_d, iN_hv);
      Kokkos::deep_copy(iS_d, iS_hv);
      Kokkos::deep_copy(jW_d, jW_hv);
      Kokkos::deep_copy(jE_d, jE_hv);
    }

    long long time6 = get_time();

    // Exponentiate input image
    Kokkos::parallel_for("extract", Kokkos::RangePolicy<>(0, Ne),
      KOKKOS_LAMBDA(int ei) {
        image_d(ei) = expf(image_d(ei) / (fp)255);
      });
    Kokkos::fence();

    long long time7 = get_time();

    for (int iter = 0; iter < niter; iter++) {
      // Compute ROI statistics via parallel reduction
      fp total_sum  = (fp)0;
      fp total_sum2 = (fp)0;
      Kokkos::parallel_reduce("reduce_sum", Kokkos::RangePolicy<>(0, Ne),
        KOKKOS_LAMBDA(int i, fp& val) { val += image_d(i); },
        total_sum);
      Kokkos::parallel_reduce("reduce_sum2", Kokkos::RangePolicy<>(0, Ne),
        KOKKOS_LAMBDA(int i, fp& val) { val += image_d(i) * image_d(i); },
        total_sum2);

      fp meanROI  = total_sum  / (fp)NeROI;
      fp varROI   = (total_sum2 / (fp)NeROI) - meanROI * meanROI;
      fp q0sqr    = varROI / (meanROI * meanROI);

      // Compute diffusion coefficients
      Kokkos::parallel_for("diffusion_coeff", Kokkos::RangePolicy<>(0, Ne),
        KOKKOS_LAMBDA(int ei) {
          int row = (ei + 1) % Nr - 1;
          int col = (ei + 1) / Nr + 1 - 1;
          if ((ei + 1) % Nr == 0) { row = Nr - 1; col = col - 1; }

          fp d_Jc = image_d(ei);
          fp N_loc = image_d(iN_d(row) + Nr * col) - d_Jc;
          fp S_loc = image_d(iS_d(row) + Nr * col) - d_Jc;
          fp W_loc = image_d(row + Nr * jW_d(col)) - d_Jc;
          fp E_loc = image_d(row + Nr * jE_d(col)) - d_Jc;

          fp d_G2   = (N_loc*N_loc + S_loc*S_loc + W_loc*W_loc + E_loc*E_loc) / (d_Jc * d_Jc);
          fp d_L    = (N_loc + S_loc + W_loc + E_loc) / d_Jc;
          fp d_num  = ((fp)0.5 * d_G2) - ((fp)(1.0/16.0) * (d_L * d_L));
          fp d_den  = (fp)1 + ((fp)0.25 * d_L);
          fp d_qsqr = d_num / (d_den * d_den);

          d_den = (d_qsqr - q0sqr) / (q0sqr * ((fp)1 + q0sqr));
          fp d_c_loc = (fp)1.0 / ((fp)1.0 + d_den);
          if (d_c_loc < (fp)0) d_c_loc = (fp)0;
          else if (d_c_loc > (fp)1) d_c_loc = (fp)1;

          dN_d(ei) = N_loc;
          dS_d(ei) = S_loc;
          dW_d(ei) = W_loc;
          dE_d(ei) = E_loc;
          c_d(ei)  = d_c_loc;
        });

      // Update image
      Kokkos::parallel_for("update", Kokkos::RangePolicy<>(0, Ne),
        KOKKOS_LAMBDA(int ei) {
          int row = (ei + 1) % Nr - 1;
          int col = (ei + 1) / Nr;
          if ((ei + 1) % Nr == 0) { row = Nr - 1; col = col - 1; }

          fp d_cN = c_d(ei);
          fp d_cS = c_d(iS_d(row) + Nr * col);
          fp d_cW = c_d(ei);
          fp d_cE = c_d(row + Nr * jE_d(col));

          fp d_D = d_cN*dN_d(ei) + d_cS*dS_d(ei) + d_cW*dW_d(ei) + d_cE*dE_d(ei);
          image_d(ei) += (fp)0.25 * lambda * d_D;
        });
    }
    Kokkos::fence();

    long long time8 = get_time();

    // Compress: scale back to 0-255
    Kokkos::parallel_for("compress", Kokkos::RangePolicy<>(0, Ne),
      KOKKOS_LAMBDA(int ei) {
        image_d(ei) = logf(image_d(ei)) * (fp)255;
      });
    Kokkos::fence();

    long long time9 = get_time();

    // Copy image back to host
    {
      auto img_hv = Kokkos::View<fp*, Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>(image, Ne);
      Kokkos::deep_copy(img_hv, image_d);
    }

    long long time10 = get_time();

    write_graphics("./image_out.pgm", image, Nr, Nc, 1, 255);

    long long time11 = get_time();

    free(image_ori);
    free(image);
    free(iN); free(iS); free(jW); free(jE);

    long long time12 = get_time();

    printf("Time spent in different stages of the application:\n");
    printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n",
        (float)(time1-time0)/1000000, (float)(time1-time0)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n",
        (float)(time2-time1)/1000000, (float)(time2-time1)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n",
        (float)(time3-time2)/1000000, (float)(time3-time2)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n",
        (float)(time4-time3)/1000000, (float)(time4-time3)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n",
        (float)(time5-time4)/1000000, (float)(time5-time4)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n",
        (float)(time6-time5)/1000000, (float)(time6-time5)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n",
        (float)(time7-time6)/1000000, (float)(time7-time6)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : COMPUTE (%d iterations)\n",
        (float)(time8-time7)/1000000, (float)(time8-time7)/(float)(time12-time0)*100, niter);
    printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n",
        (float)(time9-time8)/1000000, (float)(time9-time8)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n",
        (float)(time10-time9)/1000000, (float)(time10-time9)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n",
        (float)(time11-time10)/1000000, (float)(time11-time10)/(float)(time12-time0)*100);
    printf("%15.12f s, %15.12f %% : FREE MEMORY\n",
        (float)(time12-time11)/1000000, (float)(time12-time11)/(float)(time12-time0)*100);
    printf("Total time:\n%.12f s\n", (float)(time12-time0)/1000000);
  }
  Kokkos::finalize();
  return 0;
}
