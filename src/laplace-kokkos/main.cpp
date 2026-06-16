/** Laplace 2D – Red-Black Gauss-Seidel with SOR
 *  Kokkos port from the OpenMP target version.
 *
 *  Boundary conditions:
 *    T = 0 at x=0, x=L, y=0
 *    T = TN at y=H
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NUM       1024
#define Real      float
#define ZERO      0.0f
#define ONE       1.0f
#define TWO       2.0f

static const Real omega = 1.85f;

static void fill_coeffs(int rowmax, int colmax,
                         Real th_cond, Real dx, Real dy, Real width, Real TN,
                         Real* aP, Real* aW, Real* aE, Real* aS, Real* aN, Real* b)
{
  for (int col = 0; col < colmax; col++) {
    for (int row = 0; row < rowmax; row++) {
      int ind = col * rowmax + row;
      b[ind] = ZERO;
      Real SP = ZERO;

      if (col == 0)         { aW[ind] = ZERO; SP = -TWO*th_cond*width*dy/dx; }
      else                    aW[ind] = th_cond*width*dy/dx;

      if (col == colmax-1)  { aE[ind] = ZERO; SP = -TWO*th_cond*width*dy/dx; }
      else                    aE[ind] = th_cond*width*dy/dx;

      if (row == 0)         { aS[ind] = ZERO; SP = -TWO*th_cond*width*dx/dy; }
      else                    aS[ind] = th_cond*width*dx/dy;

      if (row == rowmax-1)  {
        aN[ind] = ZERO;
        b[ind]  = TWO*th_cond*width*dx*TN/dy;
        SP      = -TWO*th_cond*width*dx/dy;
      } else {
        aN[ind] = th_cond*width*dx/dy;
      }

      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
    }
  }
}

int main(void)
{
  const Real L      = 1.0f, H = 1.0f, width = 0.01f;
  const Real th_cond= 1.0f, TN = 1.0f, tol   = 1e-6f;

  const int num_rows  = (NUM / 2) + 2;
  const int num_cols  = NUM + 2;
  const int size_temp = num_rows * num_cols;
  const int size      = NUM * NUM;

  const Real dx = L / NUM, dy = H / NUM;

  // Coefficient arrays (global indexing, size = NUM*NUM)
  Real *aP = (Real*)calloc(size, sizeof(Real));
  Real *aW = (Real*)calloc(size, sizeof(Real));
  Real *aE = (Real*)calloc(size, sizeof(Real));
  Real *aS = (Real*)calloc(size, sizeof(Real));
  Real *aN = (Real*)calloc(size, sizeof(Real));
  Real *b  = (Real*)calloc(size, sizeof(Real));

  // Temperature arrays (interleaved red/black layout)
  Real *temp_red   = (Real*)calloc(size_temp, sizeof(Real));
  Real *temp_black = (Real*)calloc(size_temp, sizeof(Real));

  fill_coeffs(NUM, NUM, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);

  printf("Problem size: %d x %d\n", NUM, NUM);

  Kokkos::initialize();
  {
    Kokkos::View<Real*> d_aP  ("aP",   size);
    Kokkos::View<Real*> d_aW  ("aW",   size);
    Kokkos::View<Real*> d_aE  ("aE",   size);
    Kokkos::View<Real*> d_aS  ("aS",   size);
    Kokkos::View<Real*> d_aN  ("aN",   size);
    Kokkos::View<Real*> d_b   ("b",    size);
    Kokkos::View<Real*> d_red ("tr",   size_temp);
    Kokkos::View<Real*> d_blk ("tb",   size_temp);

    // Copy to device
    {
      auto cp = [](Kokkos::View<Real*> d, Real* h, int n) {
        auto hv = Kokkos::create_mirror_view(d);
        for (int i = 0; i < n; i++) hv(i) = h[i];
        Kokkos::deep_copy(d, hv);
      };
      cp(d_aP, aP, size); cp(d_aW, aW, size); cp(d_aE, aE, size);
      cp(d_aS, aS, size); cp(d_aN, aN, size); cp(d_b,  b,  size);
      cp(d_red,  temp_red,   size_temp);
      cp(d_blk,  temp_black, size_temp);
    }

    using Range2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    const int half1 = (NUM >> 1) + 2;  // num_rows
    const int it_max = 1000000;
    int iter = 1;

    auto t0 = std::chrono::steady_clock::now();

    for (; iter <= it_max; iter++) {
      Real norm_L2 = ZERO;

      // ---- RED kernel ----
      Real red_norm = ZERO;
      Kokkos::parallel_reduce("red",
        Range2D({1, 1}, {NUM/2 + 1, NUM + 1}),
        KOKKOS_LAMBDA(int row, int col, Real& lnorm) {
          int ind_red = col * half1 + row;
          int ind     = 2*row - (col & 1) - 1 + NUM*(col-1);

          Real temp_old = d_red(ind_red);
          Real res = d_b(ind)
            + d_aW(ind) * d_blk(row + (col-1)*half1)
            + d_aE(ind) * d_blk(row + (col+1)*half1)
            + d_aS(ind) * d_blk(row - (col & 1)      + col*half1)
            + d_aN(ind) * d_blk(row + ((col+1) & 1)  + col*half1);
          Real temp_new = temp_old*(ONE - omega) + omega*(res / d_aP(ind));
          d_red(ind_red) = temp_new;
          Real diff = temp_new - temp_old;
          lnorm += diff * diff;
        }, red_norm);

      norm_L2 += red_norm;

      // ---- BLACK kernel ----
      Real blk_norm = ZERO;
      Kokkos::parallel_reduce("black",
        Range2D({1, 1}, {NUM/2 + 1, NUM + 1}),
        KOKKOS_LAMBDA(int row, int col, Real& lnorm) {
          int ind_blk = col * half1 + row;
          int ind     = 2*row - ((col+1) & 1) - 1 + NUM*(col-1);

          Real temp_old = d_blk(ind_blk);
          Real res = d_b(ind)
            + d_aW(ind) * d_red(row + (col-1)*half1)
            + d_aE(ind) * d_red(row + (col+1)*half1)
            + d_aS(ind) * d_red(row - ((col+1) & 1) + col*half1)
            + d_aN(ind) * d_red(row + (col & 1)     + col*half1);
          Real temp_new = temp_old*(ONE - omega) + omega*(res / d_aP(ind));
          d_blk(ind_blk) = temp_new;
          Real diff = temp_new - temp_old;
          lnorm += diff * diff;
        }, blk_norm);

      norm_L2 += blk_norm;
      norm_L2 = sqrtf(norm_L2 / (Real)size);

      if (iter % 1000 == 0) printf("%5d, %0.6f\n", iter, norm_L2);
      if (norm_L2 < tol) break;
    }
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0;
    printf("Total time for %d iterations: %f s\n", iter, ms / 1000.0);

    // Copy results back
    {
      auto hr = Kokkos::create_mirror_view(d_red);
      auto hb = Kokkos::create_mirror_view(d_blk);
      Kokkos::deep_copy(hr, d_red);
      Kokkos::deep_copy(hb, d_blk);
      for (int i = 0; i < size_temp; i++) {
        temp_red  [i] = hr(i);
        temp_black[i] = hb(i);
      }
    }
  }
  Kokkos::finalize();

  // Write temperature field
  FILE* f = fopen("temperature.dat", "w");
  if (f) {
    fprintf(f, "#x\ty\ttemp(K)\n");
    for (int row = 1; row < NUM+1; row++) {
      for (int col = 1; col < NUM+1; col++) {
        Real xp = (col-1)*dx + dx/2;
        Real yp = (row-1)*dy + dy/2;
        Real T;
        if ((row+col) % 2 == 0) {
          int ind = col*num_rows + (row + (col%2))/2;
          T = temp_red[ind];
        } else {
          int ind = col*num_rows + (row + ((col+1)%2))/2;
          T = temp_black[ind];
        }
        fprintf(f, "%f\t%f\t%f\n", xp, yp, T);
      }
      fprintf(f, "\n");
    }
    fclose(f);
  }

  free(aP); free(aW); free(aE); free(aS); free(aN); free(b);
  free(temp_red); free(temp_black);
  return 0;
}
