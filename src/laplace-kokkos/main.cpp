/** GPU Laplace solver using optimized red-black Gauss–Seidel with SOR solver
 *
 * Solves Laplace's equation in 2D (e.g., heat conduction in a rectangular plate)
 * on GPU using Kokkos with the red-black Gauss–Seidel with successive overrelaxation
 * (SOR) that has been "optimized". This means that the red and black kernels
 * only loop over their respective cells, instead of over all cells and skipping
 * even/odd cells. This requires separate arrays for red and black cells.
 *
 * Boundary conditions:
 * T = 0 at x = 0, x = L, y = 0
 * T = TN at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "timer.h"

#include <Kokkos_Core.hpp>

/** Problem size along one side; total number of cells is this squared */
#define NUM 1024

#define Real float
#define ZERO 0.0f
#define ONE  1.0f
#define TWO  2.0f

/** SOR relaxation parameter */
const Real omega = 1.85f;

/** Function to evaluate coefficient matrix and right-hand side vector. */
void fill_coeffs (int rowmax, int colmax, Real th_cond, Real dx, Real dy,
    Real width, Real TN, Real * aP, Real * aW, Real * aE,
    Real * aS, Real * aN, Real * b)
{
  int col, row;
  for (col = 0; col < colmax; ++col) {
    for (row = 0; row < rowmax; ++row) {
      int ind = col * rowmax + row;

      b[ind] = ZERO;
      Real SP = ZERO;

      if (col == 0) {
        aW[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aW[ind] = th_cond * width * dy / dx;
      }

      if (col == colmax - 1) {
        aE[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aE[ind] = th_cond * width * dy / dx;
      }

      if (row == 0) {
        aS[ind] = ZERO;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aS[ind] = th_cond * width * dx / dy;
      }

      if (row == rowmax - 1) {
        aN[ind] = ZERO;
        b[ind] = TWO * th_cond * width * dx * TN / dy;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aN[ind] = th_cond * width * dx / dy;
      }

      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
    }
  }
}

/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 *
 * Contains iteration loop for red-black Gauss-Seidel with SOR Kokkos kernels
 */
int main (void) {

  Kokkos::initialize();
  {
    // size of plate
    Real L = 1.0;
    Real H = 1.0;
    Real width = 0.01;

    // thermal conductivity
    Real th_cond = 1.0;

    // temperature at top boundary
    Real TN = 1.0;

    // SOR iteration tolerance
    Real tol = 1.e-6;

    // number of cells in x and y directions (including unused boundary cells)
    int num_rows = (NUM / 2) + 2;
    int num_cols = NUM + 2;
    int size_temp = num_rows * num_cols;
    int size = NUM * NUM;

    // size of cells
    Real dx = L / NUM;
    Real dy = H / NUM;

    // iterations for Red-Black Gauss-Seidel with SOR
    int iter;
    int it_max = 1e6;

    // one norm value per temperature slot
    int size_norm = size_temp;

    // -------------------------------------------------------------------------
    // Device Views
    // -------------------------------------------------------------------------
    Kokkos::View<Real*> d_aP        ("aP",         size);
    Kokkos::View<Real*> d_aW        ("aW",         size);
    Kokkos::View<Real*> d_aE        ("aE",         size);
    Kokkos::View<Real*> d_aS        ("aS",         size);
    Kokkos::View<Real*> d_aN        ("aN",         size);
    Kokkos::View<Real*> d_b         ("b",          size);
    Kokkos::View<Real*> d_temp_red  ("temp_red",   size_temp);
    Kokkos::View<Real*> d_temp_black("temp_black", size_temp);
    Kokkos::View<Real*> d_bl_norm   ("bl_norm",    size_norm);

    // -------------------------------------------------------------------------
    // Host mirrors
    // -------------------------------------------------------------------------
    auto h_aP         = Kokkos::create_mirror_view(d_aP);
    auto h_aW         = Kokkos::create_mirror_view(d_aW);
    auto h_aE         = Kokkos::create_mirror_view(d_aE);
    auto h_aS         = Kokkos::create_mirror_view(d_aS);
    auto h_aN         = Kokkos::create_mirror_view(d_aN);
    auto h_b          = Kokkos::create_mirror_view(d_b);
    auto h_temp_red   = Kokkos::create_mirror_view(d_temp_red);
    auto h_temp_black = Kokkos::create_mirror_view(d_temp_black);

    // -------------------------------------------------------------------------
    // Initialize on host
    // -------------------------------------------------------------------------
    fill_coeffs(NUM, NUM, th_cond, dx, dy, width, TN,
                h_aP.data(), h_aW.data(), h_aE.data(),
                h_aS.data(), h_aN.data(), h_b.data());

    for (int i = 0; i < size_temp; ++i) {
      h_temp_red(i)   = ZERO;
      h_temp_black(i) = ZERO;
    }

    // -------------------------------------------------------------------------
    // Copy to device
    // -------------------------------------------------------------------------
    Kokkos::deep_copy(d_aP,         h_aP);
    Kokkos::deep_copy(d_aW,         h_aW);
    Kokkos::deep_copy(d_aE,         h_aE);
    Kokkos::deep_copy(d_aS,         h_aS);
    Kokkos::deep_copy(d_aN,         h_aN);
    Kokkos::deep_copy(d_b,          h_b);
    Kokkos::deep_copy(d_temp_red,   h_temp_red);
    Kokkos::deep_copy(d_temp_black, h_temp_black);
    Kokkos::deep_copy(d_bl_norm,    ZERO);

    printf("Problem size: %d x %d \n", NUM, NUM);

    // -------------------------------------------------------------------------
    // Iteration loop
    // -------------------------------------------------------------------------
    StartTimer();

    for (iter = 1; iter <= it_max; ++iter) {

      Real norm_L2 = ZERO;

      // --- Red kernel --------------------------------------------------------
      Kokkos::parallel_for("red_kernel",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {NUM/2 + 1, NUM + 1}),
        KOKKOS_LAMBDA(int row, int col) {
          int ind_red = col * ((NUM >> 1) + 2) + row;
          int ind     = 2 * row - (col & 1) - 1 + NUM * (col - 1);

          Real temp_old = d_temp_red(ind_red);

          Real res = d_b(ind)
              + (  d_aW(ind) * d_temp_black(row + (col - 1) * ((NUM >> 1) + 2))
                 + d_aE(ind) * d_temp_black(row + (col + 1) * ((NUM >> 1) + 2))
                 + d_aS(ind) * d_temp_black(row - (col & 1) + col * ((NUM >> 1) + 2))
                 + d_aN(ind) * d_temp_black(row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)));

          Real temp_new = temp_old * (ONE - omega) + omega * (res / d_aP(ind));

          d_temp_red(ind_red) = temp_new;
          res = temp_new - temp_old;

          d_bl_norm(ind_red) = res * res;
        });

      // Accumulate red residual
      Real norm_red = ZERO;
      Kokkos::parallel_reduce("norm_red",
        size_norm,
        KOKKOS_LAMBDA(int i, Real& s) { s += d_bl_norm(i); },
        norm_red);
      norm_L2 += norm_red;

      // --- Black kernel ------------------------------------------------------
      Kokkos::parallel_for("black_kernel",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {NUM/2 + 1, NUM + 1}),
        KOKKOS_LAMBDA(int row, int col) {
          int ind_black = col * ((NUM >> 1) + 2) + row;
          int ind       = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1);

          Real temp_old = d_temp_black(ind_black);

          Real res = d_b(ind)
              + (  d_aW(ind) * d_temp_red(row + (col - 1) * ((NUM >> 1) + 2))
                 + d_aE(ind) * d_temp_red(row + (col + 1) * ((NUM >> 1) + 2))
                 + d_aS(ind) * d_temp_red(row - ((col + 1) & 1) + col * ((NUM >> 1) + 2))
                 + d_aN(ind) * d_temp_red(row + (col & 1) + col * ((NUM >> 1) + 2)));

          Real temp_new = temp_old * (ONE - omega) + omega * (res / d_aP(ind));

          d_temp_black(ind_black) = temp_new;
          res = temp_new - temp_old;

          d_bl_norm(ind_black) = res * res;
        });

      // Accumulate black residual
      Real norm_black = ZERO;
      Kokkos::parallel_reduce("norm_black",
        size_norm,
        KOKKOS_LAMBDA(int i, Real& s) { s += d_bl_norm(i); },
        norm_black);
      norm_L2 += norm_black;

      // Calculate residual
      norm_L2 = sqrt(norm_L2 / ((Real)size));

      if (iter % 1000 == 0) printf("%5d, %0.6f\n", iter, norm_L2);

      // If tolerance reached, end SOR iterations
      if (norm_L2 < tol) break;
    }

    double runtime = GetTimer();
    printf("Total time for %i iterations: %f s\n", iter, runtime / 1000.0);

    // -------------------------------------------------------------------------
    // Copy results back to host and write temperature data to file
    // -------------------------------------------------------------------------
    Kokkos::deep_copy(h_temp_red,   d_temp_red);
    Kokkos::deep_copy(h_temp_black, d_temp_black);

    FILE * pfile = fopen("temperature.dat", "w");
    if (pfile != NULL) {
      fprintf(pfile, "#x\ty\ttemp(K)\n");

      int row, col;
      for (row = 1; row < NUM + 1; ++row) {
        for (col = 1; col < NUM + 1; ++col) {
          Real x_pos = (col - 1) * dx + (dx / 2);
          Real y_pos = (row - 1) * dy + (dy / 2);

          if ((row + col) % 2 == 0) {
            // even, so red cell
            int ind = col * num_rows + (row + (col % 2)) / 2;
            fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, h_temp_red(ind));
          } else {
            // odd, so black cell
            int ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
            fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, h_temp_black(ind));
          }
        }
        fprintf(pfile, "\n");
      }
    }
    fclose(pfile);

  } // end Kokkos scope
  Kokkos::finalize();

  return 0;
}
