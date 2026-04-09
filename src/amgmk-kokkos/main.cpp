/*BHEADER****************************************************************
 * (c) 2007   The Regents of the University of California               *
 *                                                                      *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright       *
 * notice and disclaimer.                                               *
 *                                                                      *
 *EHEADER****************************************************************/

//--------------
//  A micro kernel 
//--------------
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include <Kokkos_Core.hpp>
#include "headers.h"

// block size
#define BLOCK_SIZE 256

// 
const int testIter   = 500;
double totalWallTime = 0.0;

// 
void test_Matvec();
void test_Relax();
void test_Axpy();

//
int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {
    double del_wtime = 0.0;

    printf("\n");
    printf("//------------ \n");
    printf("// \n");
    printf("//  CORAL  AMGmk Benchmark Version 1.0 \n");
    printf("// \n");
    printf("//------------ \n");

    printf("\n testIter   = %d \n\n", testIter );  

    auto t0 = std::chrono::steady_clock::now();

    // Matvec
    totalWallTime = 0.0;
   
    test_Matvec();

    printf("\n");
    printf("//------------ \n");
    printf("// \n");
    printf("//   MATVEC\n");
    printf("// \n");
    printf("//------------ \n");

    printf("\nWall time = %f seconds. \n", totalWallTime);


    // Relax
    totalWallTime = 0.0;

    test_Relax();

    printf("\n");
    printf("//------------ \n");
    printf("// \n");
    printf("//   Relax\n");
    printf("// \n");
    printf("//------------ \n");

    printf("\nWall time = %f seconds. \n", totalWallTime);


    // Axpy
    totalWallTime = 0.0;
   
    test_Axpy();

    printf("\n");
    printf("//------------ \n");
    printf("// \n");
    printf("//   Axpy\n");
    printf("// \n");
    printf("//------------ \n");

    printf("\nWall time = %f seconds. \n", totalWallTime);

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t1 - t0;
    del_wtime = diff.count();

    printf("\nTotal Wall time = %f seconds. \n", del_wtime);
  }
  Kokkos::finalize();

  return  0;
}

void test_Matvec()
{
  hypre_CSRMatrix *A;
  hypre_Vector *x, *y, *sol;
  int nx, ny, nz, i;
  double *values;
  double *y_data, *sol_data;
  double error, diff;

  nx = 50;  /* size per proc nx*ny*nz */
  ny = 50;
  nz = 50;

  values = hypre_CTAlloc(double, 4);
  values[0] = 6; 
  values[1] = -1;
  values[2] = -1;
  values[3] = -1;

  A = GenerateSeqLaplacian(nx, ny, nz, values, &y, &x, &sol);

  hypre_SeqVectorSetConstantValues(x,1);
  hypre_SeqVectorSetConstantValues(y,0);

  auto t0 = std::chrono::steady_clock::now();

  for (i=0; i<testIter; ++i)
      hypre_CSRMatrixMatvec(1,A,x,0,y);

  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> tdiff = t1 - t0;
  totalWallTime += tdiff.count();

  y_data = hypre_VectorData(y);
  sol_data = hypre_VectorData(sol);

  error = 0;
  for (i=0; i < nx*ny*nz; i++)
  {
      diff = fabs(y_data[i]-sol_data[i]);
      if (diff > error) error = diff;
  }
     
  if (error > 0) printf(" \n Matvec: error: %e\n", error);

  hypre_TFree(values);
  hypre_CSRMatrixDestroy(A);
  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
  hypre_SeqVectorDestroy(sol);
}

void test_Relax()
{
  hypre_CSRMatrix *A;
  hypre_Vector *x, *y, *sol;
  int nx, ny, nz;
  double *values;
  double diff, error;

  nx = 50;  /* size per proc nx*ny*nz */
  ny = 50;
  nz = 50;

  values = hypre_CTAlloc(double, 4);
  values[0] = 6; 
  values[1] = -1;
  values[2] = -1;
  values[3] = -1;

  A = GenerateSeqLaplacian(nx, ny, nz, values, &y, &x, &sol);

  hypre_SeqVectorSetConstantValues(x,1);

  double         *A_diag_data  = hypre_CSRMatrixData(A);
  int            *A_diag_i     = hypre_CSRMatrixI(A);
  int            *A_diag_j     = hypre_CSRMatrixJ(A);

  int             n       = hypre_CSRMatrixNumRows(A);
  int             nonzero = hypre_CSRMatrixNumNonzeros(A);

  double         *u_data  = hypre_VectorData(x);
  double         *f_data  = hypre_VectorData(sol);

  int             grid_size = nx*ny*nz;

  // Create device views and copy data
  Kokkos::View<double*> d_A_diag_data("A_diag_data", nonzero);
  Kokkos::View<int*> d_A_diag_i("A_diag_i", grid_size+1);
  Kokkos::View<int*> d_A_diag_j("A_diag_j", nonzero);
  Kokkos::View<double*> d_f_data("f_data", grid_size);
  Kokkos::View<double*> d_u_data("u_data", grid_size);

  auto h_A_diag_data = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_diag_data, nonzero);
  auto h_A_diag_i = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_diag_i, grid_size+1);
  auto h_A_diag_j = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_diag_j, nonzero);
  auto h_f_data = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(f_data, grid_size);
  auto h_u_data = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(u_data, grid_size);

  Kokkos::deep_copy(d_A_diag_data, h_A_diag_data);
  Kokkos::deep_copy(d_A_diag_i, h_A_diag_i);
  Kokkos::deep_copy(d_A_diag_j, h_A_diag_j);
  Kokkos::deep_copy(d_f_data, h_f_data);
  Kokkos::deep_copy(d_u_data, h_u_data);

  auto t0 = std::chrono::steady_clock::now();

  for (int ti = 0; ti < testIter; ++ti) {
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
      if (d_A_diag_data(d_A_diag_i(i)) != 0.0)
      {
        double res = d_f_data(i);
        for (int jj = d_A_diag_i(i)+1; jj < d_A_diag_i(i+1); jj++)
        {
          int ii = d_A_diag_j(jj);
          res -= d_A_diag_data(jj) * d_u_data(ii);
        }
        d_u_data(i) = res / d_A_diag_data(d_A_diag_i(i));
      }
    });
  }
  Kokkos::fence();

  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> tdiff = t1 - t0;
  totalWallTime += tdiff.count();

  // Copy result back
  Kokkos::deep_copy(h_u_data, d_u_data);

  error = 0;
  for (int i=0; i < nx*ny*nz; i++)
  {
      diff = fabs(u_data[i]-1);
      if (diff > error) error = diff;
  }
     
  if (error > 0) printf(" \n Relax: error: %e\n", error);

  hypre_TFree(values);
  hypre_CSRMatrixDestroy(A);
  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
  hypre_SeqVectorDestroy(sol);
}

void test_Axpy()
{
  hypre_Vector *x, *y;
  int nx, i;
  double alpha=0.5;
  double diff, error;
  double *y_data;

  nx = 125000;  /* size per proc  */

  x = hypre_SeqVectorCreate(nx);
  y = hypre_SeqVectorCreate(nx);

  hypre_SeqVectorInitialize(x);
  hypre_SeqVectorInitialize(y);

  hypre_SeqVectorSetConstantValues(x,1);
  hypre_SeqVectorSetConstantValues(y,1);

  auto t0 = std::chrono::steady_clock::now();

  for (i=0; i<testIter; ++i)
      hypre_SeqVectorAxpy(alpha,x,y);

  auto t1 = std::chrono::steady_clock::now();

  y_data = hypre_VectorData(y);
  error = 0;
  for (i=0; i < nx; i++)
  {
    diff = fabs(y_data[i]-1-0.5*(double)testIter);
      if (diff > error) error = diff;
  }
     
  if (error > 0) printf(" \n Axpy: error: %e\n", error);

  std::chrono::duration<double> tdiff = t1 - t0;
  totalWallTime += tdiff.count();

  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
}
