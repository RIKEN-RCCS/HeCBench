#include <math.h>
#include <openacc.h>

#define max(a,b) (a) > (b) ? (a) : (b)

double lebesgue_function ( int n, double x[], int nfun, double xfun[] )
{
  double lmax = 0.0;
  double *linterp = (double*) malloc ( n * nfun * sizeof ( double ) );

#pragma acc data copy(lmax)			\
  copyin(x[0:n], xfun[0:nfun])					\
  create(linterp[0:n * nfun])
  {
    #pragma acc parallel loop vector_length(256) reduction(max:lmax)
    for (int j = 0; j < nfun; j++) {
      double t = 0.0;
      for (int i1 = 0; i1 < n; i1++ ) {
        linterp[i1*nfun+j] = 1.0;
        for (int i2 = 0; i2 < n; i2++ )
          if ( i1 != i2 )
            linterp[i1*nfun+j] = linterp[i1*nfun+j] * ( xfun[j] - x[i2] ) / ( x[i1] - x[i2] );
        t += fabs ( linterp[i1*nfun+j] );
      }
      lmax = max(lmax, t);
    }
  }

  free(linterp);
  return lmax;
}
