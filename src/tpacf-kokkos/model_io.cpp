#include "tpacf.h"

#include <cstdlib>

double* init_bins(int bins_per_dec, float min_angle, float max_angle,
                  int angle_units, int *nbins) {
  *nbins = (int)floor(bins_per_dec * (log10(max_angle) - log10(min_angle)));
  static double binb[31];
  int binoffset = 30 - (*nbins);
  for (int k = 0; k < (*nbins)+1; k++) {
    double bb = pow(10.0, log10(min_angle) + k * 1.0 / bins_per_dec);
    binb[k + binoffset] = cos(bb / (angle_units ? 60.0 : 1.0) * D2R);
  }
  for (int k = 0; k < binoffset; k++) binb[k] = -5.0;
  binb[30] = -5.0;
  return binb;
}

CartesianData generate_data(int n, unsigned seed) {
  CartesianData cd(n);
  srand(seed);
  for (int i = 0; i < n; i++) {
    double ra  = (rand() / (double)RAND_MAX) * 360.0;
    double dec = (rand() / (double)RAND_MAX) * 180.0 - 90.0;
    double ra_r  = ra  * D2R;
    double dec_r = dec * D2R;
    cd.x[i] = cos(dec_r) * cos(ra_r);
    cd.y[i] = cos(dec_r) * sin(ra_r);
    cd.z[i] = sin(dec_r);
  }
  return cd;
}
