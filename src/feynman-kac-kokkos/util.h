#pragma once
#include <Kokkos_Core.hpp>
#include <math.h>

KOKKOS_INLINE_FUNCTION
int i4_ceiling(double x)
{
  int value = (int)x;
  if (value < x) value = value + 1;
  return value;
}

KOKKOS_INLINE_FUNCTION
double potential(double a, double b, double x, double y)
{
  double value = 2.0 * (pow(x / a / a, 2.0) + pow(y / b / b, 2.0))
               + 1.0 / a / a + 1.0 / b / b;
  return value;
}

KOKKOS_INLINE_FUNCTION
double r8_uniform_01(int *seed)
{
  int k = *seed / 127773;
  *seed = 16807 * (*seed - k * 127773) - k * 2836;
  if (*seed < 0) *seed = *seed + 2147483647;
  double r = (double)(*seed) * 4.656612875E-10;
  return r;
}
