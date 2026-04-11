#if !defined(COMPLEX_H_)
#define COMPLEX_H_

#include <math.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

typedef struct __attribute__((__aligned__(8)))  { float  x, y; } float2;
typedef struct __attribute__((__aligned__(16))) { double x, y; } double2;

typedef float2  FloatComplex;
typedef double2 DoubleComplex;

KOKKOS_INLINE_FUNCTION float Crealf(FloatComplex x)  { return x.x; }
KOKKOS_INLINE_FUNCTION float Cimagf(FloatComplex x)  { return x.y; }

KOKKOS_INLINE_FUNCTION FloatComplex make_FloatComplex(float r, float i)
{ FloatComplex res; res.x = r; res.y = i; return res; }

KOKKOS_INLINE_FUNCTION FloatComplex Conjf(FloatComplex x)
{ return make_FloatComplex(Crealf(x), -Cimagf(x)); }

KOKKOS_INLINE_FUNCTION FloatComplex Caddf(FloatComplex x, FloatComplex y)
{ return make_FloatComplex(Crealf(x)+Crealf(y), Cimagf(x)+Cimagf(y)); }

KOKKOS_INLINE_FUNCTION FloatComplex Csubf(FloatComplex x, FloatComplex y)
{ return make_FloatComplex(Crealf(x)-Crealf(y), Cimagf(x)-Cimagf(y)); }

KOKKOS_INLINE_FUNCTION FloatComplex Cmulf(FloatComplex x, FloatComplex y) {
  return make_FloatComplex(
    (Crealf(x)*Crealf(y)) - (Cimagf(x)*Cimagf(y)),
    (Crealf(x)*Cimagf(y)) + (Cimagf(x)*Crealf(y)));
}

KOKKOS_INLINE_FUNCTION FloatComplex Cdivf(FloatComplex x, FloatComplex y) {
  FloatComplex quot;
  float s   = fabsf(Crealf(y)) + fabsf(Cimagf(y));
  float oos = 1.0f / s;
  float ars = Crealf(x) * oos,  ais = Cimagf(x) * oos;
  float brs = Crealf(y) * oos,  bis = Cimagf(y) * oos;
  s = (brs*brs) + (bis*bis);
  oos = 1.0f / s;
  quot = make_FloatComplex(((ars*brs)+(ais*bis))*oos, ((ais*brs)-(ars*bis))*oos);
  return quot;
}

KOKKOS_INLINE_FUNCTION float Cabsf(FloatComplex x) {
  float a = fabsf(Crealf(x)), b = fabsf(Cimagf(x)), v, w, t;
  if (a > b) { v = a; w = b; } else { v = b; w = a; }
  t = w / v; t = 1.0f + t*t; t = v * sqrtf(t);
  if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) t = v + w;
  return t;
}

/* Double precision */
KOKKOS_INLINE_FUNCTION double Creal(DoubleComplex x)  { return x.x; }
KOKKOS_INLINE_FUNCTION double Cimag(DoubleComplex x)  { return x.y; }

KOKKOS_INLINE_FUNCTION DoubleComplex make_DoubleComplex(double r, double i)
{ DoubleComplex res; res.x = r; res.y = i; return res; }

KOKKOS_INLINE_FUNCTION DoubleComplex Conj(DoubleComplex x)
{ return make_DoubleComplex(Creal(x), -Cimag(x)); }

KOKKOS_INLINE_FUNCTION DoubleComplex Cadd(DoubleComplex x, DoubleComplex y)
{ return make_DoubleComplex(Creal(x)+Creal(y), Cimag(x)+Cimag(y)); }

KOKKOS_INLINE_FUNCTION DoubleComplex Csub(DoubleComplex x, DoubleComplex y)
{ return make_DoubleComplex(Creal(x)-Creal(y), Cimag(x)-Cimag(y)); }

KOKKOS_INLINE_FUNCTION DoubleComplex Cmul(DoubleComplex x, DoubleComplex y) {
  return make_DoubleComplex(
    (Creal(x)*Creal(y)) - (Cimag(x)*Cimag(y)),
    (Creal(x)*Cimag(y)) + (Cimag(x)*Creal(y)));
}

KOKKOS_INLINE_FUNCTION DoubleComplex Cdiv(DoubleComplex x, DoubleComplex y) {
  DoubleComplex quot;
  double s   = fabs(Creal(y)) + fabs(Cimag(y));
  double oos = 1.0 / s;
  double ars = Creal(x)*oos,  ais = Cimag(x)*oos;
  double brs = Creal(y)*oos,  bis = Cimag(y)*oos;
  s = (brs*brs) + (bis*bis);
  oos = 1.0 / s;
  quot = make_DoubleComplex(((ars*brs)+(ais*bis))*oos, ((ais*brs)-(ars*bis))*oos);
  return quot;
}

KOKKOS_INLINE_FUNCTION double Cabs(DoubleComplex x) {
  double a = fabs(Creal(x)), b = fabs(Cimag(x)), v, w, t;
  if (a > b) { v = a; w = b; } else { v = b; w = a; }
  t = w / v; t = 1.0 + t*t; t = v * sqrt(t);
  if ((v == 0.0) || (v > 1.79769313486231570e+308) || (w > 1.79769313486231570e+308))
    t = v + w;
  return t;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(COMPLEX_H_) */
