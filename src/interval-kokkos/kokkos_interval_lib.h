/*
 * Kokkos-portable interval_gpu class and arithmetic operations.
 * Adapted from interval-omp/gpu_interval_lib.h:
 *   - replaced `inline` free functions with KOKKOS_INLINE_FUNCTION
 *   - class methods annotated with KOKKOS_INLINE_FUNCTION
 */

#ifndef KOKKOS_INTERVAL_LIB_H
#define KOKKOS_INTERVAL_LIB_H

#include "kokkos_interval_rounded_arith.h"

template <class T>
class interval_gpu {
 public:
  KOKKOS_INLINE_FUNCTION interval_gpu();
  KOKKOS_INLINE_FUNCTION interval_gpu(T const &v);
  KOKKOS_INLINE_FUNCTION interval_gpu(T const &l, T const &u);

  KOKKOS_INLINE_FUNCTION T const &lower() const;
  KOKKOS_INLINE_FUNCTION T const &upper() const;

  KOKKOS_INLINE_FUNCTION static interval_gpu empty();

 private:
  T low;
  T up;
};

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T>::interval_gpu() {}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T>::interval_gpu(T const &l, T const &u)
    : low(l), up(u) {}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T>::interval_gpu(T const &v)
    : low(v), up(v) {}

template <class T>
KOKKOS_INLINE_FUNCTION T const &interval_gpu<T>::lower() const { return low; }

template <class T>
KOKKOS_INLINE_FUNCTION T const &interval_gpu<T>::upper() const { return up; }

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> interval_gpu<T>::empty() {
  rounded_arith<T> rnd;
  return interval_gpu<T>(rnd.nan(), rnd.nan());
}

template <class T>
KOKKOS_INLINE_FUNCTION bool empty(interval_gpu<T> x) {
  T h = x.lower() + x.upper();
  return (h != h);
}

template <class T>
KOKKOS_INLINE_FUNCTION T width(interval_gpu<T> x) {
  if (empty(x)) return 0;
  rounded_arith<T> rnd;
  return rnd.sub_up(x.upper(), x.lower());
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> const &operator+(interval_gpu<T> const &x) {
  return x;
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> operator-(interval_gpu<T> const &x) {
  return interval_gpu<T>(-x.upper(), -x.lower());
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> operator+(interval_gpu<T> const &x,
                                                  interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  return interval_gpu<T>(rnd.add_down(x.lower(), y.lower()),
                         rnd.add_up  (x.upper(), y.upper()));
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> operator-(interval_gpu<T> const &x,
                                                  interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  return interval_gpu<T>(rnd.sub_down(x.lower(), y.upper()),
                         rnd.sub_up  (x.upper(), y.lower()));
}

KOKKOS_INLINE_FUNCTION float  min4(float  a,float  b,float  c,float  d){return fminf(fminf(a,b),fminf(c,d));}
KOKKOS_INLINE_FUNCTION float  max4(float  a,float  b,float  c,float  d){return fmaxf(fmaxf(a,b),fmaxf(c,d));}
KOKKOS_INLINE_FUNCTION double min4(double a,double b,double c,double d){return fmin(fmin(a,b),fmin(c,d));}
KOKKOS_INLINE_FUNCTION double max4(double a,double b,double c,double d){return fmax(fmax(a,b),fmax(c,d));}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> operator*(interval_gpu<T> const &x,
                                                  interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  return interval_gpu<T>(
      min4(rnd.mul_down(x.lower(),y.lower()), rnd.mul_down(x.lower(),y.upper()),
           rnd.mul_down(x.upper(),y.lower()), rnd.mul_down(x.upper(),y.upper())),
      max4(rnd.mul_up(x.lower(),y.lower()),   rnd.mul_up(x.lower(),y.upper()),
           rnd.mul_up(x.upper(),y.lower()),   rnd.mul_up(x.upper(),y.upper())));
}

template <class T>
KOKKOS_INLINE_FUNCTION T median(interval_gpu<T> const &x) {
  rounded_arith<T> rnd;
  return rnd.median(x.lower(), x.upper());
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> intersect(interval_gpu<T> const &x,
                                                   interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  T const &l = rnd.max(x.lower(), y.lower());
  T const &u = rnd.min(x.upper(), y.upper());
  if (l <= u) return interval_gpu<T>(l, u);
  return interval_gpu<T>::empty();
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> div_non_zero(interval_gpu<T> const &x,
                                                      interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  T xl, yl, xu, yu;
  if (y.upper() < 0) { xl = x.upper(); xu = x.lower(); }
  else               { xl = x.lower(); xu = x.upper(); }
  if (x.upper() < 0)      { yl = y.lower(); yu = y.upper(); }
  else if (x.lower() < 0) { yl = (y.upper()<0)?y.upper():y.lower();
                             yu = (y.upper()<0)?y.upper():y.lower(); }
  else                     { yl = y.upper(); yu = y.lower(); }
  return I(rnd.div_down(xl, yl), rnd.div_up(xu, yu));
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> div_positive(interval_gpu<T> const &x, T const &yu) {
  if (x.lower()==0 && x.upper()==0) return x;
  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl = x.lower(), &xu = x.upper();
  if      (xu < 0) return I(rnd.neg_inf(), rnd.div_up  (xu, yu));
  else if (xl < 0) return I(rnd.neg_inf(), rnd.pos_inf());
  else             return I(rnd.div_down(xl, yu), rnd.pos_inf());
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> div_negative(interval_gpu<T> const &x, T const &yl) {
  if (x.lower()==0 && x.upper()==0) return x;
  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl = x.lower(), &xu = x.upper();
  if      (xu < 0) return I(rnd.div_down(xu, yl), rnd.pos_inf());
  else if (xl < 0) return I(rnd.neg_inf(), rnd.pos_inf());
  else             return I(rnd.neg_inf(), rnd.div_up(xl, yl));
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> div_zero_part1(interval_gpu<T> const &x,
                                                        interval_gpu<T> const &y,
                                                        bool &b) {
  if (x.lower()==0 && x.upper()==0) { b = false; return x; }
  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl=x.lower(), &xu=x.upper(), &yl=y.lower(), &yu=y.upper();
  if      (xu < 0) { b=true;  return I(rnd.neg_inf(), rnd.div_up(xu, yu)); }
  else if (xl < 0) { b=false; return I(rnd.neg_inf(), rnd.pos_inf()); }
  else             { b=true;  return I(rnd.neg_inf(), rnd.div_up(xl, yl)); }
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> div_zero_part2(interval_gpu<T> const &x,
                                                        interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl=x.lower(), &xu=x.upper(), &yl=y.lower(), &yu=y.upper();
  if (xu < 0) return I(rnd.div_down(xu, yl), rnd.pos_inf());
  else        return I(rnd.div_down(xl, yu), rnd.pos_inf());
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> division_part1(interval_gpu<T> const &x,
                                                        interval_gpu<T> const &y,
                                                        bool &b) {
  b = false;
  if (y.lower() <= 0 && y.upper() >= 0) {
    if (y.lower() != 0) {
      if (y.upper() != 0) return div_zero_part1(x, y, b);
      else                 return div_negative(x, y.lower());
    } else if (y.upper() != 0) {
      return div_positive(x, y.upper());
    } else {
      return interval_gpu<T>::empty();
    }
  }
  return div_non_zero(x, y);
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> division_part2(interval_gpu<T> const &x,
                                                        interval_gpu<T> const &y,
                                                        bool b = true) {
  if (!b) return interval_gpu<T>::empty();
  return div_zero_part2(x, y);
}

template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> square(interval_gpu<T> const &x) {
  typedef interval_gpu<T> I;
  rounded_arith<T> rnd;
  const T &xl = x.lower(), &xu = x.upper();
  if      (xl >= 0) return I(rnd.mul_down(xl,xl), rnd.mul_up(xu,xu));
  else if (xu <= 0) return I(rnd.mul_down(xu,xu), rnd.mul_up(xl,xl));
  else              return I(static_cast<T>(0),
                             rnd.max(rnd.mul_up(xl,xl), rnd.mul_up(xu,xu)));
}

#endif // KOKKOS_INTERVAL_LIB_H
