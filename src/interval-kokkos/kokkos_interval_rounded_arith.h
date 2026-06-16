/*
 * Kokkos-portable rounded arithmetic for interval Newton benchmark.
 * Adapted from interval-omp/gpu_interval_rounded_arith.h:
 *   - replaced `inline` member functions with KOKKOS_INLINE_FUNCTION
 *   - replaced std::numeric_limits / nanf() with bit-pattern equivalents
 *     that compile in CUDA device code
 */

#ifndef KOKKOS_INTERVAL_ROUNDED_ARITH_H
#define KOKKOS_INTERVAL_ROUNDED_ARITH_H

#include <Kokkos_Core.hpp>
#include <cmath>

template <class T>
struct rounded_arith {
  KOKKOS_INLINE_FUNCTION T add_down(const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T add_up  (const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T sub_down(const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T sub_up  (const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T mul_down(const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T mul_up  (const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T div_down(const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T div_up  (const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T median  (const T &x, const T &y);
  KOKKOS_INLINE_FUNCTION T sqrt_down(const T &x);
  KOKKOS_INLINE_FUNCTION T sqrt_up  (const T &x);
  KOKKOS_INLINE_FUNCTION T int_down (const T &x);
  KOKKOS_INLINE_FUNCTION T int_up   (const T &x);
  KOKKOS_INLINE_FUNCTION T pos_inf();
  KOKKOS_INLINE_FUNCTION T neg_inf();
  KOKKOS_INLINE_FUNCTION T nan();
  KOKKOS_INLINE_FUNCTION T min(T const &x, T const &y);
  KOKKOS_INLINE_FUNCTION T max(T const &x, T const &y);
};

// ── float specialisation ──────────────────────────────────────────────────────
template <>
struct rounded_arith<float> {
  KOKKOS_INLINE_FUNCTION float add_down(const float &x, const float &y) { return x + y; }
  KOKKOS_INLINE_FUNCTION float add_up  (const float &x, const float &y) { return x + y; }
  KOKKOS_INLINE_FUNCTION float sub_down(const float &x, const float &y) { return x + (-y); }
  KOKKOS_INLINE_FUNCTION float sub_up  (const float &x, const float &y) { return x + (-y); }
  KOKKOS_INLINE_FUNCTION float mul_down(const float &x, const float &y) { return x * y; }
  KOKKOS_INLINE_FUNCTION float mul_up  (const float &x, const float &y) { return x * y; }
  KOKKOS_INLINE_FUNCTION float div_down(const float &x, const float &y) { return x / y; }
  KOKKOS_INLINE_FUNCTION float div_up  (const float &x, const float &y) { return x / y; }
  KOKKOS_INLINE_FUNCTION float median  (const float &x, const float &y) { return (x + y) * .5f; }
  KOKKOS_INLINE_FUNCTION float sqrt_down(const float &x) { return sqrtf(x); }
  KOKKOS_INLINE_FUNCTION float sqrt_up  (const float &x) { return sqrtf(x); }
  KOKKOS_INLINE_FUNCTION float int_down (const float &x) { return floorf(x); }
  KOKKOS_INLINE_FUNCTION float int_up   (const float &x) { return ceilf(x); }
  KOKKOS_INLINE_FUNCTION float min(float const &x, float const &y) { return fminf(x, y); }
  KOKKOS_INLINE_FUNCTION float max(float const &x, float const &y) { return fmaxf(x, y); }

  KOKKOS_INLINE_FUNCTION float pos_inf() {
    // IEEE 754 bit pattern for +infinity
    union { unsigned int i; float f; } u; u.i = 0x7F800000u; return u.f;
  }
  KOKKOS_INLINE_FUNCTION float neg_inf() {
    union { unsigned int i; float f; } u; u.i = 0xFF800000u; return u.f;
  }
  KOKKOS_INLINE_FUNCTION float nan() {
    // IEEE 754 quiet NaN
    union { unsigned int i; float f; } u; u.i = 0x7FC00000u; return u.f;
  }
};

// ── double specialisation ─────────────────────────────────────────────────────
template <>
struct rounded_arith<double> {
  KOKKOS_INLINE_FUNCTION double add_down(const double &x, const double &y) { return x + y; }
  KOKKOS_INLINE_FUNCTION double add_up  (const double &x, const double &y) { return x + y; }
  KOKKOS_INLINE_FUNCTION double sub_down(const double &x, const double &y) { return x + (-y); }
  KOKKOS_INLINE_FUNCTION double sub_up  (const double &x, const double &y) { return x + (-y); }
  KOKKOS_INLINE_FUNCTION double mul_down(const double &x, const double &y) { return x * y; }
  KOKKOS_INLINE_FUNCTION double mul_up  (const double &x, const double &y) { return x * y; }
  KOKKOS_INLINE_FUNCTION double div_down(const double &x, const double &y) { return x / y; }
  KOKKOS_INLINE_FUNCTION double div_up  (const double &x, const double &y) { return x / y; }
  KOKKOS_INLINE_FUNCTION double median  (const double &x, const double &y) { return (x + y) * .5; }
  KOKKOS_INLINE_FUNCTION double sqrt_down(const double &x) { return sqrt(x); }
  KOKKOS_INLINE_FUNCTION double sqrt_up  (const double &x) { return sqrt(x); }
  KOKKOS_INLINE_FUNCTION double int_down (const double &x) { return floor(x); }
  KOKKOS_INLINE_FUNCTION double int_up   (const double &x) { return ceil(x); }
  KOKKOS_INLINE_FUNCTION double min(double const &x, double const &y) { return fmin(x, y); }
  KOKKOS_INLINE_FUNCTION double max(double const &x, double const &y) { return fmax(x, y); }

  KOKKOS_INLINE_FUNCTION double pos_inf() {
    union { unsigned long long i; double d; } u; u.i = 0x7FF0000000000000ULL; return u.d;
  }
  KOKKOS_INLINE_FUNCTION double neg_inf() {
    union { unsigned long long i; double d; } u; u.i = 0xFFF0000000000000ULL; return u.d;
  }
  KOKKOS_INLINE_FUNCTION double nan() {
    union { unsigned long long i; double d; } u; u.i = 0x7FF8000000000000ULL; return u.d;
  }
};

#endif // KOKKOS_INTERVAL_ROUNDED_ARITH_H
