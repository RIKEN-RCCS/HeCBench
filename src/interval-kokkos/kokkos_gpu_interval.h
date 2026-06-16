/*
 * Kokkos-portable version of interval-omp/gpu_interval.h.
 * local_stack, global_stack, f, fd, helper predicates, and both
 * Newton interval solvers are annotated with KOKKOS_INLINE_FUNCTION so
 * they compile correctly for all Kokkos backends (Serial, OpenMP, CUDA, HIP).
 */

#ifndef KOKKOS_GPU_INTERVAL_H
#define KOKKOS_GPU_INTERVAL_H

#include "kokkos_interval_lib.h"
#include "../interval-omp/interval.h"   // THREADS, DEPTH_RESULT, T, etc.

// ── Stack in thread-private local memory ─────────────────────────────────────
template <class T, int N>
class local_stack {
 private:
  T   buf[N];
  int tos;
 public:
  KOKKOS_INLINE_FUNCTION local_stack() : tos(-1) {}
  KOKKOS_INLINE_FUNCTION T const &top() const { return buf[tos]; }
  KOKKOS_INLINE_FUNCTION T       &top()       { return buf[tos]; }
  KOKKOS_INLINE_FUNCTION void push(T const &v) { buf[++tos] = v; }
  KOKKOS_INLINE_FUNCTION T    pop()            { return buf[tos--]; }
  KOKKOS_INLINE_FUNCTION bool full()  const    { return tos == (N - 1); }
  KOKKOS_INLINE_FUNCTION bool empty() const    { return tos == -1; }
};

// ── Stack in global memory (interleaved by thread_id) ────────────────────────
template <class T, int N, int NTHREADS>
class global_stack {
 private:
  T   *buf;
  int  free_index;
 public:
  KOKKOS_INLINE_FUNCTION global_stack(T *buf_, int thread_id)
      : buf(buf_), free_index(thread_id) {}

  KOKKOS_INLINE_FUNCTION void push(T const &v) {
    buf[free_index] = v;
    free_index += NTHREADS;
  }
  KOKKOS_INLINE_FUNCTION T pop() {
    free_index -= NTHREADS;
    return buf[free_index];
  }
  KOKKOS_INLINE_FUNCTION bool full()  const { return free_index >= N * NTHREADS; }
  KOKKOS_INLINE_FUNCTION bool empty() const { return free_index < NTHREADS; }
  KOKKOS_INLINE_FUNCTION int  size()  const { return free_index / NTHREADS; }
};

// ── Function whose roots we seek ─────────────────────────────────────────────
template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> f(interval_gpu<T> const &x, int thread_id) {
  typedef interval_gpu<T> I;
  T alpha = -T(thread_id) / T(THREADS);
  return square(x - I(1)) + I(alpha) * x;
}

// ── First derivative ──────────────────────────────────────────────────────────
template <class T>
KOKKOS_INLINE_FUNCTION interval_gpu<T> fd(interval_gpu<T> const &x, int thread_id) {
  typedef interval_gpu<T> I;
  T alpha = -T(thread_id) / T(THREADS);
  return I(2) * x + I(alpha - 2);
}

// ── Is this interval narrow enough to stop? ──────────────────────────────────
template <class T>
KOKKOS_INLINE_FUNCTION bool is_minimal(interval_gpu<T> const &x, int thread_id) {
  T const eps_x = 1e-6f;
  T const eps_y = 1e-6f;
  return !empty(x) && (width(x) <= eps_x * fabs(median(x)) ||
                       width(f(x, thread_id)) <= eps_y);
}

// ── Should we bisect? ─────────────────────────────────────────────────────────
template <class T>
KOKKOS_INLINE_FUNCTION bool should_bisect(interval_gpu<T> const &x,
                                           interval_gpu<T> const &x1,
                                           interval_gpu<T> const &x2, T alpha) {
  T wmax = alpha * width(x);
  return (!empty(x1) && width(x1) > wmax) || (!empty(x2) && width(x2) > wmax);
}

// ── Optimised Newton interval (keep top-of-stack in registers) ───────────────
template <class T, int NTHREADS, int DEPTH_RES>
KOKKOS_INLINE_FUNCTION void newton_interval(
    global_stack<interval_gpu<T>, DEPTH_RES, NTHREADS> &result,
    interval_gpu<T> const &ix0, int thread_id)
{
  typedef interval_gpu<T> I;
  int const DEPTH_WORK = 128;
  T   const alpha      = T(.99);

  local_stack<I, DEPTH_WORK> work;
  I ix = ix0;

  while (true) {
    T  x  = median(ix);
    I  iq = f(I(x), thread_id);
    I  id = fd(ix,  thread_id);

    bool has_part2;
    I part1, part2 = I::empty();
    part1 = division_part1(iq, id, has_part2);
    part1 = intersect(I(x) - part1, ix);
    if (has_part2) {
      part2 = division_part2(iq, id);
      part2 = intersect(I(x) - part2, ix);
    }

    if (is_minimal(part1, thread_id)) { result.push(part1); part1 = I::empty(); }
    if (has_part2 && is_minimal(part2, thread_id)) { result.push(part2); part2 = I::empty(); }

    if (should_bisect(ix, part1, part2, alpha)) {
      part1 = I(ix.lower(), x);
      part2 = I(x, ix.upper());
      has_part2 = true;
    }

    if (!empty(part1)) {
      ix = part1;
      if (has_part2 && !empty(part2)) work.push(part2);
    } else if (has_part2 && !empty(part2)) {
      ix = part2;
    } else {
      if (work.empty()) break;
      ix = work.pop();
    }
  }
}

// ── Naive Newton interval (no register optimisation) ─────────────────────────
template <class T, int NTHREADS, int DEPTH_RES>
KOKKOS_INLINE_FUNCTION void newton_interval_naive(
    global_stack<interval_gpu<T>, DEPTH_RES, NTHREADS> &result,
    interval_gpu<T> const &ix0, int thread_id)
{
  typedef interval_gpu<T> I;
  int const DEPTH_WORK = 128;
  T   const alpha      = T(.99);

  local_stack<I, DEPTH_WORK> work;
  work.push(ix0);

  while (!work.empty()) {
    I ix = work.pop();

    if (is_minimal(ix, thread_id)) {
      result.push(ix);
    } else {
      T  x  = median(ix);
      I  iq = f(I(x), thread_id);
      I  id = fd(ix,  thread_id);

      bool has_part2;
      I part1, part2 = I::empty();
      part1 = division_part1(iq, id, has_part2);
      part1 = intersect(I(x) - part1, ix);
      if (has_part2) {
        part2 = division_part2(iq, id);
        part2 = intersect(I(x) - part2, ix);
      }

      if (should_bisect(ix, part1, part2, alpha)) {
        part1 = I(ix.lower(), x);
        part2 = I(x, ix.upper());
        has_part2 = true;
      }

      if (!empty(part1))               work.push(part1);
      if (has_part2 && !empty(part2))  work.push(part2);
    }
  }
}

#endif // KOKKOS_GPU_INTERVAL_H
