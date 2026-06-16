/* The MIT License
   Copyright (c) 2011 by Attractive Chaos <attractor@live.co.uk>
   Kokkos port
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ---- read_data inline (from extend2-sycl/read_data.cpp) ----
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>

struct extend2_dat {
  int qlen;
  unsigned char *query;
  int tlen;
  unsigned char *target;
  int m;
  char mat[25];
  int o_del, e_del, o_ins, e_ins;
  int w, end_bonus, zdrop, h0;
  int qle, tle, gtle, gscore, max_off, score;
  uint64_t tsc;
  double sec;
};

static int read_check(int fd, void *buf, size_t sz) {
  return read(fd, buf, sz) == (ssize_t)sz;
}

int read_data(const char *fn, struct extend2_dat *d) {
  if (!d) return 0;
  int fd = open(fn, O_RDONLY);
  if (fd < 0) { perror("Cannot open file"); return 0; }
  assert(read_check(fd, &d->qlen,      sizeof(int)));
  d->query = (unsigned char*)malloc(d->qlen);
  assert(read_check(fd, d->query,      d->qlen));
  assert(read_check(fd, &d->tlen,      sizeof(int)));
  d->target = (unsigned char*)malloc(d->tlen);
  assert(read_check(fd, d->target,     d->tlen));
  assert(read_check(fd, &d->m,         sizeof(int)));
  assert(read_check(fd, d->mat,        25));
  assert(read_check(fd, &d->o_del,     sizeof(int)));
  assert(read_check(fd, &d->e_del,     sizeof(int)));
  assert(read_check(fd, &d->o_ins,     sizeof(int)));
  assert(read_check(fd, &d->e_ins,     sizeof(int)));
  assert(read_check(fd, &d->w,         sizeof(int)));
  assert(read_check(fd, &d->end_bonus, sizeof(int)));
  assert(read_check(fd, &d->zdrop,     sizeof(int)));
  assert(read_check(fd, &d->h0,        sizeof(int)));
  assert(read_check(fd, &d->qle,       sizeof(int)));
  assert(read_check(fd, &d->tle,       sizeof(int)));
  assert(read_check(fd, &d->gtle,      sizeof(int)));
  assert(read_check(fd, &d->gscore,    sizeof(int)));
  assert(read_check(fd, &d->max_off,   sizeof(int)));
  assert(read_check(fd, &d->score,     sizeof(int)));
  assert(read_check(fd, &d->tsc,       sizeof(uint64_t)));
  assert(read_check(fd, &d->sec,       sizeof(double)));
  close(fd);
  return 1;
}
// ---- end read_data ----

typedef struct { int h, e; } eh_t;

static void check(int a, int b, const char *s) {
  if (a != b) printf("Error: %s %d %d\n", s, a, b);
}

// The extend2 DP algorithm is inherently sequential (each cell depends on
// previous). We run it on the host using Kokkos host execution while still
// using Kokkos Views for data management and timing infrastructure.
float run_extend2(struct extend2_dat *d)
{
  const int qlen      = d->qlen;
  const int tlen      = d->tlen;
  const int m         = d->m;
  const int o_del     = d->o_del;
  const int e_del     = d->e_del;
  const int o_ins     = d->o_ins;
  const int e_ins     = d->e_ins;
  int w               = d->w;
  const int end_bonus = d->end_bonus;
  const int zdrop     = d->zdrop;
  const int h0        = d->h0;

  // Host Views for inputs/outputs
  Kokkos::View<unsigned char*, Kokkos::HostSpace> query_v  ("query",  qlen);
  Kokkos::View<unsigned char*, Kokkos::HostSpace> target_v ("target", tlen);
  Kokkos::View<char*,          Kokkos::HostSpace> mat_v    ("mat",    m*m);
  Kokkos::View<eh_t*,          Kokkos::HostSpace> eh_v     ("eh",     qlen+1);
  Kokkos::View<char*,          Kokkos::HostSpace> qp_v     ("qp",     qlen*m);

  for (int i = 0; i < qlen; i++) query_v(i)  = d->query[i];
  for (int i = 0; i < tlen; i++) target_v(i) = d->target[i];
  for (int i = 0; i < m*m; i++) mat_v(i)    = d->mat[i];
  for (int i = 0; i <= qlen; i++) { eh_v(i).h = 0; eh_v(i).e = 0; }

  auto start = std::chrono::steady_clock::now();

  // Run single-threaded DP (no device parallelism possible for this DP)
  Kokkos::parallel_for("extend2_single",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, 1),
    [=](int) {
      unsigned char* query  = query_v.data();
      unsigned char* target = target_v.data();
      char*          mat    = mat_v.data();
      eh_t*          eh     = eh_v.data();
      char*          qp     = qp_v.data();

      int oe_del = o_del + e_del;
      int oe_ins = o_ins + e_ins;
      int i, j, k;
      int beg, end_v;
      int max, max_i, max_j, max_ins, max_del, max_ie;
      int gscore, max_off;
      int ww = w; // mutable local copy of band width

      // generate query profile
      for (k = i = 0; k < m; ++k) {
        char *p = mat + k * m;
        for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
      }

      eh[0].h = h0;
      eh[1].h = h0 > oe_ins ? h0 - oe_ins : 0;
      for (j = 2; j <= qlen && eh[j-1].h > e_ins; ++j)
        eh[j].h = eh[j-1].h - e_ins;

      k = m * m;
      for (i = 0, max = 0; i < k; ++i)
        max = max > mat[i] ? max : mat[i];

      max_ins = (int)((float)(qlen * max + end_bonus - o_ins) / e_ins + 1.f);
      max_ins = max_ins > 1 ? max_ins : 1;
      ww = ww < max_ins ? ww : max_ins;
      max_del = (int)((float)(qlen * max + end_bonus - o_del) / e_del + 1.f);
      max_del = max_del > 1 ? max_del : 1;
      ww = ww < max_del ? ww : max_del;

      max = h0; max_i = max_j = -1; max_ie = -1; gscore = -1;
      max_off = 0;
      beg = 0; end_v = qlen;

      for (i = 0; i < tlen; ++i) {
        int t, f = 0, h1, mm = 0, mj = -1;
        char *q = qp + target[i] * qlen;

        if (beg < i - ww)    beg = i - ww;
        if (end_v > i + ww + 1) end_v = i + ww + 1;
        if (end_v > qlen)   end_v = qlen;

        if (beg == 0) {
          h1 = h0 - (o_del + e_del * (i + 1));
          if (h1 < 0) h1 = 0;
        } else h1 = 0;

        for (j = beg; j < end_v; ++j) {
          eh_t *p = eh + j;
          int h, M = p->h, e = p->e;
          p->h = h1;
          M = M ? M + q[j] : 0;
          h = M > e ? M : e;
          h = h > f ? h : f;
          h1 = h;
          mj = mm > h ? mj : j;
          mm = mm > h ? mm : h;
          t = M - oe_del; t = t > 0 ? t : 0;
          e -= e_del; e = e > t ? e : t; p->e = e;
          t = M - oe_ins; t = t > 0 ? t : 0;
          f -= e_ins; f = f > t ? f : t;
        }
        eh[end_v].h = h1; eh[end_v].e = 0;
        if (j == qlen) {
          max_ie  = gscore > h1 ? max_ie  : i;
          gscore  = gscore > h1 ? gscore  : h1;
        }
        if (mm == 0) break;
        if (mm > max) {
          max = mm; max_i = i; max_j = mj;
          max_off = max_off > abs(mj - i) ? max_off : abs(mj - i);
        } else if (zdrop > 0) {
          bool stop = false;
          if (i - max_i > mj - max_j) {
            if (max - mm - ((i - max_i) - (mj - max_j)) * e_del > zdrop) stop = true;
          } else {
            if (max - mm - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) stop = true;
          }
          if (stop) break;
        }
        for (j = beg; j < end_v && eh[j].h == 0 && eh[j].e == 0; ++j);
        beg = j;
        for (j = end_v; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j);
        end_v = j + 2 < qlen ? j + 2 : qlen;
      }

      // store results back via aliases (captured by value, work on ptr)
      // We write results into the first eh slot (repurposed as output slot)
      // using a trick: store in qp_v first 6 ints (enough space)
      int* res = (int*)qp;
      res[0] = max_j + 1;
      res[1] = max_i + 1;
      res[2] = max_ie + 1;
      res[3] = gscore;
      res[4] = max_off;
      res[5] = max;
    });
  Kokkos::fence();

  auto stop = std::chrono::steady_clock::now();
  int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

  // Read back results from qp (repurposed)
  int* res = (int*)qp_v.data();
  check(d->qle,    res[0], "qle");
  check(d->tle,    res[1], "tle");
  check(d->gtle,   res[2], "gtle");
  check(d->gscore, res[3], "gscore");
  check(d->max_off,res[4], "max_off");
  check(d->score,  res[5], "score");

  return (float)ns;
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    struct extend2_dat d;

    const char* files[] = {
#include "filelist.txt"
    };

    float time = 0.f;
    for (int f = 0; f < repeat; f++) {
      read_data(files[f % 17], &d);
      time += run_extend2(&d);
      free(d.query);
      free(d.target);
    }
    printf("Average offload time %f (us)\n", (time * 1e-3f) / repeat);
  }
  Kokkos::finalize();
  return 0;
}
