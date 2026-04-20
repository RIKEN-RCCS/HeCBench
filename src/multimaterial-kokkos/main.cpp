// Multi-material pressure/density benchmark - Kokkos port
// Ported from multimaterial-omp.
// Usage: ./main <sizex> <sizey>   (e.g. ./main 2000 2000)

#include <Kokkos_Core.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>

// ---------------------------------------------------------------------------
// Copy helpers
// ---------------------------------------------------------------------------
template <typename T>
static void h2d(Kokkos::View<T *> &dst, T *src, int n)
{
  auto m = Kokkos::create_mirror_view(dst);
  memcpy(m.data(), src, (size_t)n * sizeof(T));
  Kokkos::deep_copy(dst, m);
}
template <typename T>
static void d2h(T *dst, Kokkos::View<T *> &src, int n)
{
  auto m = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(m, src);
  memcpy(dst, m.data(), (size_t)n * sizeof(T));
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct full_data {
  int sizex, sizey, Nmats;
  double *rho, *rho_mat_ave, *p, *Vf, *t, *V, *x, *y, *n, *rho_ave;
};
struct compact_data {
  int sizex, sizey, Nmats;
  double *rho_compact, *rho_compact_list;
  double *rho_mat_ave_compact, *rho_mat_ave_compact_list;
  double *p_compact, *p_compact_list;
  double *Vf_compact_list;
  double *t_compact, *t_compact_list;
  double *V, *x, *y, *n;
  double *rho_ave_compact;
  int *imaterial, *matids, *nextfrac, *mmc_index, *mmc_i, *mmc_j;
  int mm_len, mmc_cells;
};

// ---------------------------------------------------------------------------
// Initialisation (host-only, random layout)
// ---------------------------------------------------------------------------
static void initialise_field_rand(full_data &cc,
                                  double prob2, double prob3, double prob4)
{
  int sizex = cc.sizex, sizey = cc.sizey, Nmats = cc.Nmats;
  srand(0);
  for (int n = 0; n < sizex * sizey; n++) {
    int    i  = n % sizex, j = n / sizex;
    double r  = (double)rand() / RAND_MAX;
    int    m  = (int)((double)rand() / RAND_MAX * Nmats / 4) +
               (Nmats / 4) * (n / (sizex * sizey / 4));
    cc.rho[(i + sizex * j) * Nmats + m] =
    cc.t  [(i + sizex * j) * Nmats + m] =
    cc.p  [(i + sizex * j) * Nmats + m] = 1.0;
    if (r >= 1.0 - prob4 - prob3 - prob2) {
      int m2 = (int)((double)rand() / RAND_MAX * Nmats / 4) +
               (Nmats / 4) * (n / (sizex * sizey / 4));
      while (m2 == m)
        m2 = (int)((double)rand() / RAND_MAX * Nmats / 4) +
             (Nmats / 4) * (n / (sizex * sizey / 4));
      cc.rho[(i + sizex * j) * Nmats + m2] =
      cc.t  [(i + sizex * j) * Nmats + m2] =
      cc.p  [(i + sizex * j) * Nmats + m2] = 1.0;
    }
    if (r >= 1.0 - prob4 - prob3) {
      int m3 = (int)((double)rand() / RAND_MAX * Nmats / 4) +
               (Nmats / 4) * (n / (sizex * sizey / 4));
      cc.rho[(i + sizex * j) * Nmats + m3] =
      cc.t  [(i + sizex * j) * Nmats + m3] =
      cc.p  [(i + sizex * j) * Nmats + m3] = 1.0;
    }
    if (r >= 1.0 - prob4) {
      int m4 = (int)((double)rand() / RAND_MAX * Nmats / 4) +
               (Nmats / 4) * (n / (sizex * sizey / 4));
      cc.rho[(i + sizex * j) * Nmats + m4] =
      cc.t  [(i + sizex * j) * Nmats + m4] =
      cc.p  [(i + sizex * j) * Nmats + m4] = 1.0;
    }
  }
}

// ---------------------------------------------------------------------------
// Full-matrix cell-centric (Kokkos)
// ---------------------------------------------------------------------------
static void full_matrix_cell_centric(full_data &cc)
{
  int sizex = cc.sizex, sizey = cc.sizey, Nmats = cc.Nmats;
  int ncells = sizex * sizey;
  long NM = (long)ncells * Nmats;

  Kokkos::View<double *> d_rho("rho", NM), d_rma("rma", NM), d_p("p", NM),
      d_Vf("Vf", NM), d_t("t", NM);
  Kokkos::View<double *> d_V("V", ncells), d_x("x", ncells), d_y("y", ncells),
      d_n("n", Nmats), d_rho_ave("rho_ave", ncells);

  h2d(d_rho, cc.rho,         (int)NM); h2d(d_rma, cc.rho_mat_ave, (int)NM);
  h2d(d_p,   cc.p,           (int)NM); h2d(d_Vf,  cc.Vf,          (int)NM);
  h2d(d_t,   cc.t,           (int)NM);
  h2d(d_V, cc.V, ncells); h2d(d_x, cc.x, ncells); h2d(d_y, cc.y, ncells);
  h2d(d_n, cc.n, Nmats);

  auto t0 = std::chrono::system_clock::now();
  Kokkos::parallel_for("cc1", ncells, KOKKOS_LAMBDA(int cell) {
    double ave = 0.0;
    for (int mat = 0; mat < Nmats; mat++)
      if (d_Vf(cell * Nmats + mat) > 0.0)
        ave += d_rho(cell * Nmats + mat) * d_Vf(cell * Nmats + mat);
    d_rho_ave(cell) = ave / d_V(cell);
  });
  Kokkos::fence();
  printf("Full matrix, cell centric, alg 1: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  t0 = std::chrono::system_clock::now();
  Kokkos::parallel_for("cc2", ncells, KOKKOS_LAMBDA(int cell) {
    for (int mat = 0; mat < Nmats; mat++) {
      if (d_Vf(cell * Nmats + mat) > 0.0)
        d_p(cell * Nmats + mat) =
            (d_n(mat) * d_rho(cell * Nmats + mat) * d_t(cell * Nmats + mat)) /
            d_Vf(cell * Nmats + mat);
      else
        d_p(cell * Nmats + mat) = 0.0;
    }
  });
  Kokkos::fence();
  printf("Full matrix, cell centric, alg 2: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  t0 = std::chrono::system_clock::now();
  Kokkos::parallel_for(
      "cc3", (sizex - 2) * (sizey - 2), KOKKOS_LAMBDA(int idx) {
        int i = 1 + idx % (sizex - 2), j = 1 + idx / (sizex - 2);
        int cell = i + sizex * j;
        double xo = d_x(cell), yo = d_y(cell);
        double dsqr[9];
        for (int nj = -1; nj <= 1; nj++)
          for (int ni = -1; ni <= 1; ni++) {
            double xi = d_x((i+ni) + sizex*(j+nj));
            double yi = d_y((i+ni) + sizex*(j+nj));
            dsqr[(nj+1)*3+(ni+1)] = (xo-xi)*(xo-xi)+(yo-yi)*(yo-yi);
          }
        for (int mat = 0; mat < Nmats; mat++) {
          if (d_Vf(cell * Nmats + mat) > 0.0) {
            double rs = 0.0; int Nn = 0;
            for (int nj = -1; nj <= 1; nj++) {
              if (j+nj < 0 || j+nj >= sizey) continue;
              for (int ni = -1; ni <= 1; ni++) {
                if (i+ni < 0 || i+ni >= sizex) continue;
                int nc = (i+ni)+sizex*(j+nj);
                if (d_Vf(nc*Nmats+mat) > 0.0) {
                  rs += d_rho(nc*Nmats+mat) / dsqr[(nj+1)*3+(ni+1)]; Nn++;
                }
              }
            }
            d_rma(cell*Nmats+mat) = rs / Nn;
          } else {
            d_rma(cell*Nmats+mat) = 0.0;
          }
        }
      });
  Kokkos::fence();
  printf("Full matrix, cell centric, alg 3: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  d2h(cc.rho_ave,     d_rho_ave, ncells);
  d2h(cc.p,           d_p,       (int)NM);
  d2h(cc.rho_mat_ave, d_rma,     (int)NM);
}

// ---------------------------------------------------------------------------
// Full-matrix material-centric (Kokkos)
// ---------------------------------------------------------------------------
static void full_matrix_material_centric(full_data &cc, full_data &mc)
{
  int sizex = mc.sizex, sizey = mc.sizey, Nmats = mc.Nmats;
  int ncells = sizex * sizey;
  long NM = (long)ncells * Nmats;

  Kokkos::View<double *> d_rho("mc_rho", NM), d_rma("mc_rma", NM),
      d_p("mc_p", NM), d_Vf("mc_Vf", NM), d_t("mc_t", NM);
  Kokkos::View<double *> d_V("mc_V", ncells), d_x("mc_x", ncells),
      d_y("mc_y", ncells), d_n("mc_n", Nmats), d_rho_ave("mc_rho_ave", ncells);

  h2d(d_rho, mc.rho, (int)NM); h2d(d_rma, mc.rho_mat_ave, (int)NM);
  h2d(d_p,   mc.p,   (int)NM); h2d(d_Vf,  mc.Vf,          (int)NM);
  h2d(d_t,   mc.t,   (int)NM);
  h2d(d_V, mc.V, ncells); h2d(d_x, mc.x, ncells);
  h2d(d_y, mc.y, ncells); h2d(d_n, mc.n, Nmats);

  auto t0 = std::chrono::system_clock::now();
  Kokkos::deep_copy(d_rho_ave, 0.0);
  for (int mat = 0; mat < Nmats; mat++) {
    Kokkos::parallel_for("mc1_acc", ncells, KOKKOS_LAMBDA(int cell) {
      if (d_Vf((long)ncells*mat + cell) > 0.0)
        Kokkos::atomic_add(&d_rho_ave(cell),
            d_rho((long)ncells*mat+cell) * d_Vf((long)ncells*mat+cell));
    });
  }
  Kokkos::fence();
  Kokkos::parallel_for("mc1_div", ncells,
      KOKKOS_LAMBDA(int cell) { d_rho_ave(cell) /= d_V(cell); });
  Kokkos::fence();
  printf("Full matrix, material centric, alg 1: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  t0 = std::chrono::system_clock::now();
  Kokkos::parallel_for("mc2", NM, KOKKOS_LAMBDA(long idx) {
    int mat = (int)(idx / ncells), cell = (int)(idx % ncells);
    if (d_Vf((long)ncells*mat+cell) > 0.0)
      d_p((long)ncells*mat+cell) =
          (d_n(mat) * d_rho((long)ncells*mat+cell) *
           d_t((long)ncells*mat+cell)) /
          d_Vf((long)ncells*mat+cell);
    else
      d_p((long)ncells*mat+cell) = 0.0;
  });
  Kokkos::fence();
  printf("Full matrix, material centric, alg 2: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  t0 = std::chrono::system_clock::now();
  Kokkos::parallel_for(
      "mc3", (long)Nmats * (sizex-2) * (sizey-2), KOKKOS_LAMBDA(long idx) {
        int mat  = (int)(idx / ((sizex-2)*(sizey-2)));
        int flat = (int)(idx % ((sizex-2)*(sizey-2)));
        int i = 1 + flat % (sizex-2), j = 1 + flat / (sizex-2);
        int cell = i + sizex*j;
        if (d_Vf((long)ncells*mat+cell) > 0.0) {
          double xo = d_x(cell), yo = d_y(cell);
          double rs = 0.0; int Nn = 0;
          for (int nj=-1; nj<=1; nj++) {
            if (j+nj<0||j+nj>=sizey) continue;
            for (int ni=-1; ni<=1; ni++) {
              if (i+ni<0||i+ni>=sizex) continue;
              int nc=(i+ni)+sizex*(j+nj);
              if (d_Vf((long)ncells*mat+nc) > 0.0) {
                double xi=d_x(nc), yi=d_y(nc);
                double d=(xo-xi)*(xo-xi)+(yo-yi)*(yo-yi);
                rs += d_rho((long)ncells*mat+nc) / d; Nn++;
              }
            }
          }
          d_rma((long)ncells*mat+cell) = rs / Nn;
        } else {
          d_rma((long)ncells*mat+cell) = 0.0;
        }
      });
  Kokkos::fence();
  printf("Full matrix, material centric, alg 3: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  d2h(mc.rho_ave,     d_rho_ave, ncells);
  d2h(mc.p,           d_p,       (int)NM);
  d2h(mc.rho_mat_ave, d_rma,     (int)NM);
}

// ---------------------------------------------------------------------------
// Compact cell-centric (Kokkos, CSR layout)
// ---------------------------------------------------------------------------
static void compact_cell_centric(full_data &cc, compact_data &ccc)
{
  int sizex = ccc.sizex, sizey = ccc.sizey, Nmats = ccc.Nmats;
  int ncells = sizex * sizey;
  int mm_len = ccc.mm_len, mmc_cells = ccc.mmc_cells;

  Kokkos::View<int    *> d_imaterial("imaterial", ncells);
  Kokkos::View<int    *> d_matids("matids",        mm_len);
  Kokkos::View<int    *> d_mmc_index("mmc_index",  mmc_cells+1);
  Kokkos::View<int    *> d_mmc_i("mmc_i",          mmc_cells+1);
  Kokkos::View<int    *> d_mmc_j("mmc_j",          mmc_cells+1);
  Kokkos::View<double *> d_rho_c("rho_c",   ncells), d_rho_cl("rho_cl",   mm_len);
  Kokkos::View<double *> d_rma_c("rma_c",   ncells), d_rma_cl("rma_cl",   mm_len);
  Kokkos::View<double *> d_p_c("p_c",       ncells), d_p_cl("p_cl",       mm_len);
  Kokkos::View<double *> d_Vf_cl("Vf_cl",   mm_len);
  Kokkos::View<double *> d_t_c("t_c",       ncells), d_t_cl("t_cl",       mm_len);
  Kokkos::View<double *> d_V("ccc_V", ncells), d_x("ccc_x", ncells),
      d_y("ccc_y", ncells), d_n("ccc_n", Nmats);
  Kokkos::View<double *> d_rho_ave("rho_ave_c", ncells);

  h2d(d_imaterial,  ccc.imaterial,               ncells);
  h2d(d_matids,     ccc.matids,                   mm_len);
  h2d(d_mmc_index,  ccc.mmc_index,                mmc_cells+1);
  h2d(d_mmc_i,      ccc.mmc_i,                    mmc_cells+1);
  h2d(d_mmc_j,      ccc.mmc_j,                    mmc_cells+1);
  h2d(d_rho_c,      ccc.rho_compact,              ncells);
  h2d(d_rho_cl,     ccc.rho_compact_list,          mm_len);
  h2d(d_rma_c,      ccc.rho_mat_ave_compact,       ncells);
  h2d(d_rma_cl,     ccc.rho_mat_ave_compact_list,  mm_len);
  h2d(d_p_c,        ccc.p_compact,                ncells);
  h2d(d_p_cl,       ccc.p_compact_list,            mm_len);
  h2d(d_Vf_cl,      ccc.Vf_compact_list,           mm_len);
  h2d(d_t_c,        ccc.t_compact,                ncells);
  h2d(d_t_cl,       ccc.t_compact_list,            mm_len);
  h2d(d_V,  ccc.V, ncells); h2d(d_x, ccc.x, ncells);
  h2d(d_y,  ccc.y, ncells); h2d(d_n, ccc.n, Nmats);

  // ---- Loop 1: average density ----
  auto t0 = std::chrono::system_clock::now();
  // Pure cells
  Kokkos::parallel_for("ccc1_pure", ncells, KOKKOS_LAMBDA(int cell) {
    if (d_imaterial(cell) > 0)
      d_rho_ave(cell) = d_rho_c(cell) / d_V(cell);
  });
  Kokkos::fence();
  // Mixed cells (CSR)
  Kokkos::parallel_for("ccc1_mix", mmc_cells, KOKKOS_LAMBDA(int c) {
    double ave = 0.0;
    for (int m = d_mmc_index(c); m < d_mmc_index(c+1); m++)
      ave += d_rho_cl(m) * d_Vf_cl(m);
    d_rho_ave(d_mmc_i(c) + sizex * d_mmc_j(c)) =
        ave / d_V(d_mmc_i(c) + sizex * d_mmc_j(c));
  });
  Kokkos::fence();
  printf("Compact matrix, cell centric, alg 1: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  // ---- Loop 2: pressure ----
  t0 = std::chrono::system_clock::now();
  Kokkos::parallel_for("ccc2_pure", ncells, KOKKOS_LAMBDA(int cell) {
    int ix = d_imaterial(cell);
    if (ix > 0)
      d_p_c(cell) = d_n(ix-1) * d_rho_c(cell) * d_t_c(cell);
  });
  Kokkos::fence();
  Kokkos::parallel_for("ccc2_mix", mm_len, KOKKOS_LAMBDA(int idx) {
    int mat = d_matids(idx);
    d_p_cl(idx) = (d_n(mat) * d_rho_cl(idx) * d_t_cl(idx)) / d_Vf_cl(idx);
  });
  Kokkos::fence();
  printf("Compact matrix, cell centric, alg 2: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  // ---- Loop 3: neighbourhood average ----
  t0 = std::chrono::system_clock::now();
  Kokkos::parallel_for(
      "ccc3", (sizex-2)*(sizey-2), KOKKOS_LAMBDA(int flat) {
        int i = 1 + flat%(sizex-2), j = 1 + flat/(sizex-2);
        int cell = i + sizex*j;
        double xo = d_x(cell), yo = d_y(cell);
        double dsqr[9];
        for (int nj=-1; nj<=1; nj++)
          for (int ni=-1; ni<=1; ni++) {
            double xi=d_x((i+ni)+sizex*(j+nj)), yi=d_y((i+ni)+sizex*(j+nj));
            dsqr[(nj+1)*3+(ni+1)]=(xo-xi)*(xo-xi)+(yo-yi)*(yo-yi);
          }
        int ix = d_imaterial(cell);
        if (ix <= 0) {
          for (int lx=d_mmc_index(-ix); lx<d_mmc_index(-ix+1); lx++) {
            int mat=d_matids(lx); double rs=0.0; int Nn=0;
            for (int nj=-1; nj<=1; nj++)
              for (int ni=-1; ni<=1; ni++) {
                int ci=i+ni, cj=j+nj, nc=ci+sizex*cj;
                int jx=d_imaterial(nc);
                if (jx<=0) {
                  for (int jl=d_mmc_index(-jx); jl<d_mmc_index(-jx+1); jl++)
                    if (d_matids(jl)==mat) {
                      rs+=d_rho_cl(jl)/dsqr[(nj+1)*3+(ni+1)]; Nn++; break;
                    }
                } else if (jx-1==mat) {
                  rs+=d_rho_c(nc)/dsqr[(nj+1)*3+(ni+1)]; Nn++;
                }
              }
            d_rma_cl(lx) = rs / Nn;
          }
        } else {
          int mat=ix-1; double rs=0.0; int Nn=0;
          for (int nj=-1; nj<=1; nj++) {
            if (j+nj<0||j+nj>=sizey) continue;
            for (int ni=-1; ni<=1; ni++) {
              if (i+ni<0||i+ni>=sizex) continue;
              int ci=i+ni, cj=j+nj, nc=ci+sizex*cj;
              int jx=d_imaterial(nc);
              if (jx<=0) {
                for (int jl=d_mmc_index(-jx); jl<d_mmc_index(-jx+1); jl++)
                  if (d_matids(jl)==mat) {
                    rs+=d_rho_cl(jl)/dsqr[(nj+1)*3+(ni+1)]; Nn++; break;
                  }
              } else if (jx-1==mat) {
                rs+=d_rho_c(nc)/dsqr[(nj+1)*3+(ni+1)]; Nn++;
              }
            }
          }
          d_rma_c(cell) = rs / Nn;
        }
      });
  Kokkos::fence();
  printf("Compact matrix, cell centric, alg 3: %g msec\n",
         std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count() * 1000);

  d2h(ccc.rho_ave_compact,           d_rho_ave, ncells);
  d2h(ccc.p_compact,                 d_p_c,     ncells);
  d2h(ccc.p_compact_list,            d_p_cl,    mm_len);
  d2h(ccc.rho_mat_ave_compact,       d_rma_c,   ncells);
  d2h(ccc.rho_mat_ave_compact_list,  d_rma_cl,  mm_len);
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------
static bool full_matrix_check_results(full_data &cc, full_data &mc)
{
  int sizex=cc.sizex, sizey=cc.sizey, Nmats=cc.Nmats, ncells=sizex*sizey;
  for (int j=0; j<sizey; j++)
    for (int i=0; i<sizex; i++) {
      int cell=i+sizex*j;
      if (fabs(cc.rho_ave[cell]-mc.rho_ave[cell])>0.0001) {
        printf("rho_ave mismatch (%d,%d): %f %f\n",i,j,cc.rho_ave[cell],mc.rho_ave[cell]);
        return false;
      }
      for (int mat=0; mat<Nmats; mat++) {
        if (fabs(cc.p[cell*Nmats+mat]-mc.p[(long)ncells*mat+cell])>0.0001) {
          printf("p mismatch (%d,%d,m=%d)\n",i,j,mat); return false;
        }
        if (fabs(cc.rho_mat_ave[cell*Nmats+mat]-mc.rho_mat_ave[(long)ncells*mat+cell])>0.0001) {
          printf("rho_mat_ave mismatch (%d,%d,m=%d)\n",i,j,mat); return false;
        }
      }
    }
  return true;
}

static bool compact_check_results(full_data &cc, compact_data &ccc)
{
  int sizex=cc.sizex, sizey=cc.sizey, Nmats=cc.Nmats;
  for (int j=0; j<sizey; j++)
    for (int i=0; i<sizex; i++) {
      int cell=i+sizex*j;
      if (fabs(cc.rho_ave[cell]-ccc.rho_ave_compact[cell])>0.0001) {
        printf("rho_ave mismatch (%d,%d): %f %f\n",i,j,
               cc.rho_ave[cell],ccc.rho_ave_compact[cell]);
        return false;
      }
      int ix=ccc.imaterial[cell];
      if (ix<=0) {
        for (int lx=ccc.mmc_index[-ix]; lx<ccc.mmc_index[-ix+1]; lx++) {
          int mat=ccc.matids[lx];
          if (fabs(cc.p[cell*Nmats+mat]-ccc.p_compact_list[lx])>0.0001) {
            printf("p mismatch (%d,%d,m=%d)\n",i,j,mat); return false;
          }
          if (fabs(cc.rho_mat_ave[cell*Nmats+mat]-ccc.rho_mat_ave_compact_list[lx])>0.0001) {
            printf("rho_mat_ave mismatch (%d,%d,m=%d)\n",i,j,mat); return false;
          }
        }
      } else {
        int mat=ix-1;
        if (fabs(cc.p[cell*Nmats+mat]-ccc.p_compact[cell])>0.0001) {
          printf("p mismatch (%d,%d,m=%d)\n",i,j,mat); return false;
        }
        if (fabs(cc.rho_mat_ave[cell*Nmats+mat]-ccc.rho_mat_ave_compact[cell])>0.0001) {
          printf("rho_mat_ave mismatch (%d,%d,m=%d)\n",i,j,mat); return false;
        }
      }
    }
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int sizex=1000, sizey=1000;
  if (argc>1) sizex=atoi(argv[1]);
  if (argc>2) sizey=atoi(argv[2]);
  int ncells=sizex*sizey, Nmats=50;
  long NM=(long)ncells*Nmats;

  full_data cc, mc; compact_data ccc;
  cc.sizex=mc.sizex=ccc.sizex=sizex;
  cc.sizey=mc.sizey=ccc.sizey=sizey;
  cc.Nmats=mc.Nmats=ccc.Nmats=Nmats;

  cc.rho=(double*)calloc(NM,8); cc.rho_mat_ave=(double*)calloc(NM,8);
  cc.p  =(double*)calloc(NM,8); cc.Vf         =(double*)calloc(NM,8);
  cc.t  =(double*)calloc(NM,8);
  mc.rho=(double*)malloc(NM*8); mc.rho_mat_ave=(double*)calloc(NM,8);
  mc.p  =(double*)malloc(NM*8); mc.Vf         =(double*)malloc(NM*8);
  mc.t  =(double*)malloc(NM*8);
  cc.V=(double*)malloc(ncells*8); cc.x=(double*)malloc(ncells*8);
  cc.y=(double*)malloc(ncells*8); cc.n=(double*)malloc(Nmats*8);
  cc.rho_ave=(double*)malloc(ncells*8); mc.rho_ave=(double*)malloc(ncells*8);
  ccc.rho_ave_compact=(double*)malloc(ncells*8);
  ccc.rho_compact=(double*)malloc(ncells*8);
  ccc.rho_mat_ave_compact=(double*)calloc(ncells,8);
  ccc.p_compact=(double*)malloc(ncells*8); ccc.t_compact=(double*)malloc(ncells*8);
  int *nmats_per_cell=(int*)malloc(ncells*4);
  ccc.imaterial=(int*)malloc(ncells*4);
  int list_size=(int)(ceil((double)sizex/1000)*ceil((double)sizey/1000)*49000*2+600*3+400*4+16);
  ccc.nextfrac=(int*)malloc(list_size*4);
  int *frac2cell=(int*)malloc(list_size*4);
  ccc.matids=(int*)malloc(list_size*4);
  ccc.mmc_index=(int*)malloc(list_size*4);
  ccc.mmc_i=(int*)malloc(list_size*4); ccc.mmc_j=(int*)malloc(list_size*4);
  ccc.Vf_compact_list=(double*)malloc(list_size*8);
  ccc.rho_compact_list=(double*)malloc(list_size*8);
  ccc.rho_mat_ave_compact_list=(double*)calloc(list_size,8);
  ccc.t_compact_list=(double*)malloc(list_size*8);
  ccc.p_compact_list=(double*)malloc(list_size*8);

  double dx=1.0/sizex, dy=1.0/sizey;
  for (int j=0; j<sizey; j++)
    for (int i=0; i<sizex; i++) {
      cc.V[i+j*sizex]=dx*dy; cc.x[i+j*sizex]=dx*i; cc.y[i+j*sizex]=dy*j;
    }
  for (int m=0; m<Nmats; m++) cc.n[m]=1.0;
  ccc.V=mc.V=cc.V; ccc.x=mc.x=cc.x; ccc.y=mc.y=cc.y; ccc.n=mc.n=cc.n;

  initialise_field_rand(cc, 0.1, 0.05, 0.02);

  // Count materials, set Vf
  int cell_counts[4]={0,0,0,0}; ccc.mmc_cells=0;
  for (int j=0; j<sizey; j++)
    for (int i=0; i<sizex; i++) {
      int cell=i+sizex*j, count=0;
      for (int mat=0; mat<Nmats; mat++) count+=(cc.rho[cell*Nmats+mat]!=0.0)?1:0;
      if (!count) { cc.rho[cell*Nmats+1]=cc.t[cell*Nmats+1]=cc.p[cell*Nmats+1]=1.0; count=1; }
      if (count>1) ccc.mmc_cells++;
      cell_counts[std::min(count,4)-1]++;
      for (int mat=0; mat<Nmats; mat++)
        if (cc.rho[cell*Nmats+mat]!=0.0) cc.Vf[cell*Nmats+mat]=1.0/count;
    }
  printf("Pure %d, 2-mat %d, 3-mat %d, 4-mat %d: MMC %d\n",
         cell_counts[0],cell_counts[1],cell_counts[2],cell_counts[3],ccc.mmc_cells);

  // Material-centric conversion
  for (int j=0; j<sizey; j++)
    for (int i=0; i<sizex; i++) {
      int cell=i+sizex*j;
      for (int mat=0; mat<Nmats; mat++) {
        mc.rho[ncells*mat+cell]=cc.rho[cell*Nmats+mat];
        mc.p  [ncells*mat+cell]=cc.p  [cell*Nmats+mat];
        mc.Vf [ncells*mat+cell]=cc.Vf [cell*Nmats+mat];
        mc.t  [ncells*mat+cell]=cc.t  [cell*Nmats+mat];
      }
    }

  // Build compact data structure
  int imc=0; ccc.mmc_cells=0;
  for (int j=0; j<sizey; j++)
    for (int i=0; i<sizex; i++) {
      int cell=i+sizex*j, mi[4]={-1,-1,-1,-1}, midx=0, count=0;
      for (int mat=0; mat<Nmats; mat++) if (cc.rho[cell*Nmats+mat]!=0.0) mi[midx++]=mat, count++;
      if (!count) { cc.rho[cell*Nmats+1]=cc.t[cell*Nmats+1]=cc.p[cell*Nmats+1]=cc.Vf[cell*Nmats+1]=1.0; mi[0]=1; count=1; }
      if (count==1) {
        int mat=mi[0];
        ccc.rho_compact[cell]=cc.rho[cell*Nmats+mat];
        ccc.p_compact  [cell]=cc.p  [cell*Nmats+mat];
        ccc.t_compact  [cell]=cc.t  [cell*Nmats+mat];
        nmats_per_cell[cell]=-1; ccc.imaterial[cell]=mat+1;
      } else {
        nmats_per_cell[cell]=count; ccc.imaterial[cell]=-ccc.mmc_cells;
        ccc.mmc_index[ccc.mmc_cells]=imc;
        ccc.mmc_i[ccc.mmc_cells]=i; ccc.mmc_j[ccc.mmc_cells]=j;
        ccc.mmc_cells++;
        for (int li=imc; li<imc+count; li++) {
          ccc.nextfrac[li]=(li==imc+count-1)?-1:li+1;
          frac2cell[li]=cell;
          int mat=mi[li-imc];
          ccc.matids[li]=mat;
          ccc.Vf_compact_list [li]=cc.Vf [cell*Nmats+mat];
          ccc.rho_compact_list[li]=cc.rho[cell*Nmats+mat];
          ccc.p_compact_list  [li]=cc.p  [cell*Nmats+mat];
          ccc.t_compact_list  [li]=cc.t  [cell*Nmats+mat];
        }
        imc+=count;
      }
    }
  ccc.mmc_index[ccc.mmc_cells]=imc; ccc.mm_len=imc;

  bool pass=true;
  Kokkos::initialize(argc, argv);
  {
    full_matrix_cell_centric(cc);
    full_matrix_material_centric(cc, mc);
    printf("Checking full matrix results... ");
    if (full_matrix_check_results(cc,mc)) printf("PASS\n");
    else { printf("FAIL\n"); pass=false; }

    compact_cell_centric(cc, ccc);
    printf("Checking compact results... ");
    if (compact_check_results(cc, ccc)) printf("PASS\n");
    else { printf("FAIL\n"); pass=false; }
  }
  Kokkos::finalize();

  free(mc.rho); free(mc.p); free(mc.Vf); free(mc.t);
  free(cc.rho_mat_ave); free(mc.rho_mat_ave); free(ccc.rho_mat_ave_compact);
  free(ccc.rho_mat_ave_compact_list);
  free(cc.rho); free(cc.p); free(cc.Vf); free(cc.t);
  free(cc.V); free(cc.x); free(cc.y); free(cc.n);
  free(cc.rho_ave); free(mc.rho_ave); free(ccc.rho_ave_compact);
  free(ccc.rho_compact); free(ccc.p_compact); free(ccc.t_compact);
  free(nmats_per_cell); free(ccc.imaterial);
  free(ccc.nextfrac); free(frac2cell); free(ccc.matids);
  free(ccc.mmc_index); free(ccc.mmc_i); free(ccc.mmc_j);
  free(ccc.Vf_compact_list); free(ccc.rho_compact_list);
  free(ccc.t_compact_list); free(ccc.p_compact_list);
  return pass ? 0 : 1;
}
