#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <Kokkos_Core.hpp>

#define max(a,b) ((a<b)?b:a)
#define min(a,b) ((a<b)?a:b)

const int WSIZE = 12000;
const int NSIZE = 2003;
const int MSIZE = NSIZE*3+3;
const int OSIZE = NSIZE*9+9;

const int NSIZE_round = NSIZE%16 ? NSIZE+16-NSIZE%16 : NSIZE;
const size_t SSIZE = (size_t)NSIZE_round*48*48*48;

KOKKOS_INLINE_FUNCTION
void eval_abc(const float *Af, float tx, float *a) {
  a[0] = ((Af[0]*tx + Af[1])*tx + Af[2])*tx + Af[3];
  a[1] = ((Af[4]*tx + Af[5])*tx + Af[6])*tx + Af[7];
  a[2] = ((Af[8]*tx + Af[9])*tx + Af[10])*tx + Af[11];
  a[3] = ((Af[12]*tx + Af[13])*tx + Af[14])*tx + Af[15];
}

KOKKOS_INLINE_FUNCTION
void eval_UBspline_3d_s_vgh(
    const float * __restrict coefs_init,
    const intptr_t xs, const intptr_t ys, const intptr_t zs,
    float *vals, float *grads, float *hess,
    const float *a, const float *b, const float *c,
    const float *da, const float *db, const float *dc,
    const float *d2a, const float *d2b, const float *d2c,
    const float dxInv, const float dyInv, const float dzInv)
{
  float h[9] = {0};
  float v0 = 0.0f;

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      float pre20 = d2a[i]*b[j];
      float pre10 = da[i]*b[j];
      float pre00 = a[i]*b[j];
      float pre11 = da[i]*db[j];
      float pre01 = a[i]*db[j];
      float pre02 = a[i]*d2b[j];

      const float *coefs = coefs_init + i*xs + j*ys;
      float sum0 =   c[0]*coefs[0] +   c[1]*coefs[zs] +   c[2]*coefs[zs*2] +   c[3]*coefs[zs*3];
      float sum1 =  dc[0]*coefs[0] +  dc[1]*coefs[zs] +  dc[2]*coefs[zs*2] +  dc[3]*coefs[zs*3];
      float sum2 = d2c[0]*coefs[0] + d2c[1]*coefs[zs] + d2c[2]*coefs[zs*2] + d2c[3]*coefs[zs*3];

      h[0] += pre20*sum0; h[1] += pre11*sum0; h[2] += pre10*sum1;
      h[4] += pre02*sum0; h[5] += pre01*sum1; h[8] += pre00*sum2;
      h[3] += pre10*sum0; h[6] += pre01*sum0; h[7] += pre00*sum1;
      v0   += pre00*sum0;
    }

  vals[0]  = v0;
  grads[0] = h[3]*dxInv; grads[1] = h[6]*dyInv; grads[2] = h[7]*dzInv;
  hess[0] = h[0]*dxInv*dxInv; hess[1] = h[1]*dxInv*dyInv; hess[2] = h[2]*dxInv*dzInv;
  hess[3] = h[1]*dxInv*dyInv; hess[4] = h[4]*dyInv*dyInv; hess[5] = h[5]*dyInv*dzInv;
  hess[6] = h[2]*dxInv*dzInv; hess[7] = h[5]*dyInv*dzInv; hess[8] = h[8]*dzInv*dzInv;
}

int main(int argc, char **argv) {
  float Af[16] = {
    -0.166667f, 0.500000f, -0.500000f, 0.166667f,
     0.500000f,-1.000000f,  0.000000f, 0.666667f,
    -0.500000f, 0.500000f,  0.500000f, 0.166667f,
     0.166667f, 0.000000f,  0.000000f, 0.000000f
  };
  float dAf[16] = {
    0.0f,-0.5f, 1.0f,-0.5f,
    0.0f, 1.5f,-2.0f, 0.0f,
    0.0f,-1.5f, 1.0f, 0.5f,
    0.0f, 0.5f, 0.0f, 0.0f
  };
  float d2Af[16] = {
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 3.0f,-2.0f,
    0.0f, 0.0f,-3.0f, 1.0f,
    0.0f, 0.0f, 1.0f, 0.0f
  };

  float x = 0.822387f, y = 0.989919f, z = 0.104573f;

  float *walkers_vals  = (float*) malloc(sizeof(float)*WSIZE*NSIZE);
  float *walkers_grads = (float*) malloc(sizeof(float)*WSIZE*MSIZE);
  float *walkers_hess  = (float*) malloc(sizeof(float)*WSIZE*OSIZE);
  float *walkers_x = (float*) malloc(sizeof(float)*WSIZE);
  float *walkers_y = (float*) malloc(sizeof(float)*WSIZE);
  float *walkers_z = (float*) malloc(sizeof(float)*WSIZE);

  for (int i = 0; i < WSIZE; i++) {
    walkers_x[i] = x + i*1.0f/WSIZE;
    walkers_y[i] = y + i*1.0f/WSIZE;
    walkers_z[i] = z + i*1.0f/WSIZE;
  }

  float *spline_coefs = (float*) malloc(sizeof(float)*SSIZE);
  for (size_t i = 0; i < SSIZE; i++)
    spline_coefs[i] = sqrtf(0.22f + i*1.0f) * sinf(i*1.0f);

  const int num_splines = NSIZE;
  const int x_grid_num = 45, y_grid_num = 45, z_grid_num = 45;
  const intptr_t xs = NSIZE_round*48*48;
  const intptr_t ys = NSIZE_round*48;
  const intptr_t zs = NSIZE_round;
  const float delta_inv = 45.0f;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_coefs("coefs", SSIZE);
    Kokkos::View<float*> d_vals("vals",   WSIZE*NSIZE);
    Kokkos::View<float*> d_grads("grads", WSIZE*MSIZE);
    Kokkos::View<float*> d_hess("hess",   WSIZE*OSIZE);

    auto h_coefs = Kokkos::create_mirror_view(d_coefs);
    for (size_t i = 0; i < SSIZE; i++) h_coefs(i) = spline_coefs[i];
    Kokkos::deep_copy(d_coefs, h_coefs);

    double total_time = 0.0;

    for (int w = 0; w < WSIZE; w++) {
      float wx = walkers_x[w], wy = walkers_y[w], wz = walkers_z[w];

      float ux = wx * delta_inv, uy = wy * delta_inv, uz = wz * delta_inv;
      float ipartx = (int)ux, iparty = (int)uy, ipartz = (int)uz;
      float tx = ux - ipartx, ty = uy - iparty, tz = uz - ipartz;
      int ix = min(max(0,(int)ipartx), x_grid_num-1);
      int iy = min(max(0,(int)iparty), y_grid_num-1);
      int iz = min(max(0,(int)ipartz), z_grid_num-1);

      float a[4],b[4],c[4],da[4],db[4],dc[4],d2a[4],d2b[4],d2c[4];
      eval_abc(Af, tx, a); eval_abc(Af, ty, b); eval_abc(Af, tz, c);
      eval_abc(dAf, tx, da); eval_abc(dAf, ty, db); eval_abc(dAf, tz, dc);
      eval_abc(d2Af, tx, d2a); eval_abc(d2Af, ty, d2b); eval_abc(d2Af, tz, d2c);

      // Copy spline coefficients to device views for a,b,c etc. (small, just lambda capture)
      Kokkos::View<float[4]> d_a("a"),d_b("b"),d_c("c"),d_da("da"),d_db("db"),d_dc("dc"),
                              d_d2a("d2a"),d_d2b("d2b"),d_d2c("d2c");
      {
        auto ha = Kokkos::create_mirror_view(d_a);
        auto hb = Kokkos::create_mirror_view(d_b);
        auto hc = Kokkos::create_mirror_view(d_c);
        auto hda = Kokkos::create_mirror_view(d_da);
        auto hdb = Kokkos::create_mirror_view(d_db);
        auto hdc = Kokkos::create_mirror_view(d_dc);
        auto hd2a = Kokkos::create_mirror_view(d_d2a);
        auto hd2b = Kokkos::create_mirror_view(d_d2b);
        auto hd2c = Kokkos::create_mirror_view(d_d2c);
        for (int k=0;k<4;k++) {
          ha(k)=a[k]; hb(k)=b[k]; hc(k)=c[k];
          hda(k)=da[k]; hdb(k)=db[k]; hdc(k)=dc[k];
          hd2a(k)=d2a[k]; hd2b(k)=d2b[k]; hd2c(k)=d2c[k];
        }
        Kokkos::deep_copy(d_a,ha); Kokkos::deep_copy(d_b,hb); Kokkos::deep_copy(d_c,hc);
        Kokkos::deep_copy(d_da,hda); Kokkos::deep_copy(d_db,hdb); Kokkos::deep_copy(d_dc,hdc);
        Kokkos::deep_copy(d_d2a,hd2a); Kokkos::deep_copy(d_d2b,hd2b); Kokkos::deep_copy(d_d2c,hd2c);
      }

      intptr_t base_offset = (intptr_t)ix*xs + (intptr_t)iy*ys + (intptr_t)iz*zs;
      int w_vals  = w * NSIZE;
      int w_grads = w * MSIZE;
      int w_hess  = w * OSIZE;

      auto t0 = std::chrono::steady_clock::now();

      Kokkos::parallel_for("bspline", num_splines, KOKKOS_LAMBDA(int n) {
        eval_UBspline_3d_s_vgh(
          d_coefs.data() + base_offset + n,
          xs, ys, zs,
          d_vals.data()  + w_vals + n,
          d_grads.data() + w_grads + n*3,
          d_hess.data()  + w_hess + n*9,
          d_a.data(), d_b.data(), d_c.data(),
          d_da.data(), d_db.data(), d_dc.data(),
          d_d2a.data(), d_d2b.data(), d_d2c.data(),
          delta_inv, delta_inv, delta_inv);
      });
      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }

    printf("Total kernel execution time %lf (s)\n", total_time * 1e-9);

    auto h_vals  = Kokkos::create_mirror_view(d_vals);
    auto h_grads = Kokkos::create_mirror_view(d_grads);
    auto h_hess  = Kokkos::create_mirror_view(d_hess);
    Kokkos::deep_copy(h_vals,  d_vals);
    Kokkos::deep_copy(h_grads, d_grads);
    Kokkos::deep_copy(h_hess,  d_hess);

    for (int i = 0; i < WSIZE*NSIZE; i++) walkers_vals[i]  = h_vals(i);
    for (int i = 0; i < WSIZE*MSIZE; i++) walkers_grads[i] = h_grads(i);
    for (int i = 0; i < WSIZE*OSIZE; i++) walkers_hess[i]  = h_hess(i);
  }
  Kokkos::finalize();

  float resVal = 0, resGrad = 0, resHess = 0;
  for (int i = 0; i < NSIZE; i++) resVal  += walkers_vals[i];
  for (int i = 0; i < MSIZE; i++) resGrad += walkers_grads[i];
  for (int i = 0; i < OSIZE; i++) resHess += walkers_hess[i];
  printf("walkers[0]->collect([resVal resGrad resHess]) = [%e %e %e]\n",
         resVal, resGrad, resHess);

  free(walkers_vals); free(walkers_grads); free(walkers_hess);
  free(walkers_x); free(walkers_y); free(walkers_z); free(spline_coefs);
  return 0;
}
