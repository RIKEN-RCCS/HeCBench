#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// Kokkos-compatible timestep
KOKKOS_INLINE_FUNCTION int k_timestep(float t, float dt) {
  return (int)((t + 1e-3f * dt) / dt);
}

// Host reference timestep
inline int h_timestep(float t, float dt) {
  return (int)((t + 1e-3f * dt) / dt);
}

void reference_cobahh(
    float* h, float* m, float* n, float* ge,
    float* v, float* gi, const float* lastspike,
    char* not_refractory, const int N,
    const float dt, const float t,
    const int _lio_1, const float _lio_2, const float _lio_3,
    const float _lio_4, const float _lio_5, const float _lio_6,
    const float _lio_7, const float _lio_8, const float _lio_9,
    const float _lio_10, const float _lio_11, const float _lio_12,
    const float _lio_13, const float _lio_14, const float _lio_15,
    const float _lio_16, const float _lio_17, const float _lio_18,
    const float _lio_19, const float _lio_20, const float _lio_21,
    const float _lio_22, const float _lio_23, const float _lio_24,
    const float _lio_25, const float _lio_26, const float _lio_27,
    const float _lio_28, const float _lio_29, const float _lio_30,
    const float _lio_31, const float _lio_32, const float _lio_33)
{
  for (int _idx = 0; _idx < N; _idx++) {
    float hv = h[_idx], mv = m[_idx], nv = n[_idx];
    float gev = ge[_idx], vv = v[_idx], giv = gi[_idx];
    const float ls = lastspike[_idx];
    char nr = h_timestep(t - ls, dt) >= _lio_1;
    const float _BA_h = (_lio_2 * expf(_lio_3 * vv))/(((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * vv)))) - (_lio_2 * expf(_lio_3 * vv)));
    const float _h = (-_BA_h) + ((_BA_h + hv) * expf(dt * (((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * vv)))) - (_lio_2 * expf(_lio_3 * vv)))));
    const float _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * expf(_lio_9 * vv)))) + (_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * vv))))) - ((0.32f * vv)/(_lio_7 + (_lio_8 * expf(_lio_9 * vv)))))/(((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * vv)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * vv))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * vv))))) + ((0.32f * vv)/(_lio_7 + (_lio_8 * expf(_lio_9 * vv))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * vv)))) + ((0.28f * vv)/(_lio_13 + (_lio_14 * expf(_lio_15 * vv))))));
    const float _m = (-_BA_m) + ((_BA_m + mv) * expf(dt * (((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * vv)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * vv))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * vv))))) + ((0.32f * vv)/(_lio_7 + (_lio_8 * expf(_lio_9 * vv))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * vv)))) + ((0.28f * vv)/(_lio_13 + (_lio_14 * expf(_lio_15 * vv))))))));
    const float _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * expf(_lio_5 * vv)))) + (_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * vv))))) - ((0.032f * vv)/(_lio_7 + (_lio_18 * expf(_lio_5 * vv)))))/(((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * vv)))) + ((0.032f * vv)/(_lio_7 + (_lio_18 * expf(_lio_5 * vv))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * vv)))) + (_lio_21 * expf(_lio_22 * vv))));
    const float _n = (-_BA_n) + ((_BA_n + nv) * expf(dt * (((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * vv)))) + ((0.032f * vv)/(_lio_7 + (_lio_18 * expf(_lio_5 * vv))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * vv)))) + (_lio_21 * expf(_lio_22 * vv))))));
    const float _ge = _lio_23 * gev;
    const float _BA_v = (_lio_24 + ((((_lio_25 * (nv*nv*nv*nv)) + (_lio_26 * (hv * (mv*mv*mv)))) + (_lio_27 * gev)) + (_lio_28 * giv)))/((_lio_29 + (_lio_30 * (nv*nv*nv*nv))) - (((_lio_31 * (hv * (mv*mv*mv))) + (_lio_32 * gev)) + (_lio_32 * giv)));
    const float _v = (-_BA_v) + ((_BA_v + vv) * expf(dt * ((_lio_29 + (_lio_30 * (nv*nv*nv*nv))) - (((_lio_31 * (hv * (mv*mv*mv))) + (_lio_32 * gev)) + (_lio_32 * giv)))));
    const float _gi = _lio_33 * giv;
    h[_idx] = _h; m[_idx] = _m; n[_idx] = _n;
    ge[_idx] = _ge; v[_idx] = _v; gi[_idx] = _gi;
    not_refractory[_idx] = nr;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <neurons> <repeat>\n", argv[0]);
    return 1;
  }
  const int N         = atoi(argv[1]);
  const int iteration = atoi(argv[2]);
  srand(2);

  float *h_ge = new float[N], *h_gi = new float[N];
  float *h_h  = new float[N], *h_m  = new float[N];
  float *h_n  = new float[N], *h_v  = new float[N];
  float *h_lastspike = new float[N];
  char  *h_not_refract = new char[N];

  float *ge = new float[N], *gi = new float[N];
  float *hh = new float[N], *m  = new float[N];
  float *n  = new float[N], *v  = new float[N];
  float *lastspike   = new float[N];
  char  *not_refract = new char[N];

  printf("initializing ... ");
  for (int i = 1; i < N; i++) {
    h_ge[i] = ge[i] = 0.15f + ((rand()%2==0) ? 0.1f : -0.1f);
    h_gi[i] = gi[i] = 0.25f + ((rand()%2==0) ? 0.2f : -0.2f);
    h_h[i]  = hh[i] = 0.35f + ((rand()%2==0) ? 0.3f : -0.3f);
    h_m[i]  =  m[i] = 0.45f + ((rand()%2==0) ? 0.4f : -0.4f);
    h_n[i]  =  n[i] = 0.55f + ((rand()%2==0) ? 0.5f : -0.5f);
    h_v[i]  =  v[i] = 0.65f + ((rand()%2==0) ? 0.6f : -0.6f);
    h_lastspike[i] = lastspike[i] = 1.0f / (rand() % 1000 + 1);
  }
  float dt = 0.0001f, t = 0.01f;
  printf("done.\n");

  // Host constants
  const int    _lio_1  = h_timestep(0.003f, dt);
  const float  _lio_2  = 9.939082f;
  const float  _lio_3  = -55.555556f;
  const float  _lio_4  = 0.00001f;
  const float  _lio_5  = -200.0f;
  const float  _lio_6  = -0.02016f;
  const float  _lio_7  = -0.000001f;
  const float  _lio_8  = 0.0f;
  const float  _lio_9  = -250.0f;
  const float  _lio_10 = 0.00416f;
  const float  _lio_11 = 0.02016f;
  const float  _lio_12 = -0.01764f;
  const float  _lio_13 = -0.000001f;
  const float  _lio_14 = 0.000099f;
  const float  _lio_15 = 200.0f;
  const float  _lio_16 = 0.0112f;
  const float  _lio_17 = -0.002016f;
  const float  _lio_18 = 0.0f;
  const float  _lio_19 = 0.00048f;
  const float  _lio_20 = 0.002016f;
  const float  _lio_21 = 132.901474f;
  const float  _lio_22 = -25.0f;
  const float  _lio_23 = expf(-2000.0f * dt);
  const float  _lio_24 = -3.0f;
  const float  _lio_25 = -2700.0f;
  const float  _lio_26 = 5000.0f;
  const float  _lio_27 = 0.0f;
  const float  _lio_28 = -400000000.0f;
  const float  _lio_29 = -50.0f;
  const float  _lio_30 = -30000.0f;
  const float  _lio_31 = 100000.0f;
  const float  _lio_32 = 5000000000.0f;
  const float  _lio_33 = expf(-100.0f * dt);

  // Host reference run
  for (int iter = 0; iter < iteration; iter++) {
    reference_cobahh(h_h, h_m, h_n, h_ge, h_v, h_gi, h_lastspike, h_not_refract, N,
      dt, t, _lio_1, _lio_2, _lio_3, _lio_4, _lio_5, _lio_6, _lio_7,
      _lio_8, _lio_9, _lio_10, _lio_11, _lio_12, _lio_13, _lio_14, _lio_15,
      _lio_16, _lio_17, _lio_18, _lio_19, _lio_20, _lio_21, _lio_22, _lio_23,
      _lio_24, _lio_25, _lio_26, _lio_27, _lio_28, _lio_29, _lio_30, _lio_31,
      _lio_32, _lio_33);
  }

  // Kokkos run
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_ge("ge", N), d_gi("gi", N);
    Kokkos::View<float*> d_h("h", N),   d_m("m", N);
    Kokkos::View<float*> d_n("n", N),   d_v("v", N);
    Kokkos::View<float*> d_ls("ls", N);
    Kokkos::View<char*>  d_nr("nr", N);

    {
      auto mge = Kokkos::create_mirror_view(d_ge);
      auto mgi = Kokkos::create_mirror_view(d_gi);
      auto mh  = Kokkos::create_mirror_view(d_h);
      auto mm  = Kokkos::create_mirror_view(d_m);
      auto mn  = Kokkos::create_mirror_view(d_n);
      auto mv  = Kokkos::create_mirror_view(d_v);
      auto mls = Kokkos::create_mirror_view(d_ls);
      for (int i = 0; i < N; i++) {
        mge(i)=ge[i]; mgi(i)=gi[i]; mh(i)=hh[i]; mm(i)=m[i];
        mn(i)=n[i]; mv(i)=v[i]; mls(i)=lastspike[i];
      }
      Kokkos::deep_copy(d_ge, mge); Kokkos::deep_copy(d_gi, mgi);
      Kokkos::deep_copy(d_h,  mh);  Kokkos::deep_copy(d_m,  mm);
      Kokkos::deep_copy(d_n,  mn);  Kokkos::deep_copy(d_v,  mv);
      Kokkos::deep_copy(d_ls, mls);
    }

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iteration; iter++) {
      Kokkos::parallel_for("cobahh", N, KOKKOS_LAMBDA(int _idx) {
        float hv = d_h(_idx), mv = d_m(_idx), nv = d_n(_idx);
        float gev = d_ge(_idx), vv = d_v(_idx), giv = d_gi(_idx);
        const float ls = d_ls(_idx);
        char nr = k_timestep(t - ls, dt) >= _lio_1;
        const float _BA_h = (_lio_2 * Kokkos::exp(_lio_3 * vv))/(((-4.0f)/(0.001f + (_lio_4 * Kokkos::exp(_lio_5 * vv)))) - (_lio_2 * Kokkos::exp(_lio_3 * vv)));
        const float _h = (-_BA_h) + ((_BA_h + hv) * Kokkos::exp(dt * (((-4.0f)/(0.001f + (_lio_4 * Kokkos::exp(_lio_5 * vv)))) - (_lio_2 * Kokkos::exp(_lio_3 * vv)))));
        const float _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv)))) + (_lio_10/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv))))) - ((0.32f * vv)/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv)))))/(((((_lio_11/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv)))) + (_lio_12/(_lio_13 + (_lio_14 * Kokkos::exp(_lio_15 * vv))))) + (_lio_16/(_lio_13 + (_lio_14 * Kokkos::exp(_lio_15 * vv))))) + ((0.32f * vv)/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv))))) - ((_lio_10/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv)))) + ((0.28f * vv)/(_lio_13 + (_lio_14 * Kokkos::exp(_lio_15 * vv))))));
        const float _m = (-_BA_m) + ((_BA_m + mv) * Kokkos::exp(dt * (((((_lio_11/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv)))) + (_lio_12/(_lio_13 + (_lio_14 * Kokkos::exp(_lio_15 * vv))))) + (_lio_16/(_lio_13 + (_lio_14 * Kokkos::exp(_lio_15 * vv))))) + ((0.32f * vv)/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv))))) - ((_lio_10/(_lio_7 + (_lio_8 * Kokkos::exp(_lio_9 * vv)))) + ((0.28f * vv)/(_lio_13 + (_lio_14 * Kokkos::exp(_lio_15 * vv))))))));
        const float _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv)))) + (_lio_19/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv))))) - ((0.032f * vv)/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv)))))/(((_lio_20/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv)))) + ((0.032f * vv)/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv))))) - ((_lio_19/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv)))) + (_lio_21 * Kokkos::exp(_lio_22 * vv))));
        const float _n = (-_BA_n) + ((_BA_n + nv) * Kokkos::exp(dt * (((_lio_20/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv)))) + ((0.032f * vv)/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv))))) - ((_lio_19/(_lio_7 + (_lio_18 * Kokkos::exp(_lio_5 * vv)))) + (_lio_21 * Kokkos::exp(_lio_22 * vv))))));
        const float _ge = _lio_23 * gev;
        const float _BA_v = (_lio_24 + ((((_lio_25 * (nv*nv*nv*nv)) + (_lio_26 * (hv * (mv*mv*mv)))) + (_lio_27 * gev)) + (_lio_28 * giv)))/((_lio_29 + (_lio_30 * (nv*nv*nv*nv))) - (((_lio_31 * (hv * (mv*mv*mv))) + (_lio_32 * gev)) + (_lio_32 * giv)));
        const float _v = (-_BA_v) + ((_BA_v + vv) * Kokkos::exp(dt * ((_lio_29 + (_lio_30 * (nv*nv*nv*nv))) - (((_lio_31 * (hv * (mv*mv*mv))) + (_lio_32 * gev)) + (_lio_32 * giv)))));
        const float _gi = _lio_33 * giv;
        d_h(_idx) = _h; d_m(_idx) = _m; d_n(_idx) = _n;
        d_ge(_idx) = _ge; d_v(_idx) = _v; d_gi(_idx) = _gi;
        d_nr(_idx) = nr;
      });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / iteration);

    // Copy back and compare
    auto mge = Kokkos::create_mirror_view(d_ge);
    auto mgi = Kokkos::create_mirror_view(d_gi);
    auto mh  = Kokkos::create_mirror_view(d_h);
    auto mm  = Kokkos::create_mirror_view(d_m);
    auto mn  = Kokkos::create_mirror_view(d_n);
    auto mv  = Kokkos::create_mirror_view(d_v);
    auto mnr = Kokkos::create_mirror_view(d_nr);
    Kokkos::deep_copy(mge, d_ge); Kokkos::deep_copy(mgi, d_gi);
    Kokkos::deep_copy(mh, d_h);   Kokkos::deep_copy(mm, d_m);
    Kokkos::deep_copy(mn, d_n);   Kokkos::deep_copy(mv, d_v);
    Kokkos::deep_copy(mnr, d_nr);

    double rsme = 0.0;
    for (int i = 0; i < N; i++) {
      rsme += (mge(i)-h_ge[i])*(mge(i)-h_ge[i]);
      rsme += (mgi(i)-h_gi[i])*(mgi(i)-h_gi[i]);
      rsme += (mh(i)-h_h[i])*(mh(i)-h_h[i]);
      rsme += (mm(i)-h_m[i])*(mm(i)-h_m[i]);
      rsme += (mn(i)-h_n[i])*(mn(i)-h_n[i]);
      rsme += (mv(i)-h_v[i])*(mv(i)-h_v[i]);
      rsme += (mnr(i)-h_not_refract[i])*(mnr(i)-h_not_refract[i]);
    }
    printf("RSME = %lf\n", sqrt(rsme / N));
  }
  Kokkos::finalize();

  delete[] h_ge; delete[] h_gi; delete[] h_h; delete[] h_m;
  delete[] h_n; delete[] h_v; delete[] h_lastspike; delete[] h_not_refract;
  delete[] ge; delete[] gi; delete[] hh; delete[] m;
  delete[] n; delete[] v; delete[] lastspike; delete[] not_refract;
  return 0;
}
