////////////////////////////////////////////////////////////////////////////////
/**
 * @file main.cpp  (cmp-kokkos)
 * Kokkos port of the CMP (Common Mid-Point) seismic processing benchmark.
 * Original OMP-target version: cmp-omp/main.cpp
 * Support headers/sources from: cmp-cuda/
 */
////////////////////////////////////////////////////////////////////////////////

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#include "log.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "su_gather.hpp"

// ---------------------------------------------------------------------------
// Kokkos kernel: compute_semblances
// Replaces:  #pragma omp target teams distribute parallel for over ns*nc
// ---------------------------------------------------------------------------
void compute_semblances(
    Kokkos::View<const real*> d_h,
    Kokkos::View<const real*> d_c,
    Kokkos::View<const real*> d_samples,
    Kokkos::View<real*>       d_num,
    Kokkos::View<real*>       d_stt,
    int  t_id0,
    int  t_idf,
    real _idt,
    real _dt,
    int  _tau,
    int  _w,
    int  nc,
    int  ns)
{
    const int sample_base = t_id0 * ns;

    Kokkos::parallel_for("compute_semblances", ns * nc,
        KOKKOS_LAMBDA(int i) {
            real _den = 0.0f, _ac_linear = 0.0f, _ac_squared = 0.0f;
            real _num[MAX_W];
            real m = 0.0f;
            int  err = 0;

            int t0   = i / nc;
            int c_id = i % nc;

            real _c  = d_c[c_id];
            real _t0 = _dt * t0;
            _t0 *= _t0;

            for (int j = 0; j < _w; j++) _num[j] = 0.0f;

            for (int t_id = t_id0; t_id < t_idf; t_id++) {
                real t = sqrtf(_t0 + _c * d_h[t_id]) * _idt;

                int  it    = (int)(t);
                int  ittau = it - _tau;
                real x     = t - (real)it;

                if (ittau >= 0 && it + _tau + 1 < ns) {
                    // k1 is relative to the start of this CDP's sample block
                    int  k1   = ittau + (t_id - t_id0) * ns;
                    real sk1p1 = d_samples[sample_base + k1], sk1;

                    for (int j = 0; j < _w; j++) {
                        k1++;
                        sk1   = sk1p1;
                        sk1p1 = d_samples[sample_base + k1];
                        real v = (sk1p1 - sk1) * x + sk1;
                        _num[j]     += v;
                        _den        += v * v;
                        _ac_linear  += v;
                    }
                    m += 1;
                } else {
                    err++;
                }
            }

            for (int j = 0; j < _w; j++) _ac_squared += _num[j] * _num[j];

            if (_den > EPSILON && m > EPSILON && _w > EPSILON && err < 2) {
                d_num[i] = _ac_squared / (_den * m);
                d_stt[i] = _ac_linear  / (_w   * m);
            } else {
                d_num[i] = -1.0f;
                d_stt[i] = -1.0f;
            }
        });
    Kokkos::fence();
}

// ---------------------------------------------------------------------------
// Kokkos kernel: redux_semblances
// Replaces:  #pragma omp target teams distribute parallel for over ns
// ---------------------------------------------------------------------------
void redux_semblances(
    Kokkos::View<const real*> d_num,
    Kokkos::View<const real*> d_stt,
    Kokkos::View<int*>        d_ctr,
    Kokkos::View<real*>       d_str,
    Kokkos::View<real*>       d_stk,
    int nc,
    int cdp_id,
    int ns)
{
    const int cdp_offset = cdp_id * ns;

    Kokkos::parallel_for("redux_semblances", ns,
        KOKKOS_LAMBDA(int t0) {
            real max_sem = 0.0f;
            int  max_c   = -1;

            for (int it = t0 * nc; it < (t0 + 1) * nc; it++) {
                real n = d_num[it];
                if (n > max_sem) {
                    max_sem = n;
                    max_c   = it;
                }
            }

            d_ctr[cdp_offset + t0] = max_c % nc;
            d_str[cdp_offset + t0] = max_sem;
            d_stk[cdp_offset + t0] = max_c > -1 ? d_stt[max_c] : 0;
        });
    Kokkos::fence();
}

// ---------------------------------------------------------------------------
int main(int argc, const char** argv) {
    Kokkos::initialize(argc, const_cast<char**>(argv));
    {
#ifdef SAVE
        std::ofstream c_out("cmp.c.su",     std::ofstream::out | std::ios::binary);
        std::ofstream s_out("cmp.coher.su", std::ofstream::out | std::ios::binary);
        std::ofstream stack("cmp.stack.su", std::ofstream::out | std::ios::binary);
#endif

        // Parse command line
        parser::add_argument("-c0",  "C0 constant");
        parser::add_argument("-c1",  "C1 constant");
        parser::add_argument("-nc",  "NC constant");
        parser::add_argument("-aph", "APH constant");
        parser::add_argument("-tau", "Tau constant");
        parser::add_argument("-i",   "Data path");
        parser::add_argument("-v",   "Verbosity Level 0-3");

        parser::parse(argc, argv);

        const real c0   = std::stof(parser::get("-c0",  true)) * FACTOR;
        const real c1   = std::stof(parser::get("-c1",  true)) * FACTOR;
        const real itau = std::stof(parser::get("-tau", true));
        const int  nc   = std::stoi(parser::get("-nc",  true));
        const int  aph  = std::stoi(parser::get("-aph", true));
        std::string path = parser::get("-i", true);
        logger::verbosity_level(std::stoi(parser::get("-v", false)));

        // Read SU data
        su_gather gather(path, aph, nc);

        real *h_gx, *h_gy, *h_sx, *h_sy, *h_scalco, *h_samples, dt;
        int  *ntraces_by_cdp_id;

        gather.linearize(ntraces_by_cdp_id, h_samples, dt,
                         h_gx, h_gy, h_sx, h_sy, h_scalco, nc);

        const int  ttraces = gather.ttraces();
        const int  ncdps   = gather().size();
        const int  ns      = gather.ns();
        const int  ntrs    = gather.ntrs();
        const real inc     = (c1 - c0) * (1.0f / (real)nc);

        dt = dt / 1000000.0f;
        real idt = 1.0f / dt;
        int  tau = ((int)(itau * idt) > 0) ? ((int)(itau * idt)) : 0;
        int  w   = (2 * tau) + 1;

        int number_of_semblances = 0;

        LOG(INFO, "Starting CMP-Kokkos execution");

        // ---- Allocate device Views ----------------------------------------
        Kokkos::View<real*> d_gx    ("d_gx",     ttraces);
        Kokkos::View<real*> d_gy    ("d_gy",     ttraces);
        Kokkos::View<real*> d_sx    ("d_sx",     ttraces);
        Kokkos::View<real*> d_sy    ("d_sy",     ttraces);
        Kokkos::View<real*> d_scalco("d_scalco", ttraces);
        Kokkos::View<real*> d_samples_v("d_samples", ttraces * ns);
        Kokkos::View<real*> d_c    ("d_c",   nc);
        Kokkos::View<real*> d_h    ("d_h",   ttraces);
        Kokkos::View<real*> d_num  ("d_num", ns * nc);
        Kokkos::View<real*> d_stt  ("d_stt", ns * nc);
        Kokkos::View<int*>  d_ctr  ("d_ctr", ncdps * ns);
        Kokkos::View<real*> d_str  ("d_str", ncdps * ns);
        Kokkos::View<real*> d_stk  ("d_stk", ncdps * ns);

        // ---- Host mirrors for copy-in -------------------------------------
        auto h_gx_m     = Kokkos::create_mirror_view(d_gx);
        auto h_gy_m     = Kokkos::create_mirror_view(d_gy);
        auto h_sx_m     = Kokkos::create_mirror_view(d_sx);
        auto h_sy_m     = Kokkos::create_mirror_view(d_sy);
        auto h_scalco_m = Kokkos::create_mirror_view(d_scalco);
        auto h_samp_m   = Kokkos::create_mirror_view(d_samples_v);

        for (int i = 0; i < ttraces;         i++) h_gx_m(i)     = h_gx[i];
        for (int i = 0; i < ttraces;         i++) h_gy_m(i)     = h_gy[i];
        for (int i = 0; i < ttraces;         i++) h_sx_m(i)     = h_sx[i];
        for (int i = 0; i < ttraces;         i++) h_sy_m(i)     = h_sy[i];
        for (int i = 0; i < ttraces;         i++) h_scalco_m(i) = h_scalco[i];
        for (int i = 0; i < ttraces * ns;    i++) h_samp_m(i)   = h_samples[i];

        Kokkos::deep_copy(d_gx,       h_gx_m);
        Kokkos::deep_copy(d_gy,       h_gy_m);
        Kokkos::deep_copy(d_sx,       h_sx_m);
        Kokkos::deep_copy(d_sy,       h_sy_m);
        Kokkos::deep_copy(d_scalco,   h_scalco_m);
        Kokkos::deep_copy(d_samples_v, h_samp_m);

        // Const views for read-only kernels
        Kokkos::View<const real*> d_gx_c    = d_gx;
        Kokkos::View<const real*> d_gy_c    = d_gy;
        Kokkos::View<const real*> d_sx_c    = d_sx;
        Kokkos::View<const real*> d_sy_c    = d_sy;
        Kokkos::View<const real*> d_scalco_c = d_scalco;

        // ---- Start timing -------------------------------------------------
        auto beg = std::chrono::high_resolution_clock::now();

        // Init d_c  (linspace c0..c1)
        Kokkos::parallel_for("init_c", nc,
            KOKKOS_LAMBDA(int i) { d_c(i) = c0 + inc * i; });
        Kokkos::fence();

        // Init d_h  (half-offset)
        Kokkos::parallel_for("init_h", ttraces,
            KOKKOS_LAMBDA(int i) {
                real _s = d_scalco_c[i];
                if (-EPSILON < _s && _s < EPSILON) _s = 1.0f;
                else if (_s < 0)                   _s = 1.0f / _s;
                real hx = (d_gx_c[i] - d_sx_c[i]) * _s;
                real hy = (d_gy_c[i] - d_sy_c[i]) * _s;
                d_h(i) = 0.25f * (hx * hx + hy * hy) / FACTOR;
            });
        Kokkos::fence();

        Kokkos::View<const real*> d_h_c  = d_h;
        Kokkos::View<const real*> d_c_c  = d_c;
        Kokkos::View<const real*> d_samp_c = d_samples_v;

        // ---- Main CDP loop ------------------------------------------------
        for (int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
            int t_id0  = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id - 1] : 0;
            int t_idf  = ntraces_by_cdp_id[cdp_id];
            int stride = t_idf - t_id0;

            compute_semblances(d_h_c, d_c_c, d_samp_c, d_num, d_stt,
                               t_id0, t_idf, idt, dt, tau, w, nc, ns);

            redux_semblances(
                Kokkos::View<const real*>(d_num),
                Kokkos::View<const real*>(d_stt),
                d_ctr, d_str, d_stk, nc, cdp_id, ns);

            number_of_semblances += stride;

#ifdef DEBUG
            std::cout << "Progress: " + std::to_string(cdp_id)
                      + "/" + std::to_string(ncdps) << std::endl;
#endif
        }

        auto end = std::chrono::high_resolution_clock::now();

        // ---- Copy results back to host ------------------------------------
        auto h_ctr_m = Kokkos::create_mirror_view(d_ctr);
        auto h_str_m = Kokkos::create_mirror_view(d_str);
        auto h_stk_m = Kokkos::create_mirror_view(d_stk);
        Kokkos::deep_copy(h_ctr_m, d_ctr);
        Kokkos::deep_copy(h_str_m, d_str);
        Kokkos::deep_copy(h_stk_m, d_stk);

        // ---- Reference verification --------------------------------------
        real *h_c   = (real*) malloc(sizeof(real) * nc);
        real *h_h   = (real*) malloc(sizeof(real) * ttraces);
        real *h_num = (real*) malloc(sizeof(real) * ns * nc);
        real *h_stt = (real*) malloc(sizeof(real) * ns * nc);
        int  *r_ctr = (int*)  malloc(sizeof(int)  * ncdps * ns);
        real *r_str = (real*) malloc(sizeof(real) * ncdps * ns);
        real *r_stk = (real*) malloc(sizeof(real) * ncdps * ns);

        h_init_c(nc, h_c, inc, c0);
        h_init_half(ttraces, h_scalco, h_gx, h_gy, h_sx, h_sy, h_h);

        for (int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
            int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id - 1] : 0;
            int t_idf = ntraces_by_cdp_id[cdp_id];
            h_compute_semblances(h_h, h_c, h_samples + t_id0 * ns,
                                 h_num, h_stt, t_id0, t_idf,
                                 idt, dt, tau, w, nc, ns);
            h_redux_semblances(h_num, h_stt, r_ctr, r_str, r_stk, nc, cdp_id, ns);
        }

        int err_ctr = 0, err_str = 0, err_stk = 0;
        for (int i = 0; i < ncdps * ns; i++) {
            if (r_ctr[i] != h_ctr_m(i))              err_ctr++;
            if (r_str[i] - h_str_m(i) > 1e-3f)       err_str++;
            if (r_stk[i] - h_stk_m(i) > 1e-3f)       err_stk++;
        }
        printf("Error rate: ctr=%e str=%e stk=%e\n",
               (float)err_ctr / (ncdps * ns),
               (float)err_str / (ncdps * ns),
               (float)err_stk / (ncdps * ns));

        double time = std::chrono::duration_cast<
                          std::chrono::duration<double>>(end - beg).count();
        double stps = (number_of_semblances / 1e9) * (ns * nc / time);
        LOG(INFO, "Giga semblances traces per second: " + std::to_string(stps));

#ifdef SAVE
        for (int i = 0; i < ncdps; i++) {
            su_trace ctr_t = gather[i].traces()[0];
            su_trace str_t = gather[i].traces()[0];
            su_trace stk_t = gather[i].traces()[0];
            ctr_t.offset() = 0;
            ctr_t.sx() = ctr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
            ctr_t.sy() = ctr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;
            for (int k = 0; k < ns; k++)
                ctr_t.data()[k] = h_ctr_m(i*ns+k) < 0 ? 0.0f
                                  : (c0 + inc * h_ctr_m(i*ns+k)) / FACTOR;
            str_t.data().assign(&h_str_m(i*ns), &h_str_m((i+1)*ns));
            stk_t.data().assign(&h_stk_m(i*ns), &h_stk_m((i+1)*ns));
            ctr_t.fputtr(c_out);
            str_t.fputtr(s_out);
            stk_t.fputtr(stack);
        }
#endif

        free(h_h); free(h_c); free(h_num); free(h_stt);
        free(r_ctr); free(r_str); free(r_stk);
        delete[] h_gx; delete[] h_gy; delete[] h_sx; delete[] h_sy;
        delete[] h_scalco; delete[] h_samples; delete[] ntraces_by_cdp_id;
    }
    Kokkos::finalize();
    return EXIT_SUCCESS;
}
