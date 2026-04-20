// WSM5 microphysics benchmark – Kokkos port
// Ported from wsm5-omp by replacing OpenMP target offload with Kokkos TeamPolicy.

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// Thread-block dimensions (same as original)
#define XXX 8
#define YYY 8

// MKX must be defined at compile time (pass -DMKX=4 via Makefile)
#ifndef MKX
#  error "MKX must be defined (e.g. -DMKX=4)"
#endif

// ---- Indexing macros (matching spt.h from wsm5-omp) ------------------------
// Inside the Kokkos lambda, bi/bj/ti/tj/bx/by are local int variables.
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define I2(i,j,m)       ((i)+((j)*(m)))
#define I3(i,j,m,k,n)   (I2(i,j,m)+((k)*(m)*(n)))
#define TtoP(i,a,b,c,d) ((i)+(a)*(b)+(c)-(d))
#define P2(i,j)    I2(TtoP(i,bi,bx,ips,ims),TtoP(j,bj,by,jps,jms),ime-ims+1)
#define P3(i,k,j)  I3(TtoP(i,bi,bx,ips,ims),k,ime-ims+1,TtoP(j,bj,by,jps,jms),kme-kms+1)
#define ig         (TtoP(ti,bi,bx,ips,ims))
#define jg         (TtoP(tj,bj,by,jps,jms))

// Host memory helpers
#define ALLOC(A,s) A = (float*) malloc((s) * sizeof(float)); \
                   for (int _i = 0; _i < (s); _i++) A[_i] = 0.001f;
#define FREE(A)    free(A)
#define ALLOC3(A)  ALLOC(A,d3)
#define ALLOC2(A)  ALLOC(A,d2)

// ============================================================================
// The wsm Kokkos kernel launcher
// All float* pointers must be device-accessible (from Kokkos Views).
// ============================================================================

void wsm(
        float * th,
  const float * pii,
        float * q,
        float * qc,
        float * qi,
        float * qr,
        float * qs,
  const float * den,
  const float * p,
  const float * delz,
        float * rain,
        float * rainncv,
        float * sr,
        float * snow,
        float * snowncv,
  const float delt,
  const int ids, const int ide,
  const int jds, const int jde,
  const int kds, const int kde,
  const int ims, const int ime,
  const int jms, const int jme,
  const int kms, const int kme,
  const int ips, const int ipe,
  const int jps, const int jpe,
  const int kps, const int kpe,
  const int teamX, const int teamY)
{
  using exec_space  = Kokkos::DefaultExecutionSpace;
  using TeamPolicy  = Kokkos::TeamPolicy<exec_space>;
  using member_type = typename TeamPolicy::member_type;

  Kokkos::parallel_for("wsm5",
    TeamPolicy(teamX * teamY, XXX * YYY),
    KOKKOS_LAMBDA(const member_type& team) {

      const int bi = team.league_rank() % teamX;
      const int bj = team.league_rank() / teamX;
      const int ti = team.team_rank()   % XXX;
      const int tj = team.team_rank()   / XXX;
      const int bx = XXX;
      const int by = YYY;

      if (ig < ide - ids + 1 && jg < jde - jds + 1) {

        float xlf, xmi, acrfac, vt2i, vt2s, supice, diameter;
        float roqi0, xni0, qimax, value, source, factor, xlwork2;
        float t_k, q_k, qr_k, qc_k, qs_k, qi_k, qs1_k, qs2_k,
              cpm_k, xl_k, w1_k, w2_k, w3_k;

#define hsub  xls
#define hvap  xlv0
#define cvap  cpv

        float ttp, dldt, xa, xb, dldti, xai, xbi;
        float qs1[MKX], qs2[MKX], rh1[MKX], rh2[MKX];
        int k;

        // ---- constants.h content (wsm5-cuda/constants.h) -------------------
#define epsilon    1.e-15f
#define r_d        287.f
#define rhoair0    1.28f
#define rhosnow    100.f
#define dens       rhosnow
#define rhowater   1000.f
#define svpt0      .27314999389648438e+03f
#define xlv        2.5e6f

#define g_const     0.981000041961670E+01f
#define r_v         0.461600006103516E+03f
#define rv          r_v
#define cice        0.210600000000000E+04f
#define cliq        0.419000000000000E+04f
#define denr        0.100000000000000E+04f
#define den0        0.127999997138977E+01f
#define xlf0        0.350000000000000E+06f
#define xlv0        0.250000000000000E+07f
#define xls         0.285000000000000E+07f
#define t0c         0.273149993896484E+03f
#define qmin        0.100000000362749E-14f
#define ep1         0.608362436294556E+00f
#define ep2         0.621750414371490E+00f
#define psat        0.610780029296875E+03f
#define alpha_c     0.120000000000000E+00f
#define n0smax      0.100000000000000E+12f
#define n0s         0.200000000000000E+07f
#define n0r         0.800000000000000E+07f
#define qcrmin      0.100000000000000E-08f
#define avtr        0.841900000000000E+03f
#define bvtr        0.800000000000000E+00f
#define g1pbr       0.931232915622909E+00f
#define g3pbr       0.469078683336385E+01f
#define g4pbr       0.178173289058329E+02f
#define g5pbro2     0.182658695197891E+01f
#define avts        0.117200000000000E+02f
#define bvts        0.410000000000000E+00f
#define g1pbs       0.886676521690526E+00f
#define g3pbs       0.301156382231086E+01f
#define g4pbs       0.102654190601850E+02f
#define g5pbso2     1.550308f
#define r0          0.800000000000000E-05f
#define peaut       0.550000000000000E+00f
#define xncr        0.300000000000000E+09f
#define xmyu        0.171800000000000E-04f
#define lamdarmax   0.800000000000000E+05f
#define lamdasmax   0.100000000000000E+06f
#define lamdagmax   0.600000000000000E+05f
#define pi          0.314159265358979E+01f
#define dicon       0.119000000000000E+02f
#define dimax       0.500000000000000E-03f
#define pfrz1       0.100000000000000E+03f
#define pfrz2       0.660000000000000E+00f
#define eacrr       0.100000000000000E+01f
#define eacrc       0.100000000000000E+01f

        float cpv  = 4.f * r_v;
        float cp   = 7.f * r_d / 2.f;
        float cv   = cp - r_d;
        float cpd  = cp;
        float pvtr = avtr * g4pbr / 6.f;
        float pvts = avts * g4pbs / 6.f;
        float xlv1 = cliq - cv;

        float rslopermax = 1.f / lamdarmax;
        float rslopesmax = .10000000000000001e-04f;
        float rsloperbmax = 0.11954406247375457E-03f;
        float rslopesbmax = .89125093813374589e-02f;
        float rsloper2max = rslopermax * rslopermax;
        float rslopes2max = rslopesmax * rslopesmax;
        float rsloper3max = rsloper2max * rslopermax;
        float rslopes3max = rslopes2max * rslopesmax;

        float pidn0r = pi * denr * n0r;
        float pidn0s = pi * dens * n0s;

        float precs1 = 4.f * n0s * .65f;
        float precs2 = 4.f * n0s * .44f * sqrtf(avts) * g5pbso2;
        float qc0    = 4.f / 3.f * pi * denr * (r0 * r0 * r0) * xncr / den0;
        float qck1   = .104f * 9.8f * peaut /
                       powf((xncr * denr), (1.f / 3.f)) / xmyu *
                       powf(den0, (4.f / 3.f));
        float precr1 = 2.f * pi * n0r * .78f;
        float precr2 = 2.f * pi * n0r * .31f * sqrtf(avtr) * g5pbro2;
        float pacrr  = pi * n0r * avtr * g3pbr * .25f * eacrr;
        float pacrc  = pi * n0s * avts * g3pbs * .25f * eacrc;
        float roqimax = 2.08e22f * powf(dimax, 8.0f);
        // ---- end constants -------------------------------------------------

        float t[MKX], cpm[MKX], xl[MKX];

        for (k = kps - 1; k <= kpe - 1; k++)
          t[k] = th[P3(ti, k, tj)] * pii[P3(ti, k, tj)];

        for (k = kps - 1; k <= kpe - 1; k++) {
          if (qc[P3(ti,k,tj)] < 0.f) qc[P3(ti,k,tj)] = 0.f;
          if (qi[P3(ti,k,tj)] < 0.f) qi[P3(ti,k,tj)] = 0.f;
          if (qr[P3(ti,k,tj)] < 0.f) qr[P3(ti,k,tj)] = 0.f;
          if (qs[P3(ti,k,tj)] < 0.f) qs[P3(ti,k,tj)] = 0.f;
        }

#define CPMCAL(x) (cpd*(1.f-MAX(x,qmin))+MAX(x,qmin)*cpv)
#define XLCAL(x)  (xlv0-xlv1*((x)-t0c))

        for (k = kps - 1; k <= kpe - 1; k++) {
          cpm[k] = CPMCAL(q[P3(ti,k,tj)]);
          xl[k]  = XLCAL(t[k]);
        }

        float dtcldcr = 120.f;
        int loops = (int)(delt / dtcldcr + .5f + .5f);
        loops = MAX(loops, 1);
        float dtcld = delt / loops;
        if (delt <= dtcldcr) dtcld = delt;

        int loop;
        for (loop = 1; loop <= loops; loop++) {
          int mstep = 1;

          ttp   = t0c + 0.01f;
          dldt  = cvap - cliq;
          xa    = -dldt / rv;
          xb    = xa + hvap / (rv * ttp);
          dldti = cvap - cice;
          xai   = -dldti / rv;
          xbi   = xai + hsub / (rv * ttp);

          float tr, ltr, tt, pp, qq;

          for (k = kps - 1; k <= kpe - 1; k++) {
            pp  = p[P3(ti,k,tj)];
            tt  = t[k];
            tr  = ttp / tt;
            ltr = logf(tr);

            qq = psat * expf(ltr * xa + xb * (1.f - tr));
            qq = ep2 * qq / (pp - qq);
            qs1[k] = MAX(qq, qmin);
            rh1[k] = MAX(q[P3(ti,k,tj)] / qs1[k], qmin);

            if (tt < ttp)
              qq = psat * expf(ltr * xai + xbi * (1.f - tr));
            else
              qq = psat * expf(ltr * xa  + xb  * (1.f - tr));
            qq    = ep2 * qq / (pp - qq);
            qs2[k] = MAX(qq, qmin);
            rh2[k] = MAX(q[P3(ti,k,tj)] / qs2[k], qmin);
          }

          float prevp_reg = 0.f, psdep_reg = 0.f, praut_reg = 0.f;
          float psaut_reg = 0.f, pracw_reg = 0.f, psaci_reg = 0.f;
          float psacw_reg = 0.f, pigen_reg = 0.f, pidep_reg = 0.f;
          float pcond_reg = 0.f, psmlt_reg = 0.f, psevp_reg = 0.f;
          float xni[MKX];

          for (k = kps - 1; k <= kpe - 1; k++) xni[k] = 1.e3f;

#define DIFFUS(x,y)  (8.794e-5f * expf(logf(x)*(1.81f)) / (y))
#define VISCOS(x,y)  (1.496e-6f * ((x)*sqrtf(x)) / ((x)+120.f) / (y))
#define XKA(x,y)     (1.414e3f * VISCOS((x),(y)) * (y))
#define DIFFAC(a,b,c,d,e) \
  ((d)*(a)*(a)/(XKA((c),(d))*rv*(c)*(c)) + 1.f/((e)*DIFFUS((c),(b))))
#define VENFAC(a,b,c) \
  (expf(logf((VISCOS((b),(c))/DIFFUS((b),(a))))*((.3333333f))) * \
   (1.f/sqrtf(VISCOS((b),(c)))) * sqrtf(sqrtf(den0/(c))))
#define LAMDAR(x,y)    sqrtf(sqrtf(pidn0r/((x)*(y))))
#define LAMDAS(x,y,z)  sqrtf(sqrtf(pidn0s*(z)/((x)*(y))))

          float rsloper[MKX], rslopebr[MKX], rslope2r[MKX], rslope3r[MKX];
          float rslopes[MKX], rslopebs[MKX], rslope2s[MKX], rslope3s[MKX];
          float denfac[MKX], n0sfac[MKX];
          float w1[MKX], w2[MKX], w3[MKX];

          float w, rmstep;
          int numdt;
          for (k = kps - 1; k <= kpe - 1; k++) {
            float supcol = t0c - t[k];
            n0sfac[k] = MAX(MIN(expf(alpha_c * supcol), n0smax / n0s), 1.f);

            if (qr[P3(ti,k,tj)] <= qcrmin) {
              rsloper[k]  = rslopermax;
              rslopebr[k] = rsloperbmax;
              rslope2r[k] = rsloper2max;
              rslope3r[k] = rsloper3max;
            } else {
              rsloper[k]  = 1.f / LAMDAR(qr[P3(ti,k,tj)], den[P3(ti,k,tj)]);
              rslopebr[k] = expf(logf(rsloper[k]) * bvtr);
              rslope2r[k] = rsloper[k] * rsloper[k];
              rslope3r[k] = rslope2r[k] * rsloper[k];
            }
            if (qs[P3(ti,k,tj)] <= qcrmin) {
              rslopes[k]  = rslopesmax;
              rslopebs[k] = rslopesbmax;
              rslope2s[k] = rslopes2max;
              rslope3s[k] = rslopes3max;
            } else {
              rslopes[k]  = 1.f / LAMDAS(qs[P3(ti,k,tj)], den[P3(ti,k,tj)], n0sfac[k]);
              rslopebs[k] = expf(logf(rslopes[k]) * bvts);
              rslope2s[k] = rslopes[k] * rslopes[k];
              rslope3s[k] = rslope2s[k] * rslopes[k];
            }
            denfac[k] = sqrtf(den0 / den[P3(ti,k,tj)]);
            w1[k] = pvtr * rslopebr[k] * denfac[k] / delz[P3(ti,k,tj)];
            w2[k] = pvts * rslopebs[k] * denfac[k] / delz[P3(ti,k,tj)];

            w     = MAX(w1[k], w2[k]);
            numdt = MAX((int)(w * dtcld + .5f + .5f), 1);
            if (numdt >= mstep) mstep = numdt;

            float temp = den[P3(ti,k,tj)] * MAX(qi[P3(ti,k,tj)], qmin);
            temp = sqrtf(sqrtf(temp * temp * temp));
            xni[k] = MIN(MAX(5.38e7f * temp, 1.e3f), 1.e6f);
          }
          rmstep = 1.f / mstep;

          int n;
          float dtcldden, coeres, rdelz;
          float den_k, falk1_k, falk1_kp1, fall1_k, delz_k, delz_kp1;
          float        falk2_k, falk2_kp1, fall2_k;

          for (n = 1; n <= mstep; n++) {
            k           = kpe - 1;
            den_k       = den[P3(ti,k,tj)];
            falk1_kp1   = den_k * qr[P3(ti,k,tj)] * w1[k] * rmstep;
            falk2_kp1   = den_k * qs[P3(ti,k,tj)] * w2[k] * rmstep;
            dtcldden    = dtcld / den_k;
            qr[P3(ti,k,tj)] = MAX(qr[P3(ti,k,tj)] - falk1_kp1 * dtcldden, 0.f);
            qs[P3(ti,k,tj)] = MAX(qs[P3(ti,k,tj)] - falk2_kp1 * dtcldden, 0.f);
            delz_kp1    = delz[P3(ti,k,tj)];

            for (k = kpe - 2; k >= kps - 1; k--) {
              den_k    = den[P3(ti,k,tj)];
              falk1_k  = den_k * qr[P3(ti,k,tj)] * w1[k] * rmstep;
              fall1_k  = falk1_k;
              falk2_k  = den_k * qs[P3(ti,k,tj)] * w2[k] * rmstep;
              fall2_k  = falk2_k;
              dtcldden = dtcld / den_k;
              delz_k   = delz[P3(ti,k,tj)];
              rdelz    = 1.f / delz_k;
              qr[P3(ti,k,tj)] = MAX(qr[P3(ti,k,tj)] -
                (falk1_k - falk1_kp1 * delz_kp1 * rdelz) * dtcldden, 0.f);
              qs[P3(ti,k,tj)] = MAX(qs[P3(ti,k,tj)] -
                (falk2_k - falk2_kp1 * delz_kp1 * rdelz) * dtcldden, 0.f);
              delz_kp1  = delz_k;
              falk1_kp1 = falk1_k;
              falk2_kp1 = falk2_k;
            }

            for (k = kpe - 1; k >= kps - 1; k--) {
              if (t[k] > t0c && qs[P3(ti,k,tj)] > 0.f) {
                xlf    = xlf0;
                w3[k]  = VENFAC(p[P3(ti,k,tj)], t[k], den[P3(ti,k,tj)]);
                coeres = rslope2s[k] * sqrtf(rslopes[k] * rslopebs[2]);
                psmlt_reg = XKA(t[k], den[P3(ti,k,tj)]) / xlf * (t0c - t[k]) *
                  pi / 2.f * n0sfac[k] *
                  (precs1 * rslope2s[k] + precs2 * w3[k] * coeres);
                psmlt_reg = MIN(MAX(psmlt_reg * dtcld * rmstep,
                                    -qs[P3(ti,k,tj)] * rmstep), 0.f);
                qs[P3(ti,k,tj)] += psmlt_reg;
                qr[P3(ti,k,tj)] -= psmlt_reg;
                t[k] += xlf / CPMCAL(q[P3(ti,k,tj)]) * psmlt_reg;
              }
            }
          }

          // Vice: fallout of ice crystals
          mstep = 1; numdt = 1;
          for (k = kpe - 1; k >= kps - 1; k--) {
            if (qi[P3(ti,k,tj)] <= 0.f) {
              w2[k] = 0.f;
            } else {
              xmi       = den[P3(ti,k,tj)] * qi[P3(ti,k,tj)] / xni[k];
              diameter  = MAX(MIN(dicon * sqrtf(xmi), dimax), 1.e-25f);
              w1[k]     = 1.49e4f * expf(logf(diameter) * (1.31f));
              w2[k]     = w1[k] / delz[P3(ti,k,tj)];
            }
            numdt = MAX((int)(w2[k] * dtcld + .5f + .5f), 1);
            if (numdt > mstep) mstep = numdt;
          }
          rmstep = 1.f / mstep;

          float falkc_k = 0.f, falkc_kp1 = 0.f, fallc_k = 0.f, fallc_kp1 = 0.f;
          for (n = 1; n <= mstep; n++) {
            k          = kpe - 1;
            den_k      = den[P3(ti,k,tj)];
            falkc_kp1  = den_k * qi[P3(ti,k,tj)] * w2[k] * rmstep;
            fallc_kp1  = fallc_kp1 + falkc_kp1;
            qi[P3(ti,k,tj)] = MAX(qi[P3(ti,k,tj)] - falkc_kp1 * dtcld / den_k, 0.f);
            delz_kp1   = delz[P3(ti,k,tj)];

            for (k = kpe - 2; k >= kps - 1; k--) {
              den_k    = den[P3(ti,k,tj)];
              falkc_k  = den_k * qi[P3(ti,k,tj)] * w2[k] * rmstep;
              fallc_k  = fallc_k + falkc_k;
              delz_k   = delz[P3(ti,k,tj)];
              qi[P3(ti,k,tj)] = MAX(qi[P3(ti,k,tj)] -
                (falkc_k - falkc_kp1 * delz_kp1 / delz_k) * dtcld / den_k, 0.f);
              delz_kp1  = delz_k;
              falkc_kp1 = falkc_k;
              fallc_kp1 = fallc_k;
            }
          }

          float fallsum     = fall1_k + fall2_k + fallc_k;
          float fallsum_qsi = fall2_k + fallc_k;

          rainncv[P2(ti,tj)] = 0.f;
          if (fallsum > 0.f) {
            rainncv[P2(ti,tj)] = fallsum * delz[P3(ti,1,tj)] / denr * dtcld * 1000.f;
            rain[P2(ti,tj)]    = fallsum * delz[P3(ti,1,tj)] / denr * dtcld * 1000.f +
                                 rain[P2(ti,tj)];
          }
          snowncv[P2(ti,tj)] = 0.f;
          if (fallsum_qsi > 0.f) {
            snowncv[P2(ti,tj)] = fallsum_qsi * delz[P3(ti,0,tj)] / denr * dtcld * 1000.f;
            snow[P2(ti,tj)]    = fallsum_qsi * delz[P3(ti,0,tj)] / denr * dtcld * 1000.f +
                                 snow[P2(ti,tj)];
          }
          sr[P2(ti,tj)] = 0.f;
          if (fallsum > 0.f)
            sr[P2(ti,tj)] = fallsum_qsi * delz[P3(ti,0,tj)] / denr * dtcld * 1000.f /
                            (rainncv[P2(ti,tj)] + 1.e-12f);

          // ---- microphysics source/sink k-loop ----------------------------
          for (k = kps - 1; k <= kpe - 1; k++) {
            prevp_reg = 0.f; psdep_reg = 0.f; praut_reg = 0.f;
            psaut_reg = 0.f; pracw_reg = 0.f; psaci_reg = 0.f;
            psacw_reg = 0.f; pigen_reg = 0.f; pidep_reg = 0.f;
            pcond_reg = 0.f; psevp_reg = 0.f;

            q_k   = q[P3(ti,k,tj)];
            t_k   = t[k];
            qr_k  = qr[P3(ti,k,tj)];
            qc_k  = qc[P3(ti,k,tj)];
            qs_k  = qs[P3(ti,k,tj)];
            qi_k  = qi[P3(ti,k,tj)];
            qs1_k = qs1[k];
            qs2_k = qs2[k];
            cpm_k = cpm[k];
            xl_k  = xl[k];

            float supcol = t0c - t_k;
            xlf = xls - xl_k;
            if (supcol < 0.f) xlf = xlf0;

            // pimlt: instantaneous melting of cloud ice (T>T0: I->C)
            if (supcol < 0.f && qi_k > 0.f) {
              qc_k = qc_k + qi_k;
              t_k  = t_k - xlf / cpm_k * qi_k;
              qi_k = 0.f;
            }
            // pihmf: homogeneous freezing (T<-40C: C->I)
            if (supcol > 40.f && qc_k > 0.f) {
              qi_k = qi_k + qc_k;
              t_k  = t_k + xlf / cpm_k * qc_k;
              qc_k = 0.f;
            }
            // pihtf: heterogeneous freezing (T0>T>-40C: C->I)
            if (supcol > 0.f && qc_k > 0.f) {
              float pfrzdtc = MIN(pfrz1 * (expf(pfrz2 * supcol) - 1.f) *
                den[P3(ti,k,tj)] / denr / xncr * qc_k * qc_k * dtcld, qc_k);
              qi_k = qi_k + pfrzdtc;
              t_k  = t_k + xlf / cpm_k * pfrzdtc;
              qc_k = qc_k - pfrzdtc;
            }
            // psfrz: freezing of rain water (T<T0, R->S)
            if (supcol > 0.f && qr_k > 0.f) {
              float temp = rsloper[k];
              temp = temp * temp * temp * temp * temp * temp * temp;
              float pfrzdtr = MIN(20.f * pi * pi * pfrz1 * n0r * denr /
                den[P3(ti,k,tj)] * (expf(pfrz2 * supcol) - 1.f) * temp * dtcld, qr_k);
              qs_k = qs_k + pfrzdtr;
              t_k  = t_k + xlf / cpm_k * pfrzdtr;
              qr_k = qr_k - pfrzdtr;
            }

            n0sfac[k] = MAX(MIN(expf(alpha_c * supcol), n0smax / n0s), 1.f);
            if (qr_k <= qcrmin) {
              rsloper[k]  = rslopermax;
              rslopebr[k] = rsloperbmax;
              rslope2r[k] = rsloper2max;
              rslope3r[k] = rsloper3max;
            } else {
              rsloper[k]  = 1.f / sqrtf(sqrtf(pidn0r / (qr_k * den[P3(ti,k,tj)])));
              rslopebr[k] = expf(logf(rsloper[k]) * bvtr);
              rslope2r[k] = rsloper[k] * rsloper[k];
              rslope3r[k] = rslope2r[k] * rsloper[k];
            }
            if (qs_k <= qcrmin) {
              rslopes[k]  = rslopesmax;
              rslopebs[k] = rslopesbmax;
              rslope2s[k] = rslopes2max;
              rslope3s[k] = rslopes3max;
            } else {
              rslopes[k]  = 1.f / sqrtf(sqrtf(pidn0s * n0sfac[k] / (qs_k * den[P3(ti,k,tj)])));
              rslopebs[k] = expf(logf(rslopes[k]) * bvts);
              rslope2s[k] = rslopes[k] * rslopes[k];
              rslope3s[k] = rslope2s[k] * rslopes[k];
            }

            w1_k = DIFFAC(xl_k,  p[P3(ti,k,tj)], t_k, den[P3(ti,k,tj)], qs1_k);
            w2_k = DIFFAC(xls,   p[P3(ti,k,tj)], t_k, den[P3(ti,k,tj)], qs2_k);
            w3_k = VENFAC(p[P3(ti,k,tj)], t_k, den[P3(ti,k,tj)]);

            float supsat = MAX(q_k, qmin) - qs1_k;
            float satdt  = supsat / dtcld;

            // praut: auto conversion C->R
            if (qc_k > qc0) {
              praut_reg = qck1 * expf(logf(qc_k) * (7.f / 3.f));
              praut_reg = MIN(praut_reg, qc_k / dtcld);
            }
            // pracw: accretion of cloud water by rain C->R
            if (qr_k > qcrmin && qc_k > qmin)
              pracw_reg = MIN(pacrr * rslope3r[k] * rslopebr[k] *
                              qc_k * denfac[k], qc_k / dtcld);
            // prevp: evaporation/condensation of rain V->R or R->V
            if (qr_k > 0.f) {
              coeres    = rslope2r[k] * sqrtf(rsloper[k] * rslopebr[k]);
              prevp_reg = (rh1[k] - 1.f) *
                (precr1 * rslope2r[k] + precr2 * w3_k * coeres) / w1_k;
              if (prevp_reg < 0.f) {
                prevp_reg = MAX(prevp_reg, -qr_k / dtcld);
                prevp_reg = MAX(prevp_reg, satdt / 2.f);
              } else {
                prevp_reg = MIN(prevp_reg, satdt / 2.f);
              }
            }

            float rdtcld = 1.f / dtcld;
            supsat = MAX(q_k, qmin) - qs2_k;
            satdt  = supsat / dtcld;
            int ifsat = 0;

            float temp = den[P3(ti,k,tj)] * MAX(qi_k, qmin);
            temp  = sqrtf(sqrtf(temp * temp * temp));
            xni[k] = MIN(MAX(5.38e7f * temp, 1.e3f), 1.e6f);

            float eacrs = expf(0.07f * (-supcol));

            // psacw: accretion of cloud water by snow C->S (or C->R if T>=T0)
            if (qs_k > qcrmin && qc_k > qmin)
              psacw_reg = MIN(pacrc * n0sfac[k] * rslope3s[k] * rslopebs[k] *
                              qc_k * denfac[k], qc_k * rdtcld);

            if (supcol > 0.f) {
              if (qs_k > qcrmin && qi_k > qmin) {
                xmi      = den[P3(ti,k,tj)] * qi_k / xni[k];
                diameter = MIN(dicon * sqrtf(xmi), dimax);
                vt2i     = 1.49e4f * powf(diameter, 1.31f);
                vt2s     = pvts * rslopebs[k] * denfac[k];
                acrfac   = 2.f * rslope3s[k] + 2.f * diameter * rslope2s[k] +
                           diameter * diameter * rslopes[k];
                // psaci: accretion of cloud ice by snow I->S
                psaci_reg = pi * qi_k * eacrs * n0s * n0sfac[k] *
                            fabsf(vt2s - vt2i) * acrfac * .25f;
              }
              // pidep: deposition/sublimation of ice V->I or I->V
              if (qi_k > 0.f && ifsat != 1) {
                xmi      = den[P3(ti,k,tj)] * qi_k / xni[k];
                diameter = dicon * sqrtf(xmi);
                pidep_reg = 4.f * diameter * xni[k] * (rh2[k] - 1.f) / w2_k;
                supice    = satdt - prevp_reg;
                if (pidep_reg < 0.f) {
                  pidep_reg = MAX(MAX(pidep_reg, satdt * .5f), supice);
                  pidep_reg = MAX(pidep_reg, -qi_k * rdtcld);
                } else {
                  pidep_reg = MIN(MIN(pidep_reg, satdt * .5f), supice);
                }
                if (fabsf(prevp_reg + pidep_reg) >= fabsf(satdt)) ifsat = 1;
              }
              // psdep: deposition/sublimation of snow V->S or S->V
              if (qs_k > 0.f && ifsat != 1) {
                coeres    = rslope2s[k] * sqrtf(rslopes[k] * rslopebs[k]);
                psdep_reg = (rh2[k] - 1.f) * n0sfac[k] *
                  (precs1 * rslope2s[k] + precs2 * w3_k * coeres) / w2_k;
                supice = satdt - prevp_reg - pidep_reg;
                if (psdep_reg < 0.f) {
                  psdep_reg = MAX(psdep_reg, -qs_k * rdtcld);
                  psdep_reg = MAX(MAX(psdep_reg, satdt * .5f), supice);
                } else {
                  psdep_reg = MIN(MIN(psdep_reg, satdt * .5f), supice);
                }
                if (fabsf(prevp_reg + pidep_reg + psdep_reg) >= fabsf(satdt)) ifsat = 1;
              }
              // pigen: nucleation of ice from vapor V->I
              if (supsat > 0.f && ifsat != 1) {
                supice    = satdt - prevp_reg - pidep_reg - psdep_reg;
                xni0      = 1.e3f * expf(0.1f * supcol);
                roqi0     = 4.92e-11f * expf(logf(xni0) * (1.33f));
                pigen_reg = MAX(0.f, (roqi0 / den[P3(ti,k,tj)] - MAX(qi_k, 0.f)) * rdtcld);
                pigen_reg = MIN(MIN(pigen_reg, satdt), supice);
              }
              // psaut: conversion/aggregation of ice to snow I->S
              if (qi_k > 0.f) {
                qimax     = roqimax / den[P3(ti,k,tj)];
                psaut_reg = MAX(0.f, (qi_k - qimax) * rdtcld);
              }
            }
            // psevp: evaporation of melting snow S->V (T>T0)
            if (supcol < 0.f) {
              if (qs_k > 0.f && rh1[k] < 1.f)
                psevp_reg = psdep_reg * w2_k / w1_k;
              psevp_reg = MIN(MAX(psevp_reg, -qs_k * rdtcld), 0.f);
            }

            // Mass conservation + feedback to large scale
            if (t_k <= t0c) {
              value  = MAX(qmin, qc_k);
              source = (praut_reg + pracw_reg + psacw_reg) * dtcld;
              if (source > value) {
                factor    = value / source;
                praut_reg *= factor; pracw_reg *= factor; psacw_reg *= factor;
              }
              value  = MAX(qmin, qi_k);
              source = (psaut_reg + psaci_reg - pigen_reg - pidep_reg) * dtcld;
              if (source > value) {
                factor    = value / source;
                psaut_reg *= factor; psaci_reg *= factor;
                pigen_reg *= factor; pidep_reg *= factor;
              }
              value  = MAX(qmin, qr_k);
              source = (-praut_reg + pracw_reg - prevp_reg) * dtcld;
              if (source > value) {
                factor    = value / source;
                praut_reg *= factor; pracw_reg *= factor; prevp_reg *= factor;
              }
              value  = MAX(qmin, qs_k);
              source = (-psdep_reg + psaut_reg - psaci_reg - psacw_reg) * dtcld;
              if (source > value) {
                factor    = value / source;
                psdep_reg *= factor; psaut_reg *= factor;
                psaci_reg *= factor; psacw_reg *= factor;
              }

              w3_k  = -(prevp_reg + psdep_reg + pigen_reg + pidep_reg);
              q_k   = q_k + w3_k * dtcld;
              qc_k  = MAX(qc_k - (praut_reg + pracw_reg + psacw_reg) * dtcld, 0.f);
              qr_k  = MAX(qr_k + (praut_reg + pracw_reg + prevp_reg) * dtcld, 0.f);
              qi_k  = MAX(qi_k - (psaut_reg + psaci_reg - pigen_reg - pidep_reg) * dtcld, 0.f);
              qs_k  = MAX(qs_k + (psdep_reg + psaut_reg + psaci_reg + psacw_reg) * dtcld, 0.f);
              xlf   = xls - xl_k;
              xlwork2 = -xls * (psdep_reg + pidep_reg + pigen_reg)
                        - xl_k * prevp_reg - xlf * psacw_reg;
              t_k   = t_k - xlwork2 / cpm_k * dtcld;
            } else {
              value  = MAX(qmin, qc_k);
              source = (praut_reg + pracw_reg + psacw_reg) * dtcld;
              if (source > value) {
                factor    = value / source;
                praut_reg *= factor; pracw_reg *= factor; psacw_reg *= factor;
              }
              value  = MAX(qmin, qr_k);
              source = (-praut_reg - pracw_reg - prevp_reg - psacw_reg) * dtcld;
              if (source > value) {
                factor    = value / source;
                praut_reg *= factor; pracw_reg *= factor;
                prevp_reg *= factor; psacw_reg *= factor;
              }
              value  = MAX(qcrmin, qs_k);
              source = (-psevp_reg) * dtcld;
              if (source > value) {
                factor    = value / source;
                psevp_reg *= factor;
              }
              w3_k  = -(prevp_reg + psevp_reg);
              q_k   = q_k + w3_k * dtcld;
              qc_k  = MAX(qc_k - (praut_reg + pracw_reg + psacw_reg) * dtcld, 0.f);
              qr_k  = MAX(qr_k + (praut_reg + pracw_reg + prevp_reg + psacw_reg) * dtcld, 0.f);
              qs_k  = MAX(qs_k + psevp_reg * dtcld, 0.f);
              xlf   = xls - xl_k;
              xlwork2 = -xl_k * (prevp_reg + psevp_reg);
              t_k   = t_k - xlwork2 / cpm_k * dtcld;
            }

            // Inline fpvs expansion
            cvap  = cpv;
            ttp   = t0c + 0.01f;
            dldt  = cvap - cliq;
            xa    = -dldt / rv;
            xb    = xa + hvap / (rv * ttp);
            dldti = cvap - cice;
            xai   = -dldti / rv;
            xbi   = xai + hsub / (rv * ttp);
            tr    = ttp / t_k;
            qs1_k = psat * expf(logf(tr) * xa) * expf(xb * (1.f - tr));
            qs1_k = ep2 * qs1_k / (p[P3(ti,k,tj)] - qs1_k);
            qs1_k = MAX(qs1_k, qmin);

            // pcond: condensation/evaporation of cloud water
            w1_k = (MAX(q_k, qmin) - qs1_k) /
              (1.f + xl_k * xl_k / (rv * cpm_k) * qs1_k / (t_k * t_k));
            pcond_reg = MIN(MAX(w1_k / dtcld, 0.f), MAX(q_k, 0.f) / dtcld);
            if (qc_k > 0.f && w1_k < 0.f)
              pcond_reg = MAX(w1_k, -qc_k) / dtcld;
            q_k  = q_k  - pcond_reg * dtcld;
            qc_k = MAX(qc_k + pcond_reg * dtcld, 0.f);
            t_k  = t_k  + pcond_reg * xl_k / cpm_k * dtcld;

            if (qc_k <= qmin) qc_k = 0.f;
            if (qi_k <= qmin) qi_k = 0.f;

            q [P3(ti,k,tj)] = q_k;
            t [k]           = t_k;
            qr[P3(ti,k,tj)] = qr_k;
            qc[P3(ti,k,tj)] = qc_k;
            qs[P3(ti,k,tj)] = qs_k;
            qi[P3(ti,k,tj)] = qi_k;
            qs1[k]          = qs1_k;
          } // end k-loop
        }   // end loop (minor timesteps)

        for (k = kps - 1; k <= kpe - 1; k++)
          th[P3(ti,k,tj)] = t[k] / pii[P3(ti,k,tj)];

      } // end if (ig < ... && jg < ...)
    }); // end Kokkos::parallel_for

  Kokkos::fence();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    using exec_space = Kokkos::DefaultExecutionSpace;

    float *th, *pii, *q, *qc, *qi, *qr, *qs;
    float *den, *p, *delz;
    float *rain, *rainncv, *sr, *snow, *snowncv;

    float delt = 10.f;
    int ims = 0, ime = 59, jms = 0, jme = 45, kms = 0, kme = 2;
    int ips = 0, ipe = 59, jps = 0, jpe = 45, kps = 0, kpe = 2;
    int d3 = (ime - ims + 1) * (jme - jms + 1) * (kme - kms + 1);
    int d2 = (ime - ims + 1) * (jme - jms + 1);

    int dips = 0, dipe = (ipe - ips + 1);
    int djps = 0, djpe = (jpe - jps + 1);
    int dkps = 0, dkpe = (kpe - kps + 1);

    float rain_sum = 0.f, snow_sum = 0.f;
    long time_ns = 0;

    // Allocate device Views once
    Kokkos::View<float*, exec_space> d_th     ("th",      d3);
    Kokkos::View<float*, exec_space> d_pii    ("pii",     d3);
    Kokkos::View<float*, exec_space> d_q      ("q",       d3);
    Kokkos::View<float*, exec_space> d_qc     ("qc",      d3);
    Kokkos::View<float*, exec_space> d_qi     ("qi",      d3);
    Kokkos::View<float*, exec_space> d_qr     ("qr",      d3);
    Kokkos::View<float*, exec_space> d_qs     ("qs",      d3);
    Kokkos::View<float*, exec_space> d_den    ("den",     d3);
    Kokkos::View<float*, exec_space> d_p      ("p",       d3);
    Kokkos::View<float*, exec_space> d_delz   ("delz",    d3);
    Kokkos::View<float*, exec_space> d_rain   ("rain",    d2);
    Kokkos::View<float*, exec_space> d_rainncv("rainncv", d2);
    Kokkos::View<float*, exec_space> d_sr     ("sr",      d2);
    Kokkos::View<float*, exec_space> d_snow   ("snow",    d2);
    Kokkos::View<float*, exec_space> d_snowncv("snowncv", d2);

    for (int i = 0; i < repeat; i++) {
      // Allocate and initialise host arrays
      ALLOC3(th); ALLOC3(pii); ALLOC3(q);  ALLOC3(qc); ALLOC3(qi);
      ALLOC3(qr); ALLOC3(qs); ALLOC3(den); ALLOC3(p);  ALLOC3(delz);
      ALLOC2(rain); ALLOC2(rainncv); ALLOC2(sr); ALLOC2(snow); ALLOC2(snowncv);

      int remx = (ipe - ips + 1) % XXX != 0 ? 1 : 0;
      int remy = (jpe - jps + 1) % YYY != 0 ? 1 : 0;
      const int teamX = (ipe - ips + 1) / XXX + remx;
      const int teamY = (jpe - jps + 1) / YYY + remy;

      // Copy host data to device
      auto copy_to_device = [&](auto& dv, const float* src, int n) {
        auto hv = Kokkos::create_mirror_view(dv);
        for (int j = 0; j < n; j++) hv(j) = src[j];
        Kokkos::deep_copy(dv, hv);
      };
      copy_to_device(d_th,      th,      d3);
      copy_to_device(d_pii,     pii,     d3);
      copy_to_device(d_q,       q,       d3);
      copy_to_device(d_qc,      qc,      d3);
      copy_to_device(d_qi,      qi,      d3);
      copy_to_device(d_qr,      qr,      d3);
      copy_to_device(d_qs,      qs,      d3);
      copy_to_device(d_den,     den,     d3);
      copy_to_device(d_p,       p,       d3);
      copy_to_device(d_delz,    delz,    d3);
      copy_to_device(d_rainncv, rainncv, d2);
      copy_to_device(d_snowncv, snowncv, d2);
      copy_to_device(d_sr,      sr,      d2);
      copy_to_device(d_rain,    rain,    d2);
      copy_to_device(d_snow,    snow,    d2);

      auto start = std::chrono::steady_clock::now();

      wsm(d_th.data(), d_pii.data(), d_q.data(), d_qc.data(), d_qi.data(),
          d_qr.data(), d_qs.data(), d_den.data(), d_p.data(), d_delz.data(),
          d_rain.data(), d_rainncv.data(), d_sr.data(),
          d_snow.data(), d_snowncv.data(),
          delt,
          dips + 1, (ipe - ips + 1),   // ids, ide
          djps + 1, (jpe - jps + 1),   // jds, jde
          dkps + 1, (kpe - kps + 1),   // kds, kde
          dips + 1, dipe,              // ims, ime
          djps + 1, djpe,              // jms, jme
          dkps + 1, dkpe,              // kms, kme
          dips + 1, dipe,              // ips, ipe
          djps + 1, djpe,              // jps, jpe
          dkps + 1, dkpe,              // kps, kpe
          teamX, teamY);

      auto end = std::chrono::steady_clock::now();
      time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      // Copy results back
      {
        auto hv_rain = Kokkos::create_mirror_view(d_rain);
        auto hv_snow = Kokkos::create_mirror_view(d_snow);
        Kokkos::deep_copy(hv_rain, d_rain);
        Kokkos::deep_copy(hv_snow, d_snow);
        rain_sum = snow_sum = 0.f;
        for (int j = 0; j < d2; j++) { rain_sum += hv_rain(j); snow_sum += hv_snow(j); }
      }

      FREE(th); FREE(pii); FREE(q);  FREE(qc); FREE(qi);
      FREE(qr); FREE(qs); FREE(den); FREE(p);  FREE(delz);
      FREE(rain); FREE(rainncv); FREE(sr); FREE(snow); FREE(snowncv);
    }

    printf("Average kernel execution time: %lf (ms)\n",
           (time_ns * 1e-6) / repeat);
    printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);
  }
  Kokkos::finalize();
  return 0;
}
