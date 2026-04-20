// Kokkos port of miniDGS-cuda: Discontinuous Galerkin Maxwell solver.
// Original CUDA source: src/MaxwellsKernel3d.cu
//
// Three kernels are ported:
//   MaxwellsGPU_VOL_Kernel3D  – volume derivative computation (curl of E/H)
//   MaxwellsGPU_SURF_Kernel3D – surface flux via LIFT matrix
//   MaxwellsGPU_RK_Kernel3D   – low-storage Runge–Kutta update
//
// CUDA → Kokkos mapping:
//   texture fetch       → regular Kokkos::View read
//   __shared__ + sync   → per-element scratch View (SURF uses pre-allocated flux)
//   <<<K, BSIZE>>>      → flat Kokkos::RangePolicy(0, K * p_Np)
//   flat RK kernel      → Kokkos::RangePolicy

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

// -------------------------------------------------------------------------
// Problem parameters – match N=6, NDG3d (tetrahedron) from original Makefile.
// -------------------------------------------------------------------------
static constexpr int p_N       = 6;
static constexpr int p_Nfp     = (p_N + 1) * (p_N + 2) / 2;              // 28
static constexpr int p_Np      = (p_N + 1) * (p_N + 2) * (p_N + 3) / 6;  // 84
static constexpr int p_Nfields  = 6;   // Hx,Hy,Hz,Ex,Ey,Ez
static constexpr int p_Nfaces   = 4;   // tetrahedron
static constexpr int BSIZE      = 16 * ((p_Np + 15) / 16);               // 96

// Low-storage 5-stage RK4 coefficients
static constexpr float rk4a[5] = {
     0.0f,
    -567301805773.0f  / 1357537059087.0f,
    -2404267990393.0f / 2016746695238.0f,
    -3550918686646.0f / 2091501179385.0f,
    -1275806237668.0f /  842570457699.0f
};
static constexpr float rk4b[5] = {
     1432997174477.0f /  9575080441755.0f,
     5161836677717.0f / 13612068292357.0f,
     1720146321549.0f /  2090206949498.0f,
     3134564353537.0f /  4481467310338.0f,
     2277821191437.0f / 14882151754819.0f
};

// =========================================================================
// VOL kernel: flat parallel_for over K * p_Np work items.
// Each work item (k, n) computes the volume contribution for node n in
// element k and writes to rhsQ.  No shared  reads Q from global.memory 
// =========================================================================
struct VolKernel {
    Kokkos::View<const float*> Q;       // K * p_Nfields * BSIZE
    Kokkos::View<const float*> DrDsDt;  // BSIZE * BSIZE * 4  (Dr,Ds,Dt,0)
    Kokkos::View<const float*> vgeo;    // K * 12

    Kokkos::View<float*> rhsQ;          // K * p_Nfields * BSIZE

    KOKKOS_INLINE_FUNCTION
    void operator()(int gid) const {
        const int k = gid / p_Np;
        const int n = gid % p_Np;

        // Geometric factors (12 floats per element)
        const int gbase = k * 12;
        const float drdx = vgeo[gbase+ 0], drdy = vgeo[gbase+ 1], drdz = vgeo[gbase+ 2];
        const float dsdx = vgeo[gbase+ 4], dsdy = vgeo[gbase+ 5], dsdz = vgeo[gbase+ 6];
        const float dtdx = vgeo[gbase+ 8], dtdy = vgeo[gbase+ 9], dtdz = vgeo[gbase+10];

        const int qbase = k * p_Nfields * BSIZE;

        float dHxdr=0,dHxds=0,dHxdt=0;
        float dHydr=0,dHyds=0,dHydt=0;
        float dHzdr=0,dHzds=0,dHzdt=0;
        float dExdr=0,dExds=0,dExdt=0;
        float dEydr=0,dEyds=0,dEydt=0;
        float dEzdr=0,dEzds=0,dEzdt=0;

        // Accumulate volume derivatives: D * Q
        for (int m = 0; m < p_Np; m++) {
            // DrDsDt layout: [n + m*BSIZE] → {Dr, Ds, Dt, _}
            const int didx = 4 * (n + m * BSIZE);
            const float Dr = DrDsDt[didx + 0];
            const float Ds = DrDsDt[didx + 1];
            const float Dt = DrDsDt[didx + 2];

            const float Hx = Q[qbase + m            ];
            const float Hy = Q[qbase + m +   BSIZE  ];
            const float Hz = Q[qbase + m + 2*BSIZE  ];
            const float Ex = Q[qbase + m + 3*BSIZE  ];
            const float Ey = Q[qbase + m + 4*BSIZE  ];
            const float Ez = Q[qbase + m + 5*BSIZE  ];

            dHxdr += Dr*Hx; dHxds += Ds*Hx; dHxdt += Dt*Hx;
            dHydr += Dr*Hy; dHyds += Ds*Hy; dHydt += Dt*Hy;
            dHzdr += Dr*Hz; dHzds += Ds*Hz; dHzdt += Dt*Hz;
            dExdr += Dr*Ex; dExds += Ds*Ex; dExdt += Dt*Ex;
            dEydr += Dr*Ey; dEyds += Ds*Ey; dEydt += Dt*Ey;
            dEzdr += Dr*Ez; dEzds += Ds*Ez; dEzdt += Dt*Ez;
        }

        // Maxwell curl equations: dH/dt = -curl E, dE/dt = curl H
        const int rbase = n + p_Nfields * BSIZE * k;
        rhsQ[rbase          ] = -(drdy*dEzdr+dsdy*dEzds+dtdy*dEzdt
                                 -drdz*dEydr-dsdz*dEyds-dtdz*dEydt);
        rhsQ[rbase +   BSIZE] = -(drdz*dExdr+dsdz*dExds+dtdz*dExdt
                                 -drdx*dEzdr-dsdx*dEzds-dtdx*dEzdt);
        rhsQ[rbase + 2*BSIZE] = -(drdx*dEydr+dsdx*dEyds+dtdx*dEydt
                                 -drdy*dExdr-dsdy*dExds-dtdy*dExdt);
        rhsQ[rbase + 3*BSIZE] =  (drdy*dHzdr+dsdy*dHzds+dtdy*dHzdt
                                 -drdz*dHydr-dsdz*dHyds-dtdz*dHydt);
        rhsQ[rbase + 4*BSIZE] =  (drdz*dHxdr+dsdz*dHxds+dtdz*dHxdt
                                 -drdx*dHzdr-dsdx*dHzds-dtdx*dHzdt);
        rhsQ[rbase + 5*BSIZE] =  (drdx*dHydr+dsdx*dHyds+dtdx*dHydt
                                 -drdy*dHxdr-dsdy*dHxds-dtdy*dHxdt);
    }
};

// =========================================================================
// SURF kernel phase 1: compute face fluxes for all elements × surface nodes.
// Replaces the first half of MaxwellsGPU_SURF_Kernel3D.
// Output: fluxQ[K * p_Nfields * NfpNf]
// =========================================================================
struct SurfFluxKernel {
    static constexpr int NfpNf = p_Nfp * p_Nfaces;

    Kokkos::View<const float*> Q;        // K * p_Nfields * BSIZE
    Kokkos::View<const float*> surfinfo; // K * NfpNf * 7
    Kokkos::View<float*>       fluxQ;    // K * p_Nfields * NfpNf  (output)

    KOKKOS_INLINE_FUNCTION
    void operator()(int gid) const {
        const int k = gid / NfpNf;
        const int n = gid % NfpNf;

        const int m    = 7 * (k * NfpNf) + n;
        const int idM  = (int)surfinfo[m + 0*NfpNf];
        int       idP  = (int)surfinfo[m + 1*NfpNf];
        const float Fsc = surfinfo[m + 2*NfpNf];
        const float Bsc = surfinfo[m + 3*NfpNf];
        const float nx  = surfinfo[m + 4*NfpNf];
        const float ny  = surfinfo[m + 5*NfpNf];
        const float nz  = surfinfo[m + 6*NfpNf];

        float dHx, dHy, dHz, dEx, dEy, dEz;
        if (idP < 0) {
            // Boundary: zero external state
            dHx = Fsc*(0.0f - Q[idM + 0*BSIZE]);
            dHy = Fsc*(0.0f - Q[idM + 1*BSIZE]);
            dHz = Fsc*(0.0f - Q[idM + 2*BSIZE]);
            dEx = Fsc*(0.0f - Q[idM + 3*BSIZE]);
            dEy = Fsc*(0.0f - Q[idM + 4*BSIZE]);
            dEz = Fsc*(0.0f - Q[idM + 5*BSIZE]);
        } else {
            dHx = Fsc*(Q[idP + 0*BSIZE] - Q[idM + 0*BSIZE]);
            dHy = Fsc*(Q[idP + 1*BSIZE] - Q[idM + 1*BSIZE]);
            dHz = Fsc*(Q[idP + 2*BSIZE] - Q[idM + 2*BSIZE]);
            dEx = Fsc*(Bsc*Q[idP + 3*BSIZE] - Q[idM + 3*BSIZE]);
            dEy = Fsc*(Bsc*Q[idP + 4*BSIZE] - Q[idM + 4*BSIZE]);
            dEz = Fsc*(Bsc*Q[idP + 5*BSIZE] - Q[idM + 5*BSIZE]);
        }

        const float ndotdH = nx*dHx + ny*dHy + nz*dHz;
        const float ndotdE = nx*dEx + ny*dEy + nz*dEz;

        const int fbase = k * p_Nfields * NfpNf + n;
        fluxQ[fbase + 0*NfpNf] = -ny*dEz + nz*dEy + dHx - ndotdH*nx;
        fluxQ[fbase + 1*NfpNf] = -nz*dEx + nx*dEz + dHy - ndotdH*ny;
        fluxQ[fbase + 2*NfpNf] = -nx*dEy + ny*dEx + dHz - ndotdH*nz;
        fluxQ[fbase + 3*NfpNf] =  ny*dHz - nz*dHy + dEx - ndotdE*nx;
        fluxQ[fbase + 4*NfpNf] =  nz*dHx - nx*dHz + dEy - ndotdE*ny;
        fluxQ[fbase + 5*NfpNf] =  nx*dHy - ny*dHx + dEz - ndotdE*nz;
    }
};

// =========================================================================
// SURF kernel phase 2: apply LIFT matrix to fluxQ and accumulate into rhsQ.
// Replaces the second half of MaxwellsGPU_SURF_Kernel3D.
// =========================================================================
struct SurfLiftKernel {
    static constexpr int NfpNf = p_Nfp * p_Nfaces;

    Kokkos::View<const float*> fluxQ; // K * p_Nfields * NfpNf
    Kokkos::View<const float*> LIFT;  // p_Np * NfpNf (row-major: LIFT[n, j])
    Kokkos::View<float*>       rhsQ;  // K * p_Nfields * BSIZE

    KOKKOS_INLINE_FUNCTION
    void operator()(int gid) const {
        const int k = gid / p_Np;
        const int n = gid % p_Np;

        float rhsHx=0,rhsHy=0,rhsHz=0;
        float rhsEx=0,rhsEy=0,rhsEz=0;

        const int fbase = k * p_Nfields * NfpNf;

        for (int j = 0; j < NfpNf; j++) {
            const float L = LIFT[n + j * p_Np]; // LIFT[n, j]
            rhsHx += L * fluxQ[fbase + j + 0*NfpNf];
            rhsHy += L * fluxQ[fbase + j + 1*NfpNf];
            rhsHz += L * fluxQ[fbase + j + 2*NfpNf];
            rhsEx += L * fluxQ[fbase + j + 3*NfpNf];
            rhsEy += L * fluxQ[fbase + j + 4*NfpNf];
            rhsEz += L * fluxQ[fbase + j + 5*NfpNf];
        }

        const int rbase = n + p_Nfields * BSIZE * k;
        rhsQ[rbase          ] += rhsHx;
        rhsQ[rbase +   BSIZE] += rhsHy;
        rhsQ[rbase + 2*BSIZE] += rhsHz;
        rhsQ[rbase + 3*BSIZE] += rhsEx;
        rhsQ[rbase + 4*BSIZE] += rhsEy;
        rhsQ[rbase + 5*BSIZE] += rhsEz;
    }
};

// =========================================================================
// RK update kernel.
// =========================================================================
struct RKUpdateKernel {
    Kokkos::View<const float*> rhsQ;
    Kokkos::View<float*>       resQ;
    Kokkos::View<float*>       Q;
    float fa, fb, fdt;

    KOKKOS_INLINE_FUNCTION
    void operator()(int n) const {
        float res = fa * resQ[n] + fdt * rhsQ[n];
        resQ[n]    = res;
        Q[n]      += fb * res;
    }
};

// =========================================================================
// Synthetic mesh initialisation (standalone, no MPI/ParMetis).
// =========================================================================
static void buildSyntheticMesh(int K,
                                std::vector<float>& vgeo,
                                std::vector<float>& DrDsDt,
                                std::vector<float>& LIFT,
                                std::vector<float>& surfinfo,
                                std::vector<float>& Q) {
    const int NfpNf = p_Nfp * p_Nfaces;

    // Unit Jacobian (drdx=dsdy=dtdz=1, rest zero)
    vgeo.assign(K * 12, 0.0f);
    for (int k = 0; k < K; k++) {
        vgeo[k*12 + 0]  = 1.0f;  // drdx
        vgeo[k*12 + 5]  = 1.0f;  // dsdy
        vgeo[k*12 + 10] = 1.0f;  // dtdz
    }

    // DrDsDt: symmetric, diagonal-dominant differentiation operator
    // Stored as float4 (Dr,Ds,Dt,0) per (n,m) pair
    DrDsDt.assign(BSIZE * BSIZE * 4, 0.0f);
    for (int n = 0; n < p_Np; n++)
        for (int m = 0; m < p_Np; m++) {
            const int didx = 4 * (n + m * BSIZE);
            float v = (n == m) ? 1.0f / (float)p_Np : -1.0f / (float)(p_Np * p_Np);
            DrDsDt[didx + 0] = v; // Dr
            DrDsDt[didx + 1] = v; // Ds
            DrDsDt[didx + 2] = v; // Dt
        }

    // LIFT: p_Np × NfpNf, stored as LIFT[n + j*p_Np]
    LIFT.assign(p_Np * NfpNf, 0.0f);
    for (int n = 0; n < p_Np; n++)
        for (int j = 0; j < NfpNf; j++)
            LIFT[n + j * p_Np] = (n == (j % p_Np)) ? 0.5f : 0.0f;

    // Surface info: 7 values per face node per element
    // Use zero-flux boundary conditions (idP = -1) for simplicity
    surfinfo.assign(K * NfpNf * 7, 0.0f);
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < NfpNf; n++) {
            const int idx = 7 * (k * NfpNf) + n;
            const int nM  = n % p_Np;
            const int idM = nM + k * p_Nfields * BSIZE;
            surfinfo[idx + 0*NfpNf] = (float)idM;
            surfinfo[idx + 1*NfpNf] = -1.0f; // boundary
            surfinfo[idx + 2*NfpNf] = 0.5f;  // Fsc
            surfinfo[idx + 3*NfpNf] = 1.0f;  // Bsc
            surfinfo[idx + 4*NfpNf] = 1.0f;  // nx
            surfinfo[idx + 5*NfpNf] = 0.0f;  // ny
            surfinfo[idx + 6*NfpNf] = 0.0f;  // nz
        }
    }

    // Initial field: Ez = cos(pi * x_n) as a synthetic wave
    Q.assign(K * p_Nfields * BSIZE, 0.0f);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < p_Np; n++) {
            float x = (float)(k * p_Np + n) / (float)(K * p_Np) * 2.0f * 3.14159265f;
            Q[n + 5*BSIZE + k*p_Nfields*BSIZE] = cosf(x); // Ez component
        }
}

// =========================================================================
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int K = 100;
        if (argc > 1) K = atoi(argv[1]);

        const int NfpNf  = p_Nfp * p_Nfaces;
        const int Ntotal = K * BSIZE * p_Nfields;

        printf("miniDGS Maxwell solver (Kokkos): K=%d elements, p_N=%d\n",
               K, p_N);
        printf("  p_Np=%d  p_Nfp=%d  p_Nfaces=%d  BSIZE=%d  Ntotal=%d\n",
               p_Np, p_Nfp, p_Nfaces, BSIZE, Ntotal);

        // ---- Build synthetic mesh data on host ----
        std::vector<float> h_vgeo, h_DrDsDt, h_LIFT, h_surfinfo, h_Q;
        buildSyntheticMesh(K, h_vgeo, h_DrDsDt, h_LIFT, h_surfinfo, h_Q);

        // ---- Allocate device views ----
        Kokkos::View<float*> d_Q       ("Q",        Ntotal);
        Kokkos::View<float*> d_rhsQ    ("rhsQ",     Ntotal);
        Kokkos::View<float*> d_resQ    ("resQ",     Ntotal);
        Kokkos::View<float*> d_fluxQ   ("fluxQ",    K * p_Nfields * NfpNf);
        Kokkos::View<float*> d_vgeo    ("vgeo",     K * 12);
        Kokkos::View<float*> d_DrDsDt  ("DrDsDt",   BSIZE * BSIZE * 4);
        Kokkos::View<float*> d_LIFT    ("LIFT",      p_Np * NfpNf);
        Kokkos::View<float*> d_surfinfo("surfinfo",  K * NfpNf * 7);

        // ---- Upload host data ----
        auto upload = [](Kokkos::View<float*>& view, const std::vector<float>& h) {
            auto hm = Kokkos::create_mirror_view(view);
            for (int i = 0; i < (int)h.size(); i++) hm(i) = h[i];
            Kokkos::deep_copy(view, hm);
        };
        upload(d_Q,        h_Q);
        upload(d_vgeo,     h_vgeo);
        upload(d_DrDsDt,   h_DrDsDt);
        upload(d_LIFT,     h_LIFT);
        upload(d_surfinfo, h_surfinfo);
        Kokkos::deep_copy(d_rhsQ,  0.0f);
        Kokkos::deep_copy(d_resQ,  0.0f);
        Kokkos::deep_copy(d_fluxQ, 0.0f);

        const float dt        = 0.001f;
        const float FinalTime = 0.005f;
        const int   nSteps    = (int)(FinalTime / dt);

        printf("Running %d time steps (5 RK stages each)...\n", nSteps);

        auto t0 = std::chrono::steady_clock::now();

        for (int step = 0; step < nSteps; step++) {
            for (int rk = 0; rk < 5; rk++) {
                const float fa  = rk4a[rk];
                const float fb  = rk4b[rk];
                const float fdt = dt;

                // ---- VOL: compute volume contributions ----
                Kokkos::deep_copy(d_rhsQ, 0.0f);
                {
                    VolKernel vk;
                    vk.Q      = d_Q;
                    vk.DrDsDt = d_DrDsDt;
                    vk.vgeo   = d_vgeo;
                    vk.rhsQ   = d_rhsQ;
                    Kokkos::parallel_for("MaxwellsVOL",
                        Kokkos::RangePolicy<>(0, K * p_Np), vk);
                }

                // ---- SURF phase 1: compute face fluxes ----
                {
                    SurfFluxKernel sk;
                    sk.Q        = d_Q;
                    sk.surfinfo = d_surfinfo;
                    sk.fluxQ    = d_fluxQ;
                    Kokkos::parallel_for("MaxwellsSURFflux",
                        Kokkos::RangePolicy<>(0, K * NfpNf), sk);
                }

                // ---- SURF phase 2: lift flux into rhsQ ----
                {
                    SurfLiftKernel lk;
                    lk.fluxQ = d_fluxQ;
                    lk.LIFT  = d_LIFT;
                    lk.rhsQ  = d_rhsQ;
                    Kokkos::parallel_for("MaxwellsSURFlift",
                        Kokkos::RangePolicy<>(0, K * p_Np), lk);
                }

                // ---- RK update ----
                {
                    RKUpdateKernel rk_k;
                    rk_k.rhsQ = d_rhsQ;
                    rk_k.resQ = d_resQ;
                    rk_k.Q    = d_Q;
                    rk_k.fa   = fa;
                    rk_k.fb   = fb;
                    rk_k.fdt  = fdt;
                    Kokkos::parallel_for("MaxwellsRK",
                        Kokkos::RangePolicy<>(0, Ntotal), rk_k);
                }
            }
        }
        Kokkos::fence();

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Flop estimates from original MaxwellsRun3d.c
        double flopsV = (double)p_Np * p_Np * 36 + p_Np * 66;
        double flopsS = (double)p_Nfp * p_Nfaces * (15 + 10 + 36)
                       + p_Np * (p_Nfaces * p_Nfp * 12 + 6);
        double flopsR = (double)p_Np * p_Nfields * 4;
        double gflops = 5.0 * nSteps
                        * (flopsV + flopsS + flopsR)
                        * ((double)K / (1.0e9 * elapsed));

        printf("Elapsed time: %.4f s,  estimated GFLOPS: %.3f\n",
               elapsed, gflops);

        // Simple correctness check
        auto hm_Q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_Q);
        float maxEz = 0.0f;
        for (int k = 0; k < K; k++)
            for (int n = 0; n < p_Np; n++) {
                float ez = hm_Q(n + 5*BSIZE + k*p_Nfields*BSIZE);
                if (fabsf(ez) > maxEz) maxEz = fabsf(ez);
            }
        printf("max|Ez| = %e\n", maxEz);
    }
    Kokkos::finalize();
    return 0;
}
