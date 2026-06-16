// Minimod - Seismic wave simulation - Kokkos port
//
// Acoustic wave propagation using 8th-order finite differences (stencil radius R=4).
// Implements Perfectly Matched Layer (PML) absorbing boundary conditions.
// All functionality consolidated from the original multi-file CUDA source.
//
// Usage: ./main [--grid N] [--nsteps M] [--niters R] [--finalio] [--warm-up]
//   --grid   N  : cubic grid size (default 100)
//   --nsteps M  : time steps per iteration (default 100)
//   --niters R  : number of timing iterations (default 1)

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

// ---------------------------------------------------------------------------
// Constants (from constants.h / constants.cu)
// ---------------------------------------------------------------------------
static constexpr float _fmax = 25.0f;
static constexpr float vmin  = 1500.0f;
static constexpr float vmax  = 4500.0f;
static constexpr float cfl   = 0.8f;

#define POW2(x) ((x) * (x))

using llint = long long int;
using uint  = unsigned int;

// ---------------------------------------------------------------------------
// Grid structure (from grid.h / grid.cu)
// ---------------------------------------------------------------------------
struct grid_t {
    llint ntaperx, ntapery, ntaperz;
    llint ndampx,  ndampy,  ndampz;
    llint nx, ny, nz;
    llint ldimx, ldimy, ldimz;
    llint dx, dy, dz;
    llint x1, x2, x3, x4, x5, x6;
    llint y1, y2, y3, y4, y5, y6;
    llint z1, z2, z3, z4, z5, z6;
    llint lx, ly, lz;
    llint ntx, nty, tsx, tsy;
};

static grid_t init_grid(llint nx, llint ny, llint nz, llint tsx = 10, llint tsy = 10) {
    grid_t g;
    g.nx = nx; g.ny = ny; g.nz = nz;
    g.dx = 20;  g.dy = 20;  g.dz = 20;
    g.lx = 4;  g.ly = 4;  g.lz = 4;
    g.ntaperx = 3; g.ntapery = 3; g.ntaperz = 3;

    g.ldimx = nx + 4 * g.lx;
    g.ldimy = ny + 2 * g.ly;
    g.ldimz = ((nz + 2 * g.lz + 31) / 32) * 32;

    printf("ldimx: %lld, ldimy: %lld, ldimz: %lld\n", g.ldimx, g.ldimy, g.ldimz);

    const float lambdamax = vmax / _fmax;
    g.ndampx = (llint)(g.ntaperx * lambdamax / g.dx);
    g.ndampy = (llint)(g.ntapery * lambdamax / g.dy);
    g.ndampz = (llint)(g.ntaperz * lambdamax / g.dz);

    g.x1 = 0;              g.x2 = g.ndampx;
    g.x3 = g.ndampx;       g.x4 = g.nx - g.ndampx;
    g.x5 = g.nx - g.ndampx; g.x6 = g.nx;

    g.y1 = 0;              g.y2 = g.ndampy;
    g.y3 = g.ndampy;       g.y4 = g.ny - g.ndampy;
    g.y5 = g.ny - g.ndampy; g.y6 = g.ny;

    g.z1 = 0;              g.z2 = g.ndampz;
    g.z3 = g.ndampz;       g.z4 = g.nz - g.ndampz;
    g.z5 = g.nz - g.ndampz; g.z6 = g.nz;

    g.tsx = tsx; g.tsy = tsy;
    g.ntx = nx / tsx; g.nty = ny / tsy;

    printf("ndamp = %lld %lld %lld\n", g.ndampx, g.ndampy, g.ndampz);
    return g;
}

static size_t grid_size(const grid_t& g) {
    return (size_t)g.ldimx * g.ldimy * g.ldimz * sizeof(float);
}

// ---------------------------------------------------------------------------
// Flat index into the padded grid array (matches IDX3_grid macro)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
static int64_t IDX(int i, int j, int k, int lx, int ly, int lz, int ldimy, int ldimz) {
    return ((int64_t)(i + lx) * ldimy + (j + ly)) * ldimz + (k + lz);
}

// ---------------------------------------------------------------------------
// Stencil Laplacian (8th-order, R=4 in each dimension)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
static float stencil_lap(
    const Kokkos::View<const float*>& u,
    int i, int j, int k,
    int lx, int ly, int lz, int ldimy, int ldimz,
    float coef0,
    float cx1, float cx2, float cx3, float cx4,
    float cy1, float cy2, float cy3, float cy4,
    float cz1, float cz2, float cz3, float cz4)
{
#define U(di,dj,dk) u(IDX(i+(di),j+(dj),k+(dk),lx,ly,lz,ldimy,ldimz))
    return coef0 * U(0,0,0)
         + cx1 * (U( 1,0,0) + U(-1,0,0))
         + cy1 * (U(0, 1,0) + U(0,-1,0))
         + cz1 * (U(0,0, 1) + U(0,0,-1))
         + cx2 * (U( 2,0,0) + U(-2,0,0))
         + cy2 * (U(0, 2,0) + U(0,-2,0))
         + cz2 * (U(0,0, 2) + U(0,0,-2))
         + cx3 * (U( 3,0,0) + U(-3,0,0))
         + cy3 * (U(0, 3,0) + U(0,-3,0))
         + cz3 * (U(0,0, 3) + U(0,0,-3))
         + cx4 * (U( 4,0,0) + U(-4,0,0))
         + cy4 * (U(0, 4,0) + U(0,-4,0))
         + cz4 * (U(0,0, 4) + U(0,0,-4));
#undef U
}

// ---------------------------------------------------------------------------
// Functor: inner (no-PML) kernel
// ---------------------------------------------------------------------------
struct InnerKernel {
    Kokkos::View<const float*> u;
    Kokkos::View<float*>       v;
    Kokkos::View<const float*> vp;
    int lx, ly, lz, ldimy, ldimz;
    float coef0;
    float cx1, cx2, cx3, cx4;
    float cy1, cy2, cy3, cy4;
    float cz1, cz2, cz3, cz4;

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, int j, int k) const {
        const float lap = stencil_lap(u, i, j, k, lx, ly, lz, ldimy, ldimz,
                                      coef0, cx1, cx2, cx3, cx4,
                                             cy1, cy2, cy3, cy4,
                                             cz1, cz2, cz3, cz4);
        const int64_t idx = IDX(i, j, k, lx, ly, lz, ldimy, ldimz);
        v(idx) = 2.0f * u(idx) + vp(idx) * lap - v(idx);
    }
};

// ---------------------------------------------------------------------------
// Functor: PML kernel
// ---------------------------------------------------------------------------
struct PMLKernel {
    Kokkos::View<const float*> u;
    Kokkos::View<float*>       v;
    Kokkos::View<const float*> vp;
    Kokkos::View<float*>       phi;
    Kokkos::View<const float*> eta;
    int lx, ly, lz, ldimy, ldimz;
    float coef0;
    float cx1, cx2, cx3, cx4;
    float cy1, cy2, cy3, cy4;
    float cz1, cz2, cz3, cz4;
    float hdx_2, hdy_2, hdz_2;

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, int j, int k) const {
        const float lap = stencil_lap(u, i, j, k, lx, ly, lz, ldimy, ldimz,
                                      coef0, cx1, cx2, cx3, cx4,
                                             cy1, cy2, cy3, cy4,
                                             cz1, cz2, cz3, cz4);
        const int64_t idx = IDX(i, j, k, lx, ly, lz, ldimy, ldimz);

#define E(di,dj,dk) eta(IDX(i+(di),j+(dj),k+(dk),lx,ly,lz,ldimy,ldimz))
#define UU(di,dj,dk)  u(IDX(i+(di),j+(dj),k+(dk),lx,ly,lz,ldimy,ldimz))
        const float s_eta_c = E(0,0,0);

        v(idx) = ((2.0f * s_eta_c + 2.0f - s_eta_c * s_eta_c) * u(idx)
                  + vp(idx) * (lap + phi(idx)) - v(idx))
                 / (2.0f * s_eta_c + 1.0f);

        phi(idx) = (phi(idx)
                    - ((E(1,0,0) - E(-1,0,0)) * (UU( 1,0,0) - UU(-1,0,0)) * hdx_2
                     + (E(0,1,0) - E(0,-1,0)) * (UU(0, 1,0) - UU(0,-1,0)) * hdy_2
                     + (E(0,0,1) - E(0,0,-1)) * (UU(0,0, 1) - UU(0,0,-1)) * hdz_2))
                   / (1.0f + s_eta_c);
#undef E
#undef UU
    }
};

// Convenience: launch a 3D MDRangePolicy kernel over [x0,x1) x [y0,y1) x [z0,z1)
template <class Functor>
static void launch3d(const char* name,
                     llint x0, llint x1, llint y0, llint y1, llint z0, llint z1,
                     const Functor& f)
{
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) return;
    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    Kokkos::parallel_for(name,
        Policy({(int)x0, (int)y0, (int)z0},
               {(int)x1, (int)y1, (int)z1}),
        f);
}

// ---------------------------------------------------------------------------
// PML profile initialisation (from pml.cu)
// ---------------------------------------------------------------------------
static void pml_profile_init(float* profile, llint i_min, llint i_max,
                              llint n_first, llint n_last, float scale)
{
    const llint shift    = i_min - 1;
    const llint first_beg = 1 + shift;
    const llint first_end = n_first + shift;
    const llint n         = i_max - i_min + 1;
    const llint last_beg  = n - n_last + 1 + shift;

    for (llint i = i_min; i <= i_max; ++i) profile[i] = 0.0f;

    float tmp = scale / POW2(first_end - first_beg + 1);
    for (llint i = 1; i <= first_end - first_beg + 1; ++i)
        profile[first_end - i + 1] = POW2(i) * tmp;
    for (llint i = 1; i <= n - (last_beg - i_min); ++i)
        profile[last_beg + i - 1] = POW2(i) * tmp;
}

static void pml_extend_region(
    float* eta, const float* etax, const float* etay, const float* etaz,
    llint ldimy, llint ldimz, llint lx, llint ly, llint lz,
    llint xbeg, llint xend, llint ybeg, llint yend, llint zbeg, llint zend)
{
    const llint ng = 1;
    for (llint ix = xbeg - ng; ix <= xend + ng; ++ix)
        for (llint iy = ybeg - ng; iy <= yend + ng; ++iy)
            for (llint iz = zbeg - ng; iz <= zend + ng; ++iz)
                eta[IDX((int)ix, (int)iy, (int)iz,
                        (int)lx, (int)ly, (int)lz, (int)ldimy, (int)ldimz)]
                    = etax[ix] + etay[iy] + etaz[iz];
}

static void init_eta(const grid_t& g, float dt_sch, float* eta)
{
    const llint total = g.ldimx * g.ldimy * g.ldimz;
    memset(eta, 0, total * sizeof(float));

    float param;
    float* etax = new float[g.nx + 2]();
    float* etay = new float[g.ny + 2]();
    float* etaz = new float[g.nz + 2]();

    param = dt_sch * 3.0f * vmax * logf(1000.0f) / (2.0f * g.ndampx * g.dx);
    pml_profile_init(etax, 0, g.nx + 1, g.ndampx, g.ndampx, param);

    param = dt_sch * 3.0f * vmax * logf(1000.0f) / (2.0f * g.ndampy * g.dy);
    pml_profile_init(etay, 0, g.ny + 1, g.ndampy, g.ndampy, param);

    param = dt_sch * 3.0f * vmax * logf(1000.0f) / (2.0f * g.ndampz * g.dz);
    pml_profile_init(etaz, 0, g.nz + 1, g.ndampz, g.ndampz, param);

    const llint lx = g.lx, ly = g.ly, lz = g.lz;
    const llint ldimy = g.ldimy, ldimz = g.ldimz;

    // Top / Bottom
    pml_extend_region(eta, etax, etay, etaz, ldimy, ldimz, lx, ly, lz,
                      1, g.nx, 1, g.ny, g.z1 + 1, g.z2);
    pml_extend_region(eta, etax, etay, etaz, ldimy, ldimz, lx, ly, lz,
                      1, g.nx, 1, g.ny, g.z5 + 1, g.z6);
    // Front / Back
    pml_extend_region(eta, etax, etay, etaz, ldimy, ldimz, lx, ly, lz,
                      1, g.nx, g.y1 + 1, g.y2, g.z3 + 1, g.z4);
    pml_extend_region(eta, etax, etay, etaz, ldimy, ldimz, lx, ly, lz,
                      1, g.nx, g.y5 + 1, g.y6, g.z3 + 1, g.z4);
    // Left / Right
    pml_extend_region(eta, etax, etay, etaz, ldimy, ldimz, lx, ly, lz,
                      g.x1 + 1, g.x2, g.y3 + 1, g.y4, g.z3 + 1, g.z4);
    pml_extend_region(eta, etax, etay, etaz, ldimy, ldimz, lx, ly, lz,
                      g.x5 + 1, g.x6, g.y3 + 1, g.y4, g.z3 + 1, g.z4);

    delete[] etax; delete[] etay; delete[] etaz;
}

// ---------------------------------------------------------------------------
// Coefficient initialisation (from main.cu)
// ---------------------------------------------------------------------------
static void init_coef(float dx, float coef[5]) {
    const float dx2 = dx * dx;
    coef[0] = -205.0f / 72.0f / dx2;
    coef[1] =    8.0f /  5.0f / dx2;
    coef[2] =   -1.0f /  5.0f / dx2;
    coef[3] =    8.0f / 315.0f / dx2;
    coef[4] =   -1.0f / 560.0f / dx2;
}

static float compute_dt_sch(const float cx[5], const float cy[5], const float cz[5]) {
    float ftmp = fabsf(cx[0]) + fabsf(cy[0]) + fabsf(cz[0]);
    for (uint i = 1; i < 5; i++) {
        ftmp += 2.0f * (fabsf(cx[i]) + fabsf(cy[i]) + fabsf(cz[i]));
    }
    return 2.0f * cfl / (sqrtf(ftmp) * vmax);
}

static void gaussian_source(uint nt, float dt, float* src) {
    const float sigma = 0.6f * _fmax;
    const float tau   = 1.0f;
    const float scale = 8.0f;
    for (uint it = 1; it <= nt; ++it) {
        const float t = dt * (it - 1);
        src[it - 1] = -2.0f * scale * sigma
            * (sigma - 2.0f * sigma * scale * POW2(sigma * t - tau))
            * expf(-scale * POW2(sigma * t - tau));
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        llint nx = 100, ny = 100, nz = 100;
        uint  nsteps = 100;
        uint  niters = 1;
        bool  warm_up_enabled = false;
        bool  finalio = false;

        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
                nx = ny = nz = strtoll(argv[++i], nullptr, 10);
            } else if (strcmp(argv[i], "--nsteps") == 0 && i + 1 < argc) {
                nsteps = (uint)strtol(argv[++i], nullptr, 10);
                printf("nsteps = %u\n", nsteps);
            } else if (strcmp(argv[i], "--niters") == 0 && i + 1 < argc) {
                niters = (uint)strtol(argv[++i], nullptr, 10);
                printf("niters = %u\n", niters);
            } else if (strcmp(argv[i], "--warm-up") == 0) {
                warm_up_enabled = true;
            } else if (strcmp(argv[i], "--finalio") == 0) {
                finalio = true;
            }
        }

        double total_kernel_time = 0.0;
        bool   warm_up_iter = warm_up_enabled;

        const uint total_iters = niters + (warm_up_enabled ? 1u : 0u);

        for (uint iiter = 0; iiter < total_iters; ++iiter) {

            const grid_t g = init_grid(nx, ny, nz);
            printf("grid = %lld %lld %lld\n", g.nx, g.ny, g.nz);

            const llint sx = nx / 2, sy = ny / 2, sz = nz / 2;
            const llint total_cells = g.ldimx * g.ldimy * g.ldimz;

            // Allocate host arrays for setup (vp, phi, eta, source)
            float* h_vp  = new float[total_cells]();
            float* h_phi = new float[total_cells]();
            float* h_eta = new float[total_cells]();
            float* source = new float[nsteps];

            float coefx[5], coefy[5], coefz[5];
            init_coef((float)g.dx, coefx);
            init_coef((float)g.dy, coefy);
            init_coef((float)g.dz, coefz);
            const float dt_sch = compute_dt_sch(coefx, coefy, coefz);

            // Velocity model (homogeneous)
            const float vp_all = POW2(2000.0f * dt_sch);
            for (llint i = 0; i < nx; ++i)
                for (llint j = 0; j < ny; ++j)
                    for (llint k = 0; k < nz; ++k) {
                        const int64_t idx = IDX((int)i,(int)j,(int)k,
                                                (int)g.lx,(int)g.ly,(int)g.lz,
                                                (int)g.ldimy,(int)g.ldimz);
                        h_vp[idx]  = vp_all;
                        h_phi[idx] = 0.0f;
                    }

            gaussian_source(nsteps, dt_sch, source);
            init_eta(g, dt_sch, h_eta);

            const float hdx_2 = 1.0f / (4.0f * POW2(g.dx));
            const float hdy_2 = 1.0f / (4.0f * POW2(g.dy));
            const float hdz_2 = 1.0f / (4.0f * POW2(g.dz));

            const float coef0 = coefx[0] + coefy[0] + coefz[0];

            // Allocate Kokkos views
            Kokkos::View<float*> d_u("u",   total_cells);
            Kokkos::View<float*> d_v("v",   total_cells);
            Kokkos::View<float*> d_phi("phi", total_cells);
            Kokkos::View<float*> d_eta("eta", total_cells);
            Kokkos::View<float*> d_vp("vp",  total_cells);

            // Copy setup data to device
            {
                auto hv_vp  = Kokkos::create_mirror_view(d_vp);
                auto hv_phi = Kokkos::create_mirror_view(d_phi);
                auto hv_eta = Kokkos::create_mirror_view(d_eta);
                memcpy(hv_vp.data(),  h_vp,  total_cells * sizeof(float));
                memcpy(hv_phi.data(), h_phi, total_cells * sizeof(float));
                memcpy(hv_eta.data(), h_eta, total_cells * sizeof(float));
                Kokkos::deep_copy(d_vp,  hv_vp);
                Kokkos::deep_copy(d_phi, hv_phi);
                Kokkos::deep_copy(d_eta, hv_eta);
            }
            Kokkos::deep_copy(d_u, 0.0f);
            Kokkos::deep_copy(d_v, 0.0f);
            Kokkos::fence();

            // Build functor parameter structs
            const int lx = (int)g.lx, ly = (int)g.ly, lz = (int)g.lz;
            const int ldimy = (int)g.ldimy, ldimz = (int)g.ldimz;

            InnerKernel inner_k;
            inner_k.lx = lx; inner_k.ly = ly; inner_k.lz = lz;
            inner_k.ldimy = ldimy; inner_k.ldimz = ldimz;
            inner_k.coef0 = coef0;
            inner_k.cx1 = coefx[1]; inner_k.cx2 = coefx[2];
            inner_k.cx3 = coefx[3]; inner_k.cx4 = coefx[4];
            inner_k.cy1 = coefy[1]; inner_k.cy2 = coefy[2];
            inner_k.cy3 = coefy[3]; inner_k.cy4 = coefy[4];
            inner_k.cz1 = coefz[1]; inner_k.cz2 = coefz[2];
            inner_k.cz3 = coefz[3]; inner_k.cz4 = coefz[4];

            PMLKernel pml_k;
            pml_k.lx = lx; pml_k.ly = ly; pml_k.lz = lz;
            pml_k.ldimy = ldimy; pml_k.ldimz = ldimz;
            pml_k.coef0  = coef0;
            pml_k.cx1 = coefx[1]; pml_k.cx2 = coefx[2];
            pml_k.cx3 = coefx[3]; pml_k.cx4 = coefx[4];
            pml_k.cy1 = coefy[1]; pml_k.cy2 = coefy[2];
            pml_k.cy3 = coefy[3]; pml_k.cy4 = coefy[4];
            pml_k.cz1 = coefz[1]; pml_k.cz2 = coefz[2];
            pml_k.cz3 = coefz[3]; pml_k.cz4 = coefz[4];
            pml_k.hdx_2 = hdx_2; pml_k.hdy_2 = hdy_2; pml_k.hdz_2 = hdz_2;

            Kokkos::fence();
            Kokkos::Timer timer;

            for (uint istep = 1; istep <= nsteps; ++istep) {

                // Bind current u/v to functors (views are reference-counted handles)
                inner_k.u  = d_u; inner_k.v  = d_v; inner_k.vp = d_vp;
                pml_k.u    = d_u; pml_k.v    = d_v; pml_k.vp   = d_vp;
                pml_k.phi  = d_phi; pml_k.eta = d_eta;

                // 7 PML / inner regions (matching the CUDA kernel launch order):
                // Front face  (full xy, z1..z2)
                launch3d("pml_front",  0, nx,   0, ny,   g.z1, g.z2, pml_k);
                // Top slab    (full x, y1..y2, z3..z4)
                launch3d("pml_top",    0, nx,   g.y1, g.y2, g.z3, g.z4, pml_k);
                // Left slab   (x1..x2, y3..y4, z3..z4)
                launch3d("pml_left",   g.x1, g.x2, g.y3, g.y4, g.z3, g.z4, pml_k);
                // Inner core  (x3..x4, y3..y4, z3..z4) - no PML
                launch3d("inner",      g.x3, g.x4, g.y3, g.y4, g.z3, g.z4, inner_k);
                // Right slab  (x5..x6, y3..y4, z3..z4)
                launch3d("pml_right",  g.x5, g.x6, g.y3, g.y4, g.z3, g.z4, pml_k);
                // Bottom slab (full x, y5..y6, z3..z4)
                launch3d("pml_bottom", 0, nx,   g.y5, g.y6, g.z3, g.z4, pml_k);
                // Back face   (full xy, z5..z6)
                launch3d("pml_back",   0, nx,   0, ny,   g.z5, g.z6, pml_k);

                // Add source at centre of grid (into v)
                {
                    const float src_val = source[istep - 1];
                    const int64_t src_idx = IDX((int)sx,(int)sy,(int)sz, lx,ly,lz,ldimy,ldimz);
                    Kokkos::parallel_for(
                        "add_source", 1,
                        KOKKOS_LAMBDA(int) { d_v(src_idx) += src_val; });
                }

                // Swap u and v for next step
                Kokkos::View<float*> tmp = d_u;
                d_u = d_v;
                d_v = tmp;
                // Also keep pml_k / inner_k up-to-date (next iteration will rebind)
            }

            Kokkos::fence();
            const double kernel_time = warm_up_iter ? 0.0 : timer.seconds();

            // Checksum
            float min_u = FLT_MAX, max_u = -FLT_MAX;
            Kokkos::parallel_reduce(
                "min_u", total_cells,
                KOKKOS_LAMBDA(int i, float& lmin) {
                    if (d_u(i) < lmin) lmin = d_u(i);
                },
                Kokkos::Min<float>(min_u));
            Kokkos::parallel_reduce(
                "max_u", total_cells,
                KOKKOS_LAMBDA(int i, float& lmax) {
                    if (d_u(i) > lmax) lmax = d_u(i);
                },
                Kokkos::Max<float>(max_u));
            Kokkos::fence();

            printf("Checksum: min_u, max_u = %f, %f\n", min_u, max_u);

            if (finalio && !warm_up_iter) {
                auto hv_u = Kokkos::create_mirror_view(d_u);
                Kokkos::deep_copy(hv_u, d_u);
                char fname[64];
                snprintf(fname, sizeof(fname), "snapshot.it%u.n%lld.raw", nsteps, nz);
                FILE* f = fopen(fname, "wb");
                for (llint i = 0; i < nx; ++i)
                    for (llint j = 0; j < ny; ++j) {
                        const int64_t base =
                            IDX((int)i,(int)j,0, lx,ly,lz, ldimy,ldimz);
                        fwrite(hv_u.data() + base, sizeof(float), nz, f);
                    }
                fclose(f);
            }

            total_kernel_time += kernel_time;
            warm_up_iter = false;

            delete[] h_vp; delete[] h_phi; delete[] h_eta; delete[] source;
        }

        printf("Average kernel time per iteration: %g s\n", total_kernel_time / niters);
    }
    Kokkos::finalize();
    return 0;
}
