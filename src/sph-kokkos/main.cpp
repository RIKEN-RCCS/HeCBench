#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct double3 { double x, y, z; };

struct boundary_particle {
    double3 pos;
    double3 n;
};

struct fluid_particle {
    double density;
    double pressure;
    double3 pos;
    double3 v;
    double3 v_half;
    double3 a;
};

struct param {
    double rest_density;
    double mass_particle;
    double spacing_particle;
    double smoothing_radius;
    double g;
    double time_step;
    double alpha;
    double surface_tension;
    double speed_sound;
    int number_particles;
    int number_fluid_particles;
    int number_boundary_particles;
    int number_steps;
    int steps_per_frame;
};

struct AABB {
    double min_x, max_x;
    double min_y, max_y;
    double min_z, max_z;
};

// ---------------------------------------------------------------------------
// Smoothing kernel helpers (KOKKOS_INLINE_FUNCTION for use in parallel regions)
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
double W(double3 p, double3 q, double h) {
    double dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
    double r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    double C = 1.0 / (M_PI * h * h * h);
    double u = r / h;
    double val = 0.0;
    if      (u >= 2.0)              val = 0.0;
    else if (u < 1.0)               val = 1.0 - 1.5*u*u + 0.75*u*u*u;
    else /* 1 <= u < 2 */           val = 0.25 * Kokkos::pow(2.0 - u, 3.0);
    return val * C;
}

KOKKOS_INLINE_FUNCTION
double del_W(double3 p, double3 q, double h) {
    double dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
    double r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    double C = 1.0 / (M_PI * h * h * h);
    double u = r / h;
    double val = 0.0;
    if      (u >= 2.0) val = 0.0;
    else if (u < 1.0)  val = -1.0 / (h * h) * (3.0 - 2.25 * u);
    else               val = -3.0 / (4.0 * h * r) * Kokkos::pow(2.0 - u, 2.0);
    return val * C;
}

KOKKOS_INLINE_FUNCTION
double boundaryGamma(double3 p_pos, double3 k_pos, double3 k_n,
                     double h, double speed_sound) {
    double dx = p_pos.x - k_pos.x;
    double dy = p_pos.y - k_pos.y;
    double dz = p_pos.z - k_pos.z;
    double r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    double y = Kokkos::sqrt(dx*dx*k_n.x*k_n.x + dy*dy*k_n.y*k_n.y + dz*dz*k_n.z*k_n.z);
    double x = r - y;
    double u = y / h;
    double xi = (x < h) ? (1.0 - x / h) : 0.0;
    double C = xi * 2.0 * 0.02 * speed_sound * speed_sound / y;
    double val = 0.0;
    if      (u > 0.0 && u < 2.0/3.0)  val = 2.0/3.0;
    else if (u < 1.0)                  val = 2.0*u - 1.5*u*u;
    else if (u < 2.0)                  val = 0.5*(2.0-u)*(2.0-u);
    return val * C;
}

KOKKOS_INLINE_FUNCTION
double computeDensity(double3 p_pos, double3 p_v, double3 q_pos, double3 q_v,
                      double mass_particle, double smoothing_radius, double time_step) {
    double vx = p_v.x - q_v.x, vy = p_v.y - q_v.y, vz = p_v.z - q_v.z;
    double dw = del_W(p_pos, q_pos, smoothing_radius);
    double d = mass_particle * dw;
    double dx = p_pos.x - q_pos.x, dy = p_pos.y - q_pos.y, dz = p_pos.z - q_pos.z;
    return (d*vx*dx + d*vy*dy + d*vz*dz) * time_step;
}

KOKKOS_INLINE_FUNCTION
double computePressure(double density, double rest_density, double speed_sound) {
    double gam = 7.0;
    double B = rest_density * speed_sound * speed_sound / gam;
    return B * (Kokkos::pow(density / rest_density, gam) - 1.0);
}

KOKKOS_INLINE_FUNCTION
double3 computeBoundaryAcceleration(double3 p_pos, double3 k_pos, double3 k_n,
                                    double h, double speed_sound) {
    double bg = boundaryGamma(p_pos, k_pos, k_n, h, speed_sound);
    return {bg * k_n.x, bg * k_n.y, bg * k_n.z};
}

KOKKOS_INLINE_FUNCTION
double3 computeAcceleration(double3 p_pos, double3 p_v, double p_density, double p_pressure,
                             double3 q_pos, double3 q_v, double q_density, double q_pressure,
                             double h, double alpha, double speed_sound,
                             double mass_particle, double surface_tension) {
    double3 a;
    double dw = del_W(p_pos, q_pos, h);
    double accel = (p_pressure / (p_density * p_density) + q_pressure / (q_density * q_density))
                   * mass_particle * dw;
    double dx = p_pos.x - q_pos.x, dy = p_pos.y - q_pos.y, dz = p_pos.z - q_pos.z;
    a.x = -accel * dx;
    a.y = -accel * dy;
    a.z = -accel * dz;

    // Viscosity
    double VdotR = (p_v.x-q_v.x)*dx + (p_v.y-q_v.y)*dy + (p_v.z-q_v.z)*dz;
    if (VdotR < 0.0) {
        double nu = 2.0 * alpha * h * speed_sound / (p_density + q_density);
        double r2 = dx*dx + dy*dy + dz*dz;
        double eps = h / 10.0;
        double stress = nu * VdotR / (r2 + eps * h * h);
        accel = mass_particle * stress * dw;
        a.x += accel * dx;
        a.y += accel * dy;
        a.z += accel * dz;
    }

    // Surface tension
    accel = surface_tension * W(p_pos, q_pos, h);
    a.x += accel * dx;
    a.y += accel * dy;
    a.z += accel * dz;

    return a;
}

// ---------------------------------------------------------------------------
// Boundary box construction (host-only)
// ---------------------------------------------------------------------------
static void constructBoundaryBox(boundary_particle* bp, AABB* bnd, param* params) {
    double spacing = params->spacing_particle;
    int num_x = (int)std::ceil((bnd->max_x - bnd->min_x) / spacing);
    int num_y = (int)std::ceil((bnd->max_y - bnd->min_y) / spacing);
    int num_z = (int)std::ceil((bnd->max_z - bnd->min_z) / spacing);
    double min_x = bnd->min_x, min_y = bnd->min_y, min_z = bnd->min_z;
    double max_x = min_x + (num_x - 1) * spacing;
    double max_y = min_y + (num_y - 1) * spacing;
    double max_z = min_z + (num_z - 1) * spacing;
    bnd->max_x = max_x; bnd->max_y = max_y; bnd->max_z = max_z;

    double r3 = 1.0 / std::sqrt(3.0);
    double r2 = 1.0 / std::sqrt(2.0);
    int i = 0;

    // 8 corners
    auto corner = [&](double x, double y, double z, double nx, double ny, double nz) {
        bp[i] = {{x,y,z},{nx,ny,nz}}; i++;
    };
    corner(min_x, max_y, min_z,  r3,-r3, r3);
    corner(max_x, max_y, min_z, -r3,-r3, r3);
    corner(min_x, max_y, max_z,  r3,-r3,-r3);
    corner(max_x, max_y, max_z, -r3,-r3,-r3);
    corner(min_x, min_y, min_z,  r3, r3, r3);
    corner(max_x, min_y, min_z, -r3, r3, r3);
    corner(min_x, min_y, max_z,  r3, r3,-r3);
    corner(max_x, min_y, max_z, -r3, r3,-r3);

    for (int nx = 0; nx < num_x - 2; nx++) {
        double px = min_x + spacing + nx * spacing;
        bp[i++] = {{px, max_y, min_z}, {0, -r2,  r2}};
        bp[i++] = {{px, max_y, max_z}, {0, -r2, -r2}};
        bp[i++] = {{px, min_y, min_z}, {0,  r2,  r2}};
        bp[i++] = {{px, min_y, max_z}, {0,  r2, -r2}};
    }
    for (int ny = 0; ny < num_y - 2; ny++) {
        double py = min_y + spacing + ny * spacing;
        bp[i++] = {{max_x, py, min_z}, {-r2, 0,  r2}};
        bp[i++] = {{max_x, py, max_z}, {-r2, 0, -r2}};
        bp[i++] = {{min_x, py, min_z}, { r2, 0,  r2}};
        bp[i++] = {{min_x, py, max_z}, { r2, 0, -r2}};
        for (int nx = 0; nx < num_x - 2; nx++) {
            double px = min_x + spacing + nx * spacing;
            bp[i++] = {{px, py, max_z}, {0, 0, -1}};
            bp[i++] = {{px, py, min_z}, {0, 0,  1}};
        }
    }
    for (int nz = 0; nz < num_z - 2; nz++) {
        double pz = min_z + spacing + nz * spacing;
        bp[i++] = {{min_x, max_y, pz}, { r2, -r2, 0}};
        bp[i++] = {{max_x, max_y, pz}, {-r2, -r2, 0}};
        bp[i++] = {{min_x, min_y, pz}, { r2,  r2, 0}};
        bp[i++] = {{max_x, min_y, pz}, {-r2,  r2, 0}};
        for (int nx = 0; nx < num_x - 2; nx++) {
            double px = min_x + spacing + nx * spacing;
            bp[i++] = {{px, max_y, pz}, {0, -1, 0}};
            bp[i++] = {{px, min_y, pz}, {0,  1, 0}};
        }
        for (int ny = 0; ny < num_y - 2; ny++) {
            double py = min_y + spacing + ny * spacing;
            bp[i++] = {{min_x, py, pz}, { 1, 0, 0}};
            bp[i++] = {{max_x, py, pz}, {-1, 0, 0}};
        }
    }
    params->number_boundary_particles = i;
    params->number_particles = params->number_fluid_particles + i;
}

// ---------------------------------------------------------------------------
// Simulation setup
// ---------------------------------------------------------------------------
static void initParams(AABB* water, AABB* bnd, param* p) {
    bnd->min_x=0.0; bnd->max_x=1.1;
    bnd->min_y=0.0; bnd->max_y=1.1;
    bnd->min_z=0.0; bnd->max_z=1.1;

    water->min_x=0.1; water->max_x=0.5;
    water->min_y=0.1; water->max_y=0.5;
    water->min_z=0.08; water->max_z=0.8;

    p->number_fluid_particles = 2048;
    p->rest_density    = 1000.0;
    p->g               = 9.8;
    p->alpha           = 0.02;
    p->surface_tension = 0.01;
    p->number_steps    = 500;
    p->time_step       = 0.00035;

    double vol = (water->max_x - water->min_x)
               * (water->max_y - water->min_y)
               * (water->max_z - water->min_z);
    p->mass_particle    = p->rest_density * (vol / p->number_fluid_particles);
    p->spacing_particle = std::pow(vol / p->number_fluid_particles, 1.0/3.0);
    p->smoothing_radius = p->spacing_particle;

    int nx = (int)std::ceil((bnd->max_x - bnd->min_x) / p->spacing_particle);
    int ny = (int)std::ceil((bnd->max_y - bnd->min_y) / p->spacing_particle);
    int nz = (int)std::ceil((bnd->max_z - bnd->min_z) / p->spacing_particle);
    int nb = (2*nx*nz) + (2*ny*nz) + (2*ny*nz);
    p->number_boundary_particles = nb;
    p->number_particles = p->number_boundary_particles + p->number_fluid_particles;
    p->steps_per_frame  = (int)(1.0 / (p->time_step * 30.0));

    double max_height   = water->max_y;
    double max_velocity = std::sqrt(2.0 * p->g * max_height);
    p->speed_sound      = max_velocity / std::sqrt(0.01);

    double rec_step = 0.4 * p->smoothing_radius / (p->speed_sound * (1 + 0.6*p->alpha));
    printf("Using time step: %f, Minimum recomended %f\n", p->time_step, rec_step);
}

static void initParticles(fluid_particle** fp, boundary_particle** bp,
                          AABB* water, AABB* bnd, param* params) {
    *fp = (fluid_particle*)malloc(params->number_fluid_particles * sizeof(fluid_particle));
    *bp = (boundary_particle*)malloc(params->number_boundary_particles * sizeof(boundary_particle));

    double sp = params->spacing_particle;
    for (int i = 0; i < params->number_fluid_particles; i++) {
        (*fp)[i].a   = {0,0,0};
        (*fp)[i].v   = {0,0,0};
        (*fp)[i].density = params->rest_density;
        (*fp)[i].pressure = 0;
    }

    int i = 0;
    for (double z = water->min_z; z <= water->max_z; z += sp)
        for (double y = water->min_y; y <= water->max_y; y += sp)
            for (double x = water->min_x; x <= water->max_x; x += sp)
                if (i < params->number_fluid_particles) {
                    (*fp)[i].pos = {x, y, z};
                    i++;
                }
    params->number_fluid_particles = i;

    constructBoundaryBox(*bp, bnd, params);
}

static void eulerStart(fluid_particle* fp, param* params) {
    double dt_half = params->time_step / 2.0;
    for (int i = 0; i < params->number_fluid_particles; i++) {
        fp[i].v_half.x = fp[i].v.x;
        fp[i].v_half.y = fp[i].v.y;
        fp[i].v_half.z = fp[i].v.z - params->g * dt_half;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    param params;
    AABB water_volume, boundary_volume;
    fluid_particle*   fluid_particles    = nullptr;
    boundary_particle* boundary_particles = nullptr;

    initParams(&water_volume, &boundary_volume, &params);
    initParticles(&fluid_particles, &boundary_particles,
                  &water_volume, &boundary_volume, &params);
    eulerStart(fluid_particles, &params);

    Kokkos::initialize(argc, argv);
    {
        const int nfp = params.number_fluid_particles;
        const int nbp = params.number_boundary_particles;
        const int nkern = (nfp < nbp) ? nfp : nbp;

        // Views wrapping existing host data (HostSpace, unmanaged)
        using HostView_FP = Kokkos::View<fluid_particle*,   Kokkos::HostSpace>;
        using HostView_BP = Kokkos::View<boundary_particle*, Kokkos::HostSpace>;
        HostView_FP fp(fluid_particles, nfp);
        HostView_BP bp(boundary_particles, nbp);

        // Copy param fields by value for lambda capture
        const double rest_density    = params.rest_density;
        const double mass_particle   = params.mass_particle;
        const double smoothing_radius = params.smoothing_radius;
        const double g               = params.g;
        const double time_step       = params.time_step;
        const double alpha           = params.alpha;
        const double surface_tension = params.surface_tension;
        const double speed_sound     = params.speed_sound;

        auto t0 = std::chrono::steady_clock::now();

        for (int step = 0; step < params.number_steps; step++) {

            // --- Update densities ---
            Kokkos::parallel_for("update_density", nfp, KOKKOS_LAMBDA(int i) {
                double3 p_pos = fp(i).pos;
                double3 p_v   = fp(i).v;
                double density = fp(i).density;
                for (int j = 0; j < nfp; j++) {
                    density += computeDensity(p_pos, p_v,
                                             fp(j).pos, fp(j).v,
                                             mass_particle, smoothing_radius, time_step);
                }
                fp(i).density  = density;
                fp(i).pressure = computePressure(density, rest_density, speed_sound);
            });
            Kokkos::fence();

            // --- Update fluid-fluid accelerations ---
            Kokkos::parallel_for("update_accel_fp", nfp, KOKKOS_LAMBDA(int i) {
                double ax = 0.0, ay = 0.0, az = -9.8;
                double3 p_pos      = fp(i).pos;
                double3 p_v        = fp(i).v;
                double  p_density  = fp(i).density;
                double  p_pressure = fp(i).pressure;
                for (int j = 0; j < nfp; j++) {
                    if (i == j) continue;
                    double3 a = computeAcceleration(
                        p_pos, p_v, p_density, p_pressure,
                        fp(j).pos, fp(j).v, fp(j).density, fp(j).pressure,
                        smoothing_radius, alpha, speed_sound, mass_particle, surface_tension);
                    ax += a.x; ay += a.y; az += a.z;
                }
                fp(i).a = {ax, ay, az};
            });
            Kokkos::fence();

            // --- Update boundary accelerations ---
            Kokkos::parallel_for("update_accel_bp", nkern, KOKKOS_LAMBDA(int i) {
                double ax = fp(i).a.x, ay = fp(i).a.y, az = fp(i).a.z;
                double3 p_pos = fp(i).pos;
                for (int j = 0; j < nbp; j++) {
                    double3 a = computeBoundaryAcceleration(
                        p_pos, bp(j).pos, bp(j).n,
                        smoothing_radius, speed_sound);
                    ax += a.x; ay += a.y; az += a.z;
                }
                fp(i).a = {ax, ay, az};
            });
            Kokkos::fence();

            // --- Update positions (leapfrog) ---
            Kokkos::parallel_for("update_pos", nfp, KOKKOS_LAMBDA(int i) {
                double dt = time_step;
                double3 v_half = fp(i).v_half;
                double3 v      = fp(i).v;
                double3 pos    = fp(i).pos;
                double3 a      = fp(i).a;

                v_half.x += dt * a.x;
                v_half.y += dt * a.y;
                v_half.z += dt * a.z;

                v.x = v_half.x + a.x * (dt / 2.0);
                v.y = v_half.y + a.y * (dt / 2.0);
                v.z = v_half.z + a.z * (dt / 2.0);

                pos.x += dt * v_half.x;
                pos.y += dt * v_half.y;
                pos.z += dt * v_half.z;

                fp(i).v_half = v_half;
                fp(i).v      = v;
                fp(i).pos    = pos;
            });
            Kokkos::fence();
        }

        auto t1 = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("Average execution time of sph kernels: %f (ms)\n",
               (ns * 1e-6) / params.number_steps);
    }
    Kokkos::finalize();

    printf("Simulation complete. Fluid particles: %d\n", params.number_fluid_particles);

    free(fluid_particles);
    free(boundary_particles);
    return 0;
}
