/** Kokkos solver for 2D lid-driven cavity problem, using finite difference method
 * \file main.cpp
 *
 * Solve the incompressible, isothermal 2D Navier-Stokes equations for a square
 * lid-driven cavity using the finite difference method with a Kokkos backend.
 *
 * Based on the methodology given in Chapter 3 of "Numerical Simulation in Fluid
 * Dynamics", by M. Griebel, T. Dornseifer, and T. Neunhoeffer. SIAM, 1998.
 *
 * Ported from the OpenMP target offload version.
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>

#define NUM        512
#define BLOCK_SIZE 128

typedef double Real;

// Physics / discretisation parameters
constexpr Real Re_num   = 1000.0;
constexpr Real omega    = 1.7;
constexpr Real mix_param = 0.9;
constexpr Real tau      = 0.5;
constexpr Real gx       = 0.0;
constexpr Real gy       = 0.0;
constexpr Real dx       = 1.0 / NUM;
constexpr Real dy       = 1.0 / NUM;

constexpr Real ZERO = 0.0;
constexpr Real ONE  = 1.0;
constexpr Real TWO  = 2.0;
constexpr Real FOUR = 4.0;

using RealView2D = Kokkos::View<Real**>;
using MDPolicy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        const int  it_max  = 1000000;
        const Real tol     = 0.001;
        const Real time_end = 0.001;

        // All Views are zero-initialised by Kokkos on construction.
        RealView2D u_d         ("u",          NUM+2, NUM+2);
        RealView2D v_d         ("v",          NUM+2, NUM+2);
        RealView2D F_d         ("F",          NUM+2, NUM+2);
        RealView2D G_d         ("G",          NUM+2, NUM+2);
        RealView2D pres_red_d  ("pres_red",   NUM+2, (NUM/2)+2);
        RealView2D pres_black_d("pres_black", NUM+2, (NUM/2)+2);

        printf("Problem size: %d x %d\n", NUM, NUM);

        // -----------------------------------------------------------------------
        // set_BCs – velocity boundary conditions
        // -----------------------------------------------------------------------
        auto do_set_BCs = [&]() {
            Kokkos::parallel_for("set_BCs", Kokkos::RangePolicy<>(1, NUM+1),
                KOKKOS_LAMBDA(int ind) {
                    // left boundary
                    u_d(0,     ind)   = ZERO;
                    v_d(0,     ind)   = -v_d(1,   ind);
                    // right boundary
                    u_d(NUM,   ind)   = ZERO;
                    v_d(NUM+1, ind)   = -v_d(NUM, ind);
                    // bottom boundary
                    u_d(ind,   0)     = -u_d(ind, 1);
                    v_d(ind,   0)     = ZERO;
                    // top boundary (lid moves at u=1)
                    u_d(ind,   NUM+1) = TWO - u_d(ind, NUM);
                    v_d(ind,   NUM)   = ZERO;

                    if (ind == NUM) {
                        // corners – left
                        u_d(0,     0)     = ZERO;
                        v_d(0,     0)     = -v_d(1,   0);
                        u_d(0,     NUM+1) = ZERO;
                        v_d(0,     NUM+1) = -v_d(1,   NUM+1);
                        // corners – right
                        u_d(NUM,   0)     = ZERO;
                        v_d(NUM+1, 0)     = -v_d(NUM, 0);
                        u_d(NUM,   NUM+1) = ZERO;
                        v_d(NUM+1, NUM+1) = -v_d(NUM, NUM+1);
                        // corners – bottom
                        u_d(0,     0)     = -u_d(0,     1);
                        v_d(0,     0)     = ZERO;
                        u_d(NUM+1, 0)     = -u_d(NUM+1, 1);
                        v_d(NUM+1, 0)     = ZERO;
                        // corners – top
                        u_d(0,     NUM+1) = TWO - u_d(0,     NUM);
                        v_d(0,     NUM)   = ZERO;
                        u_d(NUM+1, NUM+1) = TWO - u_d(NUM+1, NUM);
                        v_d(ind,   NUM+1) = ZERO;   // v_d(NUM, NUM+1)
                    }
                });
        };

        do_set_BCs();

        Real max_u = 1.0e-10;
        Real max_v = 1.0e-10;
        Real dt    = 0.02;
        Real time  = 0.0;

        // Time-step stability limit from the viscous term
        const Real dt_Re = 0.5 * Re_num / ((1.0/(dx*dx)) + (1.0/(dy*dy)));

        // Total number of red (or black) pressure cells
        const int N = NUM * (NUM/2);

        auto wct_start = std::chrono::steady_clock::now();

        // =====================================================================
        // Main time loop
        // =====================================================================
        while (time < time_end) {

            // CFL / stability time-step selection
            dt = std::min(dx / max_u, dy / max_v);
            dt = tau * std::min(dt_Re, dt);
            if (time + dt >= time_end) dt = time_end - time;

            // -----------------------------------------------------------------
            // calculate_F  (intermediate velocity in x-direction)
            // -----------------------------------------------------------------
            Kokkos::parallel_for("calcF", MDPolicy2D({1,1},{NUM+1,NUM+1}),
                KOKKOS_LAMBDA(int col, int row) {
                    if (col == NUM) {
                        F_d(0,   row) = u_d(0,   row);
                        F_d(NUM, row) = u_d(NUM, row);
                    } else {
                        const Real u_ij     = u_d(col,   row);
                        const Real u_ip1j   = u_d(col+1, row);
                        const Real u_ijp1   = u_d(col,   row+1);
                        const Real u_im1j   = u_d(col-1, row);
                        const Real u_ijm1   = u_d(col,   row-1);
                        const Real v_ij     = v_d(col,   row);
                        const Real v_ip1j   = v_d(col+1, row);
                        const Real v_ijm1   = v_d(col,   row-1);
                        const Real v_ip1jm1 = v_d(col+1, row-1);

                        const Real du2dx =
                            (((u_ij+u_ip1j)*(u_ij+u_ip1j) - (u_im1j+u_ij)*(u_im1j+u_ij))
                             + mix_param*(fabs(u_ij+u_ip1j)*(u_ij-u_ip1j)
                                        - fabs(u_im1j+u_ij)*(u_im1j-u_ij)))
                            / (FOUR*dx);
                        const Real duvdy =
                            ((v_ij+v_ip1j)*(u_ij+u_ijp1) - (v_ijm1+v_ip1jm1)*(u_ijm1+u_ij)
                             + mix_param*(fabs(v_ij+v_ip1j)*(u_ij-u_ijp1)
                                        - fabs(v_ijm1+v_ip1jm1)*(u_ijm1-u_ij)))
                            / (FOUR*dy);
                        const Real d2udx2 = (u_ip1j - TWO*u_ij + u_im1j) / (dx*dx);
                        const Real d2udy2 = (u_ijp1 - TWO*u_ij + u_ijm1) / (dy*dy);

                        F_d(col, row) = u_ij + dt*(((d2udx2+d2udy2)/Re_num)
                                                    - du2dx - duvdy + gx);
                    }
                });

            // -----------------------------------------------------------------
            // calculate_G  (intermediate velocity in y-direction)
            // -----------------------------------------------------------------
            Kokkos::parallel_for("calcG", MDPolicy2D({1,1},{NUM+1,NUM+1}),
                KOKKOS_LAMBDA(int col, int row) {
                    if (row == NUM) {
                        G_d(col, 0)   = v_d(col, 0);
                        G_d(col, NUM) = v_d(col, NUM);
                    } else {
                        const Real u_ij     = u_d(col,   row);
                        const Real u_ijp1   = u_d(col,   row+1);
                        const Real u_im1j   = u_d(col-1, row);
                        const Real u_im1jp1 = u_d(col-1, row+1);
                        const Real v_ij     = v_d(col,   row);
                        const Real v_ijp1   = v_d(col,   row+1);
                        const Real v_ip1j   = v_d(col+1, row);
                        const Real v_ijm1   = v_d(col,   row-1);
                        const Real v_im1j   = v_d(col-1, row);

                        const Real dv2dy =
                            ((v_ij+v_ijp1)*(v_ij+v_ijp1) - (v_ijm1+v_ij)*(v_ijm1+v_ij)
                             + mix_param*(fabs(v_ij+v_ijp1)*(v_ij-v_ijp1)
                                        - fabs(v_ijm1+v_ij)*(v_ijm1-v_ij)))
                            / (FOUR*dy);
                        const Real duvdx =
                            ((u_ij+u_ijp1)*(v_ij+v_ip1j) - (u_im1j+u_im1jp1)*(v_im1j+v_ij)
                             + mix_param*(fabs(u_ij+u_ijp1)*(v_ij-v_ip1j)
                                        - fabs(u_im1j+u_im1jp1)*(v_im1j-v_ij)))
                            / (FOUR*dx);
                        const Real d2vdx2 = (v_ip1j - TWO*v_ij + v_im1j) / (dx*dx);
                        const Real d2vdy2 = (v_ijp1 - TWO*v_ij + v_ijm1) / (dy*dy);

                        G_d(col, row) = v_ij + dt*(((d2vdx2+d2vdy2)/Re_num)
                                                    - dv2dy - duvdx + gy);
                    }
                });

            // -----------------------------------------------------------------
            // Compute initial pressure norm (used to normalise the residual)
            // -----------------------------------------------------------------
            Real p0_sum = 0.0;
            Kokkos::parallel_reduce("sum_pres", N,
                KOKKOS_LAMBDA(int gid, Real& lsum) {
                    const int row = (gid % (NUM/2)) + 1;
                    const int col = (gid / (NUM/2)) + 1;
                    const Real pr = pres_red_d  (col, row);
                    const Real pb = pres_black_d(col, row);
                    lsum += pr*pr + pb*pb;
                }, p0_sum);

            Real p0_norm = std::sqrt(p0_sum / (Real)(NUM*NUM));
            if (p0_norm < 0.0001) p0_norm = 1.0;

            // -----------------------------------------------------------------
            // Red-black Gauss-Seidel pressure iteration with SOR
            // -----------------------------------------------------------------
            Real norm_L2 = 0.0;
            int  iter    = 1;

            for (iter = 1; iter <= it_max; ++iter) {

                // set horizontal pressure boundary conditions
                Kokkos::parallel_for("horz_pres_BCs",
                    Kokkos::RangePolicy<>(1, NUM/2+1),
                    KOKKOS_LAMBDA(int col_half) {
                        const int col   = (col_half * 2) - 1;
                        const int NUM_2 = NUM >> 1;
                        pres_black_d(col,   0)       = pres_red_d  (col,   1);
                        pres_red_d  (col+1, 0)       = pres_black_d(col+1, 1);
                        pres_red_d  (col,   NUM_2+1) = pres_black_d(col,   NUM_2);
                        pres_black_d(col+1, NUM_2+1) = pres_red_d  (col+1, NUM_2);
                    });

                // set vertical pressure boundary conditions
                Kokkos::parallel_for("vert_pres_BCs",
                    Kokkos::RangePolicy<>(1, NUM/2+1),
                    KOKKOS_LAMBDA(int row) {
                        pres_black_d(0,     row) = pres_red_d  (1,   row);
                        pres_red_d  (0,     row) = pres_black_d(1,   row);
                        pres_black_d(NUM+1, row) = pres_red_d  (NUM, row);
                        pres_red_d  (NUM+1, row) = pres_black_d(NUM, row);
                    });

                // update red cells
                Kokkos::parallel_for("red_kernel",
                    MDPolicy2D({1,1},{NUM+1,NUM/2+1}),
                    KOKKOS_LAMBDA(int col, int row) {
                        const Real p_ij   = pres_red_d  (col,   row);
                        const Real p_im1j = pres_black_d(col-1, row);
                        const Real p_ip1j = pres_black_d(col+1, row);
                        const Real p_ijm1 = pres_black_d(col,   row - (col & 1));
                        const Real p_ijp1 = pres_black_d(col,   row + ((col+1) & 1));

                        const int  ri  = (2*row) - (col & 1);
                        const Real rhs = (((F_d(col,   ri) - F_d(col-1, ri)) / dx)
                                         + ((G_d(col,  ri) - G_d(col,   ri-1)) / dy)) / dt;

                        pres_red_d(col, row) = p_ij*(ONE-omega) + omega*(
                            ((p_ip1j+p_im1j)/(dx*dx)) + ((p_ijp1+p_ijm1)/(dy*dy)) - rhs
                        ) / ((TWO/(dx*dx)) + (TWO/(dy*dy)));
                    });

                // update black cells
                Kokkos::parallel_for("black_kernel",
                    MDPolicy2D({1,1},{NUM+1,NUM/2+1}),
                    KOKKOS_LAMBDA(int col, int row) {
                        const Real p_ij   = pres_black_d(col,   row);
                        const Real p_im1j = pres_red_d  (col-1, row);
                        const Real p_ip1j = pres_red_d  (col+1, row);
                        const Real p_ijm1 = pres_red_d  (col,   row - ((col+1) & 1));
                        const Real p_ijp1 = pres_red_d  (col,   row + (col & 1));

                        const int  ri  = (2*row) - ((col+1) & 1);
                        const Real rhs = (((F_d(col,   ri) - F_d(col-1, ri)) / dx)
                                         + ((G_d(col,  ri) - G_d(col,   ri-1)) / dy)) / dt;

                        pres_black_d(col, row) = p_ij*(ONE-omega) + omega*(
                            ((p_ip1j+p_im1j)/(dx*dx)) + ((p_ijp1+p_ijm1)/(dy*dy)) - rhs
                        ) / ((TWO/(dx*dx)) + (TWO/(dy*dy)));
                    });

                // calculate L2 residual
                Real res_sum = 0.0;
                Kokkos::parallel_reduce("residual", N,
                    KOKKOS_LAMBDA(int gid, Real& lsum) {
                        const int row = (gid % (NUM/2)) + 1;
                        const int col = (gid / (NUM/2)) + 1;

                        // red residual
                        Real p_ij   = pres_red_d  (col,   row);
                        Real p_im1j = pres_black_d(col-1, row);
                        Real p_ip1j = pres_black_d(col+1, row);
                        Real p_ijm1 = pres_black_d(col,   row - (col & 1));
                        Real p_ijp1 = pres_black_d(col,   row + ((col+1) & 1));
                        int  ri     = (2*row) - (col & 1);
                        Real rhs    = (((F_d(col,  ri) - F_d(col-1, ri)) / dx)
                                      + ((G_d(col, ri) - G_d(col,   ri-1)) / dy)) / dt;
                        Real res    = ((p_ip1j - TWO*p_ij + p_im1j) / (dx*dx))
                                    + ((p_ijp1 - TWO*p_ij + p_ijm1) / (dy*dy)) - rhs;

                        // black residual
                        p_ij   = pres_black_d(col,   row);
                        p_im1j = pres_red_d  (col-1, row);
                        p_ip1j = pres_red_d  (col+1, row);
                        p_ijm1 = pres_red_d  (col,   row - ((col+1) & 1));
                        p_ijp1 = pres_red_d  (col,   row + (col & 1));
                        ri     = (2*row) - ((col+1) & 1);
                        rhs    = (((F_d(col,  ri) - F_d(col-1, ri)) / dx)
                                 + ((G_d(col, ri) - G_d(col,   ri-1)) / dy)) / dt;
                        Real res2 = ((p_ip1j - TWO*p_ij + p_im1j) / (dx*dx))
                                  + ((p_ijp1 - TWO*p_ij + p_ijm1) / (dy*dy)) - rhs;

                        lsum += res*res + res2*res2;
                    }, res_sum);

                norm_L2 = std::sqrt(res_sum / (Real)(NUM*NUM)) / p0_norm;
                if (norm_L2 < tol) break;

            } // pressure iteration

            printf("Time = %f, delt = %e, iter = %i, res = %e\n",
                   time+dt, dt, iter, norm_L2);

            // -----------------------------------------------------------------
            // calculate_u  – update u velocity from pressure gradient
            // -----------------------------------------------------------------
            Kokkos::parallel_for("calc_u_write", N,
                KOKKOS_LAMBDA(int gid) {
                    const int row = (gid % (NUM/2)) + 1;
                    const int col = (gid / (NUM/2)) + 1;
                    if (col != NUM) {
                        // red pressure point
                        Real p_ij   = pres_red_d  (col,   row);
                        Real p_ip1j = pres_black_d(col+1, row);
                        u_d(col, (2*row)-(col&1)) =
                            F_d(col, (2*row)-(col&1)) - dt*(p_ip1j-p_ij)/dx;

                        // black pressure point
                        p_ij   = pres_black_d(col,   row);
                        p_ip1j = pres_red_d  (col+1, row);
                        u_d(col, (2*row)-((col+1)&1)) =
                            F_d(col, (2*row)-((col+1)&1)) - dt*(p_ip1j-p_ij)/dx;
                    }
                });

            // find max |u| over the domain
            Real max_u_new = 0.0;
            Kokkos::parallel_reduce("max_u", N,
                KOKKOS_LAMBDA(int gid, Real& lmax) {
                    const int row = (gid % (NUM/2)) + 1;
                    const int col = (gid / (NUM/2)) + 1;
                    Real local = ZERO;
                    if (col != NUM) {
                        local = fmax(fabs(u_d(col, (2*row)-(col&1))),
                                     fabs(u_d(col, (2*row)-((col+1)&1))));
                        if ((2*row) == NUM)
                            local = fmax(local, fabs(u_d(col, NUM+1)));
                    } else {
                        local = fmax(fabs(u_d(NUM,   (2*row))),
                                     fabs(u_d(0,     (2*row))));
                        local = fmax(local, fabs(u_d(NUM,   (2*row)-1)));
                        local = fmax(local, fabs(u_d(0,     (2*row)-1)));
                        local = fmax(local, fabs(u_d(NUM+1, (2*row))));
                        local = fmax(local, fabs(u_d(NUM+1, (2*row)-1)));
                    }
                    if (local > lmax) lmax = local;
                }, Kokkos::Max<Real>(max_u_new));

            max_u = fmax(1.0e-10, max_u_new);

            // -----------------------------------------------------------------
            // calculate_v  – update v velocity from pressure gradient
            // -----------------------------------------------------------------
            Kokkos::parallel_for("calc_v_write", N,
                KOKKOS_LAMBDA(int gid) {
                    const int row   = (gid % (NUM/2)) + 1;
                    const int col   = (gid / (NUM/2)) + 1;
                    const int NUM_2 = NUM >> 1;

                    if (row != NUM_2) {
                        // red pressure point
                        Real p_ij   = pres_red_d  (col, row);
                        Real p_ijp1 = pres_black_d(col, row + ((col+1) & 1));
                        v_d(col, (2*row)-(col&1)) =
                            G_d(col, (2*row)-(col&1)) - dt*(p_ijp1-p_ij)/dy;

                        // black pressure point
                        p_ij   = pres_black_d(col, row);
                        p_ijp1 = pres_red_d  (col, row + (col & 1));
                        v_d(col, (2*row)-((col+1)&1)) =
                            G_d(col, (2*row)-((col+1)&1)) - dt*(p_ijp1-p_ij)/dy;
                    } else {
                        // top row of pressure grid: only one interior v-cell per column
                        if ((col & 1) == 1) {
                            // col is odd: black v-point is on boundary – only update red
                            Real p_ij   = pres_red_d  (col, row);
                            Real p_ijp1 = pres_black_d(col, row + ((col+1) & 1));
                            v_d(col, (2*row)-(col&1)) =
                                G_d(col, (2*row)-(col&1)) - dt*(p_ijp1-p_ij)/dy;
                        } else {
                            // col is even: red v-point is on boundary – only update black
                            Real p_ij   = pres_black_d(col, row);
                            Real p_ijp1 = pres_red_d  (col, row + (col & 1));
                            v_d(col, (2*row)-((col+1)&1)) =
                                G_d(col, (2*row)-((col+1)&1)) - dt*(p_ijp1-p_ij)/dy;
                        }
                    }
                });

            // find max |v| over the domain
            Real max_v_new = 0.0;
            Kokkos::parallel_reduce("max_v", N,
                KOKKOS_LAMBDA(int gid, Real& lmax) {
                    const int row   = (gid % (NUM/2)) + 1;
                    const int col   = (gid / (NUM/2)) + 1;
                    const int NUM_2 = NUM >> 1;
                    Real local = ZERO;

                    if (row != NUM_2) {
                        local = fmax(fabs(v_d(col, (2*row)-(col&1))),
                                     fabs(v_d(col, (2*row)-((col+1)&1))));
                        if (col == NUM)
                            local = fmax(local, fabs(v_d(NUM+1, (2*row))));
                    } else {
                        if ((col & 1) == 1)
                            local = fabs(v_d(col, (2*row)-(col&1)));
                        else
                            local = fabs(v_d(col, (2*row)-((col+1)&1)));

                        local = fmax(local, fabs(v_d(col, NUM)));
                        local = fmax(local, fabs(v_d(col, 0)));
                        local = fmax(local, fabs(v_d(col, NUM+1)));
                    }
                    if (local > lmax) lmax = local;
                }, Kokkos::Max<Real>(max_v_new));

            max_v = fmax(1.0e-10, max_v_new);

            do_set_BCs();

            time += dt;

        } // time loop

        auto wct_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           wct_end - wct_start).count();
        printf("\nTotal execution time of the iteration loop: %f (s)\n",
               elapsed * 1.0e-9);

        // ---------------------------------------------------------------------
        // Write velocity field to file
        // ---------------------------------------------------------------------
        auto u_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_d);
        auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_d);

        FILE* pfile = fopen("velocity_gpu.dat", "w");
        if (pfile != NULL) {
            fprintf(pfile, "#x\ty\tu\tv\n");
            for (int row = 0; row < NUM; ++row) {
                for (int col = 0; col < NUM; ++col) {
                    Real u_ij   = u_h(col, row);
                    Real u_im1j = (col == 0) ? 0.0 : u_h(col-1, row);
                    u_ij = (u_ij + u_im1j) / 2.0;

                    Real v_ij   = v_h(col, row);
                    Real v_ijm1 = (row == 0) ? 0.0 : v_h(col, row-1);
                    v_ij = (v_ij + v_ijm1) / 2.0;

                    fprintf(pfile, "%f\t%f\t%f\t%f\n",
                            ((Real)col + 0.5)*dx,
                            ((Real)row + 0.5)*dy,
                            u_ij, v_ij);
                }
            }
            fclose(pfile);
        }

    } // Kokkos scope
    Kokkos::finalize();
    return 0;
}
