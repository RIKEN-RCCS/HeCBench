#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

#define TOLERANCE 1e-3
#define D 308

// -----------------------------------------------------------------------
// Random helpers
// -----------------------------------------------------------------------
static double get_random_d() {
    return ((double)rand() / (double)(RAND_MAX - 1)) + 0.02121;
}

// -----------------------------------------------------------------------
// CPU reference kernels (flat 1D indexing, same as OMP version)
// -----------------------------------------------------------------------
static void reference(
    double* flux_0, double* flux_1, double* flux_2, double* flux_3, double* flux_4,
    const double* cons_1, const double* cons_2, const double* cons_3, const double* cons_4,
    const double* q_1,    const double* q_2,    const double* q_3,    const double* q_4,
    double dxinv0, double dxinv1, double dxinv2, int N)
{
    int M = N;
    for (int k = 4; k < N-4; k++)
    for (int j = 4; j < N-4; j++)
    for (int i = 4; i < N-4; i++) {
        flux_0[k*M*N+j*N+i] =
            -((0.8f*(cons_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1])
              -0.2f*(cons_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2])
              +0.038f*(cons_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3])
              -0.0035f*(cons_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]))*dxinv0);
        flux_0[k*M*N+j*N+i] -=
            (0.8f*(cons_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i])
            -0.2f*(cons_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i])
            +0.038f*(cons_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i])
            -0.0035f*(cons_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]))*dxinv1;
        flux_0[k*M*N+j*N+i] -=
            (0.8f*(cons_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i])
            -0.2f*(cons_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i])
            +0.038f*(cons_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i])
            -0.0035f*(cons_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]))*dxinv2;
    }
    for (int k = 4; k < N-4; k++)
    for (int j = 4; j < N-4; j++)
    for (int i = 4; i < N-4; i++) {
        flux_1[k*M*N+j*N+i] =
            -((0.8f*(cons_1[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]))
              -0.2f*(cons_1[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]))
              +0.038f*(cons_1[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]))
              -0.0035f*(cons_1[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4])))*dxinv0);
        flux_1[k*M*N+j*N+i] -=
            (0.8f*(cons_1[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_1[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])
            -0.2f*(cons_1[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_1[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])
            +0.038f*(cons_1[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_1[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])
            -0.0035f*(cons_1[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_1[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
        flux_1[k*M*N+j*N+i] -=
            (0.8f*(cons_1[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_1[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])
            -0.2f*(cons_1[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_1[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])
            +0.038f*(cons_1[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_1[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])
            -0.0035f*(cons_1[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_1[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
    }
    for (int k = 4; k < N-4; k++)
    for (int j = 4; j < N-4; j++)
    for (int i = 4; i < N-4; i++) {
        flux_2[k*M*N+j*N+i] =
            -((0.8f*(cons_2[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_2[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])
              -0.2f*(cons_2[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_2[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])
              +0.038f*(cons_2[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_2[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])
              -0.0035f*(cons_2[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_2[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
        flux_2[k*M*N+j*N+i] -=
            (0.8f*(cons_2[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]))
            -0.2f*(cons_2[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]))
            +0.038f*(cons_2[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]))
            -0.0035f*(cons_2[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i])))*dxinv1;
        flux_2[k*M*N+j*N+i] -=
            (0.8f*(cons_2[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_2[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])
            -0.2f*(cons_2[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_2[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])
            +0.038f*(cons_2[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_2[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])
            -0.0035f*(cons_2[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_2[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
    }
    for (int k = 4; k < N-4; k++)
    for (int j = 4; j < N-4; j++)
    for (int i = 4; i < N-4; i++) {
        flux_3[k*M*N+j*N+i] =
            -((0.8f*(cons_3[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_3[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])
              -0.2f*(cons_3[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_3[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])
              +0.038f*(cons_3[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_3[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])
              -0.0035f*(cons_3[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_3[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
        flux_3[k*M*N+j*N+i] -=
            (0.8f*(cons_3[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_3[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])
            -0.2f*(cons_3[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_3[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])
            +0.038f*(cons_3[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_3[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])
            -0.0035f*(cons_3[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_3[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
        flux_3[k*M*N+j*N+i] -=
            (0.8f*(cons_3[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]))
            -0.2f*(cons_3[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]))
            +0.038f*(cons_3[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]))
            -0.0035f*(cons_3[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i])))*dxinv2;
    }
    for (int k = 4; k < N-4; k++)
    for (int j = 4; j < N-4; j++)
    for (int i = 4; i < N-4; i++) {
        flux_4[k*M*N+j*N+i] =
            -((0.8f*(cons_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]))
              -0.2f*(cons_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]))
              +0.038f*(cons_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]))
              -0.0035f*(cons_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4])))*dxinv0);
        flux_4[k*M*N+j*N+i] -=
            (0.8f*(cons_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]))
            -0.2f*(cons_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]))
            +0.038f*(cons_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]))
            -0.0035f*(cons_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i])))*dxinv2;
        flux_4[k*M*N+j*N+i] -=
            (0.8f*(cons_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]))
            -0.2f*(cons_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]))
            +0.038f*(cons_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]))
            -0.0035f*(cons_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i])))*dxinv1;
    }
}

// -----------------------------------------------------------------------
// Error check helper
// -----------------------------------------------------------------------
static double checkError(int N, const double* output, const double* ref,
                         int z_lb, int z_ub, int y_lb, int y_ub, int x_lb, int x_ub)
{
    int M = N;
    double error = 0.0;
    double sum   = 0.0;
    for (int k = z_lb; k < z_ub; k++)
    for (int j = y_lb; j < y_ub; j++)
    for (int i = x_lb; i < x_ub; i++) {
        sum += output[k*M*N+j*N+i];
        double curr = fabs(output[k*M*N+j*N+i] - ref[k*M*N+j*N+i]);
        error += curr * curr;
        if (curr > TOLERANCE)
            printf("Values at (%d,%d,%d) differ: %.6f vs %.6f\n",
                   k, j, i, ref[k*M*N+j*N+i], output[k*M*N+j*N+i]);
    }
    printf("checksum = %e\n", sum);
    int cnt = (z_ub-z_lb)*(y_ub-y_lb)*(x_ub-x_lb);
    return sqrt(error / cnt);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    size_t vol = (size_t)D * D * D;

    double *cons_1 = new double[vol];
    double *cons_2 = new double[vol];
    double *cons_3 = new double[vol];
    double *cons_4 = new double[vol];
    double *q_1    = new double[vol];
    double *q_2    = new double[vol];
    double *q_3    = new double[vol];
    double *q_4    = new double[vol];

    double *flux_0 = new double[vol]();
    double *flux_1 = new double[vol]();
    double *flux_2 = new double[vol]();
    double *flux_3 = new double[vol]();
    double *flux_4 = new double[vol]();

    double *fg_0   = new double[vol]();
    double *fg_1   = new double[vol]();
    double *fg_2   = new double[vol]();
    double *fg_3   = new double[vol]();
    double *fg_4   = new double[vol]();

    for (size_t i = 0; i < vol; i++) {
        cons_1[i] = get_random_d(); cons_2[i] = get_random_d();
        cons_3[i] = get_random_d(); cons_4[i] = get_random_d();
        q_1[i]    = get_random_d(); q_2[i]    = get_random_d();
        q_3[i]    = get_random_d(); q_4[i]    = get_random_d();
    }

    double dxinv0 = 0.01, dxinv1 = 0.02, dxinv2 = 0.03;

    // CPU reference (uses gold arrays, starting from zeros)
    reference(fg_0, fg_1, fg_2, fg_3, fg_4,
              cons_1, cons_2, cons_3, cons_4,
              q_1, q_2, q_3, q_4,
              dxinv0, dxinv1, dxinv2, D);

    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<double*> d_cons_1("cons_1", vol);
        Kokkos::View<double*> d_cons_2("cons_2", vol);
        Kokkos::View<double*> d_cons_3("cons_3", vol);
        Kokkos::View<double*> d_cons_4("cons_4", vol);
        Kokkos::View<double*> d_q_1("q_1", vol);
        Kokkos::View<double*> d_q_2("q_2", vol);
        Kokkos::View<double*> d_q_3("q_3", vol);
        Kokkos::View<double*> d_q_4("q_4", vol);
        Kokkos::View<double*> d_flux_0("flux_0", vol);
        Kokkos::View<double*> d_flux_1("flux_1", vol);
        Kokkos::View<double*> d_flux_2("flux_2", vol);
        Kokkos::View<double*> d_flux_3("flux_3", vol);
        Kokkos::View<double*> d_flux_4("flux_4", vol);

        // Upload inputs
        auto copy_in = [&](Kokkos::View<double*> d, const double* h) {
            auto m = Kokkos::create_mirror_view(d);
            for (size_t i = 0; i < vol; i++) m(i) = h[i];
            Kokkos::deep_copy(d, m);
        };
        copy_in(d_cons_1, cons_1); copy_in(d_cons_2, cons_2);
        copy_in(d_cons_3, cons_3); copy_in(d_cons_4, cons_4);
        copy_in(d_q_1, q_1); copy_in(d_q_2, q_2);
        copy_in(d_q_3, q_3); copy_in(d_q_4, q_4);

        using Policy3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        long t1 = 0, t2 = 0, t3 = 0;

        for (int rep = 0; rep < repeat; rep++) {
            // Reset flux arrays to zero each iteration
            Kokkos::deep_copy(d_flux_0, 0.0);
            Kokkos::deep_copy(d_flux_1, 0.0);
            Kokkos::deep_copy(d_flux_2, 0.0);
            Kokkos::deep_copy(d_flux_3, 0.0);
            Kokkos::deep_copy(d_flux_4, 0.0);

            int N = D, M = D;

            // hypterm_1: x-direction
            auto s1 = std::chrono::steady_clock::now();
            Kokkos::parallel_for("hypterm_1",
                Policy3D({4,4,4}, {N-4,N-4,N-4}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    d_flux_0[k*M*N+j*N+i] = -((0.8f*(d_cons_1[k*M*N+j*N+i+1]-d_cons_1[k*M*N+j*N+i-1])-0.2f*(d_cons_1[k*M*N+j*N+i+2]-d_cons_1[k*M*N+j*N+i-2])+0.038f*(d_cons_1[k*M*N+j*N+i+3]-d_cons_1[k*M*N+j*N+i-3])-0.0035f*(d_cons_1[k*M*N+j*N+i+4]-d_cons_1[k*M*N+j*N+i-4]))*dxinv0);
                    d_flux_1[k*M*N+j*N+i] = -((0.8f*(d_cons_1[k*M*N+j*N+i+1]*d_q_1[k*M*N+j*N+i+1]-d_cons_1[k*M*N+j*N+i-1]*d_q_1[k*M*N+j*N+i-1]+(d_q_4[k*M*N+j*N+i+1]-d_q_4[k*M*N+j*N+i-1]))-0.2f*(d_cons_1[k*M*N+j*N+i+2]*d_q_1[k*M*N+j*N+i+2]-d_cons_1[k*M*N+j*N+i-2]*d_q_1[k*M*N+j*N+i-2]+(d_q_4[k*M*N+j*N+i+2]-d_q_4[k*M*N+j*N+i-2]))+0.038f*(d_cons_1[k*M*N+j*N+i+3]*d_q_1[k*M*N+j*N+i+3]-d_cons_1[k*M*N+j*N+i-3]*d_q_1[k*M*N+j*N+i-3]+(d_q_4[k*M*N+j*N+i+3]-d_q_4[k*M*N+j*N+i-3]))-0.0035f*(d_cons_1[k*M*N+j*N+i+4]*d_q_1[k*M*N+j*N+i+4]-d_cons_1[k*M*N+j*N+i-4]*d_q_1[k*M*N+j*N+i-4]+(d_q_4[k*M*N+j*N+i+4]-d_q_4[k*M*N+j*N+i-4])))*dxinv0);
                    d_flux_2[k*M*N+j*N+i] = -((0.8f*(d_cons_2[k*M*N+j*N+i+1]*d_q_1[k*M*N+j*N+i+1]-d_cons_2[k*M*N+j*N+i-1]*d_q_1[k*M*N+j*N+i-1])-0.2f*(d_cons_2[k*M*N+j*N+i+2]*d_q_1[k*M*N+j*N+i+2]-d_cons_2[k*M*N+j*N+i-2]*d_q_1[k*M*N+j*N+i-2])+0.038f*(d_cons_2[k*M*N+j*N+i+3]*d_q_1[k*M*N+j*N+i+3]-d_cons_2[k*M*N+j*N+i-3]*d_q_1[k*M*N+j*N+i-3])-0.0035f*(d_cons_2[k*M*N+j*N+i+4]*d_q_1[k*M*N+j*N+i+4]-d_cons_2[k*M*N+j*N+i-4]*d_q_1[k*M*N+j*N+i-4]))*dxinv0);
                    d_flux_3[k*M*N+j*N+i] = -((0.8f*(d_cons_3[k*M*N+j*N+i+1]*d_q_1[k*M*N+j*N+i+1]-d_cons_3[k*M*N+j*N+i-1]*d_q_1[k*M*N+j*N+i-1])-0.2f*(d_cons_3[k*M*N+j*N+i+2]*d_q_1[k*M*N+j*N+i+2]-d_cons_3[k*M*N+j*N+i-2]*d_q_1[k*M*N+j*N+i-2])+0.038f*(d_cons_3[k*M*N+j*N+i+3]*d_q_1[k*M*N+j*N+i+3]-d_cons_3[k*M*N+j*N+i-3]*d_q_1[k*M*N+j*N+i-3])-0.0035f*(d_cons_3[k*M*N+j*N+i+4]*d_q_1[k*M*N+j*N+i+4]-d_cons_3[k*M*N+j*N+i-4]*d_q_1[k*M*N+j*N+i-4]))*dxinv0);
                    d_flux_4[k*M*N+j*N+i] = -((0.8f*(d_cons_4[k*M*N+j*N+i+1]*d_q_1[k*M*N+j*N+i+1]-d_cons_4[k*M*N+j*N+i-1]*d_q_1[k*M*N+j*N+i-1]+(d_q_4[k*M*N+j*N+i+1]*d_q_1[k*M*N+j*N+i+1]-d_q_4[k*M*N+j*N+i-1]*d_q_1[k*M*N+j*N+i-1]))-0.2f*(d_cons_4[k*M*N+j*N+i+2]*d_q_1[k*M*N+j*N+i+2]-d_cons_4[k*M*N+j*N+i-2]*d_q_1[k*M*N+j*N+i-2]+(d_q_4[k*M*N+j*N+i+2]*d_q_1[k*M*N+j*N+i+2]-d_q_4[k*M*N+j*N+i-2]*d_q_1[k*M*N+j*N+i-2]))+0.038f*(d_cons_4[k*M*N+j*N+i+3]*d_q_1[k*M*N+j*N+i+3]-d_cons_4[k*M*N+j*N+i-3]*d_q_1[k*M*N+j*N+i-3]+(d_q_4[k*M*N+j*N+i+3]*d_q_1[k*M*N+j*N+i+3]-d_q_4[k*M*N+j*N+i-3]*d_q_1[k*M*N+j*N+i-3]))-0.0035f*(d_cons_4[k*M*N+j*N+i+4]*d_q_1[k*M*N+j*N+i+4]-d_cons_4[k*M*N+j*N+i-4]*d_q_1[k*M*N+j*N+i-4]+(d_q_4[k*M*N+j*N+i+4]*d_q_1[k*M*N+j*N+i+4]-d_q_4[k*M*N+j*N+i-4]*d_q_1[k*M*N+j*N+i-4])))*dxinv0);
                });
            Kokkos::fence();
            auto e1 = std::chrono::steady_clock::now();
            t1 += std::chrono::duration_cast<std::chrono::nanoseconds>(e1 - s1).count();

            // hypterm_2: y-direction
            auto s2 = std::chrono::steady_clock::now();
            Kokkos::parallel_for("hypterm_2",
                Policy3D({4,4,4}, {N-4,N-4,N-4}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    d_flux_0[k*M*N+j*N+i] -= (0.8f*(d_cons_2[k*M*N+(j+1)*N+i]-d_cons_2[k*M*N+(j-1)*N+i])-0.2f*(d_cons_2[k*M*N+(j+2)*N+i]-d_cons_2[k*M*N+(j-2)*N+i])+0.038f*(d_cons_2[k*M*N+(j+3)*N+i]-d_cons_2[k*M*N+(j-3)*N+i])-0.0035f*(d_cons_2[k*M*N+(j+4)*N+i]-d_cons_2[k*M*N+(j-4)*N+i]))*dxinv1;
                    d_flux_1[k*M*N+j*N+i] -= (0.8f*(d_cons_1[k*M*N+(j+1)*N+i]*d_q_2[k*M*N+(j+1)*N+i]-d_cons_1[k*M*N+(j-1)*N+i]*d_q_2[k*M*N+(j-1)*N+i])-0.2f*(d_cons_1[k*M*N+(j+2)*N+i]*d_q_2[k*M*N+(j+2)*N+i]-d_cons_1[k*M*N+(j-2)*N+i]*d_q_2[k*M*N+(j-2)*N+i])+0.038f*(d_cons_1[k*M*N+(j+3)*N+i]*d_q_2[k*M*N+(j+3)*N+i]-d_cons_1[k*M*N+(j-3)*N+i]*d_q_2[k*M*N+(j-3)*N+i])-0.0035f*(d_cons_1[k*M*N+(j+4)*N+i]*d_q_2[k*M*N+(j+4)*N+i]-d_cons_1[k*M*N+(j-4)*N+i]*d_q_2[k*M*N+(j-4)*N+i]))*dxinv1;
                    d_flux_2[k*M*N+j*N+i] -= (0.8f*(d_cons_2[k*M*N+(j+1)*N+i]*d_q_2[k*M*N+(j+1)*N+i]-d_cons_2[k*M*N+(j-1)*N+i]*d_q_2[k*M*N+(j-1)*N+i]+(d_q_4[k*M*N+(j+1)*N+i]-d_q_4[k*M*N+(j-1)*N+i]))-0.2f*(d_cons_2[k*M*N+(j+2)*N+i]*d_q_2[k*M*N+(j+2)*N+i]-d_cons_2[k*M*N+(j-2)*N+i]*d_q_2[k*M*N+(j-2)*N+i]+(d_q_4[k*M*N+(j+2)*N+i]-d_q_4[k*M*N+(j-2)*N+i]))+0.038f*(d_cons_2[k*M*N+(j+3)*N+i]*d_q_2[k*M*N+(j+3)*N+i]-d_cons_2[k*M*N+(j-3)*N+i]*d_q_2[k*M*N+(j-3)*N+i]+(d_q_4[k*M*N+(j+3)*N+i]-d_q_4[k*M*N+(j-3)*N+i]))-0.0035f*(d_cons_2[k*M*N+(j+4)*N+i]*d_q_2[k*M*N+(j+4)*N+i]-d_cons_2[k*M*N+(j-4)*N+i]*d_q_2[k*M*N+(j-4)*N+i]+(d_q_4[k*M*N+(j+4)*N+i]-d_q_4[k*M*N+(j-4)*N+i])))*dxinv1;
                    d_flux_3[k*M*N+j*N+i] -= (0.8f*(d_cons_3[k*M*N+(j+1)*N+i]*d_q_2[k*M*N+(j+1)*N+i]-d_cons_3[k*M*N+(j-1)*N+i]*d_q_2[k*M*N+(j-1)*N+i])-0.2f*(d_cons_3[k*M*N+(j+2)*N+i]*d_q_2[k*M*N+(j+2)*N+i]-d_cons_3[k*M*N+(j-2)*N+i]*d_q_2[k*M*N+(j-2)*N+i])+0.038f*(d_cons_3[k*M*N+(j+3)*N+i]*d_q_2[k*M*N+(j+3)*N+i]-d_cons_3[k*M*N+(j-3)*N+i]*d_q_2[k*M*N+(j-3)*N+i])-0.0035f*(d_cons_3[k*M*N+(j+4)*N+i]*d_q_2[k*M*N+(j+4)*N+i]-d_cons_3[k*M*N+(j-4)*N+i]*d_q_2[k*M*N+(j-4)*N+i]))*dxinv1;
                    d_flux_4[k*M*N+j*N+i] -= (0.8f*(d_cons_4[(k+1)*M*N+j*N+i]*d_q_3[(k+1)*M*N+j*N+i]-d_cons_4[(k-1)*M*N+j*N+i]*d_q_3[(k-1)*M*N+j*N+i]+(d_q_4[(k+1)*M*N+j*N+i]*d_q_3[(k+1)*M*N+j*N+i]-d_q_4[(k-1)*M*N+j*N+i]*d_q_3[(k-1)*M*N+j*N+i]))-0.2f*(d_cons_4[(k+2)*M*N+j*N+i]*d_q_3[(k+2)*M*N+j*N+i]-d_cons_4[(k-2)*M*N+j*N+i]*d_q_3[(k-2)*M*N+j*N+i]+(d_q_4[(k+2)*M*N+j*N+i]*d_q_3[(k+2)*M*N+j*N+i]-d_q_4[(k-2)*M*N+j*N+i]*d_q_3[(k-2)*M*N+j*N+i]))+0.038f*(d_cons_4[(k+3)*M*N+j*N+i]*d_q_3[(k+3)*M*N+j*N+i]-d_cons_4[(k-3)*M*N+j*N+i]*d_q_3[(k-3)*M*N+j*N+i]+(d_q_4[(k+3)*M*N+j*N+i]*d_q_3[(k+3)*M*N+j*N+i]-d_q_4[(k-3)*M*N+j*N+i]*d_q_3[(k-3)*M*N+j*N+i]))-0.0035f*(d_cons_4[(k+4)*M*N+j*N+i]*d_q_3[(k+4)*M*N+j*N+i]-d_cons_4[(k-4)*M*N+j*N+i]*d_q_3[(k-4)*M*N+j*N+i]+(d_q_4[(k+4)*M*N+j*N+i]*d_q_3[(k+4)*M*N+j*N+i]-d_q_4[(k-4)*M*N+j*N+i]*d_q_3[(k-4)*M*N+j*N+i])))*dxinv2;
                });
            Kokkos::fence();
            auto e2 = std::chrono::steady_clock::now();
            t2 += std::chrono::duration_cast<std::chrono::nanoseconds>(e2 - s2).count();

            // hypterm_3: z-direction
            auto s3 = std::chrono::steady_clock::now();
            Kokkos::parallel_for("hypterm_3",
                Policy3D({4,4,4}, {N-4,N-4,N-4}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    d_flux_0[k*M*N+j*N+i] -= (0.8f*(d_cons_3[(k+1)*M*N+j*N+i]-d_cons_3[(k-1)*M*N+j*N+i])-0.2f*(d_cons_3[(k+2)*M*N+j*N+i]-d_cons_3[(k-2)*M*N+j*N+i])+0.038f*(d_cons_3[(k+3)*M*N+j*N+i]-d_cons_3[(k-3)*M*N+j*N+i])-0.0035f*(d_cons_3[(k+4)*M*N+j*N+i]-d_cons_3[(k-4)*M*N+j*N+i]))*dxinv2;
                    d_flux_1[k*M*N+j*N+i] -= (0.8f*(d_cons_1[(k+1)*M*N+j*N+i]*d_q_3[(k+1)*M*N+j*N+i]-d_cons_1[(k-1)*M*N+j*N+i]*d_q_3[(k-1)*M*N+j*N+i])-0.2f*(d_cons_1[(k+2)*M*N+j*N+i]*d_q_3[(k+2)*M*N+j*N+i]-d_cons_1[(k-2)*M*N+j*N+i]*d_q_3[(k-2)*M*N+j*N+i])+0.038f*(d_cons_1[(k+3)*M*N+j*N+i]*d_q_3[(k+3)*M*N+j*N+i]-d_cons_1[(k-3)*M*N+j*N+i]*d_q_3[(k-3)*M*N+j*N+i])-0.0035f*(d_cons_1[(k+4)*M*N+j*N+i]*d_q_3[(k+4)*M*N+j*N+i]-d_cons_1[(k-4)*M*N+j*N+i]*d_q_3[(k-4)*M*N+j*N+i]))*dxinv2;
                    d_flux_2[k*M*N+j*N+i] -= (0.8f*(d_cons_2[(k+1)*M*N+j*N+i]*d_q_3[(k+1)*M*N+j*N+i]-d_cons_2[(k-1)*M*N+j*N+i]*d_q_3[(k-1)*M*N+j*N+i])-0.2f*(d_cons_2[(k+2)*M*N+j*N+i]*d_q_3[(k+2)*M*N+j*N+i]-d_cons_2[(k-2)*M*N+j*N+i]*d_q_3[(k-2)*M*N+j*N+i])+0.038f*(d_cons_2[(k+3)*M*N+j*N+i]*d_q_3[(k+3)*M*N+j*N+i]-d_cons_2[(k-3)*M*N+j*N+i]*d_q_3[(k-3)*M*N+j*N+i])-0.0035f*(d_cons_2[(k+4)*M*N+j*N+i]*d_q_3[(k+4)*M*N+j*N+i]-d_cons_2[(k-4)*M*N+j*N+i]*d_q_3[(k-4)*M*N+j*N+i]))*dxinv2;
                    d_flux_3[k*M*N+j*N+i] -= (0.8f*(d_cons_3[(k+1)*M*N+j*N+i]*d_q_3[(k+1)*M*N+j*N+i]-d_cons_3[(k-1)*M*N+j*N+i]*d_q_3[(k-1)*M*N+j*N+i]+(d_q_4[(k+1)*M*N+j*N+i]-d_q_4[(k-1)*M*N+j*N+i]))-0.2f*(d_cons_3[(k+2)*M*N+j*N+i]*d_q_3[(k+2)*M*N+j*N+i]-d_cons_3[(k-2)*M*N+j*N+i]*d_q_3[(k-2)*M*N+j*N+i]+(d_q_4[(k+2)*M*N+j*N+i]-d_q_4[(k-2)*M*N+j*N+i]))+0.038f*(d_cons_3[(k+3)*M*N+j*N+i]*d_q_3[(k+3)*M*N+j*N+i]-d_cons_3[(k-3)*M*N+j*N+i]*d_q_3[(k-3)*M*N+j*N+i]+(d_q_4[(k+3)*M*N+j*N+i]-d_q_4[(k-3)*M*N+j*N+i]))-0.0035f*(d_cons_3[(k+4)*M*N+j*N+i]*d_q_3[(k+4)*M*N+j*N+i]-d_cons_3[(k-4)*M*N+j*N+i]*d_q_3[(k-4)*M*N+j*N+i]+(d_q_4[(k+4)*M*N+j*N+i]-d_q_4[(k-4)*M*N+j*N+i])))*dxinv2;
                    d_flux_4[k*M*N+j*N+i] -= (0.8f*(d_cons_4[k*M*N+(j+1)*N+i]*d_q_2[k*M*N+(j+1)*N+i]-d_cons_4[k*M*N+(j-1)*N+i]*d_q_2[k*M*N+(j-1)*N+i]+(d_q_4[k*M*N+(j+1)*N+i]*d_q_2[k*M*N+(j+1)*N+i]-d_q_4[k*M*N+(j-1)*N+i]*d_q_2[k*M*N+(j-1)*N+i]))-0.2f*(d_cons_4[k*M*N+(j+2)*N+i]*d_q_2[k*M*N+(j+2)*N+i]-d_cons_4[k*M*N+(j-2)*N+i]*d_q_2[k*M*N+(j-2)*N+i]+(d_q_4[k*M*N+(j+2)*N+i]*d_q_2[k*M*N+(j+2)*N+i]-d_q_4[k*M*N+(j-2)*N+i]*d_q_2[k*M*N+(j-2)*N+i]))+0.038f*(d_cons_4[k*M*N+(j+3)*N+i]*d_q_2[k*M*N+(j+3)*N+i]-d_cons_4[k*M*N+(j-3)*N+i]*d_q_2[k*M*N+(j-3)*N+i]+(d_q_4[k*M*N+(j+3)*N+i]*d_q_2[k*M*N+(j+3)*N+i]-d_q_4[k*M*N+(j-3)*N+i]*d_q_2[k*M*N+(j-3)*N+i]))-0.0035f*(d_cons_4[k*M*N+(j+4)*N+i]*d_q_2[k*M*N+(j+4)*N+i]-d_cons_4[k*M*N+(j-4)*N+i]*d_q_2[k*M*N+(j-4)*N+i]+(d_q_4[k*M*N+(j+4)*N+i]*d_q_2[k*M*N+(j+4)*N+i]-d_q_4[k*M*N+(j-4)*N+i]*d_q_2[k*M*N+(j-4)*N+i])))*dxinv1;
                });
            Kokkos::fence();
            auto e3 = std::chrono::steady_clock::now();
            t3 += std::chrono::duration_cast<std::chrono::nanoseconds>(e3 - s3).count();
        }

        printf("Average kernel execution time (k1): %f (ms)\n", t1 * 1e-6 / repeat);
        printf("Average kernel execution time (k2): %f (ms)\n", t2 * 1e-6 / repeat);
        printf("Average kernel execution time (k3): %f (ms)\n", t3 * 1e-6 / repeat);

        // Copy results back
        auto copy_out = [&](const Kokkos::View<double*>& d, double* h) {
            auto m = Kokkos::create_mirror_view(d);
            Kokkos::deep_copy(m, d);
            for (size_t i = 0; i < vol; i++) h[i] = m(i);
        };
        copy_out(d_flux_0, flux_0); copy_out(d_flux_1, flux_1);
        copy_out(d_flux_2, flux_2); copy_out(d_flux_3, flux_3);
        copy_out(d_flux_4, flux_4);
    }
    Kokkos::finalize();

    double error;
    printf("Check flux_0\n");
    error = checkError(D, flux_0, fg_0, 4, D-4, 4, D-4, 4, D-4);
    printf("RMS Error : %e\n", error);

    printf("Check flux_1\n");
    error = checkError(D, flux_1, fg_1, 4, D-4, 4, D-4, 4, D-4);
    printf("RMS Error : %e\n", error);

    printf("Check flux_2\n");
    error = checkError(D, flux_2, fg_2, 4, D-4, 4, D-4, 4, D-4);
    printf("RMS Error : %e\n", error);

    printf("Check flux_3\n");
    error = checkError(D, flux_3, fg_3, 4, D-4, 4, D-4, 4, D-4);
    printf("RMS Error : %e\n", error);

    printf("Check flux_4\n");
    error = checkError(D, flux_4, fg_4, 4, D-4, 4, D-4, 4, D-4);
    printf("RMS Error : %e\n", error);

    delete[] cons_1; delete[] cons_2; delete[] cons_3; delete[] cons_4;
    delete[] q_1;    delete[] q_2;    delete[] q_3;    delete[] q_4;
    delete[] flux_0; delete[] flux_1; delete[] flux_2; delete[] flux_3; delete[] flux_4;
    delete[] fg_0;   delete[] fg_1;   delete[] fg_2;   delete[] fg_3;   delete[] fg_4;
    return 0;
}
