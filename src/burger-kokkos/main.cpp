/*
 * 2D Burgers' equation.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define IDX(i,j) ((i)*y_points+(j))

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <dim_x> <dim_y> <nt>\n", argv[0]);
    exit(-1);
  }
  const int x_points = atoi(argv[1]);
  const int y_points = atoi(argv[2]);
  const int num_itrs = atoi(argv[3]);
  const double x_len = 2.0, y_len = 2.0;
  const double del_x = x_len / (x_points - 1);
  const double del_y = y_len / (y_points - 1);
  const int grid_pts = x_points * y_points;

  double *x = (double*) malloc(x_points * sizeof(double));
  double *y = (double*) malloc(y_points * sizeof(double));
  double *u = (double*) malloc(grid_pts * sizeof(double));
  double *v = (double*) malloc(grid_pts * sizeof(double));
  double *u_new = (double*) malloc(grid_pts * sizeof(double));
  double *v_new = (double*) malloc(grid_pts * sizeof(double));
  double *d_u = (double*) malloc(grid_pts * sizeof(double));
  double *d_v = (double*) malloc(grid_pts * sizeof(double));

  const double nu = 0.01;
  const double sigma = 0.0009;
  const double del_t = sigma * del_x * del_y / nu;

  printf("2D Burger's equation\n");
  printf("Grid dimension: x = %d y = %d\n", x_points, y_points);
  printf("Number of time steps: %d\n", num_itrs);

  for (int i = 0; i < x_points; i++) x[i] = i * del_x;
  for (int i = 0; i < y_points; i++) y[i] = i * del_y;

  for (int i = 0; i < y_points; i++) {
    for (int j = 0; j < x_points; j++) {
      u[IDX(i,j)] = u_new[IDX(i,j)] = 1.0;
      v[IDX(i,j)] = v_new[IDX(i,j)] = 1.0;
      if (x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0) {
        u[IDX(i,j)] = u_new[IDX(i,j)] = 2.0;
        v[IDX(i,j)] = v_new[IDX(i,j)] = 2.0;
      }
    }
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_u_v("d_u", grid_pts);
    Kokkos::View<double*> d_v_v("d_v", grid_pts);
    Kokkos::View<double*> d_u_new("d_u_new", grid_pts);
    Kokkos::View<double*> d_v_new("d_v_new", grid_pts);

    auto h_u     = Kokkos::create_mirror_view(d_u_v);
    auto h_v     = Kokkos::create_mirror_view(d_v_v);
    auto h_u_new = Kokkos::create_mirror_view(d_u_new);
    auto h_v_new = Kokkos::create_mirror_view(d_v_new);

    for (int i = 0; i < grid_pts; i++) {
      h_u(i) = u[i]; h_v(i) = v[i];
      h_u_new(i) = u_new[i]; h_v_new(i) = v_new[i];
    }
    Kokkos::deep_copy(d_u_v, h_u); Kokkos::deep_copy(d_v_v, h_v);
    Kokkos::deep_copy(d_u_new, h_u_new); Kokkos::deep_copy(d_v_new, h_v_new);

    const int xp = x_points, yp = y_points;
    const double nu2 = nu, dt = del_t, dx = del_x, dy = del_y;

    auto start = std::chrono::steady_clock::now();

    for (int itr = 0; itr < num_itrs; itr++) {
      // Compute interior
      Kokkos::parallel_for("burger_interior",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{yp-1,xp-1}),
          KOKKOS_LAMBDA(int i, int j) {
            d_u_new(i*yp+j) = d_u_v(i*yp+j)
              + (nu2*dt/(dx*dx)) * (d_u_v(i*yp+j+1) + d_u_v(i*yp+j-1) - 2.0*d_u_v(i*yp+j))
              + (nu2*dt/(dy*dy)) * (d_u_v((i+1)*yp+j) + d_u_v((i-1)*yp+j) - 2.0*d_u_v(i*yp+j))
              - (dt/dx)*d_u_v(i*yp+j) * (d_u_v(i*yp+j) - d_u_v(i*yp+j-1))
              - (dt/dy)*d_v_v(i*yp+j) * (d_u_v(i*yp+j) - d_u_v((i-1)*yp+j));

            d_v_new(i*yp+j) = d_v_v(i*yp+j)
              + (nu2*dt/(dx*dx)) * (d_v_v(i*yp+j+1) + d_v_v(i*yp+j-1) - 2.0*d_v_v(i*yp+j))
              + (nu2*dt/(dy*dy)) * (d_v_v((i+1)*yp+j) + d_v_v((i-1)*yp+j) - 2.0*d_v_v(i*yp+j))
              - (dt/dx)*d_u_v(i*yp+j) * (d_v_v(i*yp+j) - d_v_v(i*yp+j-1))
              - (dt/dy)*d_v_v(i*yp+j) * (d_v_v(i*yp+j) - d_v_v((i-1)*yp+j));
          });
      Kokkos::fence();

      // Boundary conditions (x boundaries)
      Kokkos::parallel_for("burger_bc_x", xp, KOKKOS_LAMBDA(int i) {
        d_u_new(0*yp+i) = d_v_new(0*yp+i) = 1.0;
        d_u_new((yp-1)*yp+i) = d_v_new((yp-1)*yp+i) = 1.0;
      });
      Kokkos::fence();

      // Boundary conditions (y boundaries)
      Kokkos::parallel_for("burger_bc_y", yp, KOKKOS_LAMBDA(int j) {
        d_u_new(j*yp+0) = d_v_new(j*yp+0) = 1.0;
        d_u_new(j*yp+(xp-1)) = d_v_new(j*yp+(xp-1)) = 1.0;
      });
      Kokkos::fence();

      // Update
      Kokkos::parallel_for("burger_update",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{yp,xp}),
          KOKKOS_LAMBDA(int i, int j) {
            d_u_v(i*yp+j) = d_u_new(i*yp+j);
            d_v_v(i*yp+j) = d_v_new(i*yp+j);
          });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time %f (s)\n", time * 1e-9f);

    auto h_u2 = Kokkos::create_mirror_view(d_u_v);
    auto h_v2 = Kokkos::create_mirror_view(d_v_v);
    Kokkos::deep_copy(h_u2, d_u_v); Kokkos::deep_copy(h_v2, d_v_v);
    for (int i = 0; i < grid_pts; i++) { d_u[i] = h_u2(i); d_v[i] = h_v2(i); }
  }
  Kokkos::finalize();

  // CPU reference
  printf("Serial computing for verification...\n");
  for (int i = 0; i < y_points; i++) {
    for (int j = 0; j < x_points; j++) {
      u[IDX(i,j)] = u_new[IDX(i,j)] = 1.0;
      v[IDX(i,j)] = v_new[IDX(i,j)] = 1.0;
      if (x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0) {
        u[IDX(i,j)] = u_new[IDX(i,j)] = 2.0;
        v[IDX(i,j)] = v_new[IDX(i,j)] = 2.0;
      }
    }
  }

  for (int itr = 0; itr < num_itrs; itr++) {
    for (int i = 1; i < y_points-1; i++) {
      for (int j = 1; j < x_points-1; j++) {
        u_new[IDX(i,j)] = u[IDX(i,j)]
          + (nu*del_t/(del_x*del_x)) * (u[IDX(i,j+1)] + u[IDX(i,j-1)] - 2.0*u[IDX(i,j)])
          + (nu*del_t/(del_y*del_y)) * (u[IDX(i+1,j)] + u[IDX(i-1,j)] - 2.0*u[IDX(i,j)])
          - (del_t/del_x)*u[IDX(i,j)] * (u[IDX(i,j)] - u[IDX(i,j-1)])
          - (del_t/del_y)*v[IDX(i,j)] * (u[IDX(i,j)] - u[IDX(i-1,j)]);
        v_new[IDX(i,j)] = v[IDX(i,j)]
          + (nu*del_t/(del_x*del_x)) * (v[IDX(i,j+1)] + v[IDX(i,j-1)] - 2.0*v[IDX(i,j)])
          + (nu*del_t/(del_y*del_y)) * (v[IDX(i+1,j)] + v[IDX(i-1,j)] - 2.0*v[IDX(i,j)])
          - (del_t/del_x)*u[IDX(i,j)] * (v[IDX(i,j)] - v[IDX(i,j-1)])
          - (del_t/del_y)*v[IDX(i,j)] * (v[IDX(i,j)] - v[IDX(i-1,j)]);
      }
    }
    for (int i = 0; i < x_points; i++) {
      u_new[IDX(0,i)] = v_new[IDX(0,i)] = 1.0;
      u_new[IDX(y_points-1,i)] = v_new[IDX(y_points-1,i)] = 1.0;
    }
    for (int j = 0; j < y_points; j++) {
      u_new[IDX(j,0)] = v_new[IDX(j,0)] = 1.0;
      u_new[IDX(j,x_points-1)] = v_new[IDX(j,x_points-1)] = 1.0;
    }
    for (int i = 0; i < y_points; i++)
      for (int j = 0; j < x_points; j++) {
        u[IDX(i,j)] = u_new[IDX(i,j)];
        v[IDX(i,j)] = v_new[IDX(i,j)];
      }
  }

  bool ok = true;
  for (int i = 0; i < y_points; i++) {
    for (int j = 0; j < x_points; j++) {
      if (fabs(d_u[IDX(i,j)] - u[IDX(i,j)]) > 1e-6 ||
          fabs(d_v[IDX(i,j)] - v[IDX(i,j)]) > 1e-6) { ok = false; break; }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(x); free(y); free(u); free(v); free(u_new); free(v_new); free(d_u); free(d_v);
  return 0;
}
