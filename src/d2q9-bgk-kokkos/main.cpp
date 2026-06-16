/*
 * D2Q9-BGK Lattice Boltzmann – Kokkos port
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE     "av_vels.dat"

typedef struct {
  int   nx, ny, maxIters, reynolds_dim;
  float density, accel, omega;
} t_param;

typedef struct {
  float speeds[NSPEEDS];
} t_speed;

void die(const char* msg, int line, const char* file) {
  fprintf(stderr, "Error at line %d of file %s:\n%s\n", line, file, msg);
  exit(EXIT_FAILURE);
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int maxItersOverride)
{
  char message[1024];
  FILE* fp = fopen(paramfile, "r");
  if (!fp) { sprintf(message, "could not open param file: %s", paramfile); die(message, __LINE__, __FILE__); }
  if (fscanf(fp, "%d\n", &params->nx)          != 1) die("nx",       __LINE__, __FILE__);
  if (fscanf(fp, "%d\n", &params->ny)          != 1) die("ny",       __LINE__, __FILE__);
  if (fscanf(fp, "%d\n", &params->maxIters)    != 1) die("maxIters", __LINE__, __FILE__);
  if (fscanf(fp, "%d\n", &params->reynolds_dim)!= 1) die("reynolds", __LINE__, __FILE__);
  if (fscanf(fp, "%f\n", &params->density)     != 1) die("density",  __LINE__, __FILE__);
  if (fscanf(fp, "%f\n", &params->accel)       != 1) die("accel",    __LINE__, __FILE__);
  if (fscanf(fp, "%f\n", &params->omega)       != 1) die("omega",    __LINE__, __FILE__);
  fclose(fp);
  if (maxItersOverride > 0) params->maxIters = maxItersOverride;

  *cells_ptr     = (t_speed*)malloc(sizeof(t_speed) * params->ny * params->nx);
  *obstacles_ptr = (int*)    malloc(sizeof(int)     * params->ny * params->nx);
  *av_vels_ptr   = (float*)  malloc(sizeof(float)   * params->maxIters);

  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density       / 9.f;
  const float w2 = params->density       / 36.f;
  for (int jj = 0; jj < params->ny; jj++)
    for (int ii = 0; ii < params->nx; ii++) {
      int idx = ii + jj * params->nx;
      (*cells_ptr)[idx].speeds[0] = w0;
      (*cells_ptr)[idx].speeds[1] = w1; (*cells_ptr)[idx].speeds[2] = w1;
      (*cells_ptr)[idx].speeds[3] = w1; (*cells_ptr)[idx].speeds[4] = w1;
      (*cells_ptr)[idx].speeds[5] = w2; (*cells_ptr)[idx].speeds[6] = w2;
      (*cells_ptr)[idx].speeds[7] = w2; (*cells_ptr)[idx].speeds[8] = w2;
      (*obstacles_ptr)[idx] = 0;
    }

  fp = fopen(obstaclefile, "r");
  if (!fp) { sprintf(message, "could not open obstacles file: %s", obstaclefile); die(message, __LINE__, __FILE__); }
  int xx, yy, blocked;
  while (fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked) == 3)
    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  fclose(fp);
  return EXIT_SUCCESS;
}

float av_velocity_cpu(const t_param params, t_speed* cells, int* obstacles) {
  float tot_u = 0.f; int tot_cells = 0;
  for (int jj = 0; jj < params.ny; jj++)
    for (int ii = 0; ii < params.nx; ii++) {
      int idx = ii + jj * params.nx;
      if (!obstacles[idx]) {
        float ld = 0.f;
        for (int k = 0; k < NSPEEDS; k++) ld += cells[idx].speeds[k];
        float u_x = (cells[idx].speeds[1]+cells[idx].speeds[5]+cells[idx].speeds[8]
                    -cells[idx].speeds[3]-cells[idx].speeds[6]-cells[idx].speeds[7])/ld;
        float u_y = (cells[idx].speeds[2]+cells[idx].speeds[5]+cells[idx].speeds[6]
                    -cells[idx].speeds[4]-cells[idx].speeds[7]-cells[idx].speeds[8])/ld;
        tot_u += sqrtf(u_x*u_x + u_y*u_y); ++tot_cells;
      }
    }
  return tot_u / (float)tot_cells;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles) {
  return av_velocity_cpu(params, cells, obstacles) * params.reynolds_dim
         / (1.f / 6.f * (2.f / params.omega - 1.f));
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels) {
  const float c_sq = 1.f / 3.f;
  FILE* fp = fopen(FINALSTATEFILE, "w");
  if (!fp) die("could not open output file", __LINE__, __FILE__);
  for (int jj = 0; jj < params.ny; jj++)
    for (int ii = 0; ii < params.nx; ii++) {
      int idx = ii + jj * params.nx;
      float u_x = 0.f, u_y = 0.f, u = 0.f, pressure, ld = 0.f;
      if (obstacles[idx]) { pressure = params.density * c_sq; }
      else {
        for (int k = 0; k < NSPEEDS; k++) ld += cells[idx].speeds[k];
        u_x = (cells[idx].speeds[1]+cells[idx].speeds[5]+cells[idx].speeds[8]
              -cells[idx].speeds[3]-cells[idx].speeds[6]-cells[idx].speeds[7])/ld;
        u_y = (cells[idx].speeds[2]+cells[idx].speeds[5]+cells[idx].speeds[6]
              -cells[idx].speeds[4]-cells[idx].speeds[7]-cells[idx].speeds[8])/ld;
        u = sqrtf(u_x*u_x + u_y*u_y); pressure = ld * c_sq;
      }
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[idx]);
    }
  fclose(fp);
  fp = fopen(AVVELSFILE, "w");
  if (!fp) die("could not open av_vels output", __LINE__, __FILE__);
  for (int ii = 0; ii < params.maxIters; ii++)
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  fclose(fp);
  return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile> [maxIters]\n", argv[0]);
    return EXIT_FAILURE;
  }
  int maxItersOverride = (argc == 4) ? atoi(argv[3]) : -1;

  t_param params; t_speed* cells = NULL; int* obstacles = NULL; float* av_vels = NULL;
  initialise(argv[1], argv[2], &params, &cells, &obstacles, &av_vels, maxItersOverride);

  const int Ny = params.ny, Nx = params.nx, MaxIters = params.maxIters;
  const float omega = params.omega;
  const float densityaccel = params.density * params.accel;
  const float w11 = densityaccel / 9.f;
  const float w21 = densityaccel / 36.f;

  Kokkos::initialize(argc, argv);
  {
    using View1D = Kokkos::View<float*>;
    View1D d_s0("s0",Ny*Nx), d_s1("s1",Ny*Nx), d_s2("s2",Ny*Nx), d_s3("s3",Ny*Nx),
           d_s4("s4",Ny*Nx), d_s5("s5",Ny*Nx), d_s6("s6",Ny*Nx), d_s7("s7",Ny*Nx),
           d_s8("s8",Ny*Nx);
    View1D d_t0("t0",Ny*Nx), d_t1("t1",Ny*Nx), d_t2("t2",Ny*Nx), d_t3("t3",Ny*Nx),
           d_t4("t4",Ny*Nx), d_t5("t5",Ny*Nx), d_t6("t6",Ny*Nx), d_t7("t7",Ny*Nx),
           d_t8("t8",Ny*Nx);
    Kokkos::View<int*> d_obs("obs", Ny*Nx);

    {
      auto h_s0=Kokkos::create_mirror_view(d_s0), h_s1=Kokkos::create_mirror_view(d_s1),
           h_s2=Kokkos::create_mirror_view(d_s2), h_s3=Kokkos::create_mirror_view(d_s3),
           h_s4=Kokkos::create_mirror_view(d_s4), h_s5=Kokkos::create_mirror_view(d_s5),
           h_s6=Kokkos::create_mirror_view(d_s6), h_s7=Kokkos::create_mirror_view(d_s7),
           h_s8=Kokkos::create_mirror_view(d_s8);
      auto h_obs = Kokkos::create_mirror_view(d_obs);
      for (int i = 0; i < Ny*Nx; i++) {
        h_s0(i)=cells[i].speeds[0]; h_s1(i)=cells[i].speeds[1];
        h_s2(i)=cells[i].speeds[2]; h_s3(i)=cells[i].speeds[3];
        h_s4(i)=cells[i].speeds[4]; h_s5(i)=cells[i].speeds[5];
        h_s6(i)=cells[i].speeds[6]; h_s7(i)=cells[i].speeds[7];
        h_s8(i)=cells[i].speeds[8]; h_obs(i)=obstacles[i];
      }
      Kokkos::deep_copy(d_s0,h_s0); Kokkos::deep_copy(d_s1,h_s1);
      Kokkos::deep_copy(d_s2,h_s2); Kokkos::deep_copy(d_s3,h_s3);
      Kokkos::deep_copy(d_s4,h_s4); Kokkos::deep_copy(d_s5,h_s5);
      Kokkos::deep_copy(d_s6,h_s6); Kokkos::deep_copy(d_s7,h_s7);
      Kokkos::deep_copy(d_s8,h_s8); Kokkos::deep_copy(d_obs,h_obs);
    }

    // Pre-count non-obstacle cells
    int tot_cells = 0;
    Kokkos::parallel_reduce("count_cells", Ny*Nx,
      KOKKOS_LAMBDA(int gid, int& cnt) { if (!d_obs(gid)) cnt++; },
      Kokkos::Sum<int>(tot_cells));

    auto t_start = std::chrono::steady_clock::now();

    for (int tt = 0; tt < MaxIters; tt++) {
      // Combined propagate + accelerate_flow check + BGK collision
      Kokkos::parallel_for("timestep", Ny*Nx,
        KOKKOS_LAMBDA(int gid) {
          const int ii = gid % Nx;
          const int jj = gid / Nx;
          const int y_n = (jj + 1) % Ny;
          const int x_e = (ii + 1) % Nx;
          const int y_s = (jj == 0) ? (Ny - 1) : (jj - 1);
          const int x_w = (ii == 0) ? (Nx - 1) : (ii - 1);

          // Pull-model streaming with inline accelerate_flow for row jj == Ny-2
          float ts0 = d_s0(ii + jj*Nx);

          // speed1: from west neighbour; accel if row Ny-2 and conditions met
          float ts1;
          {
            int src = x_w + jj*Nx;
            bool af = (jj == Ny-2) && (!d_obs(src))
                      && (d_s3(src) > w11) && (d_s6(src) > w21) && (d_s7(src) > w21);
            ts1 = d_s1(src) + (af ? w11 : 0.f);
          }
          float ts2 = d_s2(ii + y_s*Nx);
          float ts3;
          {
            int src = x_e + jj*Nx;
            bool af = (jj == Ny-2) && (!d_obs(src))
                      && (d_s3(src) > w11) && (d_s6(src) > w21) && (d_s7(src) > w21);
            ts3 = d_s3(src) - (af ? w11 : 0.f);
          }
          float ts4 = d_s4(ii + y_n*Nx);
          float ts5;
          {
            int src = x_w + y_s*Nx;
            bool af = (y_s == Ny-2) && (!d_obs(src))
                      && (d_s3(src) > w11) && (d_s6(src) > w21) && (d_s7(src) > w21);
            ts5 = d_s5(src) + (af ? w21 : 0.f);
          }
          float ts6;
          {
            int src = x_e + y_s*Nx;
            bool af = (y_s == Ny-2) && (!d_obs(src))
                      && (d_s3(src) > w11) && (d_s6(src) > w21) && (d_s7(src) > w21);
            ts6 = d_s6(src) - (af ? w21 : 0.f);
          }
          float ts7;
          {
            int src = x_e + y_n*Nx;
            bool af = (y_n == Ny-2) && (!d_obs(src))
                      && (d_s3(src) > w11) && (d_s6(src) > w21) && (d_s7(src) > w21);
            ts7 = d_s7(src) - (af ? w21 : 0.f);
          }
          float ts8;
          {
            int src = x_w + y_n*Nx;
            bool af = (y_n == Ny-2) && (!d_obs(src))
                      && (d_s3(src) > w11) && (d_s6(src) > w21) && (d_s7(src) > w21);
            ts8 = d_s8(src) + (af ? w21 : 0.f);
          }

          // BGK collision
          const float c_sq_inv = 3.f;
          const float w0f = 4.f/9.f, w1f = 1.f/9.f, w2f = 1.f/36.f;
          const float temp1 = 4.5f;
          float ld = ts0+ts1+ts2+ts3+ts4+ts5+ts6+ts7+ts8;
          float ldr = 1.f / ld;
          float u_x = (ts1+ts5+ts8-ts3-ts6-ts7)*ldr;
          float u_y = (ts2+ts5+ts6-ts4-ts7-ts8)*ldr;
          float temp2 = -(u_x*u_x + u_y*u_y) * 1.5f;  // /(2*c_sq)=*3/2
          float de0 = w0f*ld*(1.f+temp2);
          float de1 = w1f*ld*(1.f+u_x*c_sq_inv+u_x*u_x*temp1+temp2);
          float de2 = w1f*ld*(1.f+u_y*c_sq_inv+u_y*u_y*temp1+temp2);
          float de3 = w1f*ld*(1.f-u_x*c_sq_inv+u_x*u_x*temp1+temp2);
          float de4 = w1f*ld*(1.f-u_y*c_sq_inv+u_y*u_y*temp1+temp2);
          float s5=u_x+u_y, s6=-u_x+u_y, s7=-u_x-u_y, s8=u_x-u_y;
          float de5 = w2f*ld*(1.f+s5*c_sq_inv+s5*s5*temp1+temp2);
          float de6 = w2f*ld*(1.f+s6*c_sq_inv+s6*s6*temp1+temp2);
          float de7 = w2f*ld*(1.f+s7*c_sq_inv+s7*s7*temp1+temp2);
          float de8 = w2f*ld*(1.f+s8*c_sq_inv+s8*s8*temp1+temp2);

          int obs = d_obs(gid);
          d_t0(gid) = obs ? ts0 : (ts0 + omega*(de0 - ts0));
          float tmp1 = ts1;
          d_t1(gid) = obs ? ts3 : (ts1 + omega*(de1 - ts1));
          d_t3(gid) = obs ? tmp1 : (ts3 + omega*(de3 - ts3));
          float tmp2 = ts2;
          d_t2(gid) = obs ? ts4 : (ts2 + omega*(de2 - ts2));
          d_t4(gid) = obs ? tmp2 : (ts4 + omega*(de4 - ts4));
          float tmp5 = ts5;
          d_t5(gid) = obs ? ts7 : (ts5 + omega*(de5 - ts5));
          d_t7(gid) = obs ? tmp5 : (ts7 + omega*(de7 - ts7));
          float tmp6 = ts6;
          d_t6(gid) = obs ? ts8 : (ts6 + omega*(de6 - ts6));
          d_t8(gid) = obs ? tmp6 : (ts8 + omega*(de8 - ts8));
        });
      Kokkos::fence();

      // av_velocity from post-collision/propagation values (d_t*)
      float tot_u = 0.f;
      Kokkos::parallel_reduce("av_vel", Ny*Nx,
        KOKKOS_LAMBDA(int gid, float& usum) {
          if (!d_obs(gid)) {
            float ld = d_t0(gid)+d_t1(gid)+d_t2(gid)+d_t3(gid)+d_t4(gid)
                      +d_t5(gid)+d_t6(gid)+d_t7(gid)+d_t8(gid);
            float ldr = 1.f / ld;
            float u_x = (d_t1(gid)+d_t5(gid)+d_t8(gid)-d_t3(gid)-d_t6(gid)-d_t7(gid))*ldr;
            float u_y = (d_t2(gid)+d_t5(gid)+d_t6(gid)-d_t4(gid)-d_t7(gid)-d_t8(gid))*ldr;
            usum += Kokkos::sqrt(u_x*u_x + u_y*u_y);
          }
        }, Kokkos::Sum<float>(tot_u));
      av_vels[tt] = tot_u / (float)tot_cells;

      // Swap: d_s* <- d_t*
      auto swap = [](View1D& a, View1D& b){ View1D tmp = a; a = b; b = tmp; };
      swap(d_s0,d_t0); swap(d_s1,d_t1); swap(d_s2,d_t2); swap(d_s3,d_t3);
      swap(d_s4,d_t4); swap(d_s5,d_t5); swap(d_s6,d_t6); swap(d_s7,d_t7); swap(d_s8,d_t8);
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    printf("Average kernel execution time: %.6f (us)\n", elapsed / MaxIters);

    // Copy back to host
    auto h_s0=Kokkos::create_mirror_view(d_s0), h_s1=Kokkos::create_mirror_view(d_s1),
         h_s2=Kokkos::create_mirror_view(d_s2), h_s3=Kokkos::create_mirror_view(d_s3),
         h_s4=Kokkos::create_mirror_view(d_s4), h_s5=Kokkos::create_mirror_view(d_s5),
         h_s6=Kokkos::create_mirror_view(d_s6), h_s7=Kokkos::create_mirror_view(d_s7),
         h_s8=Kokkos::create_mirror_view(d_s8);
    Kokkos::deep_copy(h_s0,d_s0); Kokkos::deep_copy(h_s1,d_s1);
    Kokkos::deep_copy(h_s2,d_s2); Kokkos::deep_copy(h_s3,d_s3);
    Kokkos::deep_copy(h_s4,d_s4); Kokkos::deep_copy(h_s5,d_s5);
    Kokkos::deep_copy(h_s6,d_s6); Kokkos::deep_copy(h_s7,d_s7);
    Kokkos::deep_copy(h_s8,d_s8);
    for (int i = 0; i < Ny*Nx; i++) {
      cells[i].speeds[0]=h_s0(i); cells[i].speeds[1]=h_s1(i);
      cells[i].speeds[2]=h_s2(i); cells[i].speeds[3]=h_s3(i);
      cells[i].speeds[4]=h_s4(i); cells[i].speeds[5]=h_s5(i);
      cells[i].speeds[6]=h_s6(i); cells[i].speeds[7]=h_s7(i);
      cells[i].speeds[8]=h_s8(i);
    }
  }
  Kokkos::finalize();

  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  write_values(params, cells, obstacles, av_vels);
  free(cells); free(obstacles); free(av_vels);
  return EXIT_SUCCESS;
}
