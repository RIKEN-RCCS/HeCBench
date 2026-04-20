// miniWeather-kokkos/main.cpp
// Port of miniWeather-omp to Kokkos.
// MPI is retained; Kokkos replaces all OMP target regions.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <Kokkos_Core.hpp>

// ──────────────────────────────────────────────────────────────────────────────
// Physical / domain constants
// ──────────────────────────────────────────────────────────────────────────────
const double pi        = 3.14159265358979323846264338327;
const double grav      = 9.8;
const double cp        = 1004.;
const double cv        = 717.;
const double rd        = 287.;
const double p0        = 1.e5;
const double C0        = 27.5629410929725921310572974482;
const double gamm      = 1.40027894002789400278940027894;
const double xlen      = 2.e4;
const double zlen      = 1.e4;
const double hv_beta   = 0.25;
const double cfl       = 1.50;
const double max_speed = 450;
const int hs           = 2;
const int sten_size    = 4;
const int NUM_VARS     = 4;
const int ID_DENS      = 0;
const int ID_UMOM      = 1;
const int ID_WMOM      = 2;
const int ID_RHOT      = 3;
const int DIR_X        = 1;
const int DIR_Z        = 2;
const int DATA_SPEC_COLLISION       = 1;
const int DATA_SPEC_THERMAL         = 2;
const int DATA_SPEC_MOUNTAIN        = 3;
const int DATA_SPEC_TURBULENCE      = 4;
const int DATA_SPEC_DENSITY_CURRENT = 5;
const int DATA_SPEC_INJECTION       = 6;

const int nqpoints = 3;
double qpoints [] = {0.112701665379258311482073460022E0,0.500000000000000000000000000000E0,0.887298334620741688517926539980E0};
double qweights[] = {0.277777777777777777777777777779E0,0.444444444444444444444444444444E0,0.277777777777777777777777777779E0};

// ──────────────────────────────────────────────────────────────────────────────
// Global state
// ──────────────────────────────────────────────────────────────────────────────
double sim_time, dt;
int    nx, nz, nx_glob, nz_glob, i_beg, k_beg;
double dx, dz;
int    nranks, myrank, left_rank, right_rank, masterproc;
double data_spec_int;

// Host-side arrays (used for init / MPI)
double *h_state, *h_state_tmp, *h_flux, *h_tend;
double *h_hy_dens_cell, *h_hy_dens_theta_cell;
double *h_hy_dens_int, *h_hy_dens_theta_int, *h_hy_pressure_int;
double *h_sendbuf_l, *h_sendbuf_r, *h_recvbuf_l, *h_recvbuf_r;

// Device-side Kokkos views
Kokkos::View<double*> d_state, d_state_tmp, d_flux, d_tend;
Kokkos::View<double*> d_hy_dens_cell, d_hy_dens_theta_cell;
Kokkos::View<double*> d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int;
Kokkos::View<double*> d_sendbuf_l, d_sendbuf_r, d_recvbuf_l, d_recvbuf_r;

int num_out = 0, direction_switch = 1;
double etime = 0, output_counter = 0;
double mass0, te0, mass, te;

static double dmin(double a, double b) { return a < b ? a : b; }

// ──────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ──────────────────────────────────────────────────────────────────────────────
void init(int *argc, char ***argv);
void finalize();
void injection      (double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht);
void density_current(double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht);
void turbulence     (double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht);
void mountain_waves (double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht);
void thermal        (double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht);
void collision      (double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht);
void hydro_const_theta (double z,double &r,double &t);
void hydro_const_bvfreq(double z,double bv_freq0,double &r,double &t);
double sample_ellipse_cosine(double x,double z,double amp,double x0,double z0,double xrad,double zrad);
void perform_timestep(double dt);
void semi_discrete_step(Kokkos::View<double*> state_init,
                        Kokkos::View<double*> state_forcing,
                        Kokkos::View<double*> state_out,
                        double dt, int dir);
void compute_tendencies_x(Kokkos::View<double*> d_st,
                          Kokkos::View<double*> d_fl,
                          Kokkos::View<double*> d_tn);
void compute_tendencies_z(Kokkos::View<double*> d_st,
                          Kokkos::View<double*> d_fl,
                          Kokkos::View<double*> d_tn);
void set_halo_values_x(Kokkos::View<double*> d_st);
void set_halo_values_z(Kokkos::View<double*> d_st);
void reductions(double &mass, double &te);

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  nx_glob      = NX;
  nz_glob      = NZ;
  sim_time     = SIM_TIME;
  data_spec_int = DATA_SPEC;

  Kokkos::initialize(argc, argv);
  {
    init(&argc, &argv);

    // Upload all initial data to device
    auto upload = [](Kokkos::View<double*>& dv, double* hptr, size_t n, std::string nm) {
      dv = Kokkos::View<double*>(nm, n);
      auto hm = Kokkos::create_mirror_view(dv);
      memcpy(hm.data(), hptr, n * sizeof(double));
      Kokkos::deep_copy(dv, hm);
    };
    upload(d_state,              h_state,              (size_t)(nx+2*hs)*(nz+2*hs)*NUM_VARS, "state");
    upload(d_state_tmp,          h_state_tmp,          (size_t)(nx+2*hs)*(nz+2*hs)*NUM_VARS, "state_tmp");
    upload(d_hy_dens_cell,       h_hy_dens_cell,       (size_t)(nz+2*hs),                    "hy_dens_cell");
    upload(d_hy_dens_theta_cell, h_hy_dens_theta_cell, (size_t)(nz+2*hs),                    "hy_dens_theta_cell");
    upload(d_hy_dens_int,        h_hy_dens_int,        (size_t)(nz+1),                       "hy_dens_int");
    upload(d_hy_dens_theta_int,  h_hy_dens_theta_int,  (size_t)(nz+1),                       "hy_dens_theta_int");
    upload(d_hy_pressure_int,    h_hy_pressure_int,    (size_t)(nz+1),                       "hy_pressure_int");
    d_flux      = Kokkos::View<double*>("flux",      (size_t)(nx+1)*(nz+1)*NUM_VARS);
    d_tend      = Kokkos::View<double*>("tend",      (size_t)nx*nz*NUM_VARS);
    d_sendbuf_l = Kokkos::View<double*>("sendbuf_l", (size_t)hs*nz*NUM_VARS);
    d_sendbuf_r = Kokkos::View<double*>("sendbuf_r", (size_t)hs*nz*NUM_VARS);
    d_recvbuf_l = Kokkos::View<double*>("recvbuf_l", (size_t)hs*nz*NUM_VARS);
    d_recvbuf_r = Kokkos::View<double*>("recvbuf_r", (size_t)hs*nz*NUM_VARS);

    reductions(mass0, te0);

    auto c_start = clock();
    while (etime < sim_time) {
      if (etime + dt > sim_time) dt = sim_time - etime;
      perform_timestep(dt);
      etime += dt;
    }
    auto c_end = clock();
    if (masterproc)
      printf("CPU Time: %lf sec\n", (double)(c_end - c_start) / CLOCKS_PER_SEC);

    reductions(mass, te);
  }
  Kokkos::finalize();

  printf("d_mass: %le\n", (mass - mass0) / mass0);
  printf("d_te:   %le\n", (te   - te0  ) / te0  );

  finalize();
  return 0;
}

// ──────────────────────────────────────────────────────────────────────────────
void perform_timestep(double dt) {
  if (direction_switch) {
    semi_discrete_step(d_state, d_state,     d_state_tmp, dt/3, DIR_X);
    semi_discrete_step(d_state, d_state_tmp, d_state_tmp, dt/2, DIR_X);
    semi_discrete_step(d_state, d_state_tmp, d_state,     dt/1, DIR_X);
    semi_discrete_step(d_state, d_state,     d_state_tmp, dt/3, DIR_Z);
    semi_discrete_step(d_state, d_state_tmp, d_state_tmp, dt/2, DIR_Z);
    semi_discrete_step(d_state, d_state_tmp, d_state,     dt/1, DIR_Z);
  } else {
    semi_discrete_step(d_state, d_state,     d_state_tmp, dt/3, DIR_Z);
    semi_discrete_step(d_state, d_state_tmp, d_state_tmp, dt/2, DIR_Z);
    semi_discrete_step(d_state, d_state_tmp, d_state,     dt/1, DIR_Z);
    semi_discrete_step(d_state, d_state,     d_state_tmp, dt/3, DIR_X);
    semi_discrete_step(d_state, d_state_tmp, d_state_tmp, dt/2, DIR_X);
    semi_discrete_step(d_state, d_state_tmp, d_state,     dt/1, DIR_X);
  }
  direction_switch = direction_switch ? 0 : 1;
}

// ──────────────────────────────────────────────────────────────────────────────
void semi_discrete_step(Kokkos::View<double*> state_init,
                        Kokkos::View<double*> state_forcing,
                        Kokkos::View<double*> state_out,
                        double dt, int dir)
{
  const int lnx = nx, lnz = nz, l_hs = hs, l_NUM_VARS = NUM_VARS;

  if (dir == DIR_X) {
    set_halo_values_x(state_forcing);
    compute_tendencies_x(state_forcing, d_flux, d_tend);
  } else {
    set_halo_values_z(state_forcing);
    compute_tendencies_z(state_forcing, d_flux, d_tend);
  }

  Kokkos::parallel_for("apply_tend",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{l_NUM_VARS,lnz,lnx}),
    KOKKOS_LAMBDA(int ll, int k, int i) {
      int inds = ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i+l_hs;
      int indt = ll*lnz*lnx + k*lnx + i;
      state_out[inds] = state_init[inds] + dt * d_tend[indt];
    });
  Kokkos::fence();
}

// ──────────────────────────────────────────────────────────────────────────────
void compute_tendencies_x(Kokkos::View<double*> d_st,
                          Kokkos::View<double*> d_fl,
                          Kokkos::View<double*> d_tn)
{
  const int lnx = nx, lnz = nz, l_hs = hs, l_NUM_VARS = NUM_VARS;
  const double l_dx = dx, l_dt = dt;
  const double l_hv_beta = hv_beta;
  const double hv_coef   = -l_hv_beta * l_dx / (16*l_dt);
  const double l_C0 = C0, l_gamm = gamm;

  Kokkos::parallel_for("flux_x",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{lnz,lnx+1}),
    KOKKOS_LAMBDA(int k, int i) {
      double stencil[4], vals[4], d3_vals[4];
      for (int ll = 0; ll < l_NUM_VARS; ll++) {
        for (int s = 0; s < 4; s++) {
          int inds = ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i+s;
          stencil[s] = d_st[inds];
        }
        vals[ll]    = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        d3_vals[ll] = -stencil[0]    + 3*stencil[1]    - 3*stencil[2]    + stencil[3];
      }
      double r = vals[0] + d_hy_dens_cell[k+l_hs];
      double u = vals[1] / r;
      double w = vals[2] / r;
      double t = (vals[3] + d_hy_dens_theta_cell[k+l_hs]) / r;
      double p = l_C0 * pow(r*t, l_gamm);
      d_fl[0*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*u     - hv_coef*d3_vals[0];
      d_fl[1*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*u*u+p - hv_coef*d3_vals[1];
      d_fl[2*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*u*w   - hv_coef*d3_vals[2];
      d_fl[3*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*u*t   - hv_coef*d3_vals[3];
    });
  Kokkos::fence();

  Kokkos::parallel_for("tend_x",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{l_NUM_VARS,lnz,lnx}),
    KOKKOS_LAMBDA(int ll, int k, int i) {
      int indt  = ll*lnz*lnx + k*lnx + i;
      int indf1 = ll*(lnz+1)*(lnx+1) + k*(lnx+1) + i;
      int indf2 = ll*(lnz+1)*(lnx+1) + k*(lnx+1) + i+1;
      d_tn[indt] = -(d_fl[indf2] - d_fl[indf1]) / l_dx;
    });
  Kokkos::fence();
}

// ──────────────────────────────────────────────────────────────────────────────
void compute_tendencies_z(Kokkos::View<double*> d_st,
                          Kokkos::View<double*> d_fl,
                          Kokkos::View<double*> d_tn)
{
  const int lnx = nx, lnz = nz, l_hs = hs, l_NUM_VARS = NUM_VARS;
  const double l_dz = dz, l_dt = dt;
  const double hv_coef = -hv_beta * l_dz / (16*l_dt);
  const double l_C0 = C0, l_gamm = gamm, l_grav = grav;

  Kokkos::parallel_for("flux_z",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{lnz+1,lnx}),
    KOKKOS_LAMBDA(int k, int i) {
      double stencil[4], vals[4], d3_vals[4];
      for (int ll = 0; ll < l_NUM_VARS; ll++) {
        for (int s = 0; s < 4; s++) {
          int inds = ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+s)*(lnx+2*l_hs) + i+l_hs;
          stencil[s] = d_st[inds];
        }
        vals[ll]    = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        d3_vals[ll] = -stencil[0]    + 3*stencil[1]    - 3*stencil[2]    + stencil[3];
      }
      double r  = vals[0] + d_hy_dens_int[k];
      double u  = vals[1] / r;
      double w  = vals[2] / r;
      double t  = (vals[3] + d_hy_dens_theta_int[k]) / r;
      double p  = l_C0 * pow(r*t, l_gamm) - d_hy_pressure_int[k];
      d_fl[0*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*w     - hv_coef*d3_vals[0];
      d_fl[1*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*w*u   - hv_coef*d3_vals[1];
      d_fl[2*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*w*w+p - hv_coef*d3_vals[2];
      d_fl[3*(lnz+1)*(lnx+1)+k*(lnx+1)+i] = r*w*t   - hv_coef*d3_vals[3];
    });
  Kokkos::fence();

  Kokkos::parallel_for("tend_z",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{l_NUM_VARS,lnz,lnx}),
    KOKKOS_LAMBDA(int ll, int k, int i) {
      int indt  = ll*lnz*lnx + k*lnx + i;
      int indf1 = ll*(lnz+1)*(lnx+1) + k*(lnx+1) + i;
      int indf2 = ll*(lnz+1)*(lnx+1) + (k+1)*(lnx+1) + i;
      d_tn[indt] = -(d_fl[indf2] - d_fl[indf1]) / l_dz;
      if (ll == 2) {  // ID_WMOM
        int inds = 0*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i+l_hs; // ID_DENS
        d_tn[indt] -= d_st[inds] * l_grav;
      }
    });
  Kokkos::fence();
}

// ──────────────────────────────────────────────────────────────────────────────
void set_halo_values_x(Kokkos::View<double*> d_st)
{
  const int lnx = nx, lnz = nz, l_hs = hs, l_NUM_VARS = NUM_VARS;
  int ierr;
  MPI_Request req_r[2], req_s[2];

  ierr = MPI_Irecv(h_recvbuf_l, hs*nz*NUM_VARS, MPI_DOUBLE, left_rank,  0, MPI_COMM_WORLD, &req_r[0]);
  ierr = MPI_Irecv(h_recvbuf_r, hs*nz*NUM_VARS, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, &req_r[1]);

  // Pack send buffers on device
  Kokkos::parallel_for("pack_halo_x",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{l_NUM_VARS,lnz,l_hs}),
    KOKKOS_LAMBDA(int ll, int k, int s) {
      d_sendbuf_l[ll*lnz*l_hs + k*l_hs + s] = d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + l_hs+s];
      d_sendbuf_r[ll*lnz*l_hs + k*l_hs + s] = d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + lnx+s];
    });
  Kokkos::fence();

  // Copy send buffers to host
  {
    auto h_sl = Kokkos::create_mirror_view(d_sendbuf_l);
    auto h_sr = Kokkos::create_mirror_view(d_sendbuf_r);
    Kokkos::deep_copy(h_sl, d_sendbuf_l);
    Kokkos::deep_copy(h_sr, d_sendbuf_r);
    memcpy(h_sendbuf_l, h_sl.data(), hs*nz*NUM_VARS*sizeof(double));
    memcpy(h_sendbuf_r, h_sr.data(), hs*nz*NUM_VARS*sizeof(double));
  }

  ierr = MPI_Isend(h_sendbuf_l, hs*nz*NUM_VARS, MPI_DOUBLE, left_rank,  1, MPI_COMM_WORLD, &req_s[0]);
  ierr = MPI_Isend(h_sendbuf_r, hs*nz*NUM_VARS, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &req_s[1]);
  ierr = MPI_Waitall(2, req_r, MPI_STATUSES_IGNORE);

  // Upload recv buffers to device
  {
    auto h_rl = Kokkos::create_mirror_view(d_recvbuf_l);
    auto h_rr = Kokkos::create_mirror_view(d_recvbuf_r);
    memcpy(h_rl.data(), h_recvbuf_l, hs*nz*NUM_VARS*sizeof(double));
    memcpy(h_rr.data(), h_recvbuf_r, hs*nz*NUM_VARS*sizeof(double));
    Kokkos::deep_copy(d_recvbuf_l, h_rl);
    Kokkos::deep_copy(d_recvbuf_r, h_rr);
  }

  // Unpack recv buffers
  Kokkos::parallel_for("unpack_halo_x",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{l_NUM_VARS,lnz,l_hs}),
    KOKKOS_LAMBDA(int ll, int k, int s) {
      d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + s         ] = d_recvbuf_l[ll*lnz*l_hs + k*l_hs + s];
      d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + lnx+l_hs+s] = d_recvbuf_r[ll*lnz*l_hs + k*l_hs + s];
    });
  Kokkos::fence();

  ierr = MPI_Waitall(2, req_s, MPI_STATUSES_IGNORE);

  if ((int)data_spec_int == DATA_SPEC_INJECTION && myrank == 0) {
    const double l_dz = dz, l_k_beg = k_beg, l_zlen = zlen;
    Kokkos::parallel_for("injection_halo",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{lnz,l_hs}),
      KOKKOS_LAMBDA(int k, int i) {
        double z = (l_k_beg + k + 0.5) * l_dz;
        if (fabs(z - 3*l_zlen/4) <= l_zlen/16) {
          int ind_r = 0*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i;
          int ind_u = 1*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i;
          int ind_t = 3*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i;
          d_st[ind_u] = (d_st[ind_r] + d_hy_dens_cell[k+l_hs]) * 50.;
          d_st[ind_t] = (d_st[ind_r] + d_hy_dens_cell[k+l_hs]) * 298. - d_hy_dens_theta_cell[k+l_hs];
        }
      });
    Kokkos::fence();
  }
  (void)ierr;
}

// ──────────────────────────────────────────────────────────────────────────────
void set_halo_values_z(Kokkos::View<double*> d_st)
{
  const int lnx = nx, lnz = nz, l_hs = hs, l_NUM_VARS = NUM_VARS;
  const double l_xlen = xlen, l_dx = dx, l_i_beg = i_beg;
  const double l_pi   = pi;
  const int l_data_spec = (int)data_spec_int;

  Kokkos::parallel_for("halo_z",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{l_NUM_VARS,lnx+2*l_hs}),
    KOKKOS_LAMBDA(int ll, int i) {
      if (ll == 2) { // ID_WMOM
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (0      )*(lnx+2*l_hs) + i] = 0.;
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (1      )*(lnx+2*l_hs) + i] = 0.;
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (lnz+l_hs  )*(lnx+2*l_hs) + i] = 0.;
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (lnz+l_hs+1)*(lnx+2*l_hs) + i] = 0.;
        if (l_data_spec == DATA_SPEC_MOUNTAIN) {
          const double mnt_width = l_xlen / 8;
          double x = (l_i_beg + i - l_hs + 0.5) * l_dx;
          if (fabs(x - l_xlen/4) < mnt_width) {
            double xloc = (x - l_xlen/4) / mnt_width;
            double mnt_deriv = -l_pi * cos(l_pi*xloc/2) * sin(l_pi*xloc/2) * 10 / l_dx;
            d_st[2*(lnz+2*l_hs)*(lnx+2*l_hs) + (0)*(lnx+2*l_hs) + i]
              = mnt_deriv * d_st[1*(lnz+2*l_hs)*(lnx+2*l_hs) + l_hs*(lnx+2*l_hs) + i];
            d_st[2*(lnz+2*l_hs)*(lnx+2*l_hs) + (1)*(lnx+2*l_hs) + i]
              = mnt_deriv * d_st[1*(lnz+2*l_hs)*(lnx+2*l_hs) + l_hs*(lnx+2*l_hs) + i];
          }
        }
      } else {
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (0      )*(lnx+2*l_hs) + i]
          = d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (l_hs     )*(lnx+2*l_hs) + i];
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (1      )*(lnx+2*l_hs) + i]
          = d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (l_hs     )*(lnx+2*l_hs) + i];
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (lnz+l_hs  )*(lnx+2*l_hs) + i]
          = d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (lnz+l_hs-1)*(lnx+2*l_hs) + i];
        d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (lnz+l_hs+1)*(lnx+2*l_hs) + i]
          = d_st[ll*(lnz+2*l_hs)*(lnx+2*l_hs) + (lnz+l_hs-1)*(lnx+2*l_hs) + i];
      }
    });
  Kokkos::fence();
}

// ──────────────────────────────────────────────────────────────────────────────
void reductions(double &mass_out, double &te_out)
{
  const int lnx = nx, lnz = nz, l_hs = hs, l_NUM_VARS = NUM_VARS;
  const double l_dx = dx, l_dz = dz;
  const double l_C0 = C0, l_gamm = gamm, l_grav = grav;
  const double l_cp = cp, l_cv = cv, l_rd = rd, l_p0 = p0;

  // Upload current state to device (if needed); state is already on device
  double lmass = 0.0, lte = 0.0;
  Kokkos::parallel_reduce("reductions",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{lnz,lnx}),
    KOKKOS_LAMBDA(int k, int i, double &lm, double &lt) {
      int ind_r = 0*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i+l_hs;
      int ind_u = 1*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i+l_hs;
      int ind_w = 2*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i+l_hs;
      int ind_t = 3*(lnz+2*l_hs)*(lnx+2*l_hs) + (k+l_hs)*(lnx+2*l_hs) + i+l_hs;
      double r  = d_state[ind_r] + d_hy_dens_cell[l_hs+k];
      double u  = d_state[ind_u] / r;
      double w  = d_state[ind_w] / r;
      double th = (d_state[ind_t] + d_hy_dens_theta_cell[l_hs+k]) / r;
      double p  = l_C0 * pow(r*th, l_gamm);
      double t  = th / pow(l_p0/p, l_rd/l_cp);
      double ke = r * (u*u + w*w);
      double ie = r * l_cv * t;
      lm += r        * l_dx * l_dz;
      lt += (ke + ie)* l_dx * l_dz;
    },
    Kokkos::Sum<double>(lmass),
    Kokkos::Sum<double>(lte));

  double glob[2] = {0,0}, loc[2] = {lmass, lte};
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  mass_out = glob[0];
  te_out   = glob[1];
}

// ──────────────────────────────────────────────────────────────────────────────
// Physics helper functions (host-only, used in init)
// ──────────────────────────────────────────────────────────────────────────────
void hydro_const_theta(double z, double &r, double &t) {
  const double theta0 = 300., exner0 = 1.;
  t = theta0;
  double exner = exner0 - grav*z/(cp*theta0);
  double p  = p0 * pow(exner, cp/rd);
  double rt = pow(p/C0, 1./gamm);
  r = rt / t;
}
void hydro_const_bvfreq(double z, double bv_freq0, double &r, double &t) {
  const double theta0 = 300., exner0 = 1.;
  t = theta0 * exp(bv_freq0*bv_freq0/grav*z);
  double exner = exner0 - grav*grav/(cp*bv_freq0*bv_freq0)*(t-theta0)/(t*theta0);
  double p  = p0 * pow(exner, cp/rd);
  double rt = pow(p/C0, 1./gamm);
  r = rt / t;
}
double sample_ellipse_cosine(double x,double z,double amp,double x0,double z0,double xrad,double zrad) {
  double dist = sqrt(((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad)) * pi / 2.;
  return (dist <= pi/2.) ? amp * pow(cos(dist),2.) : 0.;
}
void injection(double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht)
  { hydro_const_theta(z,hr,ht); r=u=w=t=0.; }
void density_current(double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht)
  { hydro_const_theta(z,hr,ht); r=u=w=t=0.; t+=sample_ellipse_cosine(x,z,-20.,xlen/2,5000.,4000.,2000.); }
void turbulence(double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht)
  { hydro_const_theta(z,hr,ht); r=u=w=t=0.; }
void mountain_waves(double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht)
  { hydro_const_bvfreq(z,0.02,hr,ht); r=w=t=0.; u=15.; }
void thermal(double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht)
  { hydro_const_theta(z,hr,ht); r=u=w=t=0.; t+=sample_ellipse_cosine(x,z,3.,xlen/2,2000.,2000.,2000.); }
void collision(double x,double z,double &r,double &u,double &w,double &t,double &hr,double &ht) {
  hydro_const_theta(z,hr,ht); r=u=w=t=0.;
  t+=sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t+=sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}

// ──────────────────────────────────────────────────────────────────────────────
void init(int *argc, char ***argv) {
  int i, k, ii, kk, ll, ierr, inds, i_end;
  double x, z, r, u, w, t, hr, ht, nper;

  ierr = MPI_Init(argc, argv);
  dx = xlen / nx_glob;
  dz = zlen / nz_glob;
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  nper  = (double)nx_glob / nranks;
  i_beg = (int)round(nper * myrank);
  i_end = (int)round(nper * (myrank+1)) - 1;
  nx    = i_end - i_beg + 1;
  left_rank  = (myrank - 1 + nranks) % nranks;
  right_rank = (myrank + 1) % nranks;
  k_beg      = 0;
  nz         = nz_glob;
  masterproc = (myrank == 0);

  h_state              = (double*)malloc((size_t)(nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double));
  h_state_tmp          = (double*)malloc((size_t)(nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double));
  h_flux               = (double*)malloc((size_t)(nx+1)*(nz+1)*NUM_VARS*sizeof(double));
  h_tend               = (double*)malloc((size_t)nx*nz*NUM_VARS*sizeof(double));
  h_hy_dens_cell       = (double*)malloc((size_t)(nz+2*hs)*sizeof(double));
  h_hy_dens_theta_cell = (double*)malloc((size_t)(nz+2*hs)*sizeof(double));
  h_hy_dens_int        = (double*)malloc((size_t)(nz+1)*sizeof(double));
  h_hy_dens_theta_int  = (double*)malloc((size_t)(nz+1)*sizeof(double));
  h_hy_pressure_int    = (double*)malloc((size_t)(nz+1)*sizeof(double));
  h_sendbuf_l          = (double*)malloc((size_t)hs*nz*NUM_VARS*sizeof(double));
  h_sendbuf_r          = (double*)malloc((size_t)hs*nz*NUM_VARS*sizeof(double));
  h_recvbuf_l          = (double*)malloc((size_t)hs*nz*NUM_VARS*sizeof(double));
  h_recvbuf_r          = (double*)malloc((size_t)hs*nz*NUM_VARS*sizeof(double));

  dt = dmin(dx,dz) / max_speed * cfl;
  etime = 0.; output_counter = 0.;

  if (masterproc) {
    printf("nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf("dx,dz: %lf %lf\n", dx, dz);
    printf("dt: %lf\n", dt);
  }
  ierr = MPI_Barrier(MPI_COMM_WORLD);

  // Initialize fluid state
  for (k = 0; k < nz+2*hs; k++) {
    for (i = 0; i < nx+2*hs; i++) {
      for (ll = 0; ll < NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        h_state[inds] = 0.;
      }
      for (kk = 0; kk < nqpoints; kk++) {
        for (ii = 0; ii < nqpoints; ii++) {
          x = (i_beg + i - hs + 0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k - hs + 0.5)*dz + (qpoints[kk]-0.5)*dz;
          if ((int)data_spec_int == DATA_SPEC_COLLISION      ) collision      (x,z,r,u,w,t,hr,ht);
          if ((int)data_spec_int == DATA_SPEC_THERMAL        ) thermal        (x,z,r,u,w,t,hr,ht);
          if ((int)data_spec_int == DATA_SPEC_MOUNTAIN       ) mountain_waves (x,z,r,u,w,t,hr,ht);
          if ((int)data_spec_int == DATA_SPEC_TURBULENCE     ) turbulence     (x,z,r,u,w,t,hr,ht);
          if ((int)data_spec_int == DATA_SPEC_DENSITY_CURRENT) density_current(x,z,r,u,w,t,hr,ht);
          if ((int)data_spec_int == DATA_SPEC_INJECTION      ) injection      (x,z,r,u,w,t,hr,ht);
          h_state[0*(nz+2*hs)*(nx+2*hs)+k*(nx+2*hs)+i] += r                        * qweights[ii]*qweights[kk];
          h_state[1*(nz+2*hs)*(nx+2*hs)+k*(nx+2*hs)+i] += (r+hr)*u                 * qweights[ii]*qweights[kk];
          h_state[2*(nz+2*hs)*(nx+2*hs)+k*(nx+2*hs)+i] += (r+hr)*w                 * qweights[ii]*qweights[kk];
          h_state[3*(nz+2*hs)*(nx+2*hs)+k*(nx+2*hs)+i] += ((r+hr)*(t+ht) - hr*ht) * qweights[ii]*qweights[kk];
        }
      }
      for (ll = 0; ll < NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        h_state_tmp[inds] = h_state[inds];
      }
    }
  }

  for (k = 0; k < nz+2*hs; k++) {
    h_hy_dens_cell[k] = 0.; h_hy_dens_theta_cell[k] = 0.;
    for (kk = 0; kk < nqpoints; kk++) {
      z = (k_beg + k - hs + 0.5)*dz;
      if ((int)data_spec_int == DATA_SPEC_COLLISION      ) collision      (0.,z,r,u,w,t,hr,ht);
      if ((int)data_spec_int == DATA_SPEC_THERMAL        ) thermal        (0.,z,r,u,w,t,hr,ht);
      if ((int)data_spec_int == DATA_SPEC_MOUNTAIN       ) mountain_waves (0.,z,r,u,w,t,hr,ht);
      if ((int)data_spec_int == DATA_SPEC_TURBULENCE     ) turbulence     (0.,z,r,u,w,t,hr,ht);
      if ((int)data_spec_int == DATA_SPEC_DENSITY_CURRENT) density_current(0.,z,r,u,w,t,hr,ht);
      if ((int)data_spec_int == DATA_SPEC_INJECTION      ) injection      (0.,z,r,u,w,t,hr,ht);
      h_hy_dens_cell[k]       += hr    * qweights[kk];
      h_hy_dens_theta_cell[k] += hr*ht * qweights[kk];
    }
  }

  for (k = 0; k < nz+1; k++) {
    z = (k_beg + k)*dz;
    if ((int)data_spec_int == DATA_SPEC_COLLISION      ) collision      (0.,z,r,u,w,t,hr,ht);
    if ((int)data_spec_int == DATA_SPEC_THERMAL        ) thermal        (0.,z,r,u,w,t,hr,ht);
    if ((int)data_spec_int == DATA_SPEC_MOUNTAIN       ) mountain_waves (0.,z,r,u,w,t,hr,ht);
    if ((int)data_spec_int == DATA_SPEC_TURBULENCE     ) turbulence     (0.,z,r,u,w,t,hr,ht);
    if ((int)data_spec_int == DATA_SPEC_DENSITY_CURRENT) density_current(0.,z,r,u,w,t,hr,ht);
    if ((int)data_spec_int == DATA_SPEC_INJECTION      ) injection      (0.,z,r,u,w,t,hr,ht);
    h_hy_dens_int[k]       = hr;
    h_hy_dens_theta_int[k] = hr*ht;
    h_hy_pressure_int[k]   = C0 * pow(hr*ht, gamm);
  }
  (void)ierr;
}

void finalize() {
  free(h_state); free(h_state_tmp); free(h_flux); free(h_tend);
  free(h_hy_dens_cell); free(h_hy_dens_theta_cell);
  free(h_hy_dens_int);  free(h_hy_dens_theta_int); free(h_hy_pressure_int);
  free(h_sendbuf_l); free(h_sendbuf_r); free(h_recvbuf_l); free(h_recvbuf_r);
  MPI_Finalize();
}
