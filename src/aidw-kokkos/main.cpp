/*
 * GPU-accelerated AIDW interpolation algorithm
 *
 * Implemented with Kokkos
 *
 * By Dr.Gang Mei
 *
 * Created on 2015.11.06, China University of Geosciences,
 *                        gang.mei@cugb.edu.cn
 * Revised on 2015.12.14, China University of Geosciences,
 *                        gang.mei@cugb.edu.cn
 *
 * Related publications:
 *  1) "Evaluating the Power of GPU Acceleration for IDW Interpolation Algorithm"
 *     http://www.hindawi.com/journals/tswj/2014/171574/
 *  2) "Accelerating Adaptive IDW Interpolation Algorithm on a Single GPU"
 *     http://arxiv.org/abs/1511.02186
 *
 * License: http://creativecommons.org/licenses/by/4.0/
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "reference.h"

// Calculate the power parameter, and then weighted interpolating
// Without using shared memory
void AIDW_Kernel(
    const Kokkos::View<const float*>& dx,
    const Kokkos::View<const float*>& dy,
    const Kokkos::View<const float*>& dz,
    const int dnum,
    const Kokkos::View<const float*>& ix,
    const Kokkos::View<const float*>& iy,
    const Kokkos::View<float*>& iz,
    const int inum,
    const float area,
    const Kokkos::View<const float*>& avg_dist)
{
  Kokkos::parallel_for("AIDW_Kernel", inum, KOKKOS_LAMBDA(const int tid) {
    float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 0.f;

    float r_obs = avg_dist(tid);
    float r_exp = 1.f / (2.f * Kokkos::sqrt(dnum / area));
    float R_S0 = r_obs / r_exp;

    float u_R = 0.f;
    if(R_S0 >= R_min) u_R = 0.5f-0.5f * Kokkos::cos(3.1415926f / R_max * (R_S0 - R_min));
    if(R_S0 >= R_max) u_R = 1.f;

    if(u_R>= 0.f && u_R<=0.1f)  alpha = a1;
    if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if(u_R>0.9f && u_R<=1.f)  alpha = a5;
    alpha *= 0.5f;

    for(int j = 0; j < dnum; j++) {
      dist = (ix(tid) - dx(j)) * (ix(tid) - dx(j)) + (iy(tid) - dy(j)) * (iy(tid) - dy(j));
      t = 1.f / (Kokkos::pow(dist, alpha));  sum += t;  z += dz(j) * t;
    }
    iz(tid) = z / sum;
  });
}

// Calculate the power parameter, and then weighted interpolating
// With using shared memory (Tiled version)
void AIDW_Kernel_Tiled(
    const Kokkos::View<const float*>& dx,
    const Kokkos::View<const float*>& dy,
    const Kokkos::View<const float*>& dz,
    const int dnum,
    const Kokkos::View<const float*>& ix,
    const Kokkos::View<const float*>& iy,
    const Kokkos::View<float*>& iz,
    const int inum,
    const float area,
    const Kokkos::View<const float*>& avg_dist)
{
  using ScratchView = Kokkos::View<float*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                   Kokkos::MemoryUnmanaged>;
  int scratch_size = ScratchView::shmem_size(BLOCK_SIZE) * 3; // sdx, sdy, sdz

  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;

  int num_teams = (inum + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Kokkos::parallel_for("AIDW_Kernel_Tiled",
    team_policy(num_teams, BLOCK_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const member_type& team) {
      ScratchView sdx(team.team_scratch(0), BLOCK_SIZE);
      ScratchView sdy(team.team_scratch(0), BLOCK_SIZE);
      ScratchView sdz(team.team_scratch(0), BLOCK_SIZE);

      int lid = team.team_rank();
      int tid = team.league_rank() * BLOCK_SIZE + lid;

      if (tid < inum) {
        float dist = 0.f, t = 0.f, alpha = 0.f;
        int part = (dnum - 1) / BLOCK_SIZE;

        float sum_up = 0.f;
        float sum_dn = 0.f;
        float six_s, siy_s;

        float r_obs = avg_dist(tid);
        float r_exp = 1.f / (2.f * Kokkos::sqrt(dnum / area));
        float R_S0 = r_obs / r_exp;

        float u_R = 0.f;
        if(R_S0 >= R_min) u_R = 0.5f-0.5f * Kokkos::cos(3.1415926f / R_max * (R_S0 - R_min));
        if(R_S0 >= R_max) u_R = 1.f;

        if(u_R>= 0.f && u_R<=0.1f)  alpha = a1;
        if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
        if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
        if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
        if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
        if(u_R>0.9f && u_R<=1.f)  alpha = a5;
        alpha *= 0.5f;

        float six_t = ix(tid);
        float siy_t = iy(tid);

        for(int m = 0; m <= part; m++) {
          int num_elem = BLOCK_SIZE < (dnum - BLOCK_SIZE*m) ? BLOCK_SIZE : (dnum - BLOCK_SIZE*m);
          if (lid < num_elem) {
            sdx(lid) = dx(lid + BLOCK_SIZE * m);
            sdy(lid) = dy(lid + BLOCK_SIZE * m);
            sdz(lid) = dz(lid + BLOCK_SIZE * m);
          }
          team.team_barrier();
          for(int e = 0; e < BLOCK_SIZE; e++) {
            six_s = six_t - sdx(e);
            siy_s = siy_t - sdy(e);
            dist = (six_s * six_s + siy_s * siy_s);
            t = 1.f / (Kokkos::pow(dist, alpha));  sum_dn += t;  sum_up += t * sdz(e);
          }
          team.team_barrier();
        }
        iz(tid) = sum_up / sum_dn;
      }
    });
}

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 4) {
      printf("Usage: %s <pts> <check> <iterations>\n", argv[0]);
      printf("pts: number of points (unit: 1K)\n");
      printf("check: enable verification when the value is 1\n");
      Kokkos::finalize();
      return 1;
    }

    const int numk = atoi(argv[1]);
    const int check = atoi(argv[2]);
    const int iterations = atoi(argv[3]);

    const int dnum = numk * 1024;
    const int inum = dnum;

    const float width = 2000, height = 2000;
    const float area = width * height;

    std::vector<float> h_dx(dnum), h_dy(dnum), h_dz(dnum);
    std::vector<float> h_avg_dist(dnum);
    std::vector<float> h_ix(inum), h_iy(inum), h_iz(inum);
    std::vector<float> h_iz_ref(inum);

    srand(123);
    for(int i = 0; i < dnum; i++)
    {
      h_dx[i] = rand()/(float)RAND_MAX * 1000;
      h_dy[i] = rand()/(float)RAND_MAX * 1000;
      h_dz[i] = rand()/(float)RAND_MAX * 1000;
    }

    for(int i = 0; i < inum; i++)
    {
      h_ix[i] = rand()/(float)RAND_MAX * 1000;
      h_iy[i] = rand()/(float)RAND_MAX * 1000;
      h_iz[i] = 0.f;
    }

    for(int i = 0; i < dnum; i++)
    {
      h_avg_dist[i] = rand()/(float)RAND_MAX * 3;
    }

    printf("Size = : %d K \n", numk);
    printf("dnum = : %d\ninum = : %d\n", dnum, inum);

    if (check) {
      printf("Verification enabled\n");
      reference(h_dx.data(), h_dy.data(), h_dz.data(), dnum, h_ix.data(),
                h_iy.data(), h_iz_ref.data(), inum, area, h_avg_dist.data());
    } else {
      printf("Verification disabled\n");
    }

    // Kokkos device views
    Kokkos::View<float*> d_dx("dx", dnum);
    Kokkos::View<float*> d_dy("dy", dnum);
    Kokkos::View<float*> d_dz("dz", dnum);
    Kokkos::View<float*> d_avg_dist("avg_dist", dnum);
    Kokkos::View<float*> d_ix("ix", inum);
    Kokkos::View<float*> d_iy("iy", inum);
    Kokkos::View<float*> d_iz("iz", inum);

    // Create host mirrors and copy data
    auto hm_dx = Kokkos::create_mirror_view(d_dx);
    auto hm_dy = Kokkos::create_mirror_view(d_dy);
    auto hm_dz = Kokkos::create_mirror_view(d_dz);
    auto hm_avg_dist = Kokkos::create_mirror_view(d_avg_dist);
    auto hm_ix = Kokkos::create_mirror_view(d_ix);
    auto hm_iy = Kokkos::create_mirror_view(d_iy);
    auto hm_iz = Kokkos::create_mirror_view(d_iz);

    for (int i = 0; i < dnum; i++) {
      hm_dx(i) = h_dx[i];
      hm_dy(i) = h_dy[i];
      hm_dz(i) = h_dz[i];
      hm_avg_dist(i) = h_avg_dist[i];
    }
    for (int i = 0; i < inum; i++) {
      hm_ix(i) = h_ix[i];
      hm_iy(i) = h_iy[i];
      hm_iz(i) = h_iz[i];
    }

    Kokkos::deep_copy(d_dx, hm_dx);
    Kokkos::deep_copy(d_dy, hm_dy);
    Kokkos::deep_copy(d_dz, hm_dz);
    Kokkos::deep_copy(d_avg_dist, hm_avg_dist);
    Kokkos::deep_copy(d_ix, hm_ix);
    Kokkos::deep_copy(d_iy, hm_iy);

    // const views for read-only data
    Kokkos::View<const float*> d_dx_c = d_dx;
    Kokkos::View<const float*> d_dy_c = d_dy;
    Kokkos::View<const float*> d_dz_c = d_dz;
    Kokkos::View<const float*> d_avg_dist_c = d_avg_dist;
    Kokkos::View<const float*> d_ix_c = d_ix;
    Kokkos::View<const float*> d_iy_c = d_iy;

    // Weighted Interpolate using AIDW
    AIDW_Kernel(d_dx_c, d_dy_c, d_dz_c, dnum, d_ix_c, d_iy_c, d_iz, inum, area, d_avg_dist_c);
    Kokkos::fence();

    Kokkos::deep_copy(hm_iz, d_iz);
    for (int i = 0; i < inum; i++) h_iz[i] = hm_iz(i);

    if (check) {
      bool ok = verify(h_iz.data(), h_iz_ref.data(), inum, EPS);
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    AIDW_Kernel_Tiled(d_dx_c, d_dy_c, d_dz_c, dnum, d_ix_c, d_iy_c, d_iz, inum, area, d_avg_dist_c);
    Kokkos::fence();

    Kokkos::deep_copy(hm_iz, d_iz);
    for (int i = 0; i < inum; i++) h_iz[i] = hm_iz(i);

    if (check) {
      bool ok = verify(h_iz.data(), h_iz_ref.data(), inum, EPS);
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; i++)
      AIDW_Kernel(d_dx_c, d_dy_c, d_dz_c, dnum, d_ix_c, d_iy_c, d_iz, inum, area, d_avg_dist_c);

    Kokkos::fence();

    auto t_end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average execution time of AIDW_Kernel       %f (s)\n", (time * 1e-9f) / iterations);

    t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; i++)
      AIDW_Kernel_Tiled(d_dx_c, d_dy_c, d_dz_c, dnum, d_ix_c, d_iy_c, d_iz, inum, area, d_avg_dist_c);

    Kokkos::fence();

    t_end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average execution time of AIDW_Kernel_Tiled %f (s)\n", (time * 1e-9f) / iterations);
  }
  Kokkos::finalize();

  return 0;
}
