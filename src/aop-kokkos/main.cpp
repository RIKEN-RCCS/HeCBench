/* Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
   redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Kokkos_Core.hpp>

#ifdef WITH_FULL_W_MATRIX
#define R_W_MATRICES_SMEM_SLOTS 15
#else
#define R_W_MATRICES_SMEM_SLOTS 12
#endif

typedef struct __attribute__((__aligned__(32)))
{
  double x, y, z, w;
}
double4;

typedef struct __attribute__((__aligned__(32)))
{
  double x, y, z;
}
double3;

KOKKOS_INLINE_FUNCTION double3 operator+(const double3 &u, const double3 &v)
{
  return {u.x+v.x, u.y+v.y, u.z+v.z};
}

KOKKOS_INLINE_FUNCTION double4 operator+(const double4 &u, const double4 &v)
{
  return {u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w};
}

struct PayoffCall
{
  double m_K;
  KOKKOS_INLINE_FUNCTION PayoffCall(double K) : m_K(K) {}
  KOKKOS_INLINE_FUNCTION double operator()(double S) const { return fmax(S - m_K, 0.0); }
  KOKKOS_INLINE_FUNCTION int is_in_the_money(double S) const { return S > m_K; }
};

struct PayoffPut
{
  double m_K;
  KOKKOS_INLINE_FUNCTION PayoffPut(double K) : m_K(K) {}
  KOKKOS_INLINE_FUNCTION double operator()(double S) const { return fmax(m_K - S, 0.0); }
  KOKKOS_INLINE_FUNCTION int is_in_the_money(double S) const { return S < m_K; }
};

template< int NUM_THREADS_PER_BLOCK, typename Payoff >
void generate_paths_kernel(int num_timesteps,
                           int num_paths,
                           Payoff payoff,
                           double dt,
                           double S0,
                           double r,
                           double sigma,
                           const Kokkos::View<double*> &d_samples,
                           const Kokkos::View<double*> &d_paths)
{
  Kokkos::parallel_for("generate_paths",
    Kokkos::RangePolicy<>(0, num_paths),
    KOKKOS_LAMBDA(const int path) {
      const double r_min_half_sigma_sq_dt = (r - 0.5*sigma*sigma)*dt;
      const double sigma_sqrt_dt = sigma*sqrt(dt);

      double S = S0;
      int offset = path;

      for (int timestep = 0; timestep < num_timesteps-1; ++timestep, offset += num_paths)
      {
        S = S * exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt*d_samples(offset));
        d_paths(offset) = S;
      }

      S = S * exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt*d_samples(offset));
      d_paths(offset) = payoff(S);
    }
  );
  Kokkos::fence();
}

KOKKOS_INLINE_FUNCTION
static void assemble_R(int m, double4 &sums, double *smem_svds)
{
  double x0 = smem_svds[0];
  double x1 = smem_svds[1];
  double x2 = smem_svds[2];

  double x0_sq = x0 * x0;

  double sum1 = sums.x - x0;
  double sum2 = sums.y - x0_sq;
  double sum3 = sums.z - x0_sq*x0;
  double sum4 = sums.w - x0_sq*x0_sq;

  double m_as_dbl = (double) m;
  double sigma = m_as_dbl - 1.0;
  double mu = sqrt(m_as_dbl);
  double v0 = -sigma / (1.0 + mu);
  double v0_sq = v0*v0;
  double beta = 2.0 * v0_sq / (sigma + v0_sq);

  double inv_v0 = 1.0 / v0;
  double one_min_beta = 1.0 - beta;
  double beta_div_v0  = beta * inv_v0;

  smem_svds[0] = mu;
  smem_svds[1] = one_min_beta*x0 - beta_div_v0*sum1;
  smem_svds[2] = one_min_beta*x0_sq - beta_div_v0*sum2;

  double beta_div_v0_sq = beta_div_v0 * inv_v0;

  double c1 = beta_div_v0_sq*sum1 + beta_div_v0*x0;
  double c2 = beta_div_v0_sq*sum2 + beta_div_v0*x0_sq;

  double x1_sq = x1*x1;

  sum1 -= x1;
  sum2 -= x1_sq;
  sum3 -= x1_sq*x1;
  sum4 -= x1_sq*x1_sq;

  x0 = x1-c1;
  x0_sq = x0*x0;
  sigma = sum2 - 2.0*c1*sum1 + (m_as_dbl-2.0)*c1*c1;
  if( fabs(sigma) < 1.0e-16 )
    beta = 0.0;
  else
  {
    mu = sqrt(x0_sq + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
    beta = 2.0*v0_sq / (sigma + v0_sq);
  }

  inv_v0 = 1.0 / v0;
  beta_div_v0 = beta * inv_v0;

  double c3 = (sum3 - c1*sum2 - c2*sum1 + (m_as_dbl-2.0)*c1*c2)*beta_div_v0;
  double c4 = (x1_sq-c2)*beta_div_v0 + c3*inv_v0;
  double c5 = c1*c4 - c2;

  one_min_beta = 1.0 - beta;

  smem_svds[3] = one_min_beta*x0 - beta_div_v0*sigma;
  smem_svds[4] = one_min_beta*(x1_sq-c2) - c3;

  double x2_sq = x2*x2;

  sum1 -= x2;
  sum2 -= x2_sq;
  sum3 -= x2_sq*x2;
  sum4 -= x2_sq*x2_sq;

  x0 = x2_sq-c4*x2+c5;
  sigma = sum4 - 2.0*c4*sum3 + (c4*c4 + 2.0*c5)*sum2 - 2.0*c4*c5*sum1 + (m_as_dbl-3.0)*c5*c5;
  if( fabs(sigma) < 1.0e-12 )
    beta = 0.0;
  else
  {
    mu = sqrt(x0*x0 + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
    beta = 2.0*v0_sq / (sigma + v0_sq);
  }

  smem_svds[5] = (1.0-beta)*x0 - (beta/v0)*sigma;
}

KOKKOS_INLINE_FUNCTION
static double off_diag_norm(double A01, double A02, double A12)
{
  return sqrt(2.0 * (A01*A01 + A02*A02 + A12*A12));
}

KOKKOS_INLINE_FUNCTION
static void my_swap(double &x, double &y)
{
  double t = x; x = y; y = t;
}

KOKKOS_INLINE_FUNCTION
static void svd_3x3(int m, double4 &sums, double *smem_svds)
{
  assemble_R(m, sums, smem_svds);

  double R00 = smem_svds[0];
  double R01 = smem_svds[1];
  double R02 = smem_svds[2];
  double R11 = smem_svds[3];
  double R12 = smem_svds[4];
  double R22 = smem_svds[5];

  double A00 = R00*R00;
  double A01 = R00*R01;
  double A02 = R00*R02;
  double A11 = R01*R01 + R11*R11;
  double A12 = R01*R02 + R11*R12;
  double A22 = R02*R02 + R12*R12 + R22*R22;

  double V00 = 1.0, V01 = 0.0, V02 = 0.0;
  double V10 = 0.0, V11 = 1.0, V12 = 0.0;
  double V20 = 0.0, V21 = 0.0, V22 = 1.0;

  const int max_iters = 16;
  const double tolerance = 1.0e-12;

  for (int iter = 0; off_diag_norm(A01, A02, A12) >= tolerance && iter < max_iters; ++iter)
  {
    double c, s, B00, B01, B02, B10, B11, B12, B20, B21, B22;

    c = 1.0; s = 0.0;
    if (A01 != 0.0)
    {
      double tau = (A11 - A00) / (2.0 * A01);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));

      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }

    B00 = c*A00 - s*A01;
    B01 = s*A00 + c*A01;
    B10 = c*A01 - s*A11;
    B11 = s*A01 + c*A11;
    B02 = A02;

    A00 = c*B00 - s*B10;
    A01 = c*B01 - s*B11;
    A11 = s*B01 + c*B11;
    A02 = c*B02 - s*A12;
    A12 = s*B02 + c*A12;

    B00 = c*V00 - s*V01;
    V01 = s*V00 + c*V01;
    V00 = B00;

    B10 = c*V10 - s*V11;
    V11 = s*V10 + c*V11;
    V10 = B10;

    B20 = c*V20 - s*V21;
    V21 = s*V20 + c*V21;
    V20 = B20;

    c = 1.0; s = 0.0;
    if (A02 != 0.0)
    {
      double tau = (A22 - A00) / (2.0 * A02);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));

      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }

    B00 = c*A00 - s*A02;
    B01 = c*A01 - s*A12;
    B02 = s*A00 + c*A02;
    B20 = c*A02 - s*A22;
    B22 = s*A02 + c*A22;

    A00 = c*B00 - s*B20;
    A12 = s*A01 + c*A12;
    A02 = c*B02 - s*B22;
    A22 = s*B02 + c*B22;
    A01 = B01;

    B00 = c*V00 - s*V02;
    V02 = s*V00 + c*V02;
    V00 = B00;

    B10 = c*V10 - s*V12;
    V12 = s*V10 + c*V12;
    V10 = B10;

    B20 = c*V20 - s*V22;
    V22 = s*V20 + c*V22;
    V20 = B20;

    c = 1.0; s = 0.0;
    if (A12 != 0.0)
    {
      double tau = (A22 - A11) / (2.0 * A12);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));

      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }

    B02 = s*A01 + c*A02;
    B11 = c*A11 - s*A12;
    B12 = s*A11 + c*A12;
    B21 = c*A12 - s*A22;
    B22 = s*A12 + c*A22;

    A01 = c*A01 - s*A02;
    A02 = B02;
    A11 = c*B11 - s*B21;
    A12 = c*B12 - s*B22;
    A22 = s*B12 + c*B22;

    B01 = c*V01 - s*V02;
    V02 = s*V01 + c*V02;
    V01 = B01;

    B11 = c*V11 - s*V12;
    V12 = s*V11 + c*V12;
    V11 = B11;

    B21 = c*V21 - s*V22;
    V22 = s*V21 + c*V22;
    V21 = B21;
  }

  if (A00 < A11)
  {
    my_swap(A00, A11);
    my_swap(V00, V01);
    my_swap(V10, V11);
    my_swap(V20, V21);
  }
  if (A00 < A22)
  {
    my_swap(A00, A22);
    my_swap(V00, V02);
    my_swap(V10, V12);
    my_swap(V20, V22);
  }
  if (A11 < A22)
  {
    my_swap(A11, A22);
    my_swap(V01, V02);
    my_swap(V11, V12);
    my_swap(V21, V22);
  }

  double inv_S0 = fabs(A00) < 1.0e-12 ? 0.0 : 1.0 / A00;
  double inv_S1 = fabs(A11) < 1.0e-12 ? 0.0 : 1.0 / A11;
  double inv_S2 = fabs(A22) < 1.0e-12 ? 0.0 : 1.0 / A22;

  double U00 = V00 * inv_S0;
  double U01 = V01 * inv_S1;
  double U02 = V02 * inv_S2;
  double U10 = V10 * inv_S0;
  double U11 = V11 * inv_S1;
  double U12 = V12 * inv_S2;
  double U20 = V20 * inv_S0;
  double U21 = V21 * inv_S1;
  double U22 = V22 * inv_S2;

#ifdef WITH_FULL_W_MATRIX
  double B00 = U00*V00 + U01*V01 + U02*V02;
  double B01 = U00*V10 + U01*V11 + U02*V12;
  double B02 = U00*V20 + U01*V21 + U02*V22;
  double B10 = U10*V00 + U11*V01 + U12*V02;
  double B11 = U10*V10 + U11*V11 + U12*V12;
  double B12 = U10*V20 + U11*V21 + U12*V22;
  double B20 = U20*V00 + U21*V01 + U22*V02;
  double B21 = U20*V10 + U21*V11 + U22*V12;
  double B22 = U20*V20 + U21*V21 + U22*V22;

  smem_svds[ 6] = B00*R00 + B01*R01 + B02*R02;
  smem_svds[ 7] =           B01*R11 + B02*R12;
  smem_svds[ 8] =                     B02*R22;
  smem_svds[ 9] = B10*R00 + B11*R01 + B12*R02;
  smem_svds[10] =           B11*R11 + B12*R12;
  smem_svds[11] =                     B12*R22;
  smem_svds[12] = B20*R00 + B21*R01 + B22*R02;
  smem_svds[13] =           B21*R11 + B22*R12;
  smem_svds[14] =                     B22*R22;
#else
  double B00 = U00*V00 + U01*V01 + U02*V02;
  double B01 = U00*V10 + U01*V11 + U02*V12;
  double B02 = U00*V20 + U01*V21 + U02*V22;
  double B11 = U10*V10 + U11*V11 + U12*V12;
  double B12 = U10*V20 + U11*V21 + U12*V22;
  double B22 = U20*V20 + U21*V21 + U22*V22;

  smem_svds[ 6] = B00*R00 + B01*R01 + B02*R02;
  smem_svds[ 7] =           B01*R11 + B02*R12;
  smem_svds[ 8] =                     B02*R22;
  smem_svds[ 9] =           B11*R11 + B12*R12;
  smem_svds[10] =                     B12*R22;
  smem_svds[11] =                     B22*R22;
#endif
}

// The prepare_svd_kernel is complex because it uses shared memory, scans, and barriers.
// We implement it using TeamPolicy with scratch memory.
template< int NUM_THREADS_PER_BLOCK, typename Payoff >
void prepare_svd_kernel(const int numTeams,
                        int num_paths,
                        int min_in_the_money,
                        Payoff payoff,
                        const Kokkos::View<double*> &d_paths,
                        const Kokkos::View<int*> &d_all_out_of_the_money,
                        const Kokkos::View<double*> &d_svds)
{
  typedef Kokkos::TeamPolicy<> team_policy;
  typedef Kokkos::TeamPolicy<>::member_type member_type;
  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<int*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchViewInt;
  typedef Kokkos::View<double*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchViewDouble;

  // scan_input[NUM_THREADS_PER_BLOCK] + scan_output[1+NUM_THREADS_PER_BLOCK] + lsum (1 int)
  int int_scratch_size = ScratchViewInt::shmem_size(NUM_THREADS_PER_BLOCK + 1 + NUM_THREADS_PER_BLOCK + 1);
  // smem_svds[R_W_MATRICES_SMEM_SLOTS] + lsums (4 doubles)
  int double_scratch_size = ScratchViewDouble::shmem_size(R_W_MATRICES_SMEM_SLOTS + 4);
  int total_scratch = int_scratch_size + double_scratch_size;

  Kokkos::parallel_for("prepare_svd",
    team_policy(numTeams, NUM_THREADS_PER_BLOCK).set_scratch_size(0, Kokkos::PerTeam(total_scratch)),
    KOKKOS_LAMBDA(const member_type &team) {
      int lid = team.team_rank();
      int bid = team.league_rank();

      ScratchViewInt scratch_int(team.team_scratch(0), NUM_THREADS_PER_BLOCK + 1 + NUM_THREADS_PER_BLOCK + 1);
      ScratchViewDouble scratch_double(team.team_scratch(0), R_W_MATRICES_SMEM_SLOTS + 4);

      // Partition shared memory
      // scan_input: scratch_int[0 .. NUM_THREADS_PER_BLOCK-1]
      // scan_output: scratch_int[NUM_THREADS_PER_BLOCK .. 2*NUM_THREADS_PER_BLOCK]
      // lsum: scratch_int[2*NUM_THREADS_PER_BLOCK+1]
      // smem_svds: scratch_double[0 .. R_W_MATRICES_SMEM_SLOTS-1]
      // lsums: scratch_double[R_W_MATRICES_SMEM_SLOTS .. R_W_MATRICES_SMEM_SLOTS+3]

      const int scan_input_off = 0;
      const int scan_output_off = NUM_THREADS_PER_BLOCK;
      const int lsum_off = 2 * NUM_THREADS_PER_BLOCK + 1;
      const int smem_off = 0;
      const int lsums_off = R_W_MATRICES_SMEM_SLOTS;

      const int timestep = bid;
      const int offset = timestep * num_paths;

      int m = 0;
      double4 sums = {0.0, 0.0, 0.0, 0.0};

      if (lid < R_W_MATRICES_SMEM_SLOTS)
        scratch_double(smem_off + lid) = 0.0;
      team.team_barrier();

      int found_paths = 0;

      for (int path = lid; path < num_paths; path += NUM_THREADS_PER_BLOCK)
      {
        double S = d_paths(offset + path);
        const int in_the_money = payoff.is_in_the_money(S);

        // Scan
        scratch_int(scan_input_off + lid) = in_the_money;
        team.team_barrier();
        if (lid == 0) {
          scratch_int(scan_output_off + 0) = 0;
          for (int i = 1; i <= NUM_THREADS_PER_BLOCK; i++)
            scratch_int(scan_output_off + i) = scratch_int(scan_output_off + i - 1) + scratch_int(scan_input_off + i - 1);
        }
        team.team_barrier();
        const int partial_sum = scratch_int(scan_output_off + lid);
        const int total_sum = scratch_int(scan_output_off + NUM_THREADS_PER_BLOCK);

        if (found_paths < 3)
        {
          if (in_the_money && found_paths + partial_sum < 3)
            scratch_double(smem_off + found_paths + partial_sum) = S;
          team.team_barrier();
          found_paths += total_sum;
        }

        // Early continue check
        if (lid == 0) scratch_int(lsum_off) = 0;
        team.team_barrier();

        if (in_the_money) Kokkos::atomic_add(&scratch_int(lsum_off), 1);

        team.team_barrier();
        if (scratch_int(lsum_off) == 0) continue;

        m += in_the_money;

        double x = 0.0, x_sq = 0.0;
        if (in_the_money)
        {
          x = S;
          x_sq = S*S;
        }

        sums.x += x;
        sums.y += x_sq;
        sums.z += x_sq*x;
        sums.w += x_sq*x_sq;
      }

      // Final reduction of m
      if (lid == 0) scratch_int(lsum_off) = 0;
      team.team_barrier();

      Kokkos::atomic_add(&scratch_int(lsum_off), m);

      team.team_barrier();

      // Use shared memory to broadcast the flag to all threads
      const int not_enough = (scratch_int(lsum_off) < min_in_the_money) ? 1 : 0;

      if (not_enough)
      {
        if (lid == 0)
          d_all_out_of_the_money(bid) = 1;
      }
      else
      {
        // Final reduction of sums
        if (lid == 0) {
          scratch_double(lsums_off + 0) = 0.0;
          scratch_double(lsums_off + 1) = 0.0;
          scratch_double(lsums_off + 2) = 0.0;
          scratch_double(lsums_off + 3) = 0.0;
        }
        team.team_barrier();

        Kokkos::atomic_add(&scratch_double(lsums_off + 0), sums.x);
        Kokkos::atomic_add(&scratch_double(lsums_off + 1), sums.y);
        Kokkos::atomic_add(&scratch_double(lsums_off + 2), sums.z);
        Kokkos::atomic_add(&scratch_double(lsums_off + 3), sums.w);

        team.team_barrier();

        if (lid == 0)
        {
          double4 final_sums;
          final_sums.x = scratch_double(lsums_off + 0);
          final_sums.y = scratch_double(lsums_off + 1);
          final_sums.z = scratch_double(lsums_off + 2);
          final_sums.w = scratch_double(lsums_off + 3);

          double local_smem[R_W_MATRICES_SMEM_SLOTS];
          for (int i = 0; i < R_W_MATRICES_SMEM_SLOTS; i++)
            local_smem[i] = scratch_double(smem_off + i);

          svd_3x3(scratch_int(lsum_off), final_sums, local_smem);

          for (int i = 0; i < R_W_MATRICES_SMEM_SLOTS; i++)
            scratch_double(smem_off + i) = local_smem[i];
        }

        team.team_barrier();

        if (lid < R_W_MATRICES_SMEM_SLOTS)
          d_svds(16*bid + lid) = scratch_double(smem_off + lid);
      }
    }
  );
  Kokkos::fence();
}

template< int NUM_THREADS_PER_BLOCK, typename Payoff >
void compute_beta_kernel(int num_paths,
                         Payoff payoff,
                         const Kokkos::View<double*> &d_svd,
                         int svd_offset,
                         const Kokkos::View<double*> &d_paths,
                         int paths_offset,
                         int cashflows_offset,
                         const Kokkos::View<int*> &d_all_out_of_the_money,
                         int aootm_offset,
                         const Kokkos::View<double*> &d_beta)
{
  // First check if all_out_of_the_money
  int h_flag;
  auto d_flag_sub = Kokkos::subview(d_all_out_of_the_money, aootm_offset);
  auto h_flag_view = Kokkos::create_mirror_view(d_flag_sub);
  Kokkos::deep_copy(h_flag_view, d_flag_sub);
  h_flag = h_flag_view();

  if (h_flag != 0) return;

  // Zero beta
  Kokkos::parallel_for("zero_beta", Kokkos::RangePolicy<>(0, 3),
    KOKKOS_LAMBDA(const int i) { d_beta(i) = 0.0; });
  Kokkos::fence();

  // Compute beta using atomics
  Kokkos::parallel_for("compute_beta",
    Kokkos::RangePolicy<>(0, num_paths),
    KOKKOS_LAMBDA(const int path) {
      const double R00 = d_svd(svd_offset + 0);
      const double R01 = d_svd(svd_offset + 1);
      const double R02 = d_svd(svd_offset + 2);
      const double R11 = d_svd(svd_offset + 3);
      const double R12 = d_svd(svd_offset + 4);
      const double R22 = d_svd(svd_offset + 5);

    #ifdef WITH_FULL_W_MATRIX
      const double W00 = d_svd(svd_offset + 6);
      const double W01 = d_svd(svd_offset + 7);
      const double W02 = d_svd(svd_offset + 8);
      const double W10 = d_svd(svd_offset + 9);
      const double W11 = d_svd(svd_offset + 10);
      const double W12 = d_svd(svd_offset + 11);
      const double W20 = d_svd(svd_offset + 12);
      const double W21 = d_svd(svd_offset + 13);
      const double W22 = d_svd(svd_offset + 14);
    #else
      const double W00 = d_svd(svd_offset + 6);
      const double W01 = d_svd(svd_offset + 7);
      const double W02 = d_svd(svd_offset + 8);
      const double W11 = d_svd(svd_offset + 9);
      const double W12 = d_svd(svd_offset + 10);
      const double W22 = d_svd(svd_offset + 11);
    #endif

      const double inv_R00 = R00 != 0.0 ? 1.0 / R00 : 0.0;
      const double inv_R11 = R11 != 0.0 ? 1.0 / R11 : 0.0;
      const double inv_R22 = R22 != 0.0 ? 1.0 / R22 : 0.0;

      const double inv_R01 = inv_R00*inv_R11*R01;
      const double inv_R02 = inv_R00*inv_R22*R02;
      const double inv_R12 =         inv_R22*R12;

    #ifdef WITH_FULL_W_MATRIX
      const double inv_W00 = W00*inv_R00;
      const double inv_W10 = W10*inv_R00;
      const double inv_W20 = W20*inv_R00;
    #else
      const double inv_W00 = W00*inv_R00;
    #endif

      double S = d_paths(paths_offset + path);

      const int in_the_money = payoff.is_in_the_money(S);

      double Q1i = inv_R11*S - inv_R01;
      double Q2i = inv_R22*S*S - inv_R02 - Q1i*inv_R12;

    #ifdef WITH_FULL_W_MATRIX
      const double WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
      const double WI1 = inv_W10 + W11 * Q1i + W12 * Q2i;
      const double WI2 = inv_W20 + W21 * Q1i + W22 * Q2i;
    #else
      const double WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
      const double WI1 =           W11 * Q1i + W12 * Q2i;
      const double WI2 =                       W22 * Q2i;
    #endif

      double cashflow = in_the_money ? d_paths(cashflows_offset + path) : 0.0;

      Kokkos::atomic_add(&d_beta(0), WI0*cashflow);
      Kokkos::atomic_add(&d_beta(1), WI1*cashflow);
      Kokkos::atomic_add(&d_beta(2), WI2*cashflow);
    }
  );
  Kokkos::fence();
}

template< int NUM_THREADS_PER_BLOCK, typename Payoff >
void update_cashflow_kernel(int num_paths,
                            Payoff payoff_object,
                            double exp_min_r_dt,
                            const Kokkos::View<double*> &d_beta,
                            const Kokkos::View<double*> &d_paths,
                            int paths_offset,
                            int cashflows_offset,
                            const Kokkos::View<int*> &d_all_out_of_the_money,
                            int aootm_offset)
{
  Kokkos::parallel_for("update_cashflow",
    Kokkos::RangePolicy<>(0, num_paths),
    KOKKOS_LAMBDA(const int path) {
      const int skip_computations = d_all_out_of_the_money(aootm_offset);

      const double beta0 = d_beta(0);
      const double beta1 = d_beta(1);
      const double beta2 = d_beta(2);

      const double old_cashflow = exp_min_r_dt*d_paths(cashflows_offset + path);
      if (skip_computations)
      {
        d_paths(cashflows_offset + path) = old_cashflow;
        return;
      }

      double S  = d_paths(paths_offset + path);
      double S2 = S*S;

      double payoff = payoff_object(S);

      double estimated_payoff = beta0 + beta1*S + beta2*S2;

      estimated_payoff *= exp_min_r_dt;

      if (payoff <= 1.0e-8 || payoff <= estimated_payoff)
        payoff = old_cashflow;

      d_paths(cashflows_offset + path) = payoff;
    }
  );
  Kokkos::fence();
}

template< int NUM_THREADS_PER_BLOCK >
void compute_sums_kernel(int num_paths,
                         const Kokkos::View<double*> &d_paths,
                         int cashflows_offset,
                         double exp_min_r_dt,
                         double &price)
{
  double sum = 0.0;
  Kokkos::parallel_reduce("compute_sums",
    Kokkos::RangePolicy<>(0, num_paths),
    KOKKOS_LAMBDA(const int path, double &lsum) {
      lsum += d_paths(cashflows_offset + path);
    },
    sum
  );
  price = exp_min_r_dt * sum / (double) num_paths;
}

template< typename Payoff >
static inline
void do_run(double *h_samples,
            int num_timesteps,
            int num_paths,
            const Payoff &payoff,
            double dt,
            double S0,
            double r,
            double sigma,
            const Kokkos::View<double*> &d_samples,
            const Kokkos::View<double*> &d_paths,
            const Kokkos::View<double*> &d_svds,
            const Kokkos::View<int*>    &d_all_out_of_the_money,
            const Kokkos::View<double*> &d_temp_storage,
            double &h_price)
{
  const int cashflows_offset = (num_timesteps-1)*num_paths;

  // Copy samples to device
  {
    auto h_view = Kokkos::create_mirror_view(d_samples);
    memcpy(h_view.data(), h_samples, num_timesteps*num_paths*sizeof(double));
    Kokkos::deep_copy(d_samples, h_view);
  }

  // Generate asset prices
  const int NUM_THREADS_PER_BLOCK0 = 256;
  generate_paths_kernel<NUM_THREADS_PER_BLOCK0>(
    num_timesteps,
    num_paths,
    payoff,
    dt,
    S0,
    r,
    sigma,
    d_samples,
    d_paths);

  // Reset the all_out_of_the_money array
  Kokkos::parallel_for("reset_aootm",
    Kokkos::RangePolicy<>(0, num_timesteps),
    KOKKOS_LAMBDA(const int i) {
      d_all_out_of_the_money(i) = 0;
    }
  );
  Kokkos::fence();

  // Prepare the SVDs
  const int NUM_THREADS_PER_BLOCK1 = 256;
  prepare_svd_kernel<NUM_THREADS_PER_BLOCK1>(
    num_timesteps-1,
    num_paths,
    4,
    payoff,
    d_paths,
    d_all_out_of_the_money,
    d_svds);

  // The constant to discount the payoffs
  const double exp_min_r_dt = std::exp(-r*dt);

  const int NUM_THREADS_PER_BLOCK2 = 128;

  // Run the main loop
  for (int timestep = num_timesteps-2; timestep >= 0; --timestep)
  {
    compute_beta_kernel<NUM_THREADS_PER_BLOCK2>(
      num_paths,
      payoff,
      d_svds,
      16*timestep,
      d_paths,
      timestep*num_paths,
      cashflows_offset,
      d_all_out_of_the_money,
      timestep,
      d_temp_storage);

    update_cashflow_kernel<NUM_THREADS_PER_BLOCK2>(
      num_paths,
      payoff,
      exp_min_r_dt,
      d_temp_storage,
      d_paths,
      timestep*num_paths,
      cashflows_offset,
      d_all_out_of_the_money,
      timestep);
  }

  // Compute the final sum
  const int NUM_THREADS_PER_BLOCK4 = 128;
  compute_sums_kernel<NUM_THREADS_PER_BLOCK4>(
    num_paths,
    d_paths,
    cashflows_offset,
    exp_min_r_dt,
    h_price);
}

template< typename Payoff >
static double binomial_tree(int num_timesteps, const Payoff &payoff, double dt, double S0, double r, double sigma)
{
  double *tree = new double[num_timesteps+1];

  double u = std::exp( sigma * std::sqrt(dt));
  double d = std::exp(-sigma * std::sqrt(dt));
  double a = std::exp( r     * dt);

  double p = (a - d) / (u - d);

  double k = std::pow(d, num_timesteps);
  for (int t = 0; t <= num_timesteps; ++t)
  {
    tree[t] = payoff(S0*k);
    k *= u*u;
  }

  for (int t = num_timesteps-1; t >= 0; --t)
  {
    k = std::pow(d, t);
    for (int i = 0; i <= t; ++i)
    {
      double expected = std::exp(-r*dt) * (p*tree[i+1] + (1.0 - p)*tree[i]);
      double earlyex = payoff(S0*k);
      tree[i] = std::max(earlyex, expected);
      k *= u*u;
    }
  }

  double f = tree[0];
  delete[] tree;
  return f;
}

inline double my_normcdf(double x) {
  return (1.0 + erf(x / sqrt(2.0))) / 2.0;
}

static double black_scholes_merton_put(double T, double K, double S0, double r, double sigma)
{
  double d1 = (std::log(S0 / K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
  double d2 = d1 - sigma*std::sqrt(T);

  return K*std::exp(-r*T)*my_normcdf(-d2) - S0*my_normcdf(-d1);
}

static double black_scholes_merton_call(double T, double K, double S0, double r, double sigma)
{
  double d1 = (std::log(S0 / K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
  double d2 = d1 - sigma*std::sqrt(T);

  return S0*my_normcdf(d1) - K*std::exp(-r*T)*my_normcdf(d2);
}

int main(int argc, char **argv)
{
  const int MAX_GRID_SIZE = 2048;

  // Simulation parameters
  int num_timesteps = 100;
  int num_paths     = 32;
  int num_runs      = 1;

  // Option parameters
  double T     = 1.00;
  double K     = 4.00;
  double S0    = 3.60;
  double r     = 0.06;
  double sigma = 0.20;

  bool price_put = true;

  for (int i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-timesteps"))
      num_timesteps = strtol(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-paths"))
      num_paths = strtol(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-runs"))
      num_runs = strtol(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-T"))
      T = strtod(argv[++i], NULL);
    else if (!strcmp(argv[i], "-S0"))
      S0 = strtod(argv[++i], NULL);
    else if (!strcmp(argv[i], "-K"))
      K = strtod(argv[++i], NULL);
    else if (!strcmp(argv[i], "-r"))
      r = strtod(argv[++i], NULL);
    else if (!strcmp(argv[i], "-sigma"))
      sigma = strtod(argv[++i], NULL);
    else if (!strcmp(argv[i], "-call"))
      price_put = false;
    else
    {
      fprintf(stderr, "Unknown option %s. Aborting!!!\n", argv[i]);
      exit(1);
    }
  }

  printf("==============\n");
  printf("Num Timesteps         : %d\n",  num_timesteps);
  printf("Num Paths             : %dK\n", num_paths);
  printf("Num Runs              : %d\n",  num_runs);
  printf("T                     : %lf\n", T);
  printf("S0                    : %lf\n", S0);
  printf("K                     : %lf\n", K);
  printf("r                     : %lf\n", r);
  printf("sigma                 : %lf\n", sigma);
  printf("Option Type           : American %s\n", price_put ? "Put" : "Call");

  num_paths *= 1024;

  double dt = T / num_timesteps;

  std::default_random_engine rng;
  std::normal_distribution<double> norm_dist(0.0, 1.0);

  double *h_samples = (double*) malloc(num_timesteps*num_paths*sizeof(double));

  Kokkos::initialize(argc, argv);
  {
    int max_temp_storage = 4*MAX_GRID_SIZE;

    Kokkos::View<double*> d_samples("d_samples", num_timesteps*num_paths);
    Kokkos::View<double*> d_paths("d_paths", num_timesteps*num_paths);
    Kokkos::View<double*> d_svds("d_svds", num_timesteps*16);
    Kokkos::View<int*>    d_all_out_of_the_money("d_aootm", num_timesteps);
    Kokkos::View<double*> d_temp_storage("d_temp", max_temp_storage);

    double h_price;
    float total_elapsed_time = 0;

    for (int run = 0; run < num_runs; ++run)
    {
      for (int i = 0; i < num_timesteps*num_paths; ++i)
        h_samples[i] = norm_dist(rng);

      auto start = std::chrono::high_resolution_clock::now();
      if (price_put)
        do_run(h_samples,
               num_timesteps,
               num_paths,
               PayoffPut(K),
               dt,
               S0,
               r,
               sigma,
               d_samples,
               d_paths,
               d_svds,
               d_all_out_of_the_money,
               d_temp_storage,
               h_price);
      else
        do_run(h_samples,
               num_timesteps,
               num_paths,
               PayoffCall(K),
               dt,
               S0,
               r,
               sigma,
               d_samples,
               d_paths,
               d_svds,
               d_all_out_of_the_money,
               d_temp_storage,
               h_price);

      auto end = std::chrono::high_resolution_clock::now();
      const float elapsed_time =
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      total_elapsed_time += elapsed_time;
    }

    printf("==============\n");
    printf("GPU Longstaff-Schwartz: %.8lf\n", h_price);

    double price = 0.0;

    if (price_put)
      price = binomial_tree(num_timesteps, PayoffPut(K), dt, S0, r, sigma);
    else
      price = binomial_tree(num_timesteps, PayoffCall(K), dt, S0, r, sigma);

    printf("Binonmial             : %.8lf\n", price);

    if (price_put)
      price = black_scholes_merton_put(T, K, S0, r, sigma);
    else
      price = black_scholes_merton_call(T, K, S0, r, sigma);

    printf("European Price        : %.8lf\n", price);

    printf("==============\n");
    printf("elapsed time for each run         : %.3fms\n", total_elapsed_time / num_runs);
    printf("==============\n");
  }
  Kokkos::finalize();

  free(h_samples);

  return 0;
}
