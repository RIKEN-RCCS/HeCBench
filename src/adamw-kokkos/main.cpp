#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
float q_mapping(const float* __restrict__ qmap,
                const float* __restrict__ qmidpt,
                float x)
{
    int low = 0;
    int high = 15;

    if (x <= qmap[low]) return low;
    if (qmap[high] <= x) return high;

    while (low < high) {
        int mid = (low + high) >> 1;
        if (qmap[mid] <= x)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }

    return (qmidpt[low-1] < x) ? low : low-1;
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 3) {
      printf("Usage: %s <vector size> <number of time steps>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }

    // assume each vector element contains two 4-bit quantized numbers
    const long vector_size = atol(argv[1]);
    const int time_step = atoi(argv[2]);

    int64_t size_bytes = vector_size * 2 * sizeof(float);

    float *g = (float*) malloc (size_bytes);
    float *p = (float*) malloc (size_bytes);
    float *m_qscale = (float*) malloc (size_bytes);
    float *v_qscale = (float*) malloc (size_bytes);
    int8_t *m = (int8_t*) malloc (vector_size);
    int8_t *v = (int8_t*) malloc (vector_size);
    float *r = (float*) malloc (size_bytes);

    std::mt19937 gen(19937);
    std::uniform_real_distribution<float> dist(0, 1);
    for (int64_t i = 0; i < vector_size * 2; i++) {
      m_qscale[i] = dist(gen);
      v_qscale[i] = dist(gen);
      g[i] = dist(gen);
      r[i] = p[i] = dist(gen);
    }

    for (int64_t i = 0; i < vector_size; i++) {
      m[i] = 256 * dist(gen);
      v[i] = 256 * dist(gen);
    }

    // Quantization lookup tables
    const float h_exp_qmap[16] = {
      -0.8875f, -0.6625f, -0.4375f, -0.2125f,
      -0.0775f, -0.0325f, -0.0055f,  0.0000f,
       0.0055f,  0.0325f,  0.0775f,  0.2125f,
       0.4375f,  0.6625f,  0.8875f,  1.0000f
    };

    const float h_exp_qmidpt[15] = {
      -0.775f, -0.55f, -0.325f, -0.145f, -0.055f,
      -0.019f, -0.00275f, 0.00275f, 0.019f, 0.055f,
       0.145f,  0.325f,  0.55f,  0.775f,  0.94375f
    };

    const float h_sq_qmap[16] = {
      0.0625f, 0.1250f, 0.1875f, 0.2500f,
      0.3125f, 0.3750f, 0.4375f, 0.5000f,
      0.5625f, 0.6250f, 0.6875f, 0.7500f,
      0.8125f, 0.8750f, 0.9375f, 1.0000f
    };

    const float h_sq_qmidpt[15] = {
      0.09375f, 0.15625f, 0.21875f, 0.28125f,
      0.34375f, 0.40625f, 0.46875f, 0.53125f,
      0.59375f, 0.65625f, 0.71875f, 0.78125f,
      0.84375f, 0.90625f, 0.96875f
    };

    // Device views for data arrays
    Kokkos::View<float*> d_p("p", vector_size * 2);
    Kokkos::View<float*> d_g("g", vector_size * 2);
    Kokkos::View<float*> d_exp_qscale("exp_qscale", vector_size * 2);
    Kokkos::View<float*> d_sq_qscale("sq_qscale", vector_size * 2);
    Kokkos::View<int8_t*> d_exp("exp", vector_size);
    Kokkos::View<int8_t*> d_sq("sq", vector_size);

    // Device views for quantization tables
    Kokkos::View<float*> d_exp_qmap("exp_qmap", 16);
    Kokkos::View<float*> d_exp_qmidpt("exp_qmidpt", 15);
    Kokkos::View<float*> d_sq_qmap("sq_qmap", 16);
    Kokkos::View<float*> d_sq_qmidpt("sq_qmidpt", 15);

    // Host mirror views for copying
    auto hv_p = Kokkos::View<float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(p, vector_size * 2);
    auto hv_g = Kokkos::View<float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(g, vector_size * 2);
    auto hv_exp_qscale = Kokkos::View<float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(m_qscale, vector_size * 2);
    auto hv_sq_qscale = Kokkos::View<float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(v_qscale, vector_size * 2);
    auto hv_exp = Kokkos::View<int8_t*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(m, vector_size);
    auto hv_sq = Kokkos::View<int8_t*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(v, vector_size);
    auto hv_exp_qmap = Kokkos::View<const float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(h_exp_qmap, 16);
    auto hv_exp_qmidpt = Kokkos::View<const float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(h_exp_qmidpt, 15);
    auto hv_sq_qmap = Kokkos::View<const float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(h_sq_qmap, 16);
    auto hv_sq_qmidpt = Kokkos::View<const float*, Kokkos::HostSpace,
                  Kokkos::MemoryUnmanaged>(h_sq_qmidpt, 15);

    Kokkos::deep_copy(d_p, hv_p);
    Kokkos::deep_copy(d_g, hv_g);
    Kokkos::deep_copy(d_exp_qscale, hv_exp_qscale);
    Kokkos::deep_copy(d_sq_qscale, hv_sq_qscale);
    Kokkos::deep_copy(d_exp, hv_exp);
    Kokkos::deep_copy(d_sq, hv_sq);
    Kokkos::deep_copy(d_exp_qmap, hv_exp_qmap);
    Kokkos::deep_copy(d_exp_qmidpt, hv_exp_qmidpt);
    Kokkos::deep_copy(d_sq_qmap, hv_sq_qmap);
    Kokkos::deep_copy(d_sq_qmidpt, hv_sq_qmidpt);

    const int threadsPerBlock = 64;
    const int nblocks = (vector_size + threadsPerBlock - 1) / threadsPerBlock;

    // default constants
    const float lr = 1e-3f;
    const float weight_decay = 1e-2f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float resid_beta1 = 1.0f - beta1;
    const float resid_beta2 = 1.0f - beta2;
    const float weight_decay_update = 1.0f - lr * weight_decay;

    using member_type = Kokkos::TeamPolicy<>::member_type;

    auto start = std::chrono::steady_clock::now();

    for (int step = 1; step <= time_step; step++) {

      const float correction1 = 1.0f - powf(beta1, step);
      const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));
      const float step_size = lr / correction1;

      Kokkos::parallel_for("fused_4bit",
        Kokkos::TeamPolicy<>(nblocks, threadsPerBlock),
        KOKKOS_LAMBDA(const member_type& team) {

          const int block_id = team.league_rank();
          const int tid = team.team_rank();
          const int64_t global_id = (int64_t)block_id * threadsPerBlock + tid;
          const bool active = global_id < vector_size;

          const uint8_t bitmask = 15;
          const uint8_t right_pack_bitmask = bitmask << 4;

          float exp_left = 0.0f, exp_right = 0.0f;
          float sq_left = 0.0f, sq_right = 0.0f;

          if (active) {
            const int8_t exp_full = d_exp(global_id);
            const int8_t sq_full = d_sq(global_id);

            float p_x = d_p(global_id * 2);
            float p_y = d_p(global_id * 2 + 1);
            const float g_x = d_g(global_id * 2);
            const float g_y = d_g(global_id * 2 + 1);

            // left side processing
            const int8_t exp_left_index = exp_full & bitmask;
            const int8_t sq_left_index = sq_full & bitmask;

            p_x = p_x * weight_decay_update;

            float exp_avg_qscale = d_exp_qscale(block_id);

            exp_left = d_exp_qmap(exp_left_index) * exp_avg_qscale;
            exp_left = beta1 * exp_left + resid_beta1 * g_x;

            sq_left = d_sq_qmap(sq_left_index) * d_sq_qscale(block_id);
            sq_left = beta2 * sq_left + resid_beta2 * (g_x * g_x);

            d_p(global_id * 2) = p_x - (step_size * (exp_left / (sqrtf(sq_left) / correction2_sqrt + eps)));

            // right side processing
            const int8_t exp_right_index = (exp_full >> 4) & bitmask;
            const int8_t sq_right_index = (sq_full >> 4) & bitmask;

            p_y = p_y * weight_decay_update;

            exp_right = d_exp_qmap(exp_right_index) * exp_avg_qscale;
            exp_right = beta1 * exp_right + resid_beta1 * g_y;

            sq_right = d_sq_qmap(sq_right_index) * d_sq_qscale(block_id);
            sq_right = beta2 * sq_right + resid_beta2 * (g_y * g_y);

            d_p(global_id * 2 + 1) = p_y - (step_size * (exp_right / (sqrtf(sq_right) / correction2_sqrt + eps)));
          }

          // block-level max reduction for exp
          float local_absmax_exp = active ?
            ((exp_left > exp_right) ? exp_left : exp_right) : 0.0f;
          float absmax_exp;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, threadsPerBlock),
            [&](const int /*i*/, float& lmax) {
              if (local_absmax_exp > lmax) lmax = local_absmax_exp;
            }, Kokkos::Max<float>(absmax_exp));

          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            d_exp_qscale(block_id) = absmax_exp;
          });

          // block-level max reduction for sq
          float local_absmax_sq = active ?
            ((sq_left > sq_right) ? sq_left : sq_right) : 0.0f;
          float absmax_sq;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, threadsPerBlock),
            [&](const int /*i*/, float& lmax) {
              if (local_absmax_sq > lmax) lmax = local_absmax_sq;
            }, Kokkos::Max<float>(absmax_sq));

          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            d_sq_qscale(block_id) = absmax_sq;
          });

          team.team_barrier();

          if (active) {
            const float* exp_qmap_ptr = d_exp_qmap.data();
            const float* exp_qmidpt_ptr = d_exp_qmidpt.data();
            const float* sq_qmap_ptr = d_sq_qmap.data();
            const float* sq_qmidpt_ptr = d_sq_qmidpt.data();

            int8_t local_packed_exp = 0;
            int8_t local_packed_sq = 0;

            // quantize and pack left
            const int8_t q_exp_left = (int8_t)q_mapping(exp_qmap_ptr, exp_qmidpt_ptr,
                                                         (float)exp_left / absmax_exp);
            const int8_t q_sq_left = (int8_t)q_mapping(sq_qmap_ptr, sq_qmidpt_ptr,
                                                        (float)sq_left / absmax_sq);
            local_packed_exp |= (q_exp_left & bitmask);
            local_packed_sq |= (q_sq_left & bitmask);

            // quantize and pack right
            const int8_t q_exp_right = (int8_t)q_mapping(exp_qmap_ptr, exp_qmidpt_ptr,
                                                          (float)exp_right / absmax_exp);
            const int8_t q_sq_right = (int8_t)q_mapping(sq_qmap_ptr, sq_qmidpt_ptr,
                                                         (float)sq_right / absmax_sq);
            local_packed_exp |= (q_exp_right & right_pack_bitmask);
            local_packed_sq |= (q_sq_right & right_pack_bitmask);

            d_exp(global_id) = local_packed_exp;
            d_sq(global_id) = local_packed_sq;
          }

          team.team_barrier();
        });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (ms)\n", time * 1e-6f / time_step);

    Kokkos::deep_copy(hv_p, d_p);

    free(p);
    free(m_qscale);
    free(v_qscale);
    free(m);
    free(v);
    free(g);
    free(r);
  }
  Kokkos::finalize();
  return 0;
}
