#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>

// Inlined from scel-cuda/reference.h (HOST_DEVICE stripped for plain C++)
KOKKOS_INLINE_FUNCTION
float sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * (tgt - (lgt >= 0.f))
         - Kokkos::log(1.f + Kokkos::exp(lgt - 2.f * lgt * (lgt >= 0.f)));
}

KOKKOS_INLINE_FUNCTION
float sigmoid_partition(float lgt) {
  return lgt * (lgt >= 0.f)
         + Kokkos::log(1.f + Kokkos::exp(lgt - 2.f * lgt * (lgt >= 0.f)));
}

KOKKOS_INLINE_FUNCTION
float sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt) {
  return (2.f * tgt - 1.f) * (lgt - sigmoid_partition(lgt));
}

KOKKOS_INLINE_FUNCTION
float unjoined_sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * tgt + (tgt - 1.f) * lgt * (lgt >= 0.f)
         - (1.f - tgt) * Kokkos::log(1.f + Kokkos::exp(lgt - 2.f * lgt * (lgt >= 0.f)));
}

// CPU reference (plain math)
static float ref_sigmoid_xent_forward(float lgt, float tgt) {
  return lgt*(tgt-(lgt>=0.f)) - logf(1.f+expf(lgt-2.f*lgt*(lgt>=0.f)));
}
static float ref_sigmoid_partition(float lgt) {
  return lgt*(lgt>=0.f) + logf(1.f+expf(lgt-2.f*lgt*(lgt>=0.f)));
}
static float ref_sigmoid_xent_with_log_d(float lgt, float tgt) {
  return (2.f*tgt-1.f)*(lgt-ref_sigmoid_partition(lgt));
}
static float ref_unjoined(float lgt, float tgt) {
  return lgt*tgt+(tgt-1.f)*lgt*(lgt>=0.f)-(1.f-tgt)*logf(1.f+expf(lgt-2.f*lgt*(lgt>=0.f)));
}

void reference(int outer_size, int inner_size, bool log_D_trick, bool unjoined_lr_loss,
               const float *logits, const float *targets, float *out) {
  for (int i = 0; i < outer_size; i++) {
    float value = 0.f;
    for (int j = i * inner_size; j < (i+1) * inner_size; j++) {
      float lgt = logits[j], tgt = targets[j];
      if (unjoined_lr_loss)
        value += ref_unjoined(lgt, tgt);
      else
        value += log_D_trick ? ref_sigmoid_xent_with_log_d(lgt, tgt)
                             : ref_sigmoid_xent_forward(lgt, tgt);
    }
    out[i] = -value / inner_size;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <outer size> <inner_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int outer_size = atoi(argv[1]);
  const int inner_size = atoi(argv[2]);
  const int repeat     = atoi(argv[3]);

  const int input_size  = (outer_size + 1) * inner_size;
  const int output_size = outer_size;

  std::default_random_engine generator(123);
  std::normal_distribution<float> distribution(0.f, 1.f);

  float *h_logits  = (float*)malloc(input_size  * sizeof(float));
  float *h_targets = (float*)malloc(input_size  * sizeof(float));
  float *h_out     = (float*)malloc(output_size * sizeof(float));
  float *r_out     = (float*)malloc(output_size * sizeof(float));

  for (int i = 0; i < input_size; i++) {
    h_logits[i]  = distribution(generator);
    h_targets[i] = distribution(generator) + 1.f;
  }

  bool ok = true;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_logits("d_logits",   input_size);
    Kokkos::View<float*> d_targets("d_targets", input_size);
    Kokkos::View<float*> d_out("d_out",         output_size);

    {
      auto hL = Kokkos::create_mirror_view(d_logits);
      auto hT = Kokkos::create_mirror_view(d_targets);
      for (int i = 0; i < input_size; i++) { hL(i) = h_logits[i]; hT(i) = h_targets[i]; }
      Kokkos::deep_copy(d_logits, hL);
      Kokkos::deep_copy(d_targets, hT);
    }

    for (int unjoined_lr_loss = 0; unjoined_lr_loss <= 1; unjoined_lr_loss++) {
      int logD = (unjoined_lr_loss == 0) ? 1 : 0;

      for (int logD_trick = 0; logD_trick <= logD; logD_trick++) {
        const bool use_unjoined = (bool)unjoined_lr_loss;
        const bool use_logD     = (bool)logD_trick;

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < repeat; iter++) {
          Kokkos::parallel_for("scel",
            Kokkos::TeamPolicy<>(outer_size, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
              int i = team.league_rank();
              float value = 0.f;
              Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, inner_size),
                [=](int j, float &lsum) {
                  int in_idx = i * inner_size + j;
                  float lgt = d_logits(in_idx);
                  float tgt = d_targets(in_idx);
                  if (use_unjoined)
                    lsum += unjoined_sigmoid_xent_forward(lgt, tgt);
                  else
                    lsum += use_logD
                            ? sigmoid_xent_forward_with_log_d_trick(lgt, tgt)
                            : sigmoid_xent_forward(lgt, tgt);
                }, value);
              Kokkos::single(Kokkos::PerTeam(team),
                [=]() { d_out(i) = -value / inner_size; });
            });
          Kokkos::fence();
        }

        auto end  = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of SigmoidCrossEntropyWithLogits kernel: %f (us)\n",
               (time * 1e-3f) / repeat);

        auto hOut = Kokkos::create_mirror_view(d_out);
        Kokkos::deep_copy(hOut, d_out);
        for (int i = 0; i < output_size; i++) h_out[i] = hOut(i);

        reference(outer_size, inner_size, use_logD, use_unjoined, h_logits, h_targets, r_out);
        for (int i = 0; i < output_size; i++) {
          if (fabsf(r_out[i] - h_out[i]) > 1e-3f) { ok = false; break; }
        }
      }
    }
  }
  Kokkos::finalize();

  printf("%s\n", ok ? "PASS" : "FAIL");
  free(h_logits); free(h_targets); free(h_out); free(r_out);
  return 0;
}
