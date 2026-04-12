/*
 * Leaky Integrate-and-Fire (LIF) neuron model.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

void lif_kernel(int numNeurons, int neurons_per_item, float dt,
                Kokkos::View<const float*> encode_result,
                Kokkos::View<float*>       voltage_array,
                Kokkos::View<float*>       reftime_array,
                float tau_rc, float tau_ref,
                Kokkos::View<const float*> bias,
                Kokkos::View<const float*> gain,
                Kokkos::View<float*>       spikes)
{
  Kokkos::parallel_for("lif", numNeurons, KOKKOS_LAMBDA(int i) {
    int neuron_index = i % neurons_per_item;
    int item_index   = i / neurons_per_item;

    float voltage  = voltage_array(i);
    float ref_time = reftime_array(i);
    float current  = bias(neuron_index) + gain(neuron_index) * encode_result(item_index);
    float dV, spike, mult;

    dV = -Kokkos::expm1(-dt / tau_rc) * (current - voltage);
    voltage = Kokkos::fmax(voltage + dV, 0.f);

    ref_time -= dt;
    mult = ref_time * (-1.f / dt) + 1.f;
    mult = mult > 1.f ? 1.f : mult;
    mult = mult < 0.f ? 0.f : mult;
    voltage *= mult;

    if (voltage > 1.f) {
      spike    = 1.f / dt;
      ref_time = tau_ref + dt * (1.f - (voltage - 1.f) / dV);
      voltage  = 0.f;
    } else {
      spike = 0.f;
    }

    reftime_array(i) = ref_time;
    voltage_array(i) = voltage;
    spikes(i) = spike;
  });
  Kokkos::fence();
}

void reference(int numNeurons, int neurons_per_item, float dt,
               float *encode_result, float *voltage_array, float *reftime_array,
               float tau_rc, float tau_ref, float *bias, float *gain, float *spikes)
{
  for (int i = 0; i < numNeurons; i++) {
    int ni = i % neurons_per_item;
    int ii = i / neurons_per_item;
    float voltage  = voltage_array[i];
    float ref_time = reftime_array[i];
    float current  = bias[ni] + gain[ni] * encode_result[ii];
    float dV, spike, mult;

    dV = -expm1f(-dt / tau_rc) * (current - voltage);
    voltage = fmaxf(voltage + dV, 0.f);
    ref_time -= dt;
    mult = ref_time * (-1.f / dt) + 1.f;
    mult = mult > 1.f ? 1.f : mult;
    mult = mult < 0.f ? 0.f : mult;
    voltage *= mult;

    if (voltage > 1.f) {
      spike    = 1.f / dt;
      ref_time = tau_ref + dt * (1.f - (voltage - 1.f) / dV);
      voltage  = 0.f;
    } else {
      spike = 0.f;
    }
    reftime_array[i] = ref_time;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <neurons per item> <num_items> <num_steps>\n", argv[0]);
    return 1;
  }
  const int neurons_per_item = atoi(argv[1]);
  const int num_items        = atoi(argv[2]);
  const int num_steps        = atoi(argv[3]);
  const int num_neurons      = neurons_per_item * num_items;

  const float dt = 0.1f, tau_rc = 10.f, tau_ref = 2.f;

  float *encode_result = (float*) malloc(num_items * sizeof(float));
  float *bias          = (float*) malloc(neurons_per_item * sizeof(float));
  float *gain          = (float*) malloc(neurons_per_item * sizeof(float));
  float *voltage       = (float*) malloc(num_neurons * sizeof(float));
  float *reftime       = (float*) malloc(num_neurons * sizeof(float));
  float *spikes        = (float*) malloc(num_neurons * sizeof(float));
  float *voltage_gold  = (float*) malloc(num_neurons * sizeof(float));
  float *reftime_gold  = (float*) malloc(num_neurons * sizeof(float));
  float *spikes_gold   = (float*) malloc(num_neurons * sizeof(float));

  srand(123);
  for (int i = 0; i < num_items; i++)
    encode_result[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < num_neurons; i++) {
    voltage_gold[i] = voltage[i] = 1.f + rand() / (float)RAND_MAX;
    reftime_gold[i] = reftime[i] = (float)(rand() % 5) / 10.f;
  }
  for (int i = 0; i < neurons_per_item; i++) {
    bias[i] = rand() / (float)RAND_MAX;
    gain[i] = rand() / (float)RAND_MAX + 0.5f;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_encode("d_encode", num_items);
    Kokkos::View<float*> d_bias("d_bias", neurons_per_item);
    Kokkos::View<float*> d_gain("d_gain", neurons_per_item);
    Kokkos::View<float*> d_voltage("d_voltage", num_neurons);
    Kokkos::View<float*> d_reftime("d_reftime", num_neurons);
    Kokkos::View<float*> d_spikes("d_spikes", num_neurons);

    auto h_encode  = Kokkos::create_mirror_view(d_encode);
    auto h_bias    = Kokkos::create_mirror_view(d_bias);
    auto h_gain    = Kokkos::create_mirror_view(d_gain);
    auto h_voltage = Kokkos::create_mirror_view(d_voltage);
    auto h_reftime = Kokkos::create_mirror_view(d_reftime);

    for (int i = 0; i < num_items; i++) h_encode(i) = encode_result[i];
    for (int i = 0; i < neurons_per_item; i++) { h_bias(i) = bias[i]; h_gain(i) = gain[i]; }
    for (int i = 0; i < num_neurons; i++) { h_voltage(i) = voltage[i]; h_reftime(i) = reftime[i]; }
    Kokkos::deep_copy(d_encode, h_encode); Kokkos::deep_copy(d_bias, h_bias);
    Kokkos::deep_copy(d_gain, h_gain);     Kokkos::deep_copy(d_voltage, h_voltage);
    Kokkos::deep_copy(d_reftime, h_reftime);

    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < num_steps; step++) {
      lif_kernel(num_neurons, neurons_per_item, dt,
                 d_encode, d_voltage, d_reftime,
                 tau_rc, tau_ref, d_bias, d_gain, d_spikes);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3) / num_steps);

    auto h_spikes = Kokkos::create_mirror_view(d_spikes);
    Kokkos::deep_copy(h_spikes, d_spikes);
    for (int i = 0; i < num_neurons; i++) spikes[i] = h_spikes(i);
  }
  Kokkos::finalize();

  // Reference
  for (int step = 0; step < num_steps; step++) {
    reference(num_neurons, neurons_per_item, dt, encode_result,
              voltage_gold, reftime_gold, tau_rc, tau_ref, bias, gain, spikes_gold);
  }

  bool ok = true;
  for (int i = 0; i < num_neurons; i++) {
    if (fabsf(spikes[i] - spikes_gold[i]) > 1e-3f) { ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(encode_result); free(bias); free(gain);
  free(voltage); free(reftime); free(spikes);
  free(voltage_gold); free(reftime_gold); free(spikes_gold);
  return 0;
}
