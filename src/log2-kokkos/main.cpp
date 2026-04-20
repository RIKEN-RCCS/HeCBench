#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

typedef union type_caster_union {
  float f;
  uint32_t i;
} placeholder_name;

KOKKOS_INLINE_FUNCTION
float binary_log(float input, int precision)
{
  placeholder_name d1;
  d1.f = input;
  uint8_t exponent = ((d1.i & 0x7F800000) >> 23) - 127;
  int m = 0;
  int sum_m = 0;
  float result = 0;
  int test = (1 << exponent);
  float y = input / test;
  bool max_condition_met = 0;
  uint64_t one = 1;
  uint64_t denom = 0;
  uint64_t prev_denom = 0;
  while ((sum_m < precision + 1 && y != 1) || max_condition_met) {
    m = 0;
    while ((y < 2.f) && (sum_m + m < precision + 1)) {
      y *= y;
      m++;
    }
    sum_m += m;
    prev_denom = denom;
    denom = one << sum_m;
    if (sum_m >= precision) break;
    if (prev_denom > denom) {
      max_condition_met = 1;
      break;
    }
    result += 1.f / (float)denom;
    y /= 2.f;
  }
  return exponent + result;
}

void log2_approx(
    const std::vector<float> &inputs,
    std::vector<float> &outputs,
    const std::vector<int> &precision,
    const int num_inputs,
    const int precision_count,
    const int repeat)
{
  Kokkos::View<float*> d_inputs("d_inputs", num_inputs);
  Kokkos::View<float*> d_outputs("d_outputs", num_inputs * precision_count);

  auto h_inputs = Kokkos::create_mirror_view(d_inputs);
  for (int j = 0; j < num_inputs; j++) h_inputs(j) = inputs[j];
  Kokkos::deep_copy(d_inputs, h_inputs);

  for (int i = 0; i < precision_count; ++i) {
    const int prec = precision[i];
    const int offset = i * num_inputs;

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < repeat; ++k) {
      Kokkos::parallel_for("log2_approx",
        Kokkos::RangePolicy<>(0, num_inputs),
        KOKKOS_LAMBDA(const int j) {
          d_outputs(offset + j) = binary_log(d_inputs(j), prec);
        });
      Kokkos::fence();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto etime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nIterative approximation with " << precision[i] << " bits of precision\n";
    std::cout << "Average kernel execution time " << etime * 1e-3 / repeat << " (us)\n";
  }

  auto h_outputs = Kokkos::create_mirror_view(d_outputs);
  Kokkos::deep_copy(h_outputs, d_outputs);
  for (int j = 0; j < num_inputs * precision_count; j++)
    outputs[j] = h_outputs(j);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    std::cout << "Usage: ./main <config filename>\n";
    return 1;
  }

  std::ifstream message_file(argv[1]);
  std::string placeholder;
  long ceilingVal;
  message_file >> placeholder >> ceilingVal;

  int repeat;
  message_file >> placeholder >> repeat;

  int precision_count;
  message_file >> placeholder >> precision_count;

  std::vector<int> precision(precision_count, 0);
  message_file >> placeholder;
  for (int i = 0; i < precision_count; ++i) message_file >> precision[i];

  std::vector<float> inputs;
  long i = 1;
  while (i <= ceilingVal) {
    inputs.push_back((float)i);
    i++;
  }
  const int inputs_size = (int)inputs.size();

  std::cout << "Number of precision counts: " << precision_count
            << "\n Number of inputs: " << inputs_size
            << "\n Number of runs: " << repeat << "\n";

  std::vector<float> d_output_vals(inputs_size * precision_count);

  Kokkos::initialize(argc, argv);
  {
    log2_approx(inputs, d_output_vals, precision, inputs_size, precision_count, repeat);
  }
  Kokkos::finalize();

  // reference values
  std::vector<float> ref_vals(inputs_size);
  for (int j = 0; j < inputs_size; j++) ref_vals[j] = log2f(inputs[j]);

  std::cout << "-------------- SUMMARY (Kokkos results): --------------\n\n";
  for (int pi = 0; pi < precision_count; ++pi) {
    std::cout << "----- Iterative approximation with " << precision[pi] << " bits of precision -----\n";
    float s = 0;
    for (int j = 0; j < inputs_size; j++) {
      float diff = d_output_vals[pi * inputs_size + j] - ref_vals[j];
      s += diff * diff;
    }
    s /= inputs_size;
    std::cout << "RMSE: " << sqrtf(s) << "\n";
  }
  return 0;
}
