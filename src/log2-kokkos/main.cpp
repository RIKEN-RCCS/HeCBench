#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include "kernel.h"

int main(int argc, char* argv[]) {
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
  for (int i = 0; i < precision_count; ++i)
    message_file >> precision[i];

  std::vector<float> inputs;
  long i = 1;
  while (i <= ceilingVal) {
    inputs.push_back((float)i);
    i += 1;
  }

  size_t inputs_size = inputs.size();
  std::cout << "Number of precision counts : " << precision_count << "\n"
            << " Number of inputs to evaluate for each precision: " << inputs_size << "\n"
            << " Number of runs for each precision : " << repeat << "\n";

  std::vector<float> d_output_vals(inputs_size * precision_count);

  Kokkos::initialize(argc, argv);
  {
    log2_approx(inputs, d_output_vals, precision, (int)inputs.size(), precision_count, repeat);
  }
  Kokkos::finalize();

  std::vector<float> ref_vals(inputs_size);
  for (size_t j = 0; j < inputs_size; ++j)
    ref_vals[j] = log2f(inputs[j]);

  std::cout << "-------------- SUMMARY (Device results): --------------\n\n";
  for (int pi = 0; pi < precision_count; ++pi) {
    std::cout << "----- Iterative approximation with " << precision[pi] << " bits of precision -----\n";
    float s = 0;
    for (size_t j = 0; j < inputs_size; ++j) {
      float diff = d_output_vals[pi*inputs_size+j] - ref_vals[j];
      s += diff * diff;
    }
    s /= inputs.size();
    std::cout << "RMSE : " << sqrtf(s) << "\n";
  }

  return 0;
}
