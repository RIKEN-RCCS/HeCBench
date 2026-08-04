#pragma once

#include <cstddef>

void run_host_saxpy(const float* x, const float* y, const float* yhost,
                    int n, float a, int loops, size_t nbytes);
void run_kokkos_saxpy(const float* x, const float* y, const float* yhost,
                      int n, float a, int loops, size_t nbytes);
