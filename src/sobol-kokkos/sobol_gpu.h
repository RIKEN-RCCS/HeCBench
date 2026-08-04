#pragma once

#include <Kokkos_Core.hpp>

double sobolGPU(int repeat, int n_vectors, int n_dimensions,
                Kokkos::View<unsigned int*> d_dir,
                Kokkos::View<float*> d_out);
