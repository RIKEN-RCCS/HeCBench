#include <stdlib.h>
#include <stdio.h>
#include <Kokkos_Core.hpp>
#include "backprop.h"

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  setup(argc, argv);
  Kokkos::finalize();
  return 0;
}
