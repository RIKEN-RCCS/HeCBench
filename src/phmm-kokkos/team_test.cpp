#include <Kokkos_Core.hpp>
#include <iostream>
int main() {
  Kokkos::initialize();
  {
    auto pol = Kokkos::TeamPolicy<>(1, Kokkos::AUTO);
    std::cout << "Max team size: " << pol.team_size_max([](auto){}, Kokkos::ParallelForTag()) << std::endl;
  }
  Kokkos::finalize();
}
