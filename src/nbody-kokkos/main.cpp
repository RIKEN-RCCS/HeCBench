//==============================================================
// Copyright © 2020 Intel Corporation
// Kokkos port
// SPDX-License-Identifier: MIT
// =============================================================

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include <Kokkos_Core.hpp>

using RealType = float;

class TimeInterval {
 public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

 private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

void run_simulation(int npart, int nsteps) {
  const RealType dt = 0.1f;
  const int sfreq = 1;

  constexpr float kSofteningSquared = 1e-3f;
  constexpr float kG = 6.67259e-11f;

  // Struct-of-arrays Kokkos Views
  Kokkos::View<RealType*> pos_x("pos_x", npart);
  Kokkos::View<RealType*> pos_y("pos_y", npart);
  Kokkos::View<RealType*> pos_z("pos_z", npart);
  Kokkos::View<RealType*> vel_x("vel_x", npart);
  Kokkos::View<RealType*> vel_y("vel_y", npart);
  Kokkos::View<RealType*> vel_z("vel_z", npart);
  Kokkos::View<RealType*> acc_x("acc_x", npart);
  Kokkos::View<RealType*> acc_y("acc_y", npart);
  Kokkos::View<RealType*> acc_z("acc_z", npart);
  Kokkos::View<RealType*> mass("mass", npart);
  Kokkos::View<RealType*> e("e", npart);

  // Host mirrors for initialization
  auto h_pos_x = Kokkos::create_mirror_view(pos_x);
  auto h_pos_y = Kokkos::create_mirror_view(pos_y);
  auto h_pos_z = Kokkos::create_mirror_view(pos_z);
  auto h_vel_x = Kokkos::create_mirror_view(vel_x);
  auto h_vel_y = Kokkos::create_mirror_view(vel_y);
  auto h_vel_z = Kokkos::create_mirror_view(vel_z);
  auto h_acc_x = Kokkos::create_mirror_view(acc_x);
  auto h_acc_y = Kokkos::create_mirror_view(acc_y);
  auto h_acc_z = Kokkos::create_mirror_view(acc_z);
  auto h_mass  = Kokkos::create_mirror_view(mass);

  // InitPos
  {
    std::mt19937 gen(42);
    std::uniform_real_distribution<RealType> unif_d(0.0f, 1.0f);
    for (int i = 0; i < npart; ++i) {
      h_pos_x(i) = unif_d(gen);
      h_pos_y(i) = unif_d(gen);
      h_pos_z(i) = unif_d(gen);
    }
  }

  // InitVel
  {
    std::mt19937 gen(42);
    std::uniform_real_distribution<RealType> unif_d(-1.0f, 1.0f);
    for (int i = 0; i < npart; ++i) {
      h_vel_x(i) = unif_d(gen) * 1.0e-3f;
      h_vel_y(i) = unif_d(gen) * 1.0e-3f;
      h_vel_z(i) = unif_d(gen) * 1.0e-3f;
    }
  }

  // InitAcc
  for (int i = 0; i < npart; ++i) {
    h_acc_x(i) = 0.f;
    h_acc_y(i) = 0.f;
    h_acc_z(i) = 0.f;
  }

  // InitMass
  {
    RealType n = static_cast<RealType>(npart);
    std::mt19937 gen(42);
    std::uniform_real_distribution<RealType> unif_d(0.0f, 1.0f);
    for (int i = 0; i < npart; ++i) {
      h_mass(i) = n * unif_d(gen);
    }
  }

  // Copy to device
  Kokkos::deep_copy(pos_x, h_pos_x);
  Kokkos::deep_copy(pos_y, h_pos_y);
  Kokkos::deep_copy(pos_z, h_pos_z);
  Kokkos::deep_copy(vel_x, h_vel_x);
  Kokkos::deep_copy(vel_y, h_vel_y);
  Kokkos::deep_copy(vel_z, h_vel_z);
  Kokkos::deep_copy(acc_x, h_acc_x);
  Kokkos::deep_copy(acc_y, h_acc_y);
  Kokkos::deep_copy(acc_z, h_acc_z);
  Kokkos::deep_copy(mass,  h_mass);
  Kokkos::deep_copy(e, 0.f);

  double gflops = 1e-9 * ((11. + 18.) * (double)npart * npart + (double)npart * 19.);
  int nf = 0;
  double av = 0.0, dev = 0.0;
  RealType kenergy = 0.f;

  TimeInterval t0;

  for (int s = 1; s <= nsteps; ++s) {
    TimeInterval ts0;

    // accelerate_particles kernel
    Kokkos::parallel_for("accelerate_particles", npart,
      KOKKOS_LAMBDA(const int i) {
        RealType acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
        const RealType xi = pos_x(i), yi = pos_y(i), zi = pos_z(i);
        for (int j = 0; j < npart; ++j) {
          RealType dx = pos_x(j) - xi;
          RealType dy = pos_y(j) - yi;
          RealType dz = pos_z(j) - zi;
          RealType distance_sqr = dx*dx + dy*dy + dz*dz + kSofteningSquared;
          RealType distance_inv = 1.0f / Kokkos::sqrt(distance_sqr);
          RealType coef = kG * mass(j) * distance_inv * distance_inv * distance_inv;
          acc0 += dx * coef;
          acc1 += dy * coef;
          acc2 += dz * coef;
        }
        acc_x(i) = acc0;
        acc_y(i) = acc1;
        acc_z(i) = acc2;
      });

    // update_particles kernel
    Kokkos::parallel_for("update_particles", npart,
      KOKKOS_LAMBDA(const int i) {
        vel_x(i) += acc_x(i) * dt;
        vel_y(i) += acc_y(i) * dt;
        vel_z(i) += acc_z(i) * dt;

        pos_x(i) += vel_x(i) * dt;
        pos_y(i) += vel_y(i) * dt;
        pos_z(i) += vel_z(i) * dt;

        acc_x(i) = 0.f;
        acc_y(i) = 0.f;
        acc_z(i) = 0.f;

        e(i) = mass(i) * (vel_x(i)*vel_x(i) + vel_y(i)*vel_y(i) + vel_z(i)*vel_z(i));
      });

    // accumulate_energy kernel
    RealType total_e = 0.f;
    Kokkos::parallel_reduce("accumulate_energy", npart,
      KOKKOS_LAMBDA(const int i, RealType& sum) {
        sum += e(i);
      }, total_e);

    Kokkos::fence();
    double elapsed_seconds = ts0.Elapsed();

    kenergy = 0.5f * total_e;

    if ((s % sfreq) == 0) {
      nf += 1;
#ifdef DEBUG
      std::cout << " " << std::left << std::setw(8) << s
                << std::left << std::setprecision(5) << std::setw(8) << s * dt
                << std::left << std::setprecision(5) << std::setw(12) << kenergy
                << std::left << std::setprecision(5) << std::setw(12) << elapsed_seconds
                << std::left << std::setprecision(5) << std::setw(12)
                << gflops * sfreq / elapsed_seconds << "\n";
#endif
      if (nf > 2) {
        av  += gflops * sfreq / elapsed_seconds;
        dev += gflops * sfreq * gflops * sfreq / (elapsed_seconds * elapsed_seconds);
      }
    }
  }

  double total_time = t0.Elapsed();
  av /= (double)(nf - 2);
  dev = std::sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << "\n";
  std::cout << "# Total Energy        : " << kenergy << "\n";
  std::cout << "# Total Time (s)      : " << total_time << "\n";
  std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
  std::cout << "===============================" << "\n";
}

int main(int argc, char** argv) {
  int npart = 16000;
  int nsteps = 10;

  if (argc > 1) npart  = std::atoi(argv[1]);
  if (argc > 2) nsteps = std::atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    std::cout << "===============================" << "\n";
    std::cout << " Initialize Gravity Simulation" << "\n";
    run_simulation(npart, nsteps);
  }
  Kokkos::finalize();
  return 0;
}
