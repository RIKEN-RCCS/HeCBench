//==============================================================
// Kokkos port of the Projectile benchmark
// Original: Copyright © 2020 Intel Corporation, SPDX-License-Identifier: MIT
//==============================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <Kokkos_Core.hpp>

#ifdef DEBUG
static const int num_elements = 100;
#else
static const int num_elements = 10000000;
#endif

const float kPIValue  = 3.1415f;
const float kGValue   = 9.81f;

// Plain struct with KOKKOS_INLINE_FUNCTION methods so it is usable in device lambdas
struct Projectile {
  float m_angle;
  float m_velocity;
  float m_range;
  float m_totalTime;
  float m_maxHeight;

  KOKKOS_INLINE_FUNCTION
  Projectile()
    : m_angle(0), m_velocity(0), m_range(0), m_totalTime(0), m_maxHeight(0) {}

  KOKKOS_INLINE_FUNCTION
  Projectile(float angle, float velocity, float range, float time, float maxheight)
    : m_angle(angle), m_velocity(velocity), m_range(range),
      m_totalTime(time), m_maxHeight(maxheight) {}

  KOKKOS_INLINE_FUNCTION float getangle()    const { return m_angle;     }
  KOKKOS_INLINE_FUNCTION float getvelocity() const { return m_velocity;  }
  KOKKOS_INLINE_FUNCTION float getRange()    const { return m_range;     }
  KOKKOS_INLINE_FUNCTION float gettotalTime()  const { return m_totalTime;  }
  KOKKOS_INLINE_FUNCTION float getmaxHeight()  const { return m_maxHeight;  }

  KOKKOS_INLINE_FUNCTION
  void setRangeandTime(float frange, float ttime,
                       float angle_s, float velocity_s, float height_s) {
    m_range     = frange;
    m_totalTime = ttime;
    m_angle     = angle_s;
    m_velocity  = velocity_s;
    m_maxHeight = height_s;
  }

  friend bool operator!=(const Projectile& a, const Projectile& b) {
    return (a.m_angle     != b.m_angle)     || (a.m_velocity  != b.m_velocity) ||
           (a.m_range     != b.m_range)     || (a.m_totalTime != b.m_totalTime) ||
           (a.m_maxHeight != b.m_maxHeight);
  }

  friend std::ostream& operator<<(std::ostream& out, const Projectile& obj) {
    out << "Angle: "          << obj.getangle()
        << " Velocity: "      << obj.getvelocity()
        << " Range: "         << obj.getRange()
        << " Total time: "    << obj.gettotalTime()
        << " Maximum Height: "<< obj.getmaxHeight() << "\n";
    return out;
  }
};

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // Allocate host input/output as plain arrays (before Kokkos init)
  Projectile* h_in  = new Projectile[num_elements];
  Projectile* h_out = new Projectile[num_elements];

  srand(2);
  for (int i = 0; i < num_elements; i++) {
    float angle = (float)(rand() % 90 + 10);
    float vel   = (float)(rand() % 400 + 10);
    h_in[i]  = Projectile(angle, vel, 1.0f, 1.0f, 1.0f);
    h_out[i] = Projectile();
  }

  Kokkos::initialize(argc, argv);
  {
    // Wrap the host arrays in unmanaged views for deep_copy
    Kokkos::View<Projectile*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_in_v(h_in,   num_elements);
    Kokkos::View<Projectile*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_out_v(h_out, num_elements);

    Kokkos::View<Projectile*> d_in("d_in",   num_elements);
    Kokkos::View<Projectile*> d_out("d_out", num_elements);

    Kokkos::deep_copy(d_in, h_in_v);

    Kokkos::fence();
    auto t_start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for(
        "projectile",
        num_elements,
        KOKKOS_LAMBDA(const int i) {
          float proj_angle = d_in(i).getangle();
          float proj_vel   = d_in(i).getvelocity();
          float sin_value  = sinf(proj_angle * kPIValue / 180.0f);
          float cos_value  = cosf(proj_angle * kPIValue / 180.0f);
          float total_time = fabsf(2.0f * proj_vel * sin_value) / kGValue;
          float max_range  = fabsf(proj_vel * total_time * cos_value);
          float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f * kGValue;
          d_out(i).setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
        });
    }

    Kokkos::fence();
    auto t_end = std::chrono::steady_clock::now();
    auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    Kokkos::deep_copy(h_out_v, d_out);
  }
  Kokkos::finalize();

#ifdef DEBUG
  for (int i = 0; i < num_elements; i++)
    std::cout << "Parallel " << h_out[i];
#endif

  delete[] h_in;
  delete[] h_out;
  return 0;
}
