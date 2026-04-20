// Inverse Kinematics benchmark – Kokkos port
// Generates synthetic data (data_size=4096, iteration=100).

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <Kokkos_Core.hpp>

#define MAX_LOOP     25
#define NUM_JOINTS   3
#define PI           3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)

static void invkin_cpu(const float* xTarget, const float* yTarget,
                       float* angles, int size)
{
  for (int idx = 0; idx < size; idx++) {
    float angle_out[NUM_JOINTS] = {0.f, 0.f, 0.f};
    float xData[NUM_JOINTS_P1], yData[NUM_JOINTS_P1];
    for (int i = 0; i < NUM_JOINTS_P1; i++) { xData[i] = (float)i; yData[i] = 0.f; }

    float cx = xTarget[idx], cy = yTarget[idx];
    for (int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++) {
      for (int iter = NUM_JOINTS; iter > 0; iter--) {
        float pe_x = xData[NUM_JOINTS], pe_y = yData[NUM_JOINTS];
        float pc_x = xData[iter-1],     pc_y = yData[iter-1];
        float dpx = pe_x - pc_x, dpy = pe_y - pc_y;
        float dtx = cx   - pc_x, dty = cy   - pc_y;
        float lpe = sqrtf(dpx*dpx + dpy*dpy);
        float ltg = sqrtf(dtx*dtx + dty*dty);
        float ax = dpx/lpe, ay = dpy/lpe;
        float bx = dtx/ltg, by = dty/ltg;
        float adb = ax*bx + ay*by;
        if (adb >  1.f) adb =  1.f;
        if (adb < -1.f) adb = -1.f;
        float angle = acosf(adb) * (180.f / PI);
        if (ax*by - ay*bx < 0.f) angle = -angle;
        if (angle >  30.f) angle =  30.f;
        if (angle < -30.f) angle = -30.f;
        angle_out[iter-1] = angle;
        for (int i = 0; i < NUM_JOINTS-1; i++) angle_out[i+1] += angle_out[i];
      }
    }
    angles[idx*NUM_JOINTS+0] = angle_out[0];
    angles[idx*NUM_JOINTS+1] = angle_out[1];
    angles[idx*NUM_JOINTS+2] = angle_out[2];
  }
}

int main(int argc, char* argv[])
{
  const int data_size = 4096;
  const int iteration = 100;

  float* xTarget = new float[data_size];
  float* yTarget = new float[data_size];
  srand(42);
  for (int i = 0; i < data_size; i++) {
    xTarget[i] = ((rand() % 400) - 200) / 100.0f;  // [-2, 2]
    yTarget[i] = ((rand() % 400) - 200) / 100.0f;
  }

  float* angle_out_gpu = new float[data_size * NUM_JOINTS]();
  float* angle_out_cpu = new float[data_size * NUM_JOINTS]();

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_x("xTarget", data_size);
    Kokkos::View<float*> d_y("yTarget", data_size);
    Kokkos::View<float*> d_angles("angles", data_size * NUM_JOINTS);

    {
      auto h_x = Kokkos::create_mirror_view(d_x);
      auto h_y = Kokkos::create_mirror_view(d_y);
      for (int i = 0; i < data_size; i++) { h_x(i) = xTarget[i]; h_y(i) = yTarget[i]; }
      Kokkos::deep_copy(d_x, h_x);
      Kokkos::deep_copy(d_y, h_y);
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int n = 0; n < iteration; n++) {
      Kokkos::parallel_for("inversek2j",
        Kokkos::RangePolicy<>(0, data_size),
        KOKKOS_LAMBDA(int idx) {
          float angle_out[NUM_JOINTS] = {0.f, 0.f, 0.f};
          float xData[NUM_JOINTS_P1], yData[NUM_JOINTS_P1];
          for (int i = 0; i < NUM_JOINTS_P1; i++) { xData[i] = (float)i; yData[i] = 0.f; }

          float cx = d_x(idx), cy = d_y(idx);
          for (int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++) {
            for (int iter = NUM_JOINTS; iter > 0; iter--) {
              float pe_x = xData[NUM_JOINTS], pe_y = yData[NUM_JOINTS];
              float pc_x = xData[iter-1],     pc_y = yData[iter-1];
              float dpx = pe_x - pc_x, dpy = pe_y - pc_y;
              float dtx = cx   - pc_x, dty = cy   - pc_y;
              float lpe = sqrtf(dpx*dpx + dpy*dpy);
              float ltg = sqrtf(dtx*dtx + dty*dty);
              float ax = dpx/lpe, ay = dpy/lpe;
              float bx = dtx/ltg, by = dty/ltg;
              float adb = ax*bx + ay*by;
              if (adb >  1.f) adb =  1.f;
              if (adb < -1.f) adb = -1.f;
              float angle = acosf(adb) * (180.f / PI);
              if (ax*by - ay*bx < 0.f) angle = -angle;
              if (angle >  30.f) angle =  30.f;
              if (angle < -30.f) angle = -30.f;
              angle_out[iter-1] = angle;
              for (int i = 0; i < NUM_JOINTS-1; i++) angle_out[i+1] += angle_out[i];
            }
          }
          d_angles(idx*NUM_JOINTS+0) = angle_out[0];
          d_angles(idx*NUM_JOINTS+1) = angle_out[1];
          d_angles(idx*NUM_JOINTS+2) = angle_out[2];
        });
    }
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::cout << "Average kernel execution time: " << (ns * 1e-3f) / iteration << " us\n";

    auto h_angles = Kokkos::create_mirror_view(d_angles);
    Kokkos::deep_copy(h_angles, d_angles);
    for (int i = 0; i < data_size * NUM_JOINTS; i++) angle_out_gpu[i] = h_angles(i);
  }
  Kokkos::finalize();

  // CPU reference
  invkin_cpu(xTarget, yTarget, angle_out_cpu, data_size);

  int errors = 0;
  for (int i = 0; i < data_size; i++) {
    for (int j = 0; j < NUM_JOINTS; j++) {
      if (fabsf(angle_out_gpu[i*NUM_JOINTS+j] - angle_out_cpu[i*NUM_JOINTS+j]) > 1e-3f) {
        errors++;
        break;
      }
    }
  }
  std::cout << (errors ? "FAIL\n" : "PASS\n");

  delete[] xTarget;
  delete[] yTarget;
  delete[] angle_out_gpu;
  delete[] angle_out_cpu;
  return errors ? 1 : 0;
}
