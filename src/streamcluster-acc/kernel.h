#pragma once

/* block ID (gang lane) and local thread ID (vector lane) */
const int local_id  = tx;
const int thread_id = bx * work_group_size + local_id;

if (thread_id < num) {
  // cost between this point and point[x]: euclidean distance multiplied by weight
  float x_cost = 0.0f;
  #pragma acc loop seq
  for (int i = 0; i < dim; ++i) {
    const float a = coord_h[(i * num) + thread_id];
    const float b = coord_h[(i * num) + (int)x];  // == coord_s_acc[i] in the OpenMP version
    const float d = a - b;
    x_cost += d * d;
  }
  x_cost *= p_h[thread_id].weight;

  const float current_cost = p_h[thread_id].cost;
  const int base = thread_id * (K + 1);

  // if computed cost is less than original (it saves), mark it as to reassign
  if (x_cost < current_cost) {
    switch_membership[thread_id] = '1';
    work_mem_h[base + K] = x_cost - current_cost;
  }
  // if computed cost is larger, save the difference
  else {
    const int assign = (int)p_h[thread_id].assign;
    work_mem_h[base + center_table[assign]] += current_cost - x_cost;
  }
}
