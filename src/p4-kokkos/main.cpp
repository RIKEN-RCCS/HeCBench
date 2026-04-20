/*
 * Kokkos port of the p4 (PointPillars) postprocess benchmark.
 * The original OMP kernel uses one team per feature location and one thread
 * per anchor.  This port uses Kokkos::TeamPolicy with the same mapping.
 *
 * Args: <repeat>
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "params.h"

KOKKOS_INLINE_FUNCTION
float sigmoid_k(float x) { return 1.0f / (1.0f + expf(-x)); }

// Postprocess one (feature-location, anchor) pair.
// loc_index = team/league rank, itanchor = thread/team rank.
KOKKOS_INLINE_FUNCTION
void postprocess_cell(
    const float* __restrict__ cls_input,
          float* __restrict__ box_input,
    const float* __restrict__ dir_cls_input,
    const float* __restrict__ anchors,
    const float* __restrict__ anchor_bottom_heights,
          float* __restrict__ bndbox_output,
          int*   __restrict__ object_counter,
    float min_x_range, float max_x_range,
    float min_y_range, float max_y_range,
    int feature_x_size, int feature_y_size,
    int num_anchors, int num_classes, int num_box_values,
    float score_thresh, float dir_offset,
    int loc_index, int itanchor)
{
  if (itanchor >= num_anchors) return;

  int col = loc_index % feature_x_size;
  int row = loc_index / feature_x_size;
  float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
  float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);

  int cls_offset = loc_index * num_anchors * num_classes + itanchor * num_classes;
  const float* scores = cls_input + cls_offset;
  float max_score = sigmoid_k(scores[0]);
  int   cls_id    = 0;
  for (int i = 1; i < num_classes; i++) {
    float s = sigmoid_k(scores[i]);
    if (s > max_score) { max_score = s; cls_id = i; }
  }

  if (max_score < score_thresh) return;

  int box_offset     = loc_index * num_anchors * num_box_values + itanchor * num_box_values;
  int dir_cls_offset = loc_index * num_anchors * 2 + itanchor * 2;

  const float* anchor_ptr = anchors + itanchor * 4;
  float z_offset = anchor_ptr[2] / 2 + anchor_bottom_heights[itanchor / 2];
  float anchor[7] = {x_offset, y_offset, z_offset,
                     anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};

  float* box = box_input + box_offset;
  float dxa = anchor[3], dya = anchor[4], dza = anchor[5];
  float diag = sqrtf(dxa * dxa + dya * dya);
  box[0] = box[0] * diag + anchor[0];
  box[1] = box[1] * diag + anchor[1];
  box[2] = box[2] * dza  + anchor[2];
  box[3] = expf(box[3]) * dxa;
  box[4] = expf(box[4]) * dya;
  box[5] = expf(box[5]) * dza;
  box[6] = box[6] + anchor[6];

  int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1] ? 0 : 1;
  const float period = (float)M_PI;
  float val     = box[6] - dir_offset;
  float dir_rot = val - floorf(val / (period + 1e-8f)) * period;
  float yaw     = dir_rot + dir_offset + period * dir_label;

  int resCount = Kokkos::atomic_fetch_add(object_counter, 1);

  bndbox_output[0] = (float)(resCount + 1);  // running total (last write wins)
  float* data = bndbox_output + 1 + resCount * 9;
  data[0] = box[0]; data[1] = box[1]; data[2] = box[2];
  data[3] = box[3]; data[4] = box[4]; data[5] = box[5];
  data[6] = yaw;
  data[7] = (float)cls_id;
  data[8] = max_score;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Params p;
  const float min_x_range    = p.min_x_range;
  const float max_x_range    = p.max_x_range;
  const float min_y_range    = p.min_y_range;
  const float max_y_range    = p.max_y_range;
  const int   feature_x_size = p.feature_x_size;
  const int   feature_y_size = p.feature_y_size;
  const int   num_anchors    = p.num_anchors;
  const int   num_classes    = p.num_classes;
  const int   num_box_values = p.num_box_values;
  const float score_thresh   = p.score_thresh;
  const float dir_offset     = p.dir_offset;
  const int   len_per_anchor = p.len_per_anchor;
  const int   num_dir_bins   = p.num_dir_bins;

  const int feature_size       = feature_x_size * feature_y_size;
  const int feature_anchor_size = feature_size * num_anchors;
  const int cls_size            = feature_anchor_size * num_classes;
  const int box_size            = feature_anchor_size * num_box_values;
  const int dir_cls_size        = feature_anchor_size * num_dir_bins;
  const int bndbox_size         = feature_anchor_size * 9 + 1;
  const int anchors_size        = num_anchors * len_per_anchor;

  std::vector<float> h_cls(cls_size), h_box(box_size), h_dir(dir_cls_size);
  std::vector<float> h_bndbox(bndbox_size, 0.0f);
  std::vector<float> h_anchors(anchors_size);
  std::vector<float> h_abh(num_classes);

  srand(123);
  for (int i = 0; i < cls_size;     i++) h_cls[i]     = rand() / (float)RAND_MAX;
  for (int i = 0; i < box_size;     i++) h_box[i]     = rand() / (float)RAND_MAX;
  for (int i = 0; i < dir_cls_size; i++) h_dir[i]     = rand() / (float)RAND_MAX;
  for (int i = 0; i < anchors_size; i++) h_anchors[i] = p.anchors[i];
  for (int i = 0; i < num_classes;  i++) h_abh[i]     = p.anchor_bottom_heights[i];

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_cls    ("cls",     cls_size);
    Kokkos::View<float*> d_box    ("box",     box_size);
    Kokkos::View<float*> d_dir    ("dir",     dir_cls_size);
    Kokkos::View<float*> d_bndbox ("bndbox",  bndbox_size);
    Kokkos::View<float*> d_anchors("anchors", anchors_size);
    Kokkos::View<float*> d_abh    ("abh",     num_classes);
    Kokkos::View<int>    d_counter("counter");

    auto copy_in = [](Kokkos::View<float*> d, const std::vector<float>& h) {
      auto hv = Kokkos::create_mirror_view(d);
      for (size_t i = 0; i < h.size(); i++) hv(i) = h[i];
      Kokkos::deep_copy(d, hv);
    };
    copy_in(d_cls, h_cls);
    copy_in(d_anchors, h_anchors);
    copy_in(d_abh, h_abh);

    auto policy = Kokkos::TeamPolicy<>(feature_size, num_anchors);

    double elapsed = 0.0;

    for (int rep = 0; rep < repeat; rep++) {
      // Reset box and counter each repetition (matches original)
      copy_in(d_box, h_box);
      copy_in(d_dir, h_dir);
      Kokkos::deep_copy(d_counter, 0);
      Kokkos::deep_copy(d_bndbox, 0.0f);

      auto t0 = std::chrono::steady_clock::now();

      Kokkos::parallel_for("postprocess", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {
          int loc_index = member.league_rank();
          int itanchor  = member.team_rank();

          postprocess_cell(
              d_cls.data(), d_box.data(), d_dir.data(),
              d_anchors.data(), d_abh.data(),
              d_bndbox.data(), &d_counter(),
              min_x_range, max_x_range,
              min_y_range, max_y_range,
              feature_x_size, feature_y_size,
              num_anchors, num_classes, num_box_values,
              score_thresh, dir_offset,
              loc_index, itanchor);
        });
      Kokkos::fence();

      auto t1 = std::chrono::steady_clock::now();
      elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

      // Fix bndbox[0] = total detection count after kernel
      Kokkos::parallel_for("set_count", 1,
        KOKKOS_LAMBDA(int) { d_bndbox(0) = (float)d_counter(); });
      Kokkos::fence();
    }

    printf("Average execution time of postprocess kernel: %f (us)\n",
           (elapsed * 1e-3) / repeat);

    auto hv = Kokkos::create_mirror_view(d_bndbox);
    Kokkos::deep_copy(hv, d_bndbox);
    double checksum = 0.0;
    for (int i = 0; i < bndbox_size; i++) checksum += hv(i);
    printf("checksum = %lf\n", checksum / bndbox_size);
  }
  Kokkos::finalize();
  return 0;
}
