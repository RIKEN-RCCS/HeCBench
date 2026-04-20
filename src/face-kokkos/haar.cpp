/*
 * Haar feature face detection - Kokkos port.
 *
 * Original by Francesco Comaschi (TU Eindhoven).
 * Kokkos port: setImageForCascadeClassifier uses Kokkos::parallel_for
 * on DefaultHostExecutionSpace (pointer arithmetic must stay on host).
 *
 * Licensed under GNU GPL.
 */

#include <stdio.h>
#include <assert.h>
#include <Kokkos_Core.hpp>
#include "haar.h"
#include "image.h"
#include "stdio-wrapper.h"

/* classifier parameters */
static int *stages_array;
static int *rectangles_array;
static int *weights_array;
static int *alpha1_array;
static int *alpha2_array;
static int *tree_thresh_array;
static int *stages_thresh_array;
static int **scaled_rectangles_array;

int clock_counter = 0;
float n_features = 0;
int iter_counter = 0;

/* forward declarations */
void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum);
void ScaleImage_Invoker(myCascade *cascade, float factor, int sum_row,
                        int sum_col, std::vector<MyRect> &vec);
void nearestNeighbor(MyImage *src, MyImage *dst);

inline int myRound(float value) {
  return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}

/*******************************************************
 * detectObjects
 ******************************************************/
std::vector<MyRect> detectObjects(
    MyImage *_img, MySize minSize, MySize maxSize, myCascade *cascade,
    float scaleFactor, int minNeighbors, int total_nodes) {
  const float GROUP_EPS = 0.4f;
  MyImage *img = _img;

  MyImage image1Obj;
  MyIntImage sum1Obj, sqsum1Obj;
  MyImage *img1 = &image1Obj;
  MyIntImage *sum1 = &sum1Obj;
  MyIntImage *sqsum1 = &sqsum1Obj;

  std::vector<MyRect> allCandidates;
  float factor;

  if (maxSize.height == 0 || maxSize.width == 0) {
    maxSize.height = img->height;
    maxSize.width  = img->width;
  }

  MySize winSize0 = cascade->orig_window_size;

  createImage(img->width, img->height, img1);
  createSumImage(img->width, img->height, sum1);
  createSumImage(img->width, img->height, sqsum1);

  for (factor = 1; ; factor *= scaleFactor) {
    iter_counter++;

    MySize winSize = {myRound(winSize0.width  * factor),
                      myRound(winSize0.height * factor)};
    MySize sz      = {(int)(img->width  / factor),
                      (int)(img->height / factor)};
    MySize sz1     = {sz.width  - winSize0.width,
                      sz.height - winSize0.height};

    if (sz1.width < 0 || sz1.height < 0) break;
    if (winSize.width  < minSize.width ||
        winSize.height < minSize.height) continue;

    setImage(sz.width, sz.height, img1);
    setSumImage(sz.width, sz.height, sum1);
    setSumImage(sz.width, sz.height, sqsum1);

    nearestNeighbor(img, img1);
    integralImages(img1, sum1, sqsum1);

    setImageForCascadeClassifier(cascade, sum1, sqsum1, total_nodes);

    printf("detecting faces, iter := %d\n", iter_counter);

    ScaleImage_Invoker(cascade, factor, sum1->height, sum1->width, allCandidates);
  }

  if (minNeighbors != 0) groupRectangles(allCandidates, minNeighbors, GROUP_EPS);

  freeImage(img1);
  freeSumImage(sum1);
  freeSumImage(sqsum1);
  return allCandidates;
}

unsigned int int_sqrt(unsigned int value) {
  unsigned int a = 0, b = 0, c = 0;
  for (int i = 0; i < (32 >> 1); i++) {
    c <<= 2;
#define UPPERBITS(v) ((v) >> 30)
    c += UPPERBITS(value);
#undef UPPERBITS
    value <<= 2;
    a <<= 1;
    b = (a << 1) | 1;
    if (c >= b) { c -= b; a++; }
  }
  return a;
}

/*******************************************************
 * setImageForCascadeClassifier
 *
 * The kernel computes host pointer values from host base
 * pointer sum->data, so it must run on the host.
 * Kokkos::parallel_for with DefaultHostExecutionSpace
 * provides parallelism while keeping pointer correctness.
 ******************************************************/
void setImageForCascadeClassifier(myCascade *_cascade, MyIntImage *_sum,
                                  MyIntImage *_sqsum, int total_nodes) {
  myCascade *cascade = _cascade;
  cascade->sum   = *_sum;
  cascade->sqsum = *_sqsum;

  MyRect equRect;
  equRect.x = equRect.y = 0;
  equRect.width  = cascade->orig_window_size.width;
  equRect.height = cascade->orig_window_size.height;

  cascade->inv_window_area = equRect.width * equRect.height;

  cascade->p0 = _sum->data;
  cascade->p1 = _sum->data + equRect.width - 1;
  cascade->p2 = _sum->data + _sum->width * (equRect.height - 1);
  cascade->p3 = _sum->data + _sum->width * (equRect.height - 1) + equRect.width - 1;

  cascade->pq0 = _sqsum->data;
  cascade->pq1 = _sqsum->data + equRect.width - 1;
  cascade->pq2 = _sqsum->data + _sqsum->width * (equRect.height - 1);
  cascade->pq3 = _sqsum->data + _sqsum->width * (equRect.height - 1) + equRect.width - 1;

  int *data  = _sum->data;
  int width  = _sum->width;
  int *rects = rectangles_array;
  int **sra  = scaled_rectangles_array;

  // Parallel over nodes on host; pointer arithmetic is host-valid
  Kokkos::parallel_for(
      "setImageForCascade",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, total_nodes),
      [=](int gid) {
        int idx = gid * 12;
        for (int k = 0; k < 3; k++) {
          int tr_x      = rects[idx + k * 4];
          int tr_y      = rects[idx + 1 + k * 4];
          int tr_width  = rects[idx + 2 + k * 4];
          int tr_height = rects[idx + 3 + k * 4];

          int *p0 = data + width * tr_y + tr_x;
          int *p1 = data + width * tr_y + (tr_x + tr_width);
          int *p2 = data + width * (tr_y + tr_height) + tr_x;
          int *p3 = data + width * (tr_y + tr_height) + (tr_x + tr_width);

          if (k < 2) {
            sra[idx + k * 4]     = p0;
            sra[idx + k * 4 + 1] = p1;
            sra[idx + k * 4 + 2] = p2;
            sra[idx + k * 4 + 3] = p3;
          } else {
            bool z = (tr_x == 0 && tr_y == 0 && tr_width == 0 && tr_height == 0);
            sra[idx + k * 4]     = z ? nullptr : p0;
            sra[idx + k * 4 + 1] = z ? nullptr : p1;
            sra[idx + k * 4 + 2] = z ? nullptr : p2;
            sra[idx + k * 4 + 3] = z ? nullptr : p3;
          }
        }
      });
  Kokkos::fence();
}

/*******************************************************
 * evalWeakClassifier
 ******************************************************/
inline int evalWeakClassifier(int variance_norm_factor, int p_offset,
                              int tree_index, int w_index, int r_index) {
  int t = tree_thresh_array[tree_index] * variance_norm_factor;

  int sum = (*( scaled_rectangles_array[r_index]     + p_offset)
            - *(scaled_rectangles_array[r_index + 1] + p_offset)
            - *(scaled_rectangles_array[r_index + 2] + p_offset)
            + *(scaled_rectangles_array[r_index + 3] + p_offset))
          * weights_array[w_index];

  sum += (*(scaled_rectangles_array[r_index + 4] + p_offset)
        - *(scaled_rectangles_array[r_index + 5] + p_offset)
        - *(scaled_rectangles_array[r_index + 6] + p_offset)
        + *(scaled_rectangles_array[r_index + 7] + p_offset))
       * weights_array[w_index + 1];

  if (scaled_rectangles_array[r_index + 8] != nullptr)
    sum += (*(scaled_rectangles_array[r_index + 8]  + p_offset)
          - *(scaled_rectangles_array[r_index + 9]  + p_offset)
          - *(scaled_rectangles_array[r_index + 10] + p_offset)
          + *(scaled_rectangles_array[r_index + 11] + p_offset))
         * weights_array[w_index + 2];

  return (sum >= t) ? alpha2_array[tree_index] : alpha1_array[tree_index];
}

/*******************************************************
 * runCascadeClassifier
 ******************************************************/
int runCascadeClassifier(myCascade *_cascade, MyPoint pt, int start_stage) {
  myCascade *cascade = _cascade;
  int p_offset  = pt.y * cascade->sum.width   + pt.x;
  int pq_offset = pt.y * cascade->sqsum.width + pt.x;

  unsigned int variance_norm_factor =
      cascade->pq0[pq_offset] - cascade->pq1[pq_offset]
    - cascade->pq2[pq_offset] + cascade->pq3[pq_offset];
  unsigned int mean =
      cascade->p0[p_offset] - cascade->p1[p_offset]
    - cascade->p2[p_offset] + cascade->p3[p_offset];

  variance_norm_factor = variance_norm_factor * cascade->inv_window_area;
  variance_norm_factor = variance_norm_factor - mean * mean;

  if ((int)variance_norm_factor > 0)
    variance_norm_factor = int_sqrt(variance_norm_factor);
  else
    variance_norm_factor = 1;

  int haar_counter = 0, w_index = 0, r_index = 0;
  for (int i = start_stage; i < cascade->n_stages; i++) {
    int stage_sum = 0;
    for (int j = 0; j < stages_array[i]; j++) {
      stage_sum += evalWeakClassifier(variance_norm_factor, p_offset,
                                      haar_counter, w_index, r_index);
      n_features++;
      haar_counter++;
      w_index += 3;
      r_index += 12;
    }
    if (stage_sum < 0.4f * stages_thresh_array[i]) return -i;
  }
  return 1;
}

/*******************************************************
 * ScaleImage_Invoker
 ******************************************************/
void ScaleImage_Invoker(myCascade *_cascade, float _factor, int sum_row,
                        int sum_col, std::vector<MyRect> &_vec) {
  myCascade *cascade = _cascade;
  float factor = _factor;
  MySize winSize0 = cascade->orig_window_size;
  MySize winSize  = {myRound(winSize0.width * factor),
                     myRound(winSize0.height * factor)};

  int y1   = 0;
  int y2   = sum_row - winSize0.height;
  int x2   = sum_col - winSize0.width;
  int step = 1;

  for (int x = 0; x <= x2; x += step)
    for (int y = y1; y <= y2; y += step) {
      MyPoint p = {x, y};
      int result = runCascadeClassifier(cascade, p, 0);
      if (result > 0) {
        MyRect r = {myRound(x * factor), myRound(y * factor),
                    winSize.width, winSize.height};
        _vec.push_back(r);
      }
    }
}

/*******************************************************
 * integralImages
 ******************************************************/
void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum) {
  int height = src->height, width = src->width;
  unsigned char *data    = src->data;
  int *sumData   = sum->data;
  int *sqsumData = sqsum->data;

  for (int y = 0; y < height; y++) {
    int s = 0, sq = 0;
    for (int x = 0; x < width; x++) {
      unsigned char it = data[y * width + x];
      s  += it;
      sq += it * it;
      int t  = s  + (y != 0 ? sumData[(y-1)*width+x]   : 0);
      int tq = sq + (y != 0 ? sqsumData[(y-1)*width+x] : 0);
      sumData[y*width+x]   = t;
      sqsumData[y*width+x] = tq;
    }
  }
}

/*******************************************************
 * nearestNeighbor
 ******************************************************/
void nearestNeighbor(MyImage *src, MyImage *dst) {
  int w1 = src->width, h1 = src->height;
  int w2 = dst->width, h2 = dst->height;
  unsigned char *src_data = src->data;
  unsigned char *dst_data = dst->data;

  int x_ratio = (int)((w1 << 16) / w2) + 1;
  int y_ratio = (int)((h1 << 16) / h2) + 1;

  for (int i = 0; i < h2; i++) {
    unsigned char *t = dst_data + i * w2;
    int y = ((i * y_ratio) >> 16);
    unsigned char *p = src_data + y * w1;
    int rat = 0;
    for (int j = 0; j < w2; j++) {
      *t++ = p[rat >> 16];
      rat += x_ratio;
    }
  }
}

/*******************************************************
 * readTextClassifier
 ******************************************************/
int readTextClassifier(const char *info_file, const char *class_file) {
  int stages = 0, total_nodes = 0;
  int i, j, k, l;
  char mystring[12];
  int r_index = 0, w_index = 0, tree_index = 0;

  FILE *finfo = fopen(info_file, "r");
  if (!finfo) { fprintf(stderr, "Failed to open %s\n", info_file); return -1; }
  FILE *fp = fopen(class_file, "r");
  if (!fp)    { fprintf(stderr, "Failed to open %s\n", class_file); return -1; }

  if (fgets(mystring, 12, finfo)) stages = atoi(mystring);
  if (stages == 0) { printf("Number of stages must be positive\n"); return -1; }

  stages_array = (int *)malloc(sizeof(int) * stages);
  i = 0;
  while (fgets(mystring, 12, finfo)) { stages_array[i] = atoi(mystring); total_nodes += stages_array[i]; i++; }
  fclose(finfo);

  rectangles_array       = (int *) malloc(sizeof(int)   * total_nodes * 12);
  scaled_rectangles_array= (int **)malloc(sizeof(int *)  * total_nodes * 12);
  weights_array          = (int *) malloc(sizeof(int)   * total_nodes * 3);
  alpha1_array           = (int *) malloc(sizeof(int)   * total_nodes);
  alpha2_array           = (int *) malloc(sizeof(int)   * total_nodes);
  tree_thresh_array      = (int *) malloc(sizeof(int)   * total_nodes);
  stages_thresh_array    = (int *) malloc(sizeof(int)   * stages);

  for (i = 0; i < stages; i++) {
    for (j = 0; j < stages_array[i]; j++) {
      for (k = 0; k < 3; k++) {
        for (l = 0; l < 4; l++) {
          if (fgets(mystring, 12, fp)) rectangles_array[r_index] = atoi(mystring);
          else break;
          r_index++;
        }
        if (fgets(mystring, 12, fp)) weights_array[w_index] = atoi(mystring);
        else break;
        w_index++;
      }
      if (fgets(mystring, 12, fp)) tree_thresh_array[tree_index] = atoi(mystring); else break;
      if (fgets(mystring, 12, fp)) alpha1_array[tree_index]      = atoi(mystring); else break;
      if (fgets(mystring, 12, fp)) alpha2_array[tree_index]      = atoi(mystring); else break;
      tree_index++;
      if (j == stages_array[i] - 1) {
        if (fgets(mystring, 12, fp)) stages_thresh_array[i] = atoi(mystring); else break;
      }
    }
  }
  fclose(fp);
  return total_nodes;
}

void releaseTextClassifier() {
  free(stages_array);
  free(rectangles_array);
  free(scaled_rectangles_array);
  free(weights_array);
  free(tree_thresh_array);
  free(alpha1_array);
  free(alpha2_array);
  free(stages_thresh_array);
}

void drawRectangle(MyImage *image, MyRect r) {
  int col = r.x, row = r.y;
  int width = r.width, height = r.height;
  int i;
  for (i = 0; i < width; i++) {
    image->data[row * image->width + col + i] = 255;
    image->data[(row + height) * image->width + col + i] = 255;
  }
  for (i = 0; i < height; i++) {
    image->data[(row + i) * image->width + col] = 255;
    image->data[(row + i) * image->width + col + width] = 255;
  }
}
