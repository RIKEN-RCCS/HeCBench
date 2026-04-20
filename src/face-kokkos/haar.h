/*
 * Haar feature face detection - Kokkos port.
 *
 * Original by Francesco Comaschi (TU Eindhoven).
 * Kokkos port: uses Kokkos::parallel_for with DefaultHostExecutionSpace
 * for the setImageForCascadeClassifier kernel (pointer-based, must run on host).
 *
 * Licensed under GNU GPL.
 */

#ifndef __HAAR_H__
#define __HAAR_H__

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "image.h"
#include "stdio-wrapper.h"

#define MAXLABELS 50

typedef  int sumtype;
typedef int sqsumtype;

typedef struct MyPoint {
  int x;
  int y;
} MyPoint;

typedef struct {
  int width;
  int height;
} MySize;

typedef struct {
  int x;
  int y;
  int width;
  int height;
} MyRect;

typedef struct myCascade {
  int  n_stages;
  int  total_nodes;
  float scale;
  MySize orig_window_size;
  int inv_window_area;
  MyIntImage sum;
  MyIntImage sqsum;
  sqsumtype *pq0, *pq1, *pq2, *pq3;
  sumtype *p0, *p1, *p2, *p3;
} myCascade;

void setImageForCascadeClassifier(myCascade *cascade, MyIntImage *sum,
                                  MyIntImage *sqsum, int total_nodes);

int runCascadeClassifier(myCascade *cascade, MyPoint pt, int start_stage);

int readTextClassifier(const char *info_file, const char *class_file);
void releaseTextClassifier();

void groupRectangles(std::vector<MyRect> &_vec, int groupThreshold, float eps);

void drawRectangle(MyImage *image, MyRect r);

std::vector<MyRect> detectObjects(MyImage *image, MySize minSize, MySize maxSize,
                                  myCascade *cascade, float scale_factor,
                                  int min_neighbors, int total_nodes);

#endif /* __HAAR_H__ */
