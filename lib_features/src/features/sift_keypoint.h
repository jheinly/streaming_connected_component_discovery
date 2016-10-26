#pragma once
#ifndef SIFT_KEYPOINT_H
#define SIFT_KEYPOINT_H

#include <core/alignment.h>

namespace features {

struct ALIGN(16) SiftKeypoint
{
  explicit SiftKeypoint()
  : x(0), y(x), scale(0), orientation(0)
  {}

  explicit SiftKeypoint(
    float x_val,
    float y_val,
    float scale_val,
    float orientation_val)
  : x(x_val),
    y(y_val),
    scale(scale_val),
    orientation(orientation_val)
  {}

  float x;
  float y;
  float scale;
  float orientation;
};

} // namespace features

#endif // SIFT_KEYPOINT_H
