#pragma once
#ifndef IMAGE_DIMENSIONS_H
#define IMAGE_DIMENSIONS_H

#include <cmath>

namespace core {

struct ImageDimensions
{
  explicit ImageDimensions()
  : width(0), height(0)
  {}
  
  explicit ImageDimensions(int w, int h)
  : width(w), height(h)
  {}

  int min_dimension() const
  { return width < height ? width : height; }

  int max_dimension() const
  { return width > height ? width : height; }

  float diagonal_width() const
  {
    const float float_width = static_cast<float>(width);
    const float float_height = static_cast<float>(height);
    return sqrtf(
      float_width * float_width +
      float_height * float_height);
  }
  
  int width;
  int height;
};

} // namespace core

#endif // IMAGE_DIMENSIONS_H
