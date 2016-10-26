#pragma once
#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include <string>
#include <vector>
#include <core/colorspace.h>
#include <core/image_dimensions.h>

namespace buffer_data {

struct ImageData
{
  explicit ImageData()
  : image_index(-1),
    image_name(),
    data(),
    dimensions(0, 0),
    colorspace(core::colorspace::Grayscale),
    scale_factor(0)
  {}

  int image_index;
  std::string image_name;
  std::vector<unsigned char> data;
  core::ImageDimensions dimensions;
  core::colorspace::Colorspace colorspace;
  float scale_factor;
};

} // namespace buffer_data

#endif // IMAGE_DATA_H
