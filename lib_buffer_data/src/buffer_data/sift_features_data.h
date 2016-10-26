#pragma once
#ifndef SIFT_FEATURES_DATA_H
#define SIFT_FEATURES_DATA_H

#include <string>
#include <vector>
#include <core/image_dimensions.h>
#include <features/sift_keypoint.h>
#include <features/sift_descriptor.h>

namespace buffer_data {

struct SiftFeaturesData
{
  explicit SiftFeaturesData()
  : image_index(-1),
    image_name(),
    dimensions(0, 0),
    num_features(0),
    keypoints(),
    descriptors(),
    image_scale_factor(0)
  {}

  int image_index;
  std::string image_name;
  core::ImageDimensions dimensions;
  int num_features;
  std::vector<features::SiftKeypoint> keypoints;
  std::vector<features::SiftDescriptorFloat> descriptors;
  float image_scale_factor;
};

} // namespace buffer_data

#endif // SIFT_FEATURES_DATA_H
