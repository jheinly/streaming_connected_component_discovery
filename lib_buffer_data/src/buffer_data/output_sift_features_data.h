#pragma once
#ifndef OUTPUT_SIFT_FEATURES_DATA_H
#define OUTPUT_SIFT_FEATURES_DATA_H

#include <string>
#include <vector>
#include <features/sift_keypoint.h>
#include <features/sift_descriptor.h>

namespace buffer_data {

struct OutputSiftFeaturesData
{
  explicit OutputSiftFeaturesData()
  : image_index(-1),
    image_name(),
    num_features(0),
    keypoints(),
    descriptors_uchar(),
    image_scale_factor(0)
  {}

  int image_index;
  std::string image_name;
  int num_features;
  std::vector<features::SiftKeypoint> keypoints;
  std::vector<features::SiftDescriptorUChar> descriptors_uchar;
  float image_scale_factor;
};

} // namespace buffer_data

#endif // OUTPUT_SIFT_FEATURES_DATA_H
