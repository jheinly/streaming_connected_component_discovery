#pragma once
#ifndef SIFT_SUPPORT_H
#define SIFT_SUPPORT_H

#include <features/sift_keypoint.h>
#include <features/sift_descriptor.h>
#include <vector>

namespace features {

namespace sift_support {

void compute_indices_for_keypoints_sorted_by_scale(
  const std::vector<SiftKeypoint> & keypoints,
  std::vector<int> & indices,
  const int max_num_keypoints = 0);

template<typename T>
void rearrange_vector_based_on_indices(
  const std::vector<T> & original_vector,
  std::vector<T> & arranged_vector,
  const std::vector<int> & indices)
{
  arranged_vector.resize(indices.size());
  for (size_t i = 0; i < indices.size(); ++i)
  {
    arranged_vector[i] = original_vector[indices[i]];
  }
}

void convert_descriptors_from_float_to_uchar(
  const std::vector<SiftDescriptorFloat> & descriptors_float,
  std::vector<SiftDescriptorUChar> & descriptors_uchar);

void convert_descriptors_from_uchar_to_float(
  const std::vector<SiftDescriptorUChar> & descriptors_uchar,
  std::vector<SiftDescriptorFloat> & descriptors_float);

} // namespace sift_support

} // namespace features

#endif // SIFT_SUPPORT_H
