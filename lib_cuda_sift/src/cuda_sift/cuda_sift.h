#pragma once
#ifndef CUDA_SIFT_H
#define CUDA_SIFT_H

#include <sift_gpu/SiftGPU.h>
#include <features/sift_descriptor.h>
#include <features/sift_keypoint.h>
#include <vector>

namespace cuda_sift {

class CudaSift
{
  public:
    CudaSift(
      const int max_image_dimension,
      const int max_num_features,
      const int gpu_num = 0,
      const int port_offset = -1);

    ~CudaSift();

    int compute_using_host_image(
      const unsigned char * host_grayscale_image,
      const int image_width,
      const int image_height);

    // NOTE: the keypoints are already sorted from largest to smallest scale.
    const std::vector<features::SiftKeypoint> & host_keypoints() const
    { return m_sorted_keypoints; }

    // NOTE: the descriptors are already normalized to unit length.
    // TODO(jheinly) - try to see if the SIFT histograms can be square-rooted before normalizing to unit length
    const std::vector<features::SiftDescriptorFloat> & host_descriptors() const
    { return m_sorted_descriptors; }

  private:
    CudaSift(const CudaSift &);
    CudaSift & operator=(const CudaSift &);

    SiftGPU * m_sift_gpu;
    std::vector<features::SiftKeypoint> m_original_keypoints;
    std::vector<features::SiftDescriptorFloat> m_original_descriptors;
    std::vector<features::SiftKeypoint> m_sorted_keypoints;
    std::vector<features::SiftDescriptorFloat> m_sorted_descriptors;
    const int m_max_image_dimension;
    const int m_max_num_features;
    const int m_gpu_num;
    std::vector<int> m_sort_indices;
};

} // namespace cuda_sift

#endif // CUDA_SIFT_H
