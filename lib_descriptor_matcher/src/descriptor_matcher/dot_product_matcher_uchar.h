#pragma once
#ifndef DOT_PRODUCT_MATCHER_UCHAR_H
#define DOT_PRODUCT_MATCHER_UCHAR_H

#include <descriptor_matcher/src/descriptor_matcher_base.h>
#include <distance_matrix/dot_product_matrix_uchar.h>
#include <cuda_helper/cuda_macros.h>

namespace descriptor_matcher {

class DotProductMatcherUChar :
  public DescriptorMatcherBase<DotProductMatcherUChar, distance_matrix::DotProductMatrixUChar>
{
  public:
    explicit DotProductMatcherUChar(
      const int max_num_descriptors,
      const int num_uchars_per_descriptor,
      const int scale_factor_per_value,
      const float min_matching_distance = config::min_descriptor_matching_distance,
      const float max_matching_ratio = config::max_descriptor_matching_ratio);

    ~DotProductMatcherUChar();

    int match_host_row_major_descriptors_cpu(
      const unsigned char * const __restrict__ host_row_major_descriptors1,
      const int num_descriptors1,
      const unsigned char * const __restrict__ host_row_major_descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches);

    int match_host_column_major_descriptors_cpu(
      const unsigned char * const __restrict__ host_column_major_descriptors1,
      const int num_descriptors1,
      const int stride_in_values1,
      const unsigned char * const __restrict__ host_column_major_descriptors2,
      const int num_descriptors2,
      const int stride_in_values2,
      std::vector<std::pair<int, int> > & matches);

    int match_host_row_major_descriptors_gpu(
      const unsigned char * const __restrict__ host_row_major_descriptors1,
      const int num_descriptors1,
      const unsigned char * const __restrict__ host_row_major_descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    int match_host_column_major_descriptors_gpu(
      const unsigned char * const __restrict__ host_column_major_descriptors1,
      const int num_descriptors1,
      const int stride_in_values1,
      const unsigned char * const __restrict__ host_column_major_descriptors2,
      const int num_descriptors2,
      const int stride_in_values2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    int match_device_row_major_descriptors_gpu(
      const unsigned char * const __restrict__ device_row_major_descriptors1,
      const int num_descriptors1,
      const unsigned char * const __restrict__ device_row_major_descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    int match_device_column_major_descriptors_gpu(
      const unsigned char * const __restrict__ device_column_major_descriptors1,
      const int num_descriptors1,
      const int stride_in_values1,
      const unsigned char * const __restrict__ device_column_major_descriptors2,
      const int num_descriptors2,
      const int stride_in_values2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    struct Functions
    {
      public:
        Functions(
          const float min_matching_distance,
          const float max_matching_ratio)
        : m_min_matching_distance(min_matching_distance),
          m_max_matching_ratio(max_matching_ratio)
        {}
      
        CUDA_HOST_AND_DEVICE_FUNCTION inline distance_matrix::DotProductMatrixUChar::TemplatedDistanceType worst_distance() const
        { return 0.0f; }

        CUDA_HOST_AND_DEVICE_FUNCTION inline bool is_first_distance_better(
          const distance_matrix::DotProductMatrixUChar::TemplatedDistanceType distance1,
          const distance_matrix::DotProductMatrixUChar::TemplatedDistanceType distance2) const
        { return distance1 > distance2; }

        CUDA_HOST_AND_DEVICE_FUNCTION inline bool does_distance_fail_min_threshold(
          const distance_matrix::DotProductMatrixUChar::TemplatedDistanceType distance) const
        { return distance < m_min_matching_distance; }

        CUDA_HOST_AND_DEVICE_FUNCTION inline bool is_distance_a_perfect_match(
          const distance_matrix::DotProductMatrixUChar::TemplatedDistanceType distance) const
        { return distance >= 1.0f; }

        CUDA_HOST_AND_DEVICE_FUNCTION inline bool do_distances_fail_max_ratio(
          const distance_matrix::DotProductMatrixUChar::TemplatedDistanceType distance1,
          const distance_matrix::DotProductMatrixUChar::TemplatedDistanceType distance2) const
        { return (1.0f - distance1) / (1.0f - distance2) > m_max_matching_ratio; }
      
      private:
        Functions & operator=(const Functions &);

        const float m_min_matching_distance;
        const float m_max_matching_ratio;
    };

    const Functions & functions() const
    { return m_functions; }

  private:
    DotProductMatcherUChar(const DotProductMatcherUChar &);
    DotProductMatcherUChar & operator=(const DotProductMatcherUChar &);

    distance_matrix::DotProductMatrixUChar m_distance_matrix;
    const Functions m_functions;
};

} // namespace descriptor_matcher

#endif // DOT_PRODUCT_MATCHER_UCHAR_H
