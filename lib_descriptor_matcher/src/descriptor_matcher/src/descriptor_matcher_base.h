#pragma once
#ifndef DESCRIPTOR_MATCHER_BASE_H
#define DESCRIPTOR_MATCHER_BASE_H

#ifdef __CUDACC__ // CUDA COMPILER
  #include <descriptor_matcher/src/descriptor_matcher_kernels.cuh>
#endif
#include <cuda_helper/cuda_helper.h>
#include <cuda_helper/cuda_macros.h>
#include <cuda_helper/cuda_memcpy_buffer.h>
#include <config/config.h>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>

namespace descriptor_matcher {

template<class Derived, class DistanceMatrix>
class DescriptorMatcherBase
{
  public:
    struct MatchResult
    {
      MatchResult()
      : matching_descriptor_index(-1),
        matching_distance(static_cast<typename DistanceMatrix::TemplatedDistanceType>(-1))
      {}

      CUDA_HOST_AND_DEVICE_FUNCTION MatchResult(
        const int matching_descriptor_index,
        const typename DistanceMatrix::TemplatedDistanceType matching_distance)
      : matching_descriptor_index(matching_descriptor_index),
        matching_distance(matching_distance)
      {}

      int matching_descriptor_index;
      typename DistanceMatrix::TemplatedDistanceType matching_distance;
    };

    DescriptorMatcherBase(
      DistanceMatrix * distance_matrix_ptr,
      const int max_num_descriptors,
      const int num_values_per_descriptor,
      const float min_matching_distance = config::min_descriptor_matching_distance,
      const float max_matching_ratio = config::max_descriptor_matching_ratio);

    virtual ~DescriptorMatcherBase();

    void initialize_for_matching_host_row_major_descriptors_cpu();
    void initialize_for_matching_host_column_major_descriptors_cpu();
    void initialize_for_matching_host_row_major_descriptors_mutual_best_cpu();
    void initialize_for_matching_host_row_major_descriptors_gpu();
    void initialize_for_matching_host_column_major_descriptors_gpu();
    void initialize_for_matching_device_row_major_descriptors_gpu();
    void initialize_for_matching_device_column_major_descriptors_gpu();

  protected:
    int match_host_row_major_descriptors_cpu_implementation(
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors1,
      const int num_descriptors1,
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches);

    int match_host_column_major_descriptors_cpu_implementation(
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors1,
      const int num_descriptors1,
      const int stride_in_values1,
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors2,
      const int num_descriptors2,
      const int stride_in_values2,
      std::vector<std::pair<int, int> > & matches);

    int match_host_row_major_descriptors_mutual_best_cpu_implementation(
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors1,
      const int num_descriptors1,
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches);

    int match_host_row_major_descriptors_gpu_implementation(
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors1,
      const int num_descriptors1,
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    int match_host_column_major_descriptors_gpu_implementation(
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors1,
      const int num_descriptors1,
      const int stride_in_values1,
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors2,
      const int num_descriptors2,
      const int stride_in_values2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    int match_device_row_major_descriptors_gpu_implementation(
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_row_major_descriptors1,
      const int num_descriptors1,
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_row_major_descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    int match_device_column_major_descriptors_gpu_implementation(
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_column_major_descriptors1,
      const int num_descriptors1,
      const int stride_in_values1,
      const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_column_major_descriptors2,
      const int num_descriptors2,
      const int stride_in_values2,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream = 0);

    const int m_max_num_descriptors;
    const int m_num_values_per_descriptor;
    const float m_min_matching_distance;
    const float m_max_matching_ratio;

  private:
    DescriptorMatcherBase(const DescriptorMatcherBase &);
    DescriptorMatcherBase & operator=(const DescriptorMatcherBase &);

    bool initialize_matches(
      const int num_descriptors1,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches);

    void initialize_gpu_match_results();

    int match_cpu(
      const typename DistanceMatrix::CpuMatrix & distance_matrix_result,
      std::vector<std::pair<int, int> > & matches);

    int match_cpu_mutual_best(
      const typename DistanceMatrix::CpuMatrix & distance_matrix_result,
      std::vector<std::pair<int, int> > & matches);

    int match_gpu(
      const typename DistanceMatrix::GpuMatrix & distance_matrix_result,
      std::vector<std::pair<int, int> > & matches,
      cudaStream_t cuda_stream);

    DistanceMatrix * m_distance_matrix_ptr;
    cuda_helper::CudaMemcpyBuffer<MatchResult> m_match_results_buffer;
    MatchResult * m_device_match_results;
    std::vector<int> m_existing_matches;
    std::vector<typename DistanceMatrix::TemplatedDistanceType> m_match_distances_cpu;
    std::vector<int> m_best_indices1;
    std::vector<int> m_best_indices2;
    std::vector<typename DistanceMatrix::TemplatedDistanceType> m_best_distances2;
};

} // namespace descriptor_matcher

template<class Derived, class DistanceMatrix>
descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::DescriptorMatcherBase(
  DistanceMatrix * distance_matrix_ptr,
  const int max_num_descriptors,
  const int num_values_per_descriptor,
  const float min_matching_distance,
  const float max_matching_ratio)
: m_max_num_descriptors(max_num_descriptors),
  m_num_values_per_descriptor(num_values_per_descriptor),
  m_min_matching_distance(min_matching_distance),
  m_max_matching_ratio(max_matching_ratio),
  m_distance_matrix_ptr(distance_matrix_ptr),
  m_device_match_results(NULL)
{
  m_existing_matches.resize(max_num_descriptors);
  m_match_distances_cpu.reserve(max_num_descriptors);
  m_best_indices1.reserve(max_num_descriptors);
  m_best_indices2.reserve(max_num_descriptors);
  m_best_distances2.reserve(max_num_descriptors);
}

template<class Derived, class DistanceMatrix>
descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::~DescriptorMatcherBase()
{
  if (m_device_match_results != NULL)
  {
    CUDA_CALL(cudaFree(m_device_match_results));
  }
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_for_matching_host_row_major_descriptors_cpu()
{
  m_distance_matrix_ptr->initialize_for_computing_host_row_major_vectors_cpu();
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_for_matching_host_column_major_descriptors_cpu()
{
  m_distance_matrix_ptr->initialize_for_computing_host_column_major_vectors_cpu();
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_for_matching_host_row_major_descriptors_mutual_best_cpu()
{
  m_distance_matrix_ptr->initialize_for_computing_row_major_vectors_cpu();
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_for_matching_host_row_major_descriptors_gpu()
{
  initialize_gpu_match_results();
  m_distance_matrix_ptr->initialize_for_computing_host_row_major_vectors_gpu();
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_for_matching_host_column_major_descriptors_gpu()
{
  initialize_gpu_match_results();
  m_distance_matrix_ptr->initialize_for_computing_host_column_major_vectors_gpu();
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_for_matching_device_row_major_descriptors_gpu()
{
  initialize_gpu_match_results();
  m_distance_matrix_ptr->initialize_for_computing_device_row_major_vectors_gpu();
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_for_matching_device_column_major_descriptors_gpu()
{
  initialize_gpu_match_results();
  m_distance_matrix_ptr->initialize_for_computing_device_column_major_vectors_gpu();
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_host_row_major_descriptors_cpu_implementation(
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors1,
  const int num_descriptors1,
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches)
{
  if (!initialize_matches(num_descriptors1, num_descriptors2, matches))
  {
    return 0;
  }

  typename DistanceMatrix::CpuMatrix distance_matrix_result =
    m_distance_matrix_ptr->compute_host_row_major_vectors_cpu(
      host_row_major_descriptors1,
      num_descriptors1,
      host_row_major_descriptors2,
      num_descriptors2);

  return match_cpu(distance_matrix_result, matches);
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_host_column_major_descriptors_cpu_implementation(
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors1,
  const int num_descriptors1,
  const int stride_in_values1,
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors2,
  const int num_descriptors2,
  const int stride_in_values2,
  std::vector<std::pair<int, int> > & matches)
{
  if (!initialize_matches(num_descriptors1, num_descriptors2, matches))
  {
    return 0;
  }

  typename DistanceMatrix::CpuMatrix distance_matrix_result =
    m_distance_matrix_ptr->compute_host_column_major_vectors_cpu(
      host_column_major_descriptors1,
      num_descriptors1,
      stride_in_values1,
      host_column_major_descriptors2,
      num_descriptors2,
      stride_in_values2);

  return match_cpu(distance_matrix_result, matches);
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_host_row_major_descriptors_mutual_best_cpu_implementation(
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors1,
  const int num_descriptors1,
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches)
{
  if (!initialize_matches(num_descriptors1, num_descriptors2, matches))
  {
    return 0;
  }

  typename DistanceMatrix::CpuMatrix distance_matrix_result =
    m_distance_matrix_ptr->compute_host_row_major_vectors_cpu(
      host_row_major_descriptors1,
      num_descriptors1,
      host_row_major_descriptors2,
      num_descriptors2);

  return match_cpu_mutual_best(distance_matrix_result, matches);
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_host_row_major_descriptors_gpu_implementation(
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors1,
  const int num_descriptors1,
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_row_major_descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  if (!initialize_matches(num_descriptors1, num_descriptors2, matches))
  {
    return 0;
  }

  typename DistanceMatrix::GpuMatrix distance_matrix_result =
    m_distance_matrix_ptr->compute_host_row_major_vectors_gpu(
      host_row_major_descriptors1,
      num_descriptors1,
      host_row_major_descriptors2,
      num_descriptors2,
      cuda_stream);

  return match_gpu(distance_matrix_result, matches, cuda_stream);
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_host_column_major_descriptors_gpu_implementation(
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors1,
  const int num_descriptors1,
  const int stride_in_values1,
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ host_column_major_descriptors2,
  const int num_descriptors2,
  const int stride_in_values2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  if (!initialize_matches(num_descriptors1, num_descriptors2, matches))
  {
    return 0;
  }

  typename DistanceMatrix::GpuMatrix distance_matrix_result =
    m_distance_matrix_ptr->compute_host_column_major_vectors_gpu(
      host_column_major_descriptors1,
      num_descriptors1,
      stride_in_values1,
      host_column_major_descriptors2,
      num_descriptors2,
      stride_in_values2,
      cuda_stream);

  return match_gpu(distance_matrix_result, matches, cuda_stream);
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_device_row_major_descriptors_gpu_implementation(
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_row_major_descriptors1,
  const int num_descriptors1,
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_row_major_descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  if (!initialize_matches(num_descriptors1, num_descriptors2, matches))
  {
    return 0;
  }

  typename DistanceMatrix::GpuMatrix distance_matrix_result =
    m_distance_matrix_ptr->compute_device_row_major_vectors_gpu(
      device_row_major_descriptors1,
      num_descriptors1,
      device_row_major_descriptors2,
      num_descriptors2,
      cuda_stream);

  return match_gpu(distance_matrix_result, matches, cuda_stream);
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_device_column_major_descriptors_gpu_implementation(
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_column_major_descriptors1,
  const int num_descriptors1,
  const int stride_in_values1,
  const typename DistanceMatrix::TemplatedVectorType * const __restrict__ device_column_major_descriptors2,
  const int num_descriptors2,
  const int stride_in_values2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  if (!initialize_matches(num_descriptors1, num_descriptors2, matches))
  {
    return 0;
  }

  typename DistanceMatrix::GpuMatrix distance_matrix_result =
    m_distance_matrix_ptr->compute_device_column_major_vectors_gpu(
      device_column_major_descriptors1,
      num_descriptors1,
      stride_in_values1,
      device_column_major_descriptors2,
      num_descriptors2,
      stride_in_values2,
      cuda_stream);

  return match_gpu(distance_matrix_result, matches, cuda_stream);
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_cpu(
  const typename DistanceMatrix::CpuMatrix & distance_matrix_result,
  std::vector<std::pair<int, int> > & matches)
{
  const int num_descriptors1 = distance_matrix_result.num_rows;
  const int num_descriptors2 = distance_matrix_result.num_cols;

  const typename DistanceMatrix::TemplatedDistanceType default_distance =
    static_cast<Derived *>(this)->functions().worst_distance();

  // Reset the existing_matches array.
  for (int i = 0; i < num_descriptors2; ++i)
  {
    m_existing_matches[i] = -1;
  }

  int num_matches = 0;
  m_match_distances_cpu.clear();

  for (int descriptor1_index = 0; descriptor1_index < num_descriptors1; ++descriptor1_index)
  {
    int best_matching_index = -1;
    typename DistanceMatrix::TemplatedDistanceType best_matching_distances[2] =
      {default_distance, default_distance};

    for (int descriptor2_index = 0; descriptor2_index < num_descriptors2; ++descriptor2_index)
    {
      const typename DistanceMatrix::TemplatedDistanceType distance =
        distance_matrix_result.at(descriptor1_index, descriptor2_index);

      if (static_cast<Derived *>(this)->functions().is_first_distance_better(
        distance, best_matching_distances[0]))
      {
        best_matching_distances[1] = best_matching_distances[0];
        best_matching_distances[0] = distance;
        best_matching_index = descriptor2_index;
      }
      else if (static_cast<Derived *>(this)->functions().is_first_distance_better(
        distance, best_matching_distances[1]))
      {
        best_matching_distances[1] = distance;
      }
    }

    if (static_cast<Derived *>(this)->functions().does_distance_fail_min_threshold(
      best_matching_distances[0]))
    {
      continue;
    }

    if (static_cast<Derived *>(this)->functions().is_distance_a_perfect_match(
      best_matching_distances[1]))
    {
      continue;
    }

    if (static_cast<Derived *>(this)->functions().do_distances_fail_max_ratio(
      best_matching_distances[0], best_matching_distances[1]))
    {
      continue;
    }

    const int existing_match = m_existing_matches[best_matching_index];

    // Test to see if a match already exists to the matching descriptor.
    if (existing_match == -1)
    {
      // There is no existing match to the matching descriptor, so create a new match.
      matches.push_back(std::make_pair(descriptor1_index, best_matching_index));

      m_match_distances_cpu.push_back(best_matching_distances[0]);

      // Store that the matching descriptor has been matched by this match by storing
      // the index of the match's entry in the matches list.
      m_existing_matches[best_matching_index] = num_matches;

      ++num_matches;
    }
    else
    {
      // Test to see if this match has a larger distance, and should therefore
      // replace the existing match.
      if (static_cast<Derived *>(this)->functions().is_first_distance_better(
        best_matching_distances[0], m_match_distances_cpu[existing_match]))
      {
        matches[existing_match].first = descriptor1_index;
        m_match_distances_cpu[existing_match] = best_matching_distances[0];
      }
    }
  }

  return num_matches;
}

template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_cpu_mutual_best(
  const typename DistanceMatrix::CpuMatrix & distance_matrix_result,
  std::vector<std::pair<int, int> > & matches)
{
  const int num_descriptors1 = distance_matrix_result.num_rows;
  const int num_descriptors2 = distance_matrix_result.num_cols;

  const typename DistanceMatrix::TemplatedDistanceType default_distance =
    static_cast<Derived *>(this)->functions().worst_distance();

  m_best_indices1.resize(num_descriptors1);
  m_best_indices2.resize(num_descriptors2);
  m_best_distances2.resize(num_descriptors2);

  for (int i = 0; i < num_descriptors2; ++i)
  {
    m_best_distances2[i] = default_distance;
  }

  for (int descriptor1_index = 0; descriptor1_index < num_descriptors1; ++descriptor1_index)
  {
    int best_matching_index = -1;
    typename DistanceMatrix::TemplatedDistanceType best_matching_distances[2] =
      {default_distance, default_distance};

    for (int descriptor2_index = 0; descriptor2_index < num_descriptors2; ++descriptor2_index)
    {
      const typename DistanceMatrix::TemplatedDistanceType distance =
        distance_matrix_result.at(descriptor1_index, descriptor2_index);

      if (static_cast<Derived *>(this)->functions().is_first_distance_better(
        distance, best_matching_distances[0]))
      {
        best_matching_distances[1] = best_matching_distances[0];
        best_matching_distances[0] = distance;
        best_matching_index = descriptor2_index;
      }
      else if (static_cast<Derived *>(this)->functions().is_first_distance_better(
        distance, best_matching_distances[1]))
      {
        best_matching_distances[1] = distance;
      }

      if (static_cast<Derived *>(this)->functions().is_first_distance_better(
        distance, m_best_distances2[descriptor2_index]))
      {
        m_best_distances2[descriptor2_index] = distance;
        m_best_indices2[descriptor2_index] = descriptor1_index;
      }
    }

    if (static_cast<Derived *>(this)->functions().does_distance_fail_min_threshold(
      best_matching_distances[0]))
    {
      m_best_indices1[descriptor1_index] = -1;
      continue;
    }

    if (static_cast<Derived *>(this)->functions().is_distance_a_perfect_match(
      best_matching_distances[1]))
    {
      m_best_indices1[descriptor1_index] = -1;
      continue;
    }

    if (static_cast<Derived *>(this)->functions().do_distances_fail_max_ratio(
      best_matching_distances[0], best_matching_distances[1]))
    {
      m_best_indices1[descriptor1_index] = -1;
      continue;
    }

    m_best_indices1[descriptor1_index] = best_matching_index;
  }

  int num_matches = 0;
  for (int i = 0; i < num_descriptors1; ++i)
  {
    const int idx1 = m_best_indices1[i];
    if (idx1 != -1)
    {
      const int idx2 = m_best_indices2[idx1];
      if (i == idx2)
      {
        ++num_matches;
        matches.push_back(std::make_pair(i, idx1));
      }
    }
  }

  return num_matches;
}

#if defined(__CUDACC__) // CUDA COMPILER
template<class Derived, class DistanceMatrix>
int descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::match_gpu(
  const typename DistanceMatrix::GpuMatrix & distance_matrix_result,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  initialize_gpu_match_results();

  const int num_descriptors1 = distance_matrix_result.num_rows;
  const int num_descriptors2 = distance_matrix_result.num_cols;

  const int num_threads_per_block = 256;

  const dim3 grid_dim(num_descriptors1);
  const dim3 block_dim(num_threads_per_block);

  descriptor_matcher_kernel<Derived, DistanceMatrix, MatchResult, num_threads_per_block><<<grid_dim, block_dim, 0, cuda_stream>>>(
    num_descriptors2,
    distance_matrix_result.device_matrix_ptr,
    distance_matrix_result.num_cols_stride,
    m_device_match_results,
    static_cast<Derived *>(this)->functions());

  // Reset the m_existing_matches array while the CUDA kernel is running.
  for (int i = 0; i < num_descriptors2; ++i)
  {
    m_existing_matches[i] = -1;
  }

  // Copy the match results from the GPU to the pinned memory buffer.
  m_match_results_buffer.memcpy_from_device_async(
    m_device_match_results, num_descriptors1, cuda_stream);
  cuda_helper::synchronize_stream(cuda_stream);

  int num_matches = 0;
  for (int i = 0; i < num_descriptors1; ++i)
  {
    // Get the index of the matching descriptor.
    const int matching_descriptor_index =
      m_match_results_buffer.host_ptr()[i].matching_descriptor_index;

    // Test to see if this descriptor (i) matched to another descriptor.
    if (matching_descriptor_index >= 0)
    {
      const int existing_match = m_existing_matches[matching_descriptor_index];

      // Test to see if a match already exists to the matching descriptor.
      if (existing_match == -1)
      {
        // There is no existing match to the matching descriptor, so create a new match.
        matches.push_back(std::make_pair(i, matching_descriptor_index));

        // Store that the matching descriptor has been matched by this match by storing
        // the index of the match's entry in the matches list.
        m_existing_matches[matching_descriptor_index] = num_matches;

        ++num_matches;
      }
      else
      {
        // Get the index of the other descriptor that already matched to the matching descriptor.
        const int existing_match_descriptor_index = matches[existing_match].first;

        // Test to see if this match has a better distance, and should therefore
        // replace the existing match.
        if (static_cast<Derived *>(this)->functions().is_first_distance_better(
          m_match_results_buffer.host_ptr()[i].matching_distance,
          m_match_results_buffer.host_ptr()[existing_match_descriptor_index].matching_distance))
        {
          // Overwrite the descriptor index that matched to the matching descriptor.
          matches[existing_match].first = i;
        }
      }
    }
  }

  return num_matches;
}
#endif // CUDA COMPILER

template<class Derived, class DistanceMatrix>
bool descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_matches(
  const int num_descriptors1,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches)
{
  matches.clear();

  if (num_descriptors1 < 2 || num_descriptors2 < 2)
  {
    return false;
  }

  return true;
}

template<class Derived, class DistanceMatrix>
void descriptor_matcher::DescriptorMatcherBase<Derived, DistanceMatrix>::initialize_gpu_match_results()
{
  if (m_device_match_results != NULL)
  {
    return;
  }

  m_match_results_buffer.alloc(
    m_max_num_descriptors,
    cuda_helper::cuda_memcpy_buffer_type::Pinned);

  CUDA_CALL(cudaMalloc(&m_device_match_results,
    m_max_num_descriptors * sizeof(MatchResult)));

  if (m_device_match_results == NULL)
  {
    std::cerr << "ERROR: initialize_gpu_match_results() - failed to allocate memory," << std::endl;
    std::cerr << "  requested " << m_max_num_descriptors << " values = " <<
      m_max_num_descriptors * sizeof(MatchResult) << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
}

#endif // DESCRIPTOR_MATCHER_BASE_H
