#pragma once
#ifndef DESCRIPTOR_MATCHER_KERNELS_CUH
#define DESCRIPTOR_MATCHER_KERNELS_CUH

#include <descriptor_matcher/src/descriptor_matcher_base.h>
#include <cuda_helper/cuda_macros.h>

template<
  class Derived,
  class DistanceMatrix,
  int num_descriptors1_per_block,
  int num_descriptors2_per_block>
__global__ void compute_best_descriptor_match_per_column(
  const int num_descriptors1,
  const int num_descriptors2,
  const typename DistanceMatrix::TemplatedDistanceType * const __restrict__ distance_matrix,
  const int distance_matrix_stride,
  int * best_descriptor_match_per_column,
  const typename Derived::Functions device_functions)
{
  // TODO(jheinly)
}

template<
  class Derived,
  class DistanceMatrix,
  class MatchResult,
  int num_threads_per_block>
__global__ void descriptor_matcher_kernel(
  const int num_descriptors2,
  const typename DistanceMatrix::TemplatedDistanceType * const __restrict__ distance_matrix,
  const int distance_matrix_stride,
  MatchResult * __restrict__ match_results,
  const typename Derived::Functions device_functions)
{
  // Each block processes the distances from one descriptor in descriptors1
  // (hence we don't need to know num_descriptors1).

  const int half_block_size = num_threads_per_block >> 1;

  // Store the best matching distances for each thread.
  __shared__ typename DistanceMatrix::TemplatedDistanceType best_matching_distances[num_threads_per_block];

  // This will be used for a parallel reduction to determine the single best match.
  __shared__ typename DistanceMatrix::TemplatedDistanceType reduction_distances[half_block_size];
  __shared__ int reduction_indices[half_block_size];

  __shared__ int best_reduction_index;
  __shared__ typename DistanceMatrix::TemplatedDistanceType best_matching_distance;
  __shared__ int best_matching_index;

  const int thread_index = threadIdx.x;

  // Store this thread's best matching index and second best distance in a local register.
  int best_matching_index_current = -1;
  typename DistanceMatrix::TemplatedDistanceType best_matching_distance2_current =
    device_functions.worst_distance();

  // Get a pointer to the distances for one of the descriptors in descriptors1.
  const int descriptor1_index = blockIdx.x;
  const typename DistanceMatrix::TemplatedDistanceType * const __restrict__ distance_matrix_row =
    distance_matrix + MUL_INT24(descriptor1_index, distance_matrix_stride);

  // Initialize the first best matching distances.
  int descriptor2_index = thread_index;
  if (descriptor2_index < num_descriptors2)
  {
    best_matching_distances[thread_index] = distance_matrix_row[descriptor2_index];
    best_matching_index_current = descriptor2_index;
  }

  // Loop across the remaining distances in this row, and keep track of the first and second best matches.
  const int num_block_iterations =
    (num_descriptors2 + num_threads_per_block - 1) / num_threads_per_block;
  for (int block_iteration = 1; block_iteration < num_block_iterations; ++block_iteration)
  {
    descriptor2_index += num_threads_per_block;

    if (descriptor2_index < num_descriptors2)
    {
      const typename DistanceMatrix::TemplatedDistanceType distance = distance_matrix_row[descriptor2_index];
      if (device_functions.is_first_distance_better(
        distance, best_matching_distances[thread_index]))
      {
        best_matching_distance2_current = best_matching_distances[thread_index];
        best_matching_distances[thread_index] = distance;
        best_matching_index_current = descriptor2_index;
      }
      else if (device_functions.is_first_distance_better(
        distance, best_matching_distance2_current))
      {
        best_matching_distance2_current = distance;
      }
    }
  }

  __syncthreads();

  // Run the first parallel reduction on best_matching_distances.
  if (num_descriptors2 < num_threads_per_block)
  {
    // Handle the case where we need to make sure that there are valid values during the reduction.
    if (thread_index < half_block_size)
    {
      const int reduction_index = thread_index + half_block_size;
      if (reduction_index < num_descriptors2)
      {
        const typename DistanceMatrix::TemplatedDistanceType distance1 = best_matching_distances[thread_index];
        const typename DistanceMatrix::TemplatedDistanceType distance2 = best_matching_distances[reduction_index];
        const bool better = device_functions.is_first_distance_better(distance1, distance2);
        reduction_distances[thread_index] = better ? distance1 : distance2;
        reduction_indices[thread_index] = better ? thread_index : reduction_index;
      }
      else if (thread_index < num_descriptors2)
      {
        reduction_distances[thread_index] = best_matching_distances[thread_index];
        reduction_indices[thread_index] = thread_index;
      }
    }
    __syncthreads();

    // Run the remaining reduction iterations to find the first best distance.
    for (int i = half_block_size >> 1; i >= 1; i >>= 1)
    {
      if (thread_index < i)
      {
        const int reduction_index = thread_index + i;
        if (reduction_index < num_descriptors2)
        {
          const typename DistanceMatrix::TemplatedDistanceType distance1 = reduction_distances[thread_index];
          const typename DistanceMatrix::TemplatedDistanceType distance2 = reduction_distances[reduction_index];

          // We only need to update the reduction values if the second
          // distance is the better one.
          if (device_functions.is_first_distance_better(distance2, distance1))
          {
            reduction_distances[thread_index] = distance2;
            reduction_indices[thread_index] = reduction_indices[reduction_index];
          }
        }
      }
      __syncthreads();
    }
  }
  else
  {
    // Run the first reduction iteration to initialize the reduction_* variables.
    if (thread_index < half_block_size)
    {
      const int reduction_index = thread_index + half_block_size;
      const typename DistanceMatrix::TemplatedDistanceType distance1 = best_matching_distances[thread_index];
      const typename DistanceMatrix::TemplatedDistanceType distance2 = best_matching_distances[reduction_index];
      const bool better = device_functions.is_first_distance_better(distance1, distance2);
      reduction_distances[thread_index] = better ? distance1 : distance2;
      reduction_indices[thread_index] = better ? thread_index : reduction_index;
    }
    __syncthreads();

    // Run the remaining reduction iterations to find the first best distance.
    for (int i = half_block_size >> 1; i >= 1; i >>= 1)
    {
      if (thread_index < i)
      {
        const int reduction_index = thread_index + i;
        const typename DistanceMatrix::TemplatedDistanceType distance1 = reduction_distances[thread_index];
        const typename DistanceMatrix::TemplatedDistanceType distance2 = reduction_distances[reduction_index];

        // We only need to update the reduction values if the second
        // distance is the better one.
        if (device_functions.is_first_distance_better(distance2, distance1))
        {
          reduction_distances[thread_index] = distance2;
          reduction_indices[thread_index] = reduction_indices[reduction_index];
        }
      }
      __syncthreads();
    }
  }

  if (thread_index == 0)
  {
    if (device_functions.does_distance_fail_min_threshold(
      reduction_distances[0]))
    {
      best_reduction_index = -1;
      match_results[descriptor1_index].matching_descriptor_index = -1;
    }
    else
    {
      best_matching_distance = reduction_distances[0];
      best_reduction_index = reduction_indices[0];
    }
  }
  __syncthreads();

  // If we need to verify the second best match.
  if (best_reduction_index >= 0)
  {
    if (thread_index == best_reduction_index)
    {
      best_matching_index = best_matching_index_current;
      best_matching_distances[thread_index] = best_matching_distance2_current;
    }
    __syncthreads();

    // Run the second parallel reduction to find the second best matching distance.
    if (num_descriptors2 < num_threads_per_block)
    {
      // Handle the case where we need to make sure that there are valid values during the reduction.
      if (thread_index < half_block_size)
      {
        const int reduction_index = thread_index + half_block_size;
        if (reduction_index < num_descriptors2)
        {
          const typename DistanceMatrix::TemplatedDistanceType distance1 = best_matching_distances[thread_index];
          const typename DistanceMatrix::TemplatedDistanceType distance2 = best_matching_distances[reduction_index];
          const bool better = device_functions.is_first_distance_better(distance1, distance2);
          reduction_distances[thread_index] = better ? distance1 : distance2;
        }
        else if (thread_index < num_descriptors2)
        {
          reduction_distances[thread_index] = best_matching_distances[thread_index];
        }
      }
      __syncthreads();

      // Run the remaining reduction iterations to find the first best distance.
      for (int i = half_block_size >> 1; i >= 1; i >>= 1)
      {
        if (thread_index < i)
        {
          const int reduction_index = thread_index + i;
          if (reduction_index < num_descriptors2)
          {
            const typename DistanceMatrix::TemplatedDistanceType distance1 = reduction_distances[thread_index];
            const typename DistanceMatrix::TemplatedDistanceType distance2 = reduction_distances[reduction_index];

            // We only need to update the reduction values if the second
            // distance is the better one.
            if (device_functions.is_first_distance_better(distance2, distance1))
            {
              reduction_distances[thread_index] = distance2;
            }
          }
        }
        __syncthreads();
      }
    }
    else
    {
      // Run the first reduction iteration to initialize the reduction_* variables.
      if (thread_index < half_block_size)
      {
        const int reduction_index = thread_index + half_block_size;
        const typename DistanceMatrix::TemplatedDistanceType distance1 = best_matching_distances[thread_index];
        const typename DistanceMatrix::TemplatedDistanceType distance2 = best_matching_distances[reduction_index];
        const bool better = device_functions.is_first_distance_better(distance1, distance2);
        reduction_distances[thread_index] = better ? distance1 : distance2;
      }
      __syncthreads();

      // Run the remaining reduction iterations to find the first best distance.
      for (int i = half_block_size >> 1; i >= 1; i >>= 1)
      {
        if (thread_index < i)
        {
          const int reduction_index = thread_index + i;
          const typename DistanceMatrix::TemplatedDistanceType distance1 = reduction_distances[thread_index];
          const typename DistanceMatrix::TemplatedDistanceType distance2 = reduction_distances[reduction_index];

          // We only need to update the reduction values if the second
          // distance is the better one.
          if (device_functions.is_first_distance_better(distance2, distance1))
          {
            reduction_distances[thread_index] = distance2;
          }
        }
        __syncthreads();
      }
    }

    if (thread_index == 0)
    {
      if (device_functions.is_distance_a_perfect_match(reduction_distances[0]))
      {
        match_results[descriptor1_index].matching_descriptor_index = -1;
      }
      else
      {
        if (device_functions.do_distances_fail_max_ratio(
          best_matching_distance, reduction_distances[0]))
        {
          match_results[descriptor1_index].matching_descriptor_index = -1;
        }
        else
        {
          match_results[descriptor1_index] =
            MatchResult(best_matching_index, best_matching_distance);
          //match_results[descriptor1_index].matching_descriptor_index = best_matching_index;
          //match_results[descriptor1_index].matching_distance = best_matching_distance;
        }
      }
    }
  }
}

#endif // DESCRIPTOR_MATCHER_KERNELS_CUH
