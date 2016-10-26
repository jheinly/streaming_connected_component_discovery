#pragma once
#ifndef DISTANCE_MATRIX_UTIL_KERNELS_CUH
#define DISTANCE_MATRIX_UTIL_KERNELS_CUH

#include <cuda_helper/cuda_macros.h>

template<typename T>
__global__ void convert_row_major_to_column_major_kernel(
  const T * const __restrict__ row_major_vectors,
  const int num_vectors,
  const int num_values_per_vector,
  T * const __restrict__ column_major_vectors,
  const int column_stride_in_elements)
{
  const int vector_index = MUL_INT24(blockIdx.x, blockDim.y) + threadIdx.y;
  if (vector_index < num_vectors)
  {
    const int dim_index = threadIdx.x;

    column_major_vectors[MUL_INT24(dim_index, column_stride_in_elements) + vector_index] =
      row_major_vectors[MUL_INT24(vector_index, num_values_per_vector) + dim_index];
  }
}

template<int num_threads_per_block, int scale_factor_per_value>
__global__ void convert_column_major_uchar_to_float_kernel(
  const unsigned char * const __restrict__ column_major_uchars,
  const int num_uchars_per_column,
  const int stride,
  float * __restrict__ column_major_floats)
{
  const int idx_in_column = MUL_INT24(blockIdx.y, num_threads_per_block) + threadIdx.x;
  if (idx_in_column < num_uchars_per_column)
  {
    const int idx = MUL_INT24(blockIdx.x, stride) + idx_in_column;

    const float inv_scale = 1.0f / static_cast<float>(scale_factor_per_value);

    column_major_floats[idx] =
      static_cast<float>(column_major_uchars[idx]) * inv_scale;
  }
}

template<int block_dim_x, int block_dim_y, int num_values_per_column>
__global__ void normalize_column_major_vectors_kernel(
  float * device_column_major_vectors,
  const int num_columns,
  const int stride)
{
  const int vector_index = MUL_INT24(blockIdx.x, block_dim_x) + threadIdx.x;
  int index_within_vector = threadIdx.y;

  const int num_iterations = num_values_per_column / block_dim_y;

  __shared__ float magnitudes[block_dim_y][block_dim_x];
  magnitudes[threadIdx.y][threadIdx.x] = 0.0f;

  if (vector_index < num_columns)
  {
    for (int i = 0; i < num_iterations; ++i)
    {
      const float val =
        device_column_major_vectors[MUL_INT24(index_within_vector, stride) + vector_index];
      magnitudes[threadIdx.y][threadIdx.x] += val * val;
      index_within_vector += block_dim_y;
    }
  }

  __syncthreads();

  if (vector_index < num_columns)
  {
    if (threadIdx.y == 0)
    {
      for (int i = 1; i < block_dim_y; ++i)
      {
        magnitudes[0][threadIdx.x] += magnitudes[i][threadIdx.x];
      }
      // Compute inverse magnitude.
      magnitudes[0][threadIdx.x] = rsqrtf(magnitudes[0][threadIdx.x]);
    }
  }

  __syncthreads();

  if (vector_index < num_columns)
  {
    index_within_vector = threadIdx.y;
    for (int i = 0; i < num_iterations; ++i)
    {
      device_column_major_vectors[MUL_INT24(index_within_vector, stride) + vector_index] *=
        magnitudes[0][threadIdx.x];
      index_within_vector += block_dim_y;
    }
  }
}

#endif // DISTANCE_MATRIX_UTIL_KERNELS_CUH
