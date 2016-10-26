#pragma once
#ifndef DISTANCE_MATRIX_UTIL_H
#define DISTANCE_MATRIX_UTIL_H

#include <assert/assert.h>
#include <core/math.h>
#if defined(__CUDACC__) // CUDA COMPILER
  #include <distance_matrix/src/distance_matrix_util_kernels.cuh>
#endif

namespace distance_matrix {

template<typename T>
int convert_row_major_to_column_major_cpu(
  const T * const __restrict__ host_row_major_vectors,
  const int num_vectors,
  const int num_values_per_vector,
  T * const __restrict__ host_column_major_vectors,
  const int column_major_vectors_size, // max num elements
  const int byte_alignment)
{
  const int num_bytes_per_column = num_vectors * sizeof(T);
  const int column_stride_in_bytes = core::math::round_up_to_multiple(
    num_bytes_per_column, byte_alignment);
  const int column_stride_in_elements = column_stride_in_bytes / sizeof(T);

  ASSERT(column_stride_in_bytes % sizeof(T) == 0);
  ASSERT(column_stride_in_elements * num_values_per_vector <= column_major_vectors_size);

  for (int i = 0; i < num_vectors; ++i)
  {
    for (int j = 0; j < num_values_per_vector; ++j)
    {
      host_column_major_vectors[j * column_stride_in_elements + i] =
        host_row_major_vectors[i * num_values_per_vector + j];
    }
  }

  return column_stride_in_elements;
}

template<typename T>
void convert_column_major_to_row_major_cpu(
  const T * const __restrict__ host_column_major_vectors,
  const int num_vectors,
  const int num_values_per_vector,
  const int column_stride_in_elements,
  T * const __restrict__ host_row_major_vectors)
{
  for (int j = 0; j < num_values_per_vector; ++j)
  {
    for (int i = 0; i < num_vectors; ++i)
    {
      host_row_major_vectors[i * num_values_per_vector + j] =
        host_column_major_vectors[j * column_stride_in_elements + i];
    }
  }
}

#if defined(__CUDACC__) // CUDA COMPILER
template<typename T>
int convert_row_major_to_column_major_gpu(
  const T * const __restrict__ device_row_major_vectors,
  const int num_vectors,
  const int num_values_per_vector,
  T * const __restrict__ device_column_major_vectors,
  const int column_major_vectors_size, // max num elements
  const int byte_alignment,
  cudaStream_t cuda_stream = 0)
{
  const int num_bytes_per_column = num_vectors * sizeof(T);
  const int column_stride_in_bytes = core::math::round_up_to_multiple(
    num_bytes_per_column, byte_alignment);
  const int column_stride_in_elements = column_stride_in_bytes / sizeof(T);

  ASSERT(column_stride_in_bytes % sizeof(T) == 0);
  ASSERT(column_stride_in_elements * num_values_per_vector <= column_major_vectors_size);

  const int num_threads_per_block = 256;
  const int num_vectors_per_block = num_threads_per_block / num_values_per_vector;

  const int block_dim_x = num_values_per_vector;
  const int block_dim_y = num_vectors_per_block;

  const int num_blocks_x = (num_vectors + num_vectors_per_block - 1) / num_vectors_per_block;
  const int num_blocks_y = 1;

  const dim3 grid_dim(num_blocks_x, num_blocks_y);
  const dim3 block_dim(block_dim_x, block_dim_y);

  convert_row_major_to_column_major_kernel<T><<<grid_dim, block_dim, 0, cuda_stream>>>(
    device_row_major_vectors,
    num_vectors,
    num_values_per_vector,
    device_column_major_vectors,
    column_stride_in_elements);

  return column_stride_in_elements;
}
#endif // CUDA COMPILER

void convert_row_major_uchar_to_float_cpu(
  const unsigned char * const __restrict__ host_row_major_uchars,
  const int num_values_per_row,
  const int num_rows,
  float * const __restrict__ host_row_major_floats,
  const int scale_factor_per_value);

/* Unused
void convert_column_major_uchar_to_float_cpu(
  const unsigned char * const __restrict__ host_column_major_uchars,
  const int num_values_per_column,
  const int num_columns,
  const int stride,
  float * const __restrict__ host_column_major_floats,
  const int scale_factor_per_value);
*/

void convert_column_major_uchar_to_float_gpu(
  const unsigned char * const __restrict__ device_column_major_uchars,
  const int num_values_per_column,
  const int num_columns,
  const int stride,
  float * const __restrict__ device_column_major_floats,
  const int scale_factor_per_value,
  cudaStream_t cuda_stream);

void normalize_column_major_vectors_gpu(
  float * device_column_major_vectors,
  const int num_values_per_column,
  const int num_columns,
  const int stride,
  cudaStream_t cuda_stream);

} // namespace distance_matrix

#endif // DISTANCE_MATRIX_UTIL_H
