#include <distance_matrix/src/distance_matrix_util.h>
#include <cuda_helper/cuda_helper.h>
#include <iostream>
#include <cstdlib>

void distance_matrix::convert_row_major_uchar_to_float_cpu(
  const unsigned char * const __restrict__ host_row_major_uchars,
  const int num_values_per_row,
  const int num_rows,
  float * const __restrict__ host_row_major_floats,
  const int scale_factor_per_value)
{
  const float inv_scale = 1.0f / static_cast<float>(scale_factor_per_value);
  const int num_values = num_values_per_row * num_rows;

  const unsigned char * __restrict__ current_uchar = host_row_major_uchars;
  float * __restrict__ current_float = host_row_major_floats;

  for (int i = 0; i < num_values; ++i)
  {
    *current_float = static_cast<float>(*current_uchar) * inv_scale;
    ++current_uchar;
    ++current_float;
  }
}

/* Unused
void distance_matrix::convert_column_major_uchar_to_float_cpu(
  const unsigned char * const __restrict__ host_column_major_uchars,
  const int num_values_per_column,
  const int num_columns,
  const int stride,
  float * const __restrict__ host_column_major_floats,
  const int scale_factor_per_value)
{
  ASSERT(stride >= num_values_per_column);
  const int column_step = stride - num_values_per_column;

  const float inv_scale = 1.0f / static_cast<float>(scale_factor_per_value);

  const unsigned char * __restrict__ current_uchar = host_column_major_uchars;
  float * __restrict__ current_float = host_column_major_floats;
  for (int i = 0; i < num_columns; ++i)
  {
    for (int j = 0; j < num_values_per_column; ++j)
    {
      *current_float = static_cast<float>(*current_uchar) * inv_scale;
      ++current_uchar;
      ++current_float;
    }
    current_float += column_step;
    current_uchar += column_step;
  }
}
*/

void distance_matrix::convert_column_major_uchar_to_float_gpu(
  const unsigned char * const __restrict__ device_column_major_uchars,
  const int num_values_per_column,
  const int num_columns,
  const int stride,
  float * const __restrict__ device_column_major_floats,
  const int scale_factor_per_value,
  cudaStream_t cuda_stream)
{
  const int num_threads_per_block = 256;

  const int block_dim_x = num_threads_per_block;
  const int block_dim_y = 1;

  const int num_blocks_x = num_columns;
  const int num_blocks_y = core::math::divide_and_round_up(num_values_per_column, num_threads_per_block);

  const dim3 block_dim(block_dim_x, block_dim_y);
  const dim3 grid_dim(num_blocks_x, num_blocks_y);

  switch (scale_factor_per_value)
  {
    case 255:
      convert_column_major_uchar_to_float_kernel<num_threads_per_block, 255><<<grid_dim, block_dim, 0, cuda_stream>>>(
        device_column_major_uchars,
        num_values_per_column,
        stride,
        device_column_major_floats);
      break;
    case 256:
      convert_column_major_uchar_to_float_kernel<num_threads_per_block, 256><<<grid_dim, block_dim, 0, cuda_stream>>>(
        device_column_major_uchars,
        num_values_per_column,
        stride,
        device_column_major_floats);
      break;
    case 512:
      convert_column_major_uchar_to_float_kernel<num_threads_per_block, 512><<<grid_dim, block_dim, 0, cuda_stream>>>(
        device_column_major_uchars,
        num_values_per_column,
        stride,
        device_column_major_floats);
      break;
    default:
      std::cerr << "ERROR: convert_column_major_uchar_to_float_gpu() - no support"
        << " for scale factor of " << scale_factor_per_value << std::endl;
      exit(EXIT_FAILURE);
  }
}

void distance_matrix::normalize_column_major_vectors_gpu(
  float * device_column_major_vectors,
  const int num_values_per_column,
  const int num_columns,
  const int stride,
  cudaStream_t cuda_stream)
{
  ASSERT(stride >= num_values_per_column);
  ASSERT(num_values_per_column % 16 == 0);

  const int block_dim_x = 16;
  const int block_dim_y = 16;

  const int num_blocks_x = core::math::divide_and_round_up(num_columns, block_dim_x);
  const int num_blocks_y = 1;

  const dim3 block_dim(block_dim_x, block_dim_y);
  const dim3 grid_dim(num_blocks_x, num_blocks_y);

  switch (num_values_per_column)
  {
    case 64:
      normalize_column_major_vectors_kernel<block_dim_x, block_dim_y, 64><<<grid_dim, block_dim, 0, cuda_stream>>>(
        device_column_major_vectors,
        num_columns,
        stride);
      break;
    case 128:
      normalize_column_major_vectors_kernel<block_dim_x, block_dim_y, 128><<<grid_dim, block_dim, 0, cuda_stream>>>(
        device_column_major_vectors,
        num_columns,
        stride);
      break;
    case 256:
      normalize_column_major_vectors_kernel<block_dim_x, block_dim_y, 256><<<grid_dim, block_dim, 0, cuda_stream>>>(
        device_column_major_vectors,
        num_columns,
        stride);
      break;
    default:
      std::cerr << "ERROR: normalize_column_major_vectors_gpu() - no support"
        << " for num values per column of " << num_values_per_column << std::endl;
      exit(EXIT_FAILURE);
  }
}
