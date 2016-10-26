#pragma once
#ifndef DOT_PRODUCT_MATRIX_UCHAR_H
#define DOT_PRODUCT_MATRIX_UCHAR_H

#include <distance_matrix/src/distance_matrix_base.h>

namespace distance_matrix {

class DotProductMatrixUChar : public DistanceMatrixBase<DotProductMatrixUChar, unsigned char, float>
{
  public:
    explicit DotProductMatrixUChar(
      const int max_num_vectors1,
      const int max_num_vectors2,
      const int num_uchars_per_vector,
      const int scale_factor_per_value);

    ~DotProductMatrixUChar();

    const CpuMatrix compute_host_row_major_vectors_cpu(
      const unsigned char * const __restrict__ host_row_major_vectors1,
      const int num_vectors1,
      const unsigned char * const __restrict__ host_row_major_vectors2,
      const int num_vectors2);

    const CpuMatrix compute_host_column_major_vectors_cpu(
      const unsigned char * const __restrict__ host_column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const unsigned char * const __restrict__ host_column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2);

    const GpuMatrix compute_host_row_major_vectors_gpu(
      const unsigned char * const __restrict__ host_row_major_vectors1,
      const int num_vectors1,
      const unsigned char * const __restrict__ host_row_major_vectors2,
      const int num_vectors2,
      cudaStream_t cuda_stream = 0);

    const GpuMatrix compute_host_column_major_vectors_gpu(
      const unsigned char * const __restrict__ host_column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const unsigned char * const __restrict__ host_column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2,
      cudaStream_t cuda_stream = 0);

    const GpuMatrix compute_device_row_major_vectors_gpu(
      const unsigned char * const __restrict__ device_row_major_vectors1,
      const int num_vectors1,
      const unsigned char * const __restrict__ device_row_major_vectors2,
      const int num_vectors2,
      cudaStream_t cuda_stream = 0);

    const GpuMatrix compute_device_column_major_vectors_gpu(
      const unsigned char * const __restrict__ device_column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const unsigned char * const __restrict__ device_column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2,
      cudaStream_t cuda_stream = 0);

    const CpuMatrix compute_cpu(
      const unsigned char * const __restrict__ host_row_major_vectors1,
      const int num_vectors1,
      const unsigned char * const __restrict__ host_row_major_vectors2,
      const int num_vectors2);

    const GpuMatrix compute_gpu(
      const unsigned char * const __restrict__ device_column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const unsigned char * const __restrict__ device_column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2,
      cudaStream_t cuda_stream);

  private:
    DotProductMatrixUChar(const DotProductMatrixUChar &);
    DotProductMatrixUChar & operator=(const DotProductMatrixUChar &);

    float * m_host_vectors_float;
    float * m_device_vectors_float;
    cublasHandle_t m_cublas_handle;
    const int m_scale_factor_per_value;
};

} // namespace distance_matrix

#endif // DOT_PRODUCT_MATRIX_UCHAR_H
