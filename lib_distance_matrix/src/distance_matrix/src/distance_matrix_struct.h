#pragma once
#ifndef DISTANCE_MATRIX_STRUCT_H
#define DISTANCE_MATRIX_STRUCT_H

#include <cuda_helper/cuda_helper.h>

namespace distance_matrix {

template<typename T>
struct CpuMatrixStruct
{
  public:
    CpuMatrixStruct(
      const T * const host_matrix_ptr,
      const int num_rows,
      const int num_cols,
      const int num_cols_stride,
      const bool should_free_memory = false)
    : host_matrix_ptr(host_matrix_ptr),
      num_rows(num_rows),
      num_cols(num_cols),
      num_cols_stride(num_cols_stride),
      m_should_free_memory(should_free_memory)
    {}

    ~CpuMatrixStruct()
    {
      if (m_should_free_memory)
      {
        delete [] host_matrix_ptr;
      }
    }

    inline const T at(const int row, const int col) const
    { return host_matrix_ptr[row * num_cols_stride + col]; }

    const T * const host_matrix_ptr;
    const int num_rows;
    const int num_cols;
    const int num_cols_stride;

  private:
    CpuMatrixStruct & operator=(const CpuMatrixStruct &);
    const bool m_should_free_memory;
};

template<typename T>
struct GpuMatrixStruct
{
  public:
    GpuMatrixStruct(
      const T * const device_matrix_ptr,
      const int num_rows,
      const int num_cols,
      const int num_cols_stride)
    : device_matrix_ptr(device_matrix_ptr),
      num_rows(num_rows),
      num_cols(num_cols),
      num_cols_stride(num_cols_stride)
    {}

    const T * const device_matrix_ptr;
    const int num_rows;
    const int num_cols;
    const int num_cols_stride;

  private:
    GpuMatrixStruct & operator=(const GpuMatrixStruct &);
};

template<typename T>
CpuMatrixStruct<T> copy_gpu_matrix_to_cpu(
  const GpuMatrixStruct<T> & gpu_matrix)
{
  const int num_elements = gpu_matrix.num_rows * gpu_matrix.num_cols_stride;

  T * host_ptr = new T[num_elements];

  CUDA_CALL(cudaMemcpy(
    host_ptr, // destination
    gpu_matrix.device_matrix_ptr, // source
    num_elements * sizeof(T), // num bytes
    cudaMemcpyDeviceToHost));

  return CpuMatrixStruct<T>(
    host_ptr,
    gpu_matrix.num_rows,
    gpu_matrix.num_cols,
    gpu_matrix.num_cols_stride,
    true); // should free memory
}

} // namespace distance_matrix

#endif // DISTANCE_MATRIX_STRUCT_H
