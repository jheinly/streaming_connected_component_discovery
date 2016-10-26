#pragma once
#ifndef DISTANCE_MATRIX_BASE_H
#define DISTANCE_MATRIX_BASE_H

#include <distance_matrix/src/distance_matrix_struct.h>
#include <distance_matrix/src/distance_matrix_util.h>
#include <assert/assert.h>
#include <cuda_helper/cuda_helper.h>
#include <cuda_helper/cuda_memcpy_buffer.h>
#include <iostream>
#include <cstdlib>

namespace distance_matrix {

template<class Derived, typename VectorType, typename DistanceType>
class DistanceMatrixBase
{
  public:
    typedef CpuMatrixStruct<DistanceType> CpuMatrix;
    typedef GpuMatrixStruct<DistanceType> GpuMatrix;

    typedef VectorType TemplatedVectorType;
    typedef DistanceType TemplatedDistanceType;
    
    DistanceMatrixBase(
      const int max_num_vectors1,
      const int max_num_vectors2,
      const int num_values_per_vector);

    virtual ~DistanceMatrixBase();

    void initialize_for_computing_host_row_major_vectors_cpu();
    void initialize_for_computing_host_column_major_vectors_cpu();
    void initialize_for_computing_host_row_major_vectors_gpu();
    void initialize_for_computing_host_column_major_vectors_gpu();
    void initialize_for_computing_device_row_major_vectors_gpu();
    void initialize_for_computing_device_column_major_vectors_gpu();

  protected:
    const CpuMatrix compute_host_row_major_vectors_cpu_implementation(
      const VectorType * const __restrict__ host_row_major_vectors1,
      const int num_vectors1,
      const VectorType * const __restrict__ host_row_major_vectors2,
      const int num_vectors2);

    const CpuMatrix compute_host_column_major_vectors_cpu_implementation(
      const VectorType * const __restrict__ host_column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const VectorType * const __restrict__ host_column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2);

    const GpuMatrix compute_host_row_major_vectors_gpu_implementation(
      const VectorType * const __restrict__ host_row_major_vectors1,
      const int num_vectors1,
      const VectorType * const __restrict__ host_row_major_vectors2,
      const int num_vectors2,
      cudaStream_t cuda_stream = 0);

    const GpuMatrix compute_host_column_major_vectors_gpu_implementation(
      const VectorType * const __restrict__ host_column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const VectorType * const __restrict__ host_column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2,
      cudaStream_t cuda_stream = 0);

    const GpuMatrix compute_device_row_major_vectors_gpu_implementation(
      const VectorType * const __restrict__ device_row_major_vectors1,
      const int num_vectors1,
      const VectorType * const __restrict__ device_row_major_vectors2,
      const int num_vectors2,
      cudaStream_t cuda_stream = 0);

    const GpuMatrix compute_device_column_major_vectors_gpu_implementation(
      const VectorType * const __restrict__ device_column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const VectorType * const __restrict__ device_column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2,
      cudaStream_t cuda_stream = 0);

    const int m_max_num_vectors1;
    const int m_max_num_vectors2;
    const int m_num_values_per_vector;

    VectorType * m_host_vectors;
    cuda_helper::CudaMemcpyBuffer<VectorType> m_host_vectors_buffer;
    VectorType * m_device_vectors;
    DistanceType * m_host_distance_matrix;
    DistanceType * m_device_distance_matrix;

  private:
    DistanceMatrixBase(const DistanceMatrixBase &);
    DistanceMatrixBase & operator=(const DistanceMatrixBase &);

    void enforce_asserts(
      const VectorType * const __restrict__ row_major_vectors1,
      const int num_vectors1,
      const VectorType * const __restrict__ row_major_vectors2,
      const int num_vectors2);

    void enforce_asserts(
      const VectorType * const __restrict__ column_major_vectors1,
      const int num_vectors1,
      const int stride_in_values1,
      const VectorType * const __restrict__ column_major_vectors2,
      const int num_vectors2,
      const int stride_in_values2);
    
    void initialize_cpu_host_vectors();
    void initialize_cpu_host_distance_matrix();
    void initialize_gpu_host_vectors();
    void initialize_gpu_device_vectors();
    void initialize_gpu_device_distance_matrix();

    static const int byte_alignment = 256;
};

} // namespace distance_matrix

template<class Derived, typename VectorType, typename DistanceType>
distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::DistanceMatrixBase(
  const int max_num_vectors1,
  const int max_num_vectors2,
  const int num_values_per_vector)
: m_max_num_vectors1(max_num_vectors1),
  m_max_num_vectors2(max_num_vectors2),
  m_num_values_per_vector(num_values_per_vector),
  m_host_vectors(NULL),
  m_device_vectors(NULL),
  m_host_distance_matrix(NULL),
  m_device_distance_matrix(NULL)
{}

template<class Derived, typename VectorType, typename DistanceType>
distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::~DistanceMatrixBase()
{
  if (m_host_vectors != NULL)
  {
    delete [] m_host_vectors;
    m_host_vectors = NULL;
  }
  if (m_device_vectors != NULL)
  {
    CUDA_CALL(cudaFree(m_device_vectors));
    m_device_vectors = NULL;
  }
  if (m_host_distance_matrix != NULL)
  {
    delete [] m_host_distance_matrix;
    m_host_distance_matrix = NULL;
  }
  if (m_device_distance_matrix != NULL)
  {
    CUDA_CALL(cudaFree(m_device_distance_matrix));
    m_device_distance_matrix = NULL;
  }
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_for_computing_host_row_major_vectors_cpu()
{
  initialize_cpu_host_distance_matrix();
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_for_computing_host_column_major_vectors_cpu()
{
  initialize_cpu_host_vectors();
  initialize_cpu_host_distance_matrix();
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_for_computing_host_row_major_vectors_gpu()
{
  initialize_gpu_host_vectors();
  initialize_gpu_device_vectors();
  initialize_gpu_device_distance_matrix();
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_for_computing_host_column_major_vectors_gpu()
{
  initialize_gpu_host_vectors();
  initialize_gpu_device_vectors();
  initialize_gpu_device_distance_matrix();
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_for_computing_device_row_major_vectors_gpu()
{
  initialize_gpu_device_vectors();
  initialize_gpu_device_distance_matrix();
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_for_computing_device_column_major_vectors_gpu()
{
  initialize_gpu_device_distance_matrix();
}

template<class Derived, typename VectorType, typename DistanceType>
const typename distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::CpuMatrix
  distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::compute_host_row_major_vectors_cpu_implementation(
    const VectorType * const __restrict__ host_row_major_vectors1,
    const int num_vectors1,
    const VectorType * const __restrict__ host_row_major_vectors2,
    const int num_vectors2)
{
  enforce_asserts(
    host_row_major_vectors1,
    num_vectors1,
    host_row_major_vectors2,
    num_vectors2);

  initialize_for_computing_host_row_major_vectors_cpu();

  return static_cast<Derived *>(this)->compute_cpu(
    host_row_major_vectors1,
    num_vectors1,
    host_row_major_vectors2,
    num_vectors2);
}

template<class Derived, typename VectorType, typename DistanceType>
const typename distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::CpuMatrix
  distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::compute_host_column_major_vectors_cpu_implementation(
    const VectorType * const __restrict__ host_column_major_vectors1,
    const int num_vectors1,
    const int stride_in_values1,
    const VectorType * const __restrict__ host_column_major_vectors2,
    const int num_vectors2,
    const int stride_in_values2)
{
  enforce_asserts(
    host_column_major_vectors1,
    num_vectors1,
    stride_in_values1,
    host_column_major_vectors2,
    num_vectors2,
    stride_in_values2);

  initialize_for_computing_host_column_major_vectors_cpu();

  VectorType * const __restrict__ host_row_major_vectors1 =
    m_host_vectors;
  VectorType * const __restrict__ host_row_major_vectors2 =
    m_host_vectors + m_max_num_vectors1 * m_num_values_per_vector;

  distance_matrix::convert_column_major_to_row_major_cpu(
    host_column_major_vectors1, // source
    num_vectors1,
    m_num_values_per_vector,
    stride_in_values1,
    host_row_major_vectors1); // destination
  distance_matrix::convert_column_major_to_row_major_cpu(
    host_column_major_vectors2, // source
    num_vectors2,
    m_num_values_per_vector,
    stride_in_values2,
    host_row_major_vectors2); // destination

  return static_cast<Derived *>(this)->compute_cpu(
    host_row_major_vectors1,
    num_vectors1,
    host_row_major_vectors2,
    num_vectors2);
}

#if defined(__CUDACC__) // CUDA COMPILER
template<class Derived, typename VectorType, typename DistanceType>
const typename distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::GpuMatrix
  distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::compute_host_row_major_vectors_gpu_implementation(
    const VectorType * const __restrict__ host_row_major_vectors1,
    const int num_vectors1,
    const VectorType * const __restrict__ host_row_major_vectors2,
    const int num_vectors2,
    cudaStream_t cuda_stream)
{
  enforce_asserts(
    host_row_major_vectors1,
    num_vectors1,
    host_row_major_vectors2,
    num_vectors2);

  initialize_for_computing_host_row_major_vectors_gpu();

  const int stride_in_values1 = distance_matrix::convert_row_major_to_column_major_cpu(
    host_row_major_vectors1, // source
    num_vectors1,
    m_num_values_per_vector,
    m_host_vectors_buffer.host_ptr(), // destination
    m_max_num_vectors1 * m_num_values_per_vector, // max destination size
    byte_alignment);
  const int num_values1 = stride_in_values1 * m_num_values_per_vector;

  const int stride_in_values2 = distance_matrix::convert_row_major_to_column_major_cpu(
    host_row_major_vectors2, // source
    num_vectors2,
    m_num_values_per_vector,
    m_host_vectors_buffer.host_ptr() + num_values1, // destination
    m_max_num_vectors2 * m_num_values_per_vector, // max destination size
    byte_alignment);
  const int num_values2 = stride_in_values2 * m_num_values_per_vector;

  m_host_vectors_buffer.memcpy_to_device_async(
    m_device_vectors, // destination
    num_values1 + num_values2, // num elements
    cuda_stream);

  return static_cast<Derived *>(this)->compute_gpu(
    m_device_vectors,
    num_vectors1,
    stride_in_values1,
    m_device_vectors + num_values1,
    num_vectors2,
    stride_in_values2,
    cuda_stream);
}
#endif // CUDA COMPILER

#if defined(__CUDACC__) // CUDA COMPILER
template<class Derived, typename VectorType, typename DistanceType>
const typename distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::GpuMatrix
  distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::compute_host_column_major_vectors_gpu_implementation(
    const VectorType * const __restrict__ host_column_major_vectors1,
    const int num_vectors1,
    const int stride_in_values1,
    const VectorType * const __restrict__ host_column_major_vectors2,
    const int num_vectors2,
    const int stride_in_values2,
    cudaStream_t cuda_stream)
{
  enforce_asserts(
    host_column_major_vectors1,
    num_vectors1,
    stride_in_values1,
    host_column_major_vectors2,
    num_vectors2,
    stride_in_values2);

  initialize_for_computing_host_column_major_vectors_gpu();

  const int num_values1 = stride_in_values1 * m_num_values_per_vector;
  const int num_values2 = stride_in_values2 * m_num_values_per_vector;

  memcpy(
    m_host_vectors_buffer.host_ptr(), // destination
    host_column_major_vectors1, // source
    num_values1 * sizeof(VectorType)); // num bytes
  memcpy(
    m_host_vectors_buffer.host_ptr() + num_values1, // destination
    host_column_major_vectors2, // source
    num_values2 * sizeof(VectorType)); // num bytes

  m_host_vectors_buffer.memcpy_to_device_async(
    m_device_vectors, // destination
    num_values1 + num_values2, // num elements
    cuda_stream);

  return static_cast<Derived *>(this)->compute_gpu(
    m_device_vectors,
    num_vectors1,
    stride_in_values1,
    m_device_vectors + num_values1,
    num_vectors2,
    stride_in_values2,
    cuda_stream);
}
#endif // CUDA COMPILER

#if defined(__CUDACC__) // CUDA COMPILER
template<class Derived, typename VectorType, typename DistanceType>
const typename distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::GpuMatrix
  distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::compute_device_row_major_vectors_gpu_implementation(
    const VectorType * const __restrict__ device_row_major_vectors1,
    const int num_vectors1,
    const VectorType * const __restrict__ device_row_major_vectors2,
    const int num_vectors2,
    cudaStream_t cuda_stream)
{
  enforce_asserts(
    device_row_major_vectors1,
    num_vectors1,
    device_row_major_vectors2,
    num_vectors2);

  initialize_for_computing_device_row_major_vectors_gpu();

  const int stride_in_values1 = distance_matrix::convert_row_major_to_column_major_gpu(
    device_row_major_vectors1, // source
    num_vectors1,
    m_num_values_per_vector,
    m_device_vectors, // destination
    m_max_num_vectors1 * m_num_values_per_vector, // max destination size
    byte_alignment,
    cuda_stream);
  const int num_values1 = stride_in_values1 * m_num_values_per_vector;

  const int stride_in_values2 = distance_matrix::convert_row_major_to_column_major_gpu(
    device_row_major_vectors2,  // source
    num_vectors2,
    m_num_values_per_vector,
    m_device_vectors + num_values1, // destination
    m_max_num_vectors2 * m_num_values_per_vector, // max destination size
    byte_alignment,
    cuda_stream);

  return static_cast<Derived *>(this)->compute_gpu(
    m_device_vectors,
    num_vectors1,
    stride_in_values1,
    m_device_vectors + num_values1,
    num_vectors2,
    stride_in_values2,
    cuda_stream);
}
#endif // CUDA COMPILER

#if defined(__CUDACC__) // CUDA COMPILER
template<class Derived, typename VectorType, typename DistanceType>
const typename distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::GpuMatrix
  distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::compute_device_column_major_vectors_gpu_implementation(
    const VectorType * const __restrict__ device_column_major_vectors1,
    const int num_vectors1,
    const int stride_in_values1,
    const VectorType * const __restrict__ device_column_major_vectors2,
    const int num_vectors2,
    const int stride_in_values2,
    cudaStream_t cuda_stream)
{
  enforce_asserts(
    device_column_major_vectors1,
    num_vectors1,
    stride_in_values1,
    device_column_major_vectors2,
    num_vectors2,
    stride_in_values2);

  initialize_for_computing_device_column_major_vectors_gpu();

  return static_cast<Derived *>(this)->compute_gpu(
    device_column_major_vectors1,
    num_vectors1,
    stride_in_values1,
    device_column_major_vectors2,
    num_vectors2,
    stride_in_values2,
    cuda_stream);
}
#endif // CUDA COMPILER

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::enforce_asserts(
  const VectorType * const __restrict__ row_major_vectors1,
  const int num_vectors1,
  const VectorType * const __restrict__ row_major_vectors2,
  const int num_vectors2)
{
  ASSERT(num_vectors1 <= m_max_num_vectors1);
  ASSERT(num_vectors2 <= m_max_num_vectors2);
  ASSERT(row_major_vectors1 != NULL);
  ASSERT(row_major_vectors2 != NULL);
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::enforce_asserts(
  const VectorType * const __restrict__ column_major_vectors1,
  const int num_vectors1,
  const int stride_in_values1,
  const VectorType * const __restrict__ column_major_vectors2,
  const int num_vectors2,
  const int stride_in_values2)
{
  ASSERT(num_vectors1 <= m_max_num_vectors1);
  ASSERT(num_vectors2 <= m_max_num_vectors2);
  ASSERT(stride_in_values1 >= num_vectors1);
  ASSERT(stride_in_values2 >= num_vectors2);
  ASSERT(column_major_vectors1 != NULL);
  ASSERT(column_major_vectors2 != NULL);
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_cpu_host_vectors()
{
  if (m_host_vectors != NULL)
  {
    return;
  }

  const int num_values =
    m_max_num_vectors1 * m_num_values_per_vector +
    m_max_num_vectors2 * m_num_values_per_vector;
  m_host_vectors = new VectorType[num_values];

  if (m_host_vectors == NULL)
  {
    std::cerr << "ERROR: initialize_cpu_host_vectors() - failed to allocate memory," << std::endl;
    std::cerr << "  requested " << num_values << " values = " <<
      num_values * sizeof(VectorType) << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_cpu_host_distance_matrix()
{
  if (m_host_distance_matrix != NULL)
  {
    return;
  }

  const int num_values = m_max_num_vectors1 * m_max_num_vectors2;
  m_host_distance_matrix = new DistanceType[num_values];

  if (m_host_distance_matrix == NULL)
  {
    std::cerr << "ERROR: initialize_cpu_host_distance_matrix() - failed to allocate memory," << std::endl;
    std::cerr << "  requested " << m_max_num_vectors1 << "x" << m_max_num_vectors2 << " values = " <<
      num_values * sizeof(DistanceType) << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_gpu_host_vectors()
{
  if (m_host_vectors_buffer.host_ptr() != NULL)
  {
    return;
  }

  const int num_values =
    m_max_num_vectors1 * m_num_values_per_vector +
    m_max_num_vectors2 * m_num_values_per_vector;
  m_host_vectors_buffer.alloc(
    num_values,
    cuda_helper::cuda_memcpy_buffer_type::Pinned);

  if (m_host_vectors_buffer.host_ptr() == NULL)
  {
    std::cerr << "ERROR: initialize_gpu_host_vectors() - failed to allocate memory," << std::endl;
    std::cerr << "  requested " << num_values << " values = " <<
      num_values * sizeof(VectorType) << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_gpu_device_vectors()
{
  if (m_device_vectors != NULL)
  {
    return;
  }

  const int num_values =
    m_max_num_vectors1 * m_num_values_per_vector +
    m_max_num_vectors2 * m_num_values_per_vector;
  CUDA_CALL(cudaMalloc(&m_device_vectors,
    num_values * sizeof(VectorType)));

  if (m_device_vectors == NULL)
  {
    std::cerr << "ERROR: initialize_gpu_device_vectors() - failed to allocate memory," << std::endl;
    std::cerr << "  requested " << num_values << " values = " <<
      num_values * sizeof(VectorType) << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
}

template<class Derived, typename VectorType, typename DistanceType>
void distance_matrix::DistanceMatrixBase<Derived, VectorType, DistanceType>::initialize_gpu_device_distance_matrix()
{
  if (m_device_distance_matrix != NULL)
  {
    return;
  }

  const int num_values = m_max_num_vectors1 * m_max_num_vectors2;
  CUDA_CALL(cudaMalloc(&m_device_distance_matrix,
    num_values * sizeof(DistanceType)));

  if (m_device_distance_matrix == NULL)
  {
    std::cerr << "ERROR: initialize_gpu_device_distance_matrix() - failed to allocate memory," << std::endl;
    std::cerr << "  requested " << m_max_num_vectors1 << "x" << m_max_num_vectors2 << " values = " <<
      num_values * sizeof(DistanceType) << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
}

#endif // DISTANCE_MATRIX_BASE_H
