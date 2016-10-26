#include <distance_matrix/dot_product_matrix_uchar.h>
#include <core/math.h>

distance_matrix::DotProductMatrixUChar::DotProductMatrixUChar(
  const int max_num_vectors1,
  const int max_num_vectors2,
  const int num_uchars_per_vector,
  const int scale_factor_per_value)
: DistanceMatrixBase<DotProductMatrixUChar, unsigned char, float>(
    max_num_vectors1, max_num_vectors2, num_uchars_per_vector),
  m_host_vectors_float(NULL),
  m_device_vectors_float(NULL),
  m_cublas_handle(NULL),
  m_scale_factor_per_value(scale_factor_per_value)
{}

distance_matrix::DotProductMatrixUChar::~DotProductMatrixUChar()
{
  if (m_host_vectors_float != NULL)
  {
    delete [] m_host_vectors_float;
    m_host_vectors_float = NULL;
  }
  if (m_device_vectors_float != NULL)
  {
    CUDA_CALL(cudaFree(m_device_vectors_float));
    m_device_vectors_float = NULL;
  }
  if (m_cublas_handle != NULL)
  {
    CUBLAS_CALL(cublasDestroy(m_cublas_handle));
    m_cublas_handle = NULL;
  }
}

const distance_matrix::DotProductMatrixUChar::CpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_host_row_major_vectors_cpu(
    const unsigned char * const __restrict__ host_row_major_vectors1,
    const int num_vectors1,
    const unsigned char * const __restrict__ host_row_major_vectors2,
    const int num_vectors2)
{
  return compute_host_row_major_vectors_cpu_implementation(
    host_row_major_vectors1,
    num_vectors1,
    host_row_major_vectors2,
    num_vectors2);
}

const distance_matrix::DotProductMatrixUChar::CpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_host_column_major_vectors_cpu(
    const unsigned char * const __restrict__ host_column_major_vectors1,
    const int num_vectors1,
    const int stride_in_values1,
    const unsigned char * const __restrict__ host_column_major_vectors2,
    const int num_vectors2,
    const int stride_in_values2)
{
  return compute_host_column_major_vectors_cpu_implementation(
    host_column_major_vectors1,
    num_vectors1,
    stride_in_values1,
    host_column_major_vectors2,
    num_vectors2,
    stride_in_values2);
}

const distance_matrix::DotProductMatrixUChar::GpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_host_row_major_vectors_gpu(
    const unsigned char * const __restrict__ host_row_major_vectors1,
    const int num_vectors1,
    const unsigned char * const __restrict__ host_row_major_vectors2,
    const int num_vectors2,
    cudaStream_t cuda_stream)
{
  return compute_host_row_major_vectors_gpu_implementation(
    host_row_major_vectors1,
    num_vectors1,
    host_row_major_vectors2,
    num_vectors2,
    cuda_stream);
}

const distance_matrix::DotProductMatrixUChar::GpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_host_column_major_vectors_gpu(
    const unsigned char * const __restrict__ host_column_major_vectors1,
    const int num_vectors1,
    const int stride_in_values1,
    const unsigned char * const __restrict__ host_column_major_vectors2,
    const int num_vectors2,
    const int stride_in_values2,
    cudaStream_t cuda_stream)
{
  return compute_host_column_major_vectors_gpu_implementation(
    host_column_major_vectors1,
    num_vectors1,
    stride_in_values1,
    host_column_major_vectors2,
    num_vectors2,
    stride_in_values2,
    cuda_stream);
}

const distance_matrix::DotProductMatrixUChar::GpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_device_row_major_vectors_gpu(
    const unsigned char * const __restrict__ device_row_major_vectors1,
    const int num_vectors1,
    const unsigned char * const __restrict__ device_row_major_vectors2,
    const int num_vectors2,
    cudaStream_t cuda_stream)
{
  return compute_device_row_major_vectors_gpu_implementation(
    device_row_major_vectors1,
    num_vectors1,
    device_row_major_vectors2,
    num_vectors2,
    cuda_stream);
}

const distance_matrix::DotProductMatrixUChar::GpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_device_column_major_vectors_gpu(
    const unsigned char * const __restrict__ device_column_major_vectors1,
    const int num_vectors1,
    const int stride_in_values1,
    const unsigned char * const __restrict__ device_column_major_vectors2,
    const int num_vectors2,
    const int stride_in_values2,
    cudaStream_t cuda_stream)
{
  return compute_device_column_major_vectors_gpu_implementation(
    device_column_major_vectors1,
    num_vectors1,
    stride_in_values1,
    device_column_major_vectors2,
    num_vectors2,
    stride_in_values2,
    cuda_stream);
}

const distance_matrix::DotProductMatrixUChar::GpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_gpu(
    const unsigned char * const __restrict__ device_column_major_vectors1,
    const int num_vectors1,
    const int stride_in_values1,
    const unsigned char * const __restrict__ device_column_major_vectors2,
    const int num_vectors2,
    const int stride_in_values2,
    cudaStream_t cuda_stream)
{
  if (m_device_vectors_float == NULL)
  {
    const int num_values =
      m_max_num_vectors1 * m_num_values_per_vector +
      m_max_num_vectors2 * m_num_values_per_vector;
    CUDA_CALL(cudaMalloc(&m_device_vectors_float,
      num_values * sizeof(float)));

    if (m_device_vectors_float == NULL)
    {
      std::cerr << "ERROR: DotProductMatrixUChar::compute_gpu() - failed to allocate memory," << std::endl;
      std::cerr << "  requested " << num_values << " values = " <<
        num_values * sizeof(float) << " bytes" << std::endl;
      exit(EXIT_FAILURE);
    }

    CUBLAS_CALL(cublasCreate(&m_cublas_handle));
  }

  CUBLAS_CALL(cublasSetStream(m_cublas_handle, cuda_stream));

  float * const __restrict__ device_column_major_vectors_float1 =
    m_device_vectors_float;
  float * const __restrict__ device_column_major_vectors_float2 =
    m_device_vectors_float + m_max_num_vectors1 * m_num_values_per_vector;

  distance_matrix::convert_column_major_uchar_to_float_gpu(
    device_column_major_vectors1,
    num_vectors1,
    m_num_values_per_vector,
    stride_in_values1,
    device_column_major_vectors_float1,
    m_scale_factor_per_value,
    cuda_stream);
  distance_matrix::convert_column_major_uchar_to_float_gpu(
    device_column_major_vectors2,
    num_vectors2,
    m_num_values_per_vector,
    stride_in_values2,
    device_column_major_vectors_float2,
    m_scale_factor_per_value,
    cuda_stream);

  distance_matrix::normalize_column_major_vectors_gpu(
    device_column_major_vectors_float1,
    m_num_values_per_vector,
    num_vectors1,
    stride_in_values1,
    cuda_stream);
  distance_matrix::normalize_column_major_vectors_gpu(
    device_column_major_vectors_float2,
    m_num_values_per_vector,
    num_vectors2,
    stride_in_values2,
    cuda_stream);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // NOTE: CUBLAS stores matrices in column-major format, so interchange vectors1
  //       and vectors2 so that the resulting distance matrix has the distances
  //       for a single vector from vectors1 stored contiguously (row-major).

  CUBLAS_CALL(cublasSgemm(
    m_cublas_handle,
    CUBLAS_OP_N, // leave vectors2 as-is
    CUBLAS_OP_T, // transpose vectors1
    num_vectors2, // number of rows in vectors2 and number of rows in distance_matrix
    num_vectors1, // number of columns in vectors1 after transposing and number of columns in distance_matrix
    m_num_values_per_vector, // number of columns in vectors2 and number of rows in vectors1 transposed
    &alpha, // scalar used for multiplication
    device_column_major_vectors_float2, // pointer to first set of vectors
    stride_in_values2, // number of rows used to store vectors2
    device_column_major_vectors_float1, // pointer to second set of vectors
    stride_in_values1, // number of rows used to store vectors1
    &beta, // scalar used for multiplication
    m_device_distance_matrix, // pointer to distance matrix
    m_max_num_vectors2)); // number of rows used to store distance_matrix (stride)

  return GpuMatrix(
    m_device_distance_matrix,
    num_vectors1,
    num_vectors2,
    m_max_num_vectors2);
}
