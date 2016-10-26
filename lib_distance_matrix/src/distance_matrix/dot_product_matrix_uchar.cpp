#include <distance_matrix/dot_product_matrix_uchar.h>
#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

namespace distance_matrix {

namespace dot_product_matrix_uchar {

template<int num_values_per_vector>
void compute_cpu(
  const float * const __restrict__ host_row_major_vectors1,
  const int num_vectors1,
  const float * const __restrict__ host_row_major_vectors2,
  const int num_vectors2,
  float * __restrict__ distance_matrix)
{
  Eigen::Map<const Eigen::Matrix<float, num_values_per_vector, Eigen::Dynamic> > eigen_vectors1(
    host_row_major_vectors1, num_values_per_vector, num_vectors1);
  Eigen::Map<const Eigen::Matrix<float, num_values_per_vector, Eigen::Dynamic> > eigen_vectors2(
    host_row_major_vectors2, num_values_per_vector, num_vectors2);

  // NOTE: tried using Eigen::ColMajor, Eigen::Aligned, and Eigen::OuterStride<Eigen::Dynamic>, but
  //       none of these gave any noticeable speed increases.
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > eigen_distance_matrix(
    distance_matrix, num_vectors1, num_vectors2);

  eigen_distance_matrix.noalias() = eigen_vectors1.transpose() * eigen_vectors2;
}

} // namespace dot_product_matrix_uchar

} // namespace distance_matrix

const distance_matrix::DotProductMatrixUChar::CpuMatrix
  distance_matrix::DotProductMatrixUChar::compute_cpu(
    const unsigned char * const __restrict__ host_row_major_vectors1,
    const int num_vectors1,
    const unsigned char * const __restrict__ host_row_major_vectors2,
    const int num_vectors2)
{
  if (m_host_vectors_float == NULL)
  {
    const int num_values =
      m_max_num_vectors1 * m_num_values_per_vector +
      m_max_num_vectors2 * m_num_values_per_vector;
    m_host_vectors_float = new float[num_values];

    if (m_host_vectors_float == NULL)
    {
      std::cerr << "ERROR: DotProductMatrixUChar::compute_cpu() - failed to allocate memory," << std::endl;
      std::cerr << "  requested " << num_values << " values = " <<
        num_values * sizeof(float) << " bytes" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  float * const __restrict__ host_row_major_vectors_float1 =
    m_host_vectors_float;
  float * const __restrict__ host_row_major_vectors_float2 =
    m_host_vectors_float + m_max_num_vectors1 * m_num_values_per_vector;

  distance_matrix::convert_row_major_uchar_to_float_cpu(
    host_row_major_vectors1,
    m_num_values_per_vector,
    num_vectors1,
    host_row_major_vectors_float1,
    m_scale_factor_per_value);
  distance_matrix::convert_row_major_uchar_to_float_cpu(
    host_row_major_vectors2,
    m_num_values_per_vector,
    num_vectors2,
    host_row_major_vectors_float2,
    m_scale_factor_per_value);

  switch (m_num_values_per_vector)
  {
    case 16:
      dot_product_matrix_uchar::compute_cpu<16>(
        host_row_major_vectors_float1,
        num_vectors1,
        host_row_major_vectors_float2,
        num_vectors2,
        m_host_distance_matrix);
      break;
    case 32:
      dot_product_matrix_uchar::compute_cpu<32>(
        host_row_major_vectors_float1,
        num_vectors1,
        host_row_major_vectors_float2,
        num_vectors2,
        m_host_distance_matrix);
      break;
    case 64:
      dot_product_matrix_uchar::compute_cpu<64>(
        host_row_major_vectors_float1,
        num_vectors1,
        host_row_major_vectors_float2,
        num_vectors2,
        m_host_distance_matrix);
      break;
    case 128:
      dot_product_matrix_uchar::compute_cpu<128>(
        host_row_major_vectors_float1,
        num_vectors1,
        host_row_major_vectors_float2,
        num_vectors2,
        m_host_distance_matrix);
      break;
    default:
      std::cerr << "ERROR: DotProductMatrixUChar::compute_cpu() - no support"
        << " for vectors with " << m_num_values_per_vector << " values" << std::endl;
      exit(EXIT_FAILURE);
  }

  return CpuMatrix(
    m_host_distance_matrix,
    num_vectors1,
    num_vectors2,
    num_vectors2);
}
