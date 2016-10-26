#include <descriptor_matcher/dot_product_matcher_uchar.h>

descriptor_matcher::DotProductMatcherUChar::DotProductMatcherUChar(
  const int max_num_descriptors,
  const int num_uchars_per_descriptor,
  const int scale_factor_per_value,
  const float min_matching_distance,
  const float max_matching_ratio)
: DescriptorMatcherBase<DotProductMatcherUChar, distance_matrix::DotProductMatrixUChar>(
    &m_distance_matrix,
    max_num_descriptors,
    num_uchars_per_descriptor,
    min_matching_distance,
    max_matching_ratio),
  m_distance_matrix(
    max_num_descriptors,
    max_num_descriptors,
    num_uchars_per_descriptor,
    scale_factor_per_value),
  m_functions(
    min_matching_distance,
    max_matching_ratio)
{}

descriptor_matcher::DotProductMatcherUChar::~DotProductMatcherUChar()
{}

int descriptor_matcher::DotProductMatcherUChar::match_host_row_major_descriptors_cpu(
  const unsigned char * const __restrict__ host_row_major_descriptors1,
  const int num_descriptors1,
  const unsigned char * const __restrict__ host_row_major_descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches)
{
  return match_host_row_major_descriptors_cpu_implementation(
    host_row_major_descriptors1,
    num_descriptors1,
    host_row_major_descriptors2,
    num_descriptors2,
    matches);
}

int descriptor_matcher::DotProductMatcherUChar::match_host_column_major_descriptors_cpu(
  const unsigned char * const __restrict__ host_column_major_descriptors1,
  const int num_descriptors1,
  const int stride_in_values1,
  const unsigned char * const __restrict__ host_column_major_descriptors2,
  const int num_descriptors2,
  const int stride_in_values2,
  std::vector<std::pair<int, int> > & matches)
{
  return match_host_column_major_descriptors_cpu_implementation(
    host_column_major_descriptors1,
    num_descriptors1,
    stride_in_values1,
    host_column_major_descriptors2,
    num_descriptors2,
    stride_in_values2,
    matches);
}

int descriptor_matcher::DotProductMatcherUChar::match_host_row_major_descriptors_gpu(
  const unsigned char * const __restrict__ host_row_major_descriptors1,
  const int num_descriptors1,
  const unsigned char * const __restrict__ host_row_major_descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  return match_host_row_major_descriptors_gpu_implementation(
    host_row_major_descriptors1,
    num_descriptors1,
    host_row_major_descriptors2,
    num_descriptors2,
    matches,
    cuda_stream);
}

int descriptor_matcher::DotProductMatcherUChar::match_host_column_major_descriptors_gpu(
  const unsigned char * const __restrict__ host_column_major_descriptors1,
  const int num_descriptors1,
  const int stride_in_values1,
  const unsigned char * const __restrict__ host_column_major_descriptors2,
  const int num_descriptors2,
  const int stride_in_values2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  return match_host_column_major_descriptors_gpu_implementation(
    host_column_major_descriptors1,
    num_descriptors1,
    stride_in_values1,
    host_column_major_descriptors2,
    num_descriptors2,
    stride_in_values2,
    matches,
    cuda_stream);
}

int descriptor_matcher::DotProductMatcherUChar::match_device_row_major_descriptors_gpu(
  const unsigned char * const __restrict__ device_row_major_descriptors1,
  const int num_descriptors1,
  const unsigned char * const __restrict__ device_row_major_descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  return match_device_row_major_descriptors_gpu_implementation(
    device_row_major_descriptors1,
    num_descriptors1,
    device_row_major_descriptors2,
    num_descriptors2,
    matches,
    cuda_stream);
}

int descriptor_matcher::DotProductMatcherUChar::match_device_column_major_descriptors_gpu(
  const unsigned char * const __restrict__ device_column_major_descriptors1,
  const int num_descriptors1,
  const int stride_in_values1,
  const unsigned char * const __restrict__ device_column_major_descriptors2,
  const int num_descriptors2,
  const int stride_in_values2,
  std::vector<std::pair<int, int> > & matches,
  cudaStream_t cuda_stream)
{
  return match_device_column_major_descriptors_gpu_implementation(
    device_column_major_descriptors1,
    num_descriptors1,
    stride_in_values1,
    device_column_major_descriptors2,
    num_descriptors2,
    stride_in_values2,
    matches,
    cuda_stream);
}
