#include <features/sift_support.h>
#include <algorithm>
#include <cmath>

namespace features {

class CompareKeypointScales
{
  public:
    explicit CompareKeypointScales(
      const std::vector<SiftKeypoint> & keypoints)
    : m_keypoints(keypoints)
    {}

    inline bool operator()(const int index1, const int index2) const
    { return m_keypoints[index1].scale > m_keypoints[index2].scale; }

  private:
    CompareKeypointScales & operator=(const CompareKeypointScales &);

    const std::vector<SiftKeypoint> & m_keypoints;
};

} // namespace features

void features::sift_support::compute_indices_for_keypoints_sorted_by_scale(
  const std::vector<SiftKeypoint> & keypoints,
  std::vector<int> & indices,
  const int max_num_keypoints)
{
  const int num_keypoints = static_cast<int>(keypoints.size());
  indices.resize(num_keypoints);
  for (int i = 0; i < num_keypoints; ++i)
  {
    // SiftGPU outputs the keypoints in a roughly sorted order from smaller
    // scales to larger scales. So, initialize the indices with the reverse
    // order so that the keypoints are roughly sorted from larger to
    // smaller scales.
    indices[i] = num_keypoints - i - 1;
  }

  if (max_num_keypoints > 0 && max_num_keypoints < num_keypoints)
  {
    std::partial_sort(
      indices.begin(),
      indices.begin() + max_num_keypoints,
      indices.end(),
      CompareKeypointScales(keypoints));
    indices.resize(max_num_keypoints);
  }
  else
  {
    std::sort(
      indices.begin(),
      indices.end(),
      CompareKeypointScales(keypoints));
  }
}

void features::sift_support::convert_descriptors_from_float_to_uchar(
  const std::vector<SiftDescriptorFloat> & descriptors_float,
  std::vector<SiftDescriptorUChar> & descriptors_uchar)
{
  const int num_descriptors = static_cast<int>(descriptors_float.size());
  descriptors_uchar.resize(num_descriptors);
  for (int i = 0; i < num_descriptors; ++i)
  {
    for (int j = 0; j < SiftDescriptorFloat::num_floats_per_descriptor; ++j)
    {
      const int val = static_cast<int>(512.0f * descriptors_float[i].floats[j] + 0.5f);
      descriptors_uchar[i].uchars[j] = static_cast<unsigned char>(val > 255 ? 255 : val);
    }
  }
}

void features::sift_support::convert_descriptors_from_uchar_to_float(
  const std::vector<SiftDescriptorUChar> & descriptors_uchar,
  std::vector<SiftDescriptorFloat> & descriptors_float)
{
  const int num_descriptors = static_cast<int>(descriptors_uchar.size());
  descriptors_float.resize(num_descriptors);
  for (int i = 0; i < num_descriptors; ++i)
  {
    float norm = 0.0f;
    for (int j = 0; j < SiftDescriptorUChar::num_uchars_per_descriptor; ++j)
    {
      const float val = static_cast<float>(descriptors_uchar[i].uchars[j]);
      descriptors_float[i].floats[j] = val;
      norm += val * val;
    }

    const float inv_norm = 1.0f / sqrtf(norm);
    for (int j = 0; j < SiftDescriptorUChar::num_uchars_per_descriptor; ++j)
    {
      descriptors_float[i].floats[j] *= inv_norm;
    }
  }
}
