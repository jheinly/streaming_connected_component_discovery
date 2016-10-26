#pragma once
#ifndef ESSENTIAL_MATRIX_ESTIMATOR_H
#define ESSENTIAL_MATRIX_ESTIMATOR_H

#include <config/config.h>
#include <assert/assert.h>
#include <vector>
#include <utility>

#define USE_V3D_CODE_ESSENTIAL_MATRIX

#ifdef USE_V3D_CODE_ESSENTIAL_MATRIX
  #include <estimator/v3d_code/DataContainer.h>
  #include <estimator/v3d_code/Matcher.h>
  #include <estimator/v3d_code/Features.h>
  #include <opencv2/opencv.hpp>
#endif

namespace estimator {

class EssentialMatrixEstimator
{
  public:
    EssentialMatrixEstimator(
      const int max_num_ransac_iterations = config::max_num_ransac_iterations_for_geometric_verification,
      const int min_num_inliers = config::min_num_inliers_for_geometric_verification,
      const float max_point_to_line_distance = config::max_point_to_line_distance_in_pixels_for_essential);
    ~EssentialMatrixEstimator();

    // FeatureMatchesAccessor should define the following functions:
    //   num_matches() - the number of matches
    //   x1(int i)     - X coordinate of first feature for the i'th match
    //   y1(int i)     - Y coordinate of first feature for the i'th match
    //   x2(int i)     - X coordinate of second feature for the i'th match
    //   y2(int i)     - Y coordinate of second feature for the i'th match
    template<typename FeatureMatchesAccessor>
    int estimate(
      const FeatureMatchesAccessor & feature_matches_accessor,
      const float focal1, const int width1, const int height1, // camera parameters for first image
      const float focal2, const int width2, const int height2, // camera parameters for second image
      std::vector<int> & output_inlier_match_indices, // final set of inlier matches
      float output_essential_matrix[9]); // final computed essential matrix

  private:
    EssentialMatrixEstimator(const EssentialMatrixEstimator &);
    EssentialMatrixEstimator & operator=(const EssentialMatrixEstimator &);

    const int m_max_num_ransac_iterations;
    const int m_min_num_inliers;
    const float m_max_point_to_line_distance;

#ifdef USE_V3D_CODE_ESSENTIAL_MATRIX
    sfmData::DataContainer m_container;
    std::vector<std::pair<int, int> > m_matches;
    V3D::Features m_features1;
    V3D::Features m_features2;
    struct DummyFeature
    {
      // General information used across all features
      static const V3D::Features::FeatureType Type = V3D::Features::DummyFeatureType;
      typedef unsigned char StorageType;
      static const int NumStorageVals = 1;
      typedef unsigned char MatchingType;
      static const int NumMatchingVals = 1;

      struct Keypoint : public V3D::Features::GenericKeypoint
      {};

      struct Descriptor
      {
        DummyFeature::StorageType descriptor[DummyFeature::NumStorageVals];
      };
    };
#endif
};

} // namespace estimator

template<typename FeatureMatchesAccessor>
int estimator::EssentialMatrixEstimator::estimate(
  const FeatureMatchesAccessor & feature_matches_accessor,
  const float focal1, const int width1, const int height1, // camera parameters for first image
  const float focal2, const int width2, const int height2, // camera parameters for second image
  std::vector<int> & output_inlier_match_indices, // final set of inlier matches
  float output_essential_matrix[9]) // final computed essential matrix
{
  output_inlier_match_indices.clear();

  const int num_matches = feature_matches_accessor.num_matches();

  if (num_matches < m_min_num_inliers)
  {
    return 0;
  }

#ifdef USE_V3D_CODE_ESSENTIAL_MATRIX
  m_features1.resize<DummyFeature>(num_matches);
  m_features2.resize<DummyFeature>(num_matches);
  m_matches.resize(num_matches);
  for (int i = 0; i < num_matches; ++i)
  {
    m_features1.keypoint<DummyFeature>(i).x = feature_matches_accessor.x1(i);
    m_features1.keypoint<DummyFeature>(i).y = feature_matches_accessor.y1(i);
    m_features2.keypoint<DummyFeature>(i).x = feature_matches_accessor.x2(i);
    m_features2.keypoint<DummyFeature>(i).y = feature_matches_accessor.y2(i);
    m_matches[i].first = i;
    m_matches[i].second = i;
  }
  const double k1[3] = {focal1, width1 / 2.0, height1 / 2.0};
  const double k2[3] = {focal2, width2 / 2.0, height2 / 2.0};
  double E_array[9];
  cv::Mat E(3, 3, CV_64FC1, E_array);
  int num_inliers = static_cast<int>(sfmData::matcher::ransacE_ocv(
    m_features1,
    m_features2,
    k1,
    k2,
    m_matches,
    m_container.inlier_matches,
    m_container.E_Estimator_ocv,
    m_container.matcherContainer,
    E,
    false));

  if (num_inliers < m_min_num_inliers)
  {
    return 0;
  }

  output_inlier_match_indices.reserve(num_inliers);
  for (size_t i = 0; i < m_container.matcherContainer.inliers.size(); ++i)
  {
    if (m_container.matcherContainer.inliers[i])
    {
      output_inlier_match_indices.push_back(static_cast<int>(i));
    }
  }

  ASSERT(static_cast<int>(output_inlier_match_indices.size()) == num_inliers);

  output_essential_matrix[0] = static_cast<float>(E_array[0]);
  output_essential_matrix[1] = static_cast<float>(E_array[1]);
  output_essential_matrix[2] = static_cast<float>(E_array[2]);
  output_essential_matrix[3] = static_cast<float>(E_array[3]);
  output_essential_matrix[4] = static_cast<float>(E_array[4]);
  output_essential_matrix[5] = static_cast<float>(E_array[5]);
  output_essential_matrix[6] = static_cast<float>(E_array[6]);
  output_essential_matrix[7] = static_cast<float>(E_array[7]);
  output_essential_matrix[8] = static_cast<float>(E_array[8]);

  return num_inliers;
#endif
}

#endif // ESSENTIAL_MATRIX_ESTIMATOR_H
