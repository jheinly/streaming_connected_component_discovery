#pragma once
#ifndef CHECK_FOR_DEGENERATE_MATCH_H
#define CHECK_FOR_DEGENERATE_MATCH_H

#include <vector>
#include <core/image_dimensions.h>

namespace estimator {

template<typename Keypoint1, typename Keypoint2>
bool check_matches_for_duplicate_images(
  const std::vector<Keypoint1> & keypoints1,
  const std::vector<Keypoint2> & keypoints2,
  const std::vector<std::pair<int, int> > & matches)
{
  const int num_matches = static_cast<int>(matches.size());
  float total_residual = 0.0f;
  const float min_total_residual = 1.0f;

  for (int i = 0; i < num_matches; ++i)
  {
    const float x1 = keypoints1[matches[i].first].x;
    const float y1 = keypoints1[matches[i].first].y;
    const float x2 = keypoints2[matches[i].second].x;
    const float y2 = keypoints2[matches[i].second].y;

    const float dx = x2 - x1;
    const float dy = y2 - y1;

    total_residual += dx * dx + dy * dy;
    if (total_residual >= min_total_residual)
    {
      return false;
    }
  }

  return true;
}

template<typename Keypoint1, typename Keypoint2>
bool check_matches_for_translation_only(
  const std::vector<Keypoint1> & keypoints1,
  const std::vector<Keypoint2> & keypoints2,
  const std::vector<std::pair<int, int> > & matches)
{
  const float squared_distance_threshold = 1.9f * 1.9f;

  // 7 iterations = 99% confidence of finding solution assuming 1-sample RANSAC and 50% inlier rate
  const float min_inlier_ratio = 0.5f;
  const int num_iterations = 10;
  
  const int num_matches = static_cast<int>(matches.size());

  if (num_matches < 2)
  {
    return false;
  }

  int min_successful_matches = static_cast<int>(min_inlier_ratio * num_matches + 1);
  int max_failed_matches = static_cast<int>((1 - min_inlier_ratio) * num_matches + 1);

  for (int iteration = 0; iteration < num_iterations; ++iteration)
  {
    int num_successful_matches = 1;
    int num_failed_matches = 0;
  
    const int match_index = rand() % num_matches;

    const float x1 = keypoints1[matches[match_index].first].x;
    const float y1 = keypoints1[matches[match_index].first].y;
    const float x2 = keypoints2[matches[match_index].second].x;
    const float y2 = keypoints2[matches[match_index].second].y;

    const float translation_x = x2 - x1;
    const float translation_y = y2 - y1;

    for (int i = 0; i < num_matches; ++i)
    {
      if (i == match_index)
      {
        continue;
      }

      const float new_x = keypoints1[matches[i].first].x + translation_x;
      const float new_y = keypoints1[matches[i].first].y + translation_y;

      const float diff_x = new_x - keypoints2[matches[i].second].x;
      const float diff_y = new_y - keypoints2[matches[i].second].y;

      const float squared_distance = diff_x * diff_x + diff_y * diff_y;
      if (squared_distance < squared_distance_threshold)
      {
        ++num_successful_matches;
      }
      else
      {
        ++num_failed_matches;
      }

      if (num_successful_matches > min_successful_matches)
      {
        return true;
      }
      else if (num_failed_matches > max_failed_matches)
      {
        break;
      }
    }

    const float inlier_ratio = float(num_successful_matches) / float(num_matches);
    if (inlier_ratio >= min_inlier_ratio)
    {
      return true;
    }
  }

  return false;
}

inline bool is_point_in_border(
  const float point_x,
  const float point_y,
  const float border_threshold,
  const float width_threshold,
  const float height_threshold)
{
  if (point_x < border_threshold || point_x > width_threshold ||
      point_y < border_threshold || point_y > height_threshold)
  {
    return true;
  }
  return false;
}

template<typename Keypoint1, typename Keypoint2>
bool check_inliers_for_border_matches(
  const core::ImageDimensions dimensions1,
  const core::ImageDimensions dimensions2,
  const std::vector<Keypoint1> & keypoints1,
  const std::vector<Keypoint2> & keypoints2,
  const std::vector<std::pair<int, int> > & matches,
  const std::vector<int> & inlier_match_indices)
{
  const float border_width = 0.075f; // fraction of image's diagonal width
  const float min_inlier_ratio = 0.7f;

  const float border_threshold1 =
    border_width * dimensions1.diagonal_width();
  const float border_threshold2 =
    border_width * dimensions2.diagonal_width();

  const float width_threshold1 = dimensions1.width - border_threshold1;
  const float height_threshold1 = dimensions1.height - border_threshold1;
  const float width_threshold2 = dimensions2.width - border_threshold2;
  const float height_threshold2 = dimensions2.height - border_threshold2;

  int num_inliers = static_cast<int>(inlier_match_indices.size());

  if (num_inliers < 2)
  {
    return false;
  }

  const int min_border_matches = static_cast<int>(min_inlier_ratio * num_inliers + 1);
  const int min_non_border_matches = static_cast<int>((1 - min_inlier_ratio) * num_inliers + 1);

  int num_border_matches = 0;
  int num_non_border_matches = 0;

  for (int i = 0; i < num_inliers; ++i)
  {
    const int match_index = inlier_match_indices[i];
    const float x1 = keypoints1[matches[match_index].first].x;
    const float y1 = keypoints1[matches[match_index].first].y;
    const float x2 = keypoints2[matches[match_index].second].x;
    const float y2 = keypoints2[matches[match_index].second].y;

    if (is_point_in_border(x1, y1, border_threshold1, width_threshold1, height_threshold1) &&
        is_point_in_border(x2, y2, border_threshold2, width_threshold2, height_threshold2))
    {
      ++num_border_matches;
    }
    else
    {
      ++num_non_border_matches;
    }

    if (num_border_matches > min_border_matches)
    {
      return true;
    }
    else if (num_non_border_matches > min_non_border_matches)
    {
      return false;
    }
  }

  const float ratio = float(num_border_matches) / float(num_inliers);
  if (ratio >= min_inlier_ratio)
  {
    return true;
  }
  else
  {
    return false;
  }
}

} // namespace estimator

#endif // CHECK_FOR_DEGENERATE_MATCH_H
