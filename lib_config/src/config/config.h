#pragma once
#ifndef CONFIG_H
#define CONFIG_H

namespace config {

static const int max_num_features_per_image = 4 * 1024;
static const int max_num_ransac_iterations_for_geometric_verification = 400;
static const int min_num_inliers_for_geometric_verification = 30;
static const float min_descriptor_matching_distance = 0.7f;
static const float max_descriptor_matching_ratio = 0.8f;
static const float max_point_to_line_distance_in_pixels_for_essential = 4.0f;

} // namespace config

#endif // CONFIG_H
