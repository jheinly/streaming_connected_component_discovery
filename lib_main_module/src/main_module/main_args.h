#pragma once
#ifndef MAIN_ARGS_H
#define MAIN_ARGS_H

#include <string>
#include <vector>

struct MainArgs
{
  // Required paths.
  std::string sqlite_database_path;
  std::string vocab_tree_path;
  std::string pcdb_path;
  std::string image_lists_path;
  std::string output_visualization_folder;
  std::string output_backup_folder;
  std::string output_log_folder;

  // Optional paths.
  std::string dim_list_path;
  std::string focal_list_path;

  int num_files_to_skip;

  int batch_size;
  int max_match_attempts_per_image;
  int max_matches_per_connected_component;
  int cluster_discard_rate_in_images;

  bool iconic_selection_enabled;
  static const int num_images_for_iconic_selection = 3;
  bool cluster_merging_enabled;
  static const int num_knn_in_same_component_to_merge = 3;
  bool augmented_visual_words_enabled;

  int main_registration_stats_display_rate_in_batches;
  int main_visualization_rate_in_batches;
  int main_backup_rate_in_batches;
  int main_visualization_num_clusters;
  int main_visualization_num_images_per_cluster;
  int main_visualization_num_components;
  int main_visualization_num_images_per_component;
  int main_backup_min_images_per_component;

  // Number of buffers.
  int num_file_buffers;
  int num_image_buffers;
  int num_sift_features_buffers;
  int num_visual_words_buffers;
  int num_output_sift_buffers;

  // Number of threads.
  int num_image_loader_threads;
  int num_cuda_sift_threads;
  int num_visual_words_threads;
  int num_main_cpu_threads;
  int num_main_gpu_threads;
  int num_output_sift_threads;

  std::string sift_gpu_nums_list;
  std::string main_gpu_nums_list;
  std::vector<int> sift_gpu_nums;
  std::vector<int> main_gpu_nums;

  // Image loader settings.
  int max_image_dimension;
  int min_image_dimension;
  float max_aspect_ratio;

  // Cuda SIFT settings.
  int max_num_features;
  int min_num_features;

  // Descriptor matcher settings.
  float min_matching_distance;
  float max_matching_ratio;

  // Essential matrix settings.
  int max_num_ransac_iterations;
  float max_pixel_error_ransac;
  int min_num_inliers_for_database;
  int min_num_inliers_for_successful_match;

  // Vocab tree settings.
  int vocab_tree_num_knn;

  // Densification settings.
  std::string densification_components_path;
  std::string densification_clusters_path;
  std::string densification_output_sfm_folder;
  std::string densification_image_links_folder;
  std::string densification_mapper_project_template_path;
  std::string densification_python_exe_path;
  std::string densification_mapper_exe_path;
  int densification_vocab_tree_num_knn;
  int densification_max_component_size;
  int densification_min_component_size;
  int densification_min_num_inliers_for_export;
  int densification_num_cpu_threads;
  int densification_num_gpu_threads;
  std::string densification_gpu_nums_list;
  std::vector<int> densification_gpu_nums;
  int densification_sfm_visualization_size;
  int densification_visualization_subset_size;
};

void handle_main_args(
  const int argc,
  const char ** argv,
  MainArgs & main_args);

#endif // MAIN_ARGS_H
