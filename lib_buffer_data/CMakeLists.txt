cmake_minimum_required(VERSION 3.0)

cmh_add_header_module(
  src/buffer_data/file_data.h
  src/buffer_data/image_data.h
  src/buffer_data/output_sift_features_data.h
  src/buffer_data/sift_features_data.h
  src/buffer_data/sift_visual_words_data.h
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)
