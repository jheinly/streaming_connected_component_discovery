cmake_minimum_required(VERSION 3.0)

cmh_add_library_module(
  src/features/sift_descriptor.h
  src/features/sift_keypoint.h
  src/features/sift_support.h
  src/features/sift_support.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)
