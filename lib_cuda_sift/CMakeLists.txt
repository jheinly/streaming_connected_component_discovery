cmake_minimum_required(VERSION 3.0)

cmh_add_library_module(
  src/cuda_sift/cuda_sift.h
  src/cuda_sift/cuda_sift.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)
