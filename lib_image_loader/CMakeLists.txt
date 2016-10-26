cmake_minimum_required(VERSION 3.0)

cmh_add_library_module(
  src/image_loader/image_loader.h
  src/image_loader/image_loader.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)
