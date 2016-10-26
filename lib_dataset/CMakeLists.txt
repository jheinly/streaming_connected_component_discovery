cmake_minimum_required(VERSION 3.0)

find_package(Boost)

cmh_add_library_module(
  src/dataset/focal_lookup.h
  src/dataset/focal_lookup.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Boost_INCLUDE_DIR}
)
