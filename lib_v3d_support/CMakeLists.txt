cmake_minimum_required(VERSION 3.0)

find_package(Boost COMPONENTS)

set(SOURCE_FILES
  src/v3d_support/pcdb.h
  src/v3d_support/pcdb.cpp
  src/v3d_support/sift_parser.h
  src/v3d_support/sift_parser.cpp
)

cmh_add_library_module(${SOURCE_FILES})

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Boost_INCLUDE_DIR}
)
