cmake_minimum_required(VERSION 3.0)

cmh_add_library_module(
  src/jpeg_parser/jpeg_parser.h
  src/jpeg_parser/jpeg_parser.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)
