cmake_minimum_required(VERSION 3.0)

cmh_add_header_module(
  src/config/config.h
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)
