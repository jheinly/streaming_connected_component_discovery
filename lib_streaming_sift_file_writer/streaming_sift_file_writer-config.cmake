cmake_minimum_required(VERSION 3.0)

find_package(cmake_helper REQUIRED)

cmh_new_module_with_dependencies(
  ${CMAKE_CURRENT_LIST_DIR}/../lib_buffer_data/buffer_data-config.cmake
  ${CMAKE_CURRENT_LIST_DIR}/../lib_core/core-config.cmake
  ${CMAKE_CURRENT_LIST_DIR}/../lib_features/features-config.cmake
  ${CMAKE_CURRENT_LIST_DIR}/../lib_v3d_support/v3d_support-config.cmake
)
