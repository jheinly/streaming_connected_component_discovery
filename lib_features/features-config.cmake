cmake_minimum_required(VERSION 3.0)

find_package(cmake_helper REQUIRED)

cmh_new_module_with_dependencies(
  ${CMAKE_CURRENT_LIST_DIR}/../lib_core/core-config.cmake
  ${CMAKE_CURRENT_LIST_DIR}/../lib_cuda_helper/cuda_helper-config.cmake
)
