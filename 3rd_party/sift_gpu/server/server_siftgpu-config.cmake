cmake_minimum_required(VERSION 3.0)

find_package(cmake_helper REQUIRED)

cmh_set_as_third_party_module()

cmh_new_module_with_dependencies(
  ${CMAKE_CURRENT_LIST_DIR}/../sift_gpu-config.cmake
)
