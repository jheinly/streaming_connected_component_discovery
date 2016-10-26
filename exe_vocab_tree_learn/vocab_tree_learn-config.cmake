cmake_minimum_required(VERSION 3.0)

find_package(cmake_helper REQUIRED)

cmh_new_module_with_dependencies(
  # If this module depends on other cmake_helper modules, list either the names
  # of those modules (as in a find_package command), or by specifying the path
  # to the *-config.cmake file (if the relative location of the dependency is
  # fixed relative to this module). If specifying a relative path, the current
  # directory can be accessed via ${CMAKE_CURRENT_LIST_DIR}/.
  # ex. module_name
  # ex. ${CMAKE_CURRENT_LIST_DIR}/../module_name/module_name-config.cmake
  ${CMAKE_CURRENT_LIST_DIR}/../lib_core/core-config.cmake
  ${CMAKE_CURRENT_LIST_DIR}/../lib_vocab_tree/vocab_tree-config.cmake
  ${CMAKE_CURRENT_LIST_DIR}/../lib_v3d_support/v3d_support-config.cmake
)
