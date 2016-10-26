cmake_minimum_required(VERSION 3.0)

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/src/Eigen/*)
file(GLOB SOURCE_FILES_TO_REMOVE
  ${CMAKE_CURRENT_LIST_DIR}/src/Eigen/*.txt
  ${CMAKE_CURRENT_LIST_DIR}/src/Eigen/src
  ${CMAKE_CURRENT_LIST_DIR}/src/Eigen/.svn)
list(REMOVE_ITEM SOURCE_FILES ${SOURCE_FILES_TO_REMOVE})

cmh_add_header_module(${SOURCE_FILES} FOLDER_NAME Eigen)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)
