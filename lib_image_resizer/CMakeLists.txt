cmake_minimum_required(VERSION 3.0)

find_package(OpenCV COMPONENTS core imgproc)

cmh_add_library_module(
  src/image_resizer/image_resizer.h
  src/image_resizer/image_resizer.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
)

cmh_target_link_libraries(
  ${OpenCV_LIBS}
)
