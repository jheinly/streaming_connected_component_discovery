cmake_minimum_required(VERSION 3.0)

find_package(Boost COMPONENTS system thread date_time)

cmh_add_library_module(
  src/streaming_image_loader/streaming_image_loader.h
  src/streaming_image_loader/streaming_image_loader.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Boost_INCLUDE_DIR}
)

cmh_target_link_libraries(
  ${Boost_LIBRARIES}
)
