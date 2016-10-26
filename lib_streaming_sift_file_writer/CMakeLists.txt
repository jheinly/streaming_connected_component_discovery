cmake_minimum_required(VERSION 3.0)

find_package(Boost COMPONENTS system thread date_time)

cmh_add_library_module(
  src/streaming_sift_file_writer/streaming_sift_file_writer.h
  src/streaming_sift_file_writer/streaming_sift_file_writer.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Boost_INCLUDE_DIR}
)

cmh_target_link_libraries(
  ${Boost_LIBRARIES}
)
