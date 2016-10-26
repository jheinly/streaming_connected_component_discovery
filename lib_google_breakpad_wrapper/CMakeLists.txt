# NOTE: This file should not be executed directly using CMake, but instead
#       should be referenced via a find_package() command using the name
#       of the corresponding *-config.cmake file in the same directory.

cmake_minimum_required(VERSION 3.0)

# Create a static library with the given source files.
cmh_add_library_module(
  src/google_breakpad_wrapper/google_breakpad_wrapper.h
  src/google_breakpad_wrapper/google_breakpad_wrapper.cpp
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)