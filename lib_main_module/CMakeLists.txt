# NOTE: This file should not be executed directly using CMake, but instead
#       should be referenced via a find_package() command using the name
#       of the corresponding *-config.cmake file in the same directory.

cmake_minimum_required(VERSION 3.0)

# Find 3rd-party packages.
# ex. find_package(Boost COMPONENTS system thread)
# ex. find_package(OpenCV COMPONENTS core)
find_package(Boost COMPONENTS program_options)

# Create a static library module with the given source files.
cmh_add_library_module(
  # ex. file.h
  # ex. src/file.h
  # ex. ${CMAKE_CURRENT_LIST_DIR}/../file.h
  src/main_module/feature_database_wrapper.h
  src/main_module/streaming_ipc2sfm_types.h
  src/main_module/sift_matcher_thread_group.h
  src/main_module/sift_matcher_thread_group.cpp
  src/main_module/main_args.h
  src/main_module/main_args.cpp
)

# Set the compile definitions necessary to use this module.
cmh_target_compile_definitions(
  # ex. PREPROCESSOR_FLAG
  # ex. PREPROCESSOR_PATH="${CMAKE_CURRENT_LIST_DIR}/../path"
)

# Set the include directories necessary to use this module.
cmh_target_include_directories(
  # ex. ${Boost_INCLUDE_DIR}
  # ex. ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Boost_INCLUDE_DIR}
)

# Set the libraries that need to be linked to this module.
cmh_target_link_libraries(
  # ex. ${Boost_LIBRARIES}
  # ex. ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)
