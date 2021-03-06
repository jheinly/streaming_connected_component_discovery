# NOTE: This file should not be executed directly using CMake, but instead
#       should be referenced via a find_package() command using the name
#       of the corresponding *-config.cmake file in the same directory.

cmake_minimum_required(VERSION 3.0)

# Find 3rd-party packages.
# ex. find_package(Boost COMPONENTS system thread)
# ex. find_package(OpenCV COMPONENTS core)

# Create an executable with the given source files.
cmh_add_executable_module(
  # ex. file.h
  # ex. file.cpp
  # ex. src/file.cpp
  # ex. ${CMAKE_CURRENT_LIST_DIR}/../file.cpp
  densification.cpp
)

# Set the compile definitions necessary to use this module.
cmh_target_compile_definitions(
  # ex. PREPROCESSOR_FLAG
  # ex. PREPROCESSOR_PATH="${CMAKE_CURRENT_LIST_DIR}/../path"
  CREATE_EMPTY_FEATURE_DATABASE_BIN_FOLDER="${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}"
  STREAMING_IPC2SFM_PACKAGE_SCRIPTS_FOLDER="${CMAKE_CURRENT_LIST_DIR}/../scripts"
)

# Set the include directories necessary to use this module.
cmh_target_include_directories(
  # ex. ${CMAKE_CURRENT_LIST_DIR}/src
  # ex. ${Boost_INCLUDE_DIR}
  # ex. ${OpenCV_INCLUDE_DIRS}
)

# Set the libraries that need to be linked to this module.
cmh_target_link_libraries(
  # ex. ${Boost_LIBRARIES}
  # ex. ${OpenCV_LIBS}
)
