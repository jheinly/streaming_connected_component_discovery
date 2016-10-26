cmake_minimum_required(VERSION 3.0)

# Find 3rd-party packages.
# ex. find_package(Boost COMPONENTS system thread)
# ex. find_package(OpenCV COMPONENTS core)
find_package(OpenMP)
find_package(Boost)

# Create an executable with the given source files.
cmh_add_executable_module(
  # ex. file.h
  # ex. file.cpp
  # ex. src/file.cpp
  # ex. ${CMAKE_CURRENT_LIST_DIR}/../file.cpp
  vocab_tree_learn.cpp
)

# Set the compile definitions necessary to use this module.
cmh_target_compile_definitions(
  # ex. -DBOOST_ALL_NO_LIB
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
