cmake_minimum_required(VERSION 3.0)

# Find 3rd-party packages.
# ex. find_package(Boost COMPONENTS system thread)
# ex. find_package(OpenCV COMPONENTS core)

# Create a static library with the given source files.
cmh_add_library_module(
  # ex. file.h
  # ex. file.cpp
  # ex. src/file.cpp
  # ex. ${CMAKE_CURRENT_LIST_DIR}/../file.cpp
  include/ANN/ANN.h
  include/ANN/ANNperf.h
  include/ANN/ANNx.h
  src/ANN.cpp
  src/bd_fix_rad_search.cpp
  src/bd_pr_search.cpp
  src/bd_search.cpp
  src/bd_tree.h
  src/bd_tree.cpp
  src/brute.cpp
  src/kd_dump.cpp
  src/kd_fix_rad_search.h
  src/kd_fix_rad_search.cpp
  src/kd_pr_search.h
  src/kd_pr_search.cpp
  src/kd_search.h
  src/kd_search.cpp
  src/kd_split.h
  src/kd_split.cpp
  src/kd_tree.h
  src/kd_tree.cpp
  src/kd_util.h
  src/kd_util.cpp
  src/perf.cpp
  src/pr_queue.h
  src/pr_queue_k.h
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
  ${CMAKE_CURRENT_LIST_DIR}/include
)

# Set the libraries that need to be linked to this module.
cmh_target_link_libraries(
  # ex. ${Boost_LIBRARIES}
  # ex. ${OpenCV_LIBS}
)
