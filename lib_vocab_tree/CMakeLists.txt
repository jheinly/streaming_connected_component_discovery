cmake_minimum_required(VERSION 3.0)

# Find 3rd-party packages.
# ex. find_package(OpenCV COMPONENTS core)
find_package(Boost COMPONENTS system thread)

# Create a static library with the given source files.
cmh_add_library_module(
  # ex. file.h
  # ex. file.cpp
  # ex. src/file.cpp
  # ex. ${CMAKE_CURRENT_LIST_DIR}/../file.cpp
  src/vocab_tree/kmeans.h
  src/vocab_tree/kmeans.cpp
  src/vocab_tree/kmeans_kd.h
  src/vocab_tree/kmeans_kd.cpp
  src/vocab_tree/VocabFlatNode.cpp
  src/vocab_tree/VocabTree.h
  src/vocab_tree/VocabTree.cpp
  src/vocab_tree/VocabTreeBuild.cpp
  src/vocab_tree/VocabTreeIO.cpp
  src/vocab_tree/VocabTreeUtil.cpp
)

# Set the compile definitions necessary to use this module.
cmh_target_compile_definitions(
  # ex. -DBOOST_ALL_NO_LIB
)

# Set the include directories necessary to use this module.
cmh_target_include_directories(
  # ex. ${CMAKE_CURRENT_LIST_DIR}/src
  # ex. ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Boost_INCLUDE_DIR}
)

# Set the libraries that need to be linked to this module.
cmh_target_link_libraries(
  # ex. ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)
