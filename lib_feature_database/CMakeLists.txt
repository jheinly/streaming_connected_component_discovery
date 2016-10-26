# NOTE: This file should not be executed directly using CMake, but instead
#       should be referenced via a find_package() command using the name
#       of the corresponding *-config.cmake file in the same directory.

cmake_minimum_required(VERSION 3.0)

# Find 3rd-party packages.
# ex. find_package(OpenCV COMPONENTS core)
find_package(Boost REQUIRED)

set(BASE_2D_FILES
  src/base2d/feature_database.h
  src/base2d/feature_database.cc
  src/base2d/image_data.h
  src/base2d/camera.h
  src/base2d/camera.cc
)

set(BASE_3D_FILES
  src/base3d/camera_models.h
  src/base3d/camera_models.cc
)

set(ESTIMATORS_FILES
  src/estimators/two_view_geometry.h
)

set(UTIL_FILES
  src/util/sqlite3_utils.h
  src/util/types.h
  src/util/verify.h
  src/util/verify.cc
  src/util/misc.h
  src/util/misc.cc
)

# Set the name of this module's source folder in the IDE project (e.g. Visual Studio).
source_group(base2d FILES ${BASE_2D_FILES})
source_group(base3d FILES ${BASE_3D_FILES})
source_group(estimators FILES ${ESTIMATORS_FILES})
source_group(optim FILES ${OPTIM_FILES})
source_group(util FILES ${UTIL_FILES})

# Create a static library.
add_library(${CMH_MODULE_NAME}
  ${BASE_2D_FILES}
  ${BASE_3D_FILES}
  ${ESTIMATORS_FILES}
  ${UTIL_FILES}
)

# Set the compile definitions necessary to use this module.
target_compile_definitions(${CMH_MODULE_NAME} PUBLIC
  # ex. PREPROCESSOR_FLAG
  # ex. PREPROCESSOR_PATH="${CMAKE_CURRENT_LIST_DIR}/../path"
)

# Set the include directories necessary to use this module.
target_include_directories(${CMH_MODULE_NAME} PUBLIC
  # ex. ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${FreeImage_INCLUDE_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/src
)

# Set the libraries that need to be linked to this module.
target_link_libraries(${CMH_MODULE_NAME} PUBLIC
  # ex. ${Boost_LIBRARIES}
  # ex. ${OpenCV_LIBS}
)

# Finalize this module.
cmh_end_module()
