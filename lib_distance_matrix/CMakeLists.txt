cmake_minimum_required(VERSION 3.0)

find_package(CUDA)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)

set(DISTANCE_MATRIX_FILES
  src/distance_matrix/dot_product_matrix_uchar.h
  src/distance_matrix/dot_product_matrix_uchar.cu
  src/distance_matrix/dot_product_matrix_uchar.cpp
)

set(DISTANCE_MATRIX_SRC_FILES
  src/distance_matrix/src/distance_matrix_base.h
  src/distance_matrix/src/distance_matrix_struct.h
  src/distance_matrix/src/distance_matrix_util.h
  src/distance_matrix/src/distance_matrix_util.cu
  src/distance_matrix/src/distance_matrix_util_kernels.cuh
)

source_group(${CMH_MODULE_NAME} FILES ${DISTANCE_MATRIX_FILES})
source_group(${CMH_MODULE_NAME}\\src FILES ${DISTANCE_MATRIX_SRC_FILES})

cmh_prepare_cuda_compiler(CMH_CUDA_COMPILER_DEFINITIONS)

cuda_add_library(${CMH_MODULE_NAME}
  ${DISTANCE_MATRIX_FILES} ${DISTANCE_MATRIX_SRC_FILES}
  OPTIONS ${CMH_CUDA_COMPILER_DEFINITIONS}
)

cmh_end_module()
