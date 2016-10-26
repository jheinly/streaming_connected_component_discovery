cmake_minimum_required(VERSION 3.0)

find_package(CUDA)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)

set(DESCRIPTOR_MATCHER_FILES
  src/descriptor_matcher/dot_product_matcher_uchar.h
  src/descriptor_matcher/dot_product_matcher_uchar.cu
)

set(DESCRIPTOR_MATCHER_SRC_FILES
  src/descriptor_matcher/src/descriptor_matcher_base.h
  src/descriptor_matcher/src/descriptor_matcher_kernels.cuh
) 

source_group(${CMH_MODULE_NAME} FILES ${DESCRIPTOR_MATCHER_FILES})
source_group(${CMH_MODULE_NAME}\\src FILES ${DESCRIPTOR_MATCHER_SRC_FILES})

cmh_prepare_cuda_compiler(CMH_CUDA_COMPILER_DEFINITIONS)

cuda_add_library(${CMH_MODULE_NAME}
  ${DESCRIPTOR_MATCHER_FILES} ${DESCRIPTOR_MATCHER_SRC_FILES}
  OPTIONS ${CMH_CUDA_COMPILER_DEFINITIONS}
)

cmh_end_module()
