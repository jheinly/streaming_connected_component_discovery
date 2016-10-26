cmake_minimum_required(VERSION 3.0)

find_package(CUDA)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_SDK_ROOT_DIR}/common/inc
)

cmh_add_cuda_library_module(
  src/cuda_helper/cuda_helper.h
  src/cuda_helper/cuda_helper.cpp
  src/cuda_helper/cuda_macros.h
  src/cuda_helper/cuda_memcpy_buffer.h
)

cmh_target_link_libraries(
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cublas_LIBRARY}
)
