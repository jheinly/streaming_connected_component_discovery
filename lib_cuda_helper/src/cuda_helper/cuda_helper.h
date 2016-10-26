#pragma once
#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cstdio>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <driver_types.h>
#include <exception>
#include <string>

namespace cuda_helper {

int get_num_devices();
void set_device(int gpu_index);
int get_device();
std::string get_device_name(int gpu_index = -1);

void enable_mapped_memory_before_cuda_context_initialization();

void initialize_cuda_context();

void synchronize_device();
void synchronize_stream(cudaStream_t cuda_stream = 0);

void prefer_larger_shared_memory();
void prefer_larger_cache();

const char * cublas_get_error_string(cublasStatus_t error);

class CudaException : public std::exception
{
public:
  CudaException(const std::string & message)
    : std::exception(),
    m_message(message)
  {}

  virtual const char * what() const throw()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

#define CUDA_CALL(func) { \
  cudaError_t error = func; \
  if (error != cudaSuccess) { \
    const std::string error_string = cudaGetErrorString(error); \
    printf("CUDA Error: %s, line %i: %s\n", __FILE__, __LINE__, \
      error_string.c_str()); \
    throw(cuda_helper::CudaException(error_string)); \
  } \
}

#define CUDA_CALL_STORE_SUCCESS(func, success) { \
  cudaError_t error = func; \
  if (error != cudaSuccess) { \
    printf("CUDA Error: %s, line %i: %s\n", __FILE__, __LINE__, \
      cudaGetErrorString(error)); \
    success = false; \
  } else { \
    success = true; \
  } \
}

#define CUBLAS_CALL(func) {\
  cublasStatus_t stat = func; \
  if (stat != CUBLAS_STATUS_SUCCESS) { \
    const std::string error_string = cuda_helper::cublas_get_error_string(stat); \
    printf("CUBLAS Error: %s, line %i: %s\n", __FILE__, __LINE__, \
      error_string.c_str()); \
    throw(cuda_helper::CudaException(error_string)); \
  } \
}

#define CUBLAS_CALL_STORE_SUCCESS(func, success) {\
  cublasStatus_t stat = func; \
  if (stat != CUBLAS_STATUS_SUCCESS) { \
    printf("CUBLAS Error: %s, line %i: %s\n", __FILE__, __LINE__, \
      cuda_helper::cublas_get_error_string(stat)); \
    success = false; \
  } else { \
    success = true; \
  } \
}

#define CHECK_KERNEL_ERROR() { \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    const std::string error_string = cudaGetErrorString(error); \
    printf("CUDA Error: %s, line %i: %s\n", __FILE__, __LINE__, \
      error_string.c_str()); \
    throw(cuda_helper::CudaException(error_string)); \
  } \
}

#define CHECK_KERNEL_ERROR_STORE_SUCCESS(success) { \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    printf("CUDA Error: %s, line %i: %s\n", __FILE__, __LINE__, \
      cudaGetErrorString(error)); \
    success = false; \
  } else { \
    success = true; \
  } \
}

} // namespace cuda_helper

#endif // CUDA_HELPER_H
