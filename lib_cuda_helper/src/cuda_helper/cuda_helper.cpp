#include <cuda_helper/cuda_helper.h>

int cuda_helper::get_num_devices()
{
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  return num_devices;
}

void cuda_helper::set_device(int gpu_index)
{
  CUDA_CALL(cudaSetDevice(gpu_index));
}

int cuda_helper::get_device()
{
  int gpu_index = -1;
  CUDA_CALL(cudaGetDevice(&gpu_index));
  return gpu_index;
}

std::string cuda_helper::get_device_name(int gpu_index)
{
  if (gpu_index == -1)
  {
    gpu_index = get_device();
  }

  cudaDeviceProp properties;
  CUDA_CALL(cudaGetDeviceProperties(&properties, gpu_index));
  return properties.name;
}

void cuda_helper::enable_mapped_memory_before_cuda_context_initialization()
{
  CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
}

void cuda_helper::initialize_cuda_context()
{
  CUDA_CALL(cudaFree(NULL));
}

void cuda_helper::synchronize_device()
{
  CUDA_CALL(cudaDeviceSynchronize());
}

void cuda_helper::synchronize_stream(cudaStream_t cuda_stream)
{
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
}

void cuda_helper::prefer_larger_shared_memory()
{
  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
}

void cuda_helper::prefer_larger_cache()
{
  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
}

const char * cuda_helper::cublas_get_error_string(cublasStatus_t error)
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}
