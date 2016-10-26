#pragma once
#ifndef CUDA_MACROS_H
#define CUDA_MACROS_H

namespace cuda_helper {

// This macro indicates that a function should be compiled for both the host
// and the CUDA device, if the CUDA compiler is processing the current file.
#ifdef __CUDACC__ // CUDA COMPILER
  #define CUDA_HOST_AND_DEVICE_FUNCTION __host__ __device__
#else
  #define CUDA_HOST_AND_DEVICE_FUNCTION
#endif

#define CUDA_COMPUTE_CAPABILITY_10 100
#define CUDA_COMPUTE_CAPABILITY_11 110
#define CUDA_COMPUTE_CAPABILITY_12 120
#define CUDA_COMPUTE_CAPABILITY_13 130
#define CUDA_COMPUTE_CAPABILITY_20 200
#define CUDA_COMPUTE_CAPABILITY_21 210
#define CUDA_COMPUTE_CAPABILITY_30 300
#define CUDA_COMPUTE_CAPABILITY_35 350
#define CUDA_COMPUTE_CAPABILITY_50 500

#ifdef __CUDA_ARCH__
  #define CUDA_COMPUTE_CAPABILITY_VERSION __CUDA_ARCH__
#endif

#ifdef __CUDACC__ // CUDA COMPILER

#if CUDA_COMPUTE_CAPABILITY_VERSION < CUDA_COMPUTE_CAPABILITY_20
  #define MUL_INT24(a, b) __mul24((a), (b))
  #define MUL_UINT24(a, b) __umul24((a), (b))
#else
  #define MUL_INT24(a, b) ((a) * (b))
  #define MUL_UINT24(a, b) ((a) * (b))
#endif

#endif // __CUDACC__

} // namespace cuda_helper

#endif // CUDA_MACROS_H
