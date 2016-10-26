#pragma once
#ifndef ALIGNMENT_H
#define ALIGNMENT_H

namespace core {

#if defined(_MSC_VER) // Visual Studio
  #define ALIGN(num_bytes) __declspec(align(num_bytes))
#elif defined(__CUDACC__) // CUDA COMPILER
  #define ALIGN(num_bytes) __align__(num_bytes)
#elif defined(__GNUC__) // GCC
  #define ALIGN(num_bytes) __attribute__((aligned(num_bytes)))
#else
  #error "ALIGN macro undefined in core/alignment.h"
#endif

} // namespace core

#endif // ALIGNMENT_H
