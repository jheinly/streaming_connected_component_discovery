#pragma once
#ifndef CUDA_MEMCPY_BUFFER_H
#define CUDA_MEMCPY_BUFFER_H

#include <cuda_helper/cuda_helper.h>
#include <iostream>

namespace cuda_helper {

namespace cuda_memcpy_buffer_type {

/*
Pageable  - Default allocation (such as new or malloc) that is pageable.
Pinned    - Page-locked allocation that potentially enables overlapping
            kernels and memory transfers, and avoids a copy operation
            into page-locked memory.
Portable  - Can be shared between multiple CUDA contexts.
WriteOnly - The CPU can only write to this memory, but it provides
            potentially faster transfers over the PCI-E bus.
Mapped    - Avoids a memory copy between the CPU and GPU, but is not
            cached on the GPU so repeated reads should be avoided.
            NOTE: cuda_helper::enable_mapped_memory_before_cuda_context_initialization()
            should be called before the CUDA context has been initialized.
*/
enum BufferType {
  Pageable,
  Pinned,
  PinnedPortable,
  PinnedWriteOnly, // For some reason, using PinnedWriteOnly instead of Pinned would cause certain memory copies to hang.
  PinnedMapped,
  PinnedWriteOnlyPortable,
  PinnedWriteOnlyMapped,
  PinnedMappedPortable,
  PinnedWriteOnlyMappedPortable
};

} // namespace cuda_memcpy_buffer_type

template<typename T>
class CudaMemcpyBuffer
{
  public:
    explicit CudaMemcpyBuffer()
    : m_host_ptr(NULL), m_buffer_type(cuda_memcpy_buffer_type::Pageable)
    {}

    ~CudaMemcpyBuffer()
    { release(); }

    template<typename Integer>
    bool alloc(const Integer num_elements, cuda_memcpy_buffer_type::BufferType buffer_type, T ** mapped_device_ptr = NULL)
    {
      release();

      const Integer num_bytes = num_elements * sizeof(T);
      switch (buffer_type)
      {
        case cuda_memcpy_buffer_type::Pageable:
          m_host_ptr = new T[num_elements];
          break;
        case cuda_memcpy_buffer_type::Pinned:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocDefault));
          break;
        case cuda_memcpy_buffer_type::PinnedPortable:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocPortable));
          break;
        case cuda_memcpy_buffer_type::PinnedWriteOnly:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocWriteCombined));
          break;
        case cuda_memcpy_buffer_type::PinnedMapped:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocMapped));
          break;
        case cuda_memcpy_buffer_type::PinnedWriteOnlyPortable:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocWriteCombined | cudaHostAllocPortable));
          break;
        case cuda_memcpy_buffer_type::PinnedWriteOnlyMapped:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));
          break;
        case cuda_memcpy_buffer_type::PinnedMappedPortable:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
          break;
        case cuda_memcpy_buffer_type::PinnedWriteOnlyMappedPortable:
          CUDA_CALL(cudaHostAlloc(&m_host_ptr, num_bytes, cudaHostAllocWriteCombined | cudaHostAllocMapped | cudaHostAllocPortable));
          break;
        default:
          m_host_ptr = NULL;
          break;
      }

      if (m_host_ptr != NULL)
      {
        switch (buffer_type)
        {
          case cuda_memcpy_buffer_type::Pageable:
          case cuda_memcpy_buffer_type::Pinned:
          case cuda_memcpy_buffer_type::PinnedPortable:
          case cuda_memcpy_buffer_type::PinnedWriteOnly:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyPortable:
            break;
          case cuda_memcpy_buffer_type::PinnedMapped:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyMapped:
          case cuda_memcpy_buffer_type::PinnedMappedPortable:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyMappedPortable:
            CUDA_CALL(cudaHostGetDevicePointer(mapped_device_ptr, m_host_ptr, 0));
            if (mapped_device_ptr == NULL)
            {
              std::cerr << "ERROR: CudaMemcpyBuffer::alloc() - failed to map memory," << std::endl;
              std::cerr << "  requested " << num_elements << " values = " <<
                num_bytes << " bytes" << std::endl;
            }
            break;
        }
        m_buffer_type = buffer_type;
        return true;
      }
      else
      {
        std::cerr << "ERROR: CudaMemcpyBuffer::alloc() - failed to allocate memory," << std::endl;
        std::cerr << "  requested " << num_elements << " values = " <<
          num_bytes << " bytes" << std::endl;
        switch (buffer_type)
        {
          case cuda_memcpy_buffer_type::Pageable:
          case cuda_memcpy_buffer_type::Pinned:
          case cuda_memcpy_buffer_type::PinnedPortable:
          case cuda_memcpy_buffer_type::PinnedWriteOnly:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyPortable:
            break;
          case cuda_memcpy_buffer_type::PinnedMapped:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyMapped:
          case cuda_memcpy_buffer_type::PinnedMappedPortable:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyMappedPortable:
            *mapped_device_ptr = NULL;
            break;
        }
        m_buffer_type = cuda_memcpy_buffer_type::Pageable;
        return false;
      }
    }

    void release()
    {
      if (m_host_ptr != NULL)
      {
        switch (m_buffer_type)
        {
          case cuda_memcpy_buffer_type::Pageable:
            delete [] m_host_ptr;
            break;
          case cuda_memcpy_buffer_type::Pinned:
          case cuda_memcpy_buffer_type::PinnedPortable:
          case cuda_memcpy_buffer_type::PinnedWriteOnly:
          case cuda_memcpy_buffer_type::PinnedMapped:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyPortable:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyMapped:
          case cuda_memcpy_buffer_type::PinnedMappedPortable:
          case cuda_memcpy_buffer_type::PinnedWriteOnlyMappedPortable:
            CUDA_CALL(cudaFreeHost(m_host_ptr));
            break;
        }
        m_host_ptr = NULL;
        m_buffer_type = cuda_memcpy_buffer_type::Pageable;
      }
    }

    template<typename Integer>
    void memcpy_to_device(T * device_ptr, const Integer num_elements) const
    {
      const Integer num_bytes = num_elements * sizeof(T);
      CUDA_CALL(cudaMemcpy(device_ptr, m_host_ptr, num_bytes, cudaMemcpyHostToDevice));
    }

    template<typename Integer>
    void memcpy_to_device_async(T * device_ptr, const Integer num_elements, cudaStream_t cuda_stream) const
    {
      const Integer num_bytes = num_elements * sizeof(T);
      CUDA_CALL(cudaMemcpyAsync(device_ptr, m_host_ptr, num_bytes, cudaMemcpyHostToDevice, cuda_stream));
    }

    template<typename Integer>
    void memcpy_from_device(const T * device_ptr, const Integer num_elements)
    {
      const Integer num_bytes = num_elements * sizeof(T);
      CUDA_CALL(cudaMemcpy(m_host_ptr, device_ptr, num_bytes, cudaMemcpyDeviceToHost));
    }

    template<typename Integer>
    void memcpy_from_device_async(const T * device_ptr, const Integer num_elements, cudaStream_t cuda_stream)
    {
      const Integer num_bytes = num_elements * sizeof(T);
      CUDA_CALL(cudaMemcpyAsync(m_host_ptr, device_ptr, num_bytes, cudaMemcpyDeviceToHost, cuda_stream));
    }

    inline const T * host_ptr() const
    { return m_host_ptr; }

    inline T * host_ptr()
    { return m_host_ptr; }

    inline cuda_memcpy_buffer_type::BufferType type() const
    { return m_buffer_type; }

  private:
    T * m_host_ptr;
    cuda_memcpy_buffer_type::BufferType m_buffer_type;
};

} // namespace cuda_helper

#endif // CUDA_MEMCPY_BUFFER_H
