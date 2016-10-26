#pragma once
#ifndef SHARED_BATCH_BUFFER_H
#define SHARED_BATCH_BUFFER_H

#include <core/src/shared_buffer_base.h>
#include <core/timer.h>
#include <core/shared_counter.h>
#include <core/shared_speed_stats.h>

namespace core {

template<typename T>
class SharedBatchBuffer;

template<typename T>
class SharedBatchBufferLockstepWriter;

template<typename T>
class SharedBatchBufferLockstepReader
: public SharedBufferLockstepReaderBase<
  T,
  std::vector<T>,
  SharedBatchBuffer,
  SharedBatchBufferLockstepReader>
{
  public:
    friend class SharedBufferBase<
      T,
      std::vector<T>,
      SharedBatchBuffer,
      core::SharedBatchBufferLockstepReader,
      SharedBatchBufferLockstepWriter>;
    friend class SharedBufferLockstepReaderBase<
      T,
      std::vector<T>,
      SharedBatchBuffer,
      core::SharedBatchBufferLockstepReader>;

    ~SharedBatchBufferLockstepReader()
    {}

    void starting_to_read_from_buffer()
    {
      boost::lock_guard<boost::mutex> auto_lock(this->m_shared_buffer->m_lockstep_readers_timer_mutex);
      if (!this->m_shared_buffer->m_lockstep_readers_timer.is_running())
      {
        this->m_shared_buffer->m_lockstep_readers_timer.start();
      }
    }

    SharedCounter * read_buffer_counter()
    { return &this->m_shared_buffer->m_lockstep_readers_counter; }

  protected:
    explicit SharedBatchBufferLockstepReader(SharedBatchBuffer<T> * shared_buffer)
    : SharedBufferLockstepReaderBase<
        T,
        std::vector<T>,
        SharedBatchBuffer,
        core::SharedBatchBufferLockstepReader>
        (shared_buffer)
    {}

    void done_reading_from_buffer_thread_0_task()
    {
      this->m_shared_buffer->m_lockstep_readers_counter.reset();

      boost::lock_guard<boost::mutex> auto_lock(this->m_shared_buffer->m_lockstep_readers_timer_mutex);
      this->m_shared_buffer->m_lockstep_readers_timer.stop();
      const double elapsed_seconds = this->m_shared_buffer->m_lockstep_readers_timer.elapsed();
      const int num_items = static_cast<int>(this->read_buffer().size());
      this->m_shared_buffer->m_lockstep_readers_speed_stats.add_timing(elapsed_seconds, num_items);
    }
};

template<typename T>
class SharedBatchBufferLockstepWriter
: public SharedBufferLockstepWriterBase<
  T,
  std::vector<T>,
  SharedBatchBuffer,
  SharedBatchBufferLockstepWriter>
{
  public:
    friend class SharedBufferBase<
      T,
      std::vector<T>,
      SharedBatchBuffer,
      SharedBatchBufferLockstepReader,
      core::SharedBatchBufferLockstepWriter>;
    friend class SharedBufferLockstepWriterBase<
      T,
      std::vector<T>,
      SharedBatchBuffer,
      core::SharedBatchBufferLockstepWriter>;

    ~SharedBatchBufferLockstepWriter()
    {}

    void starting_to_write_to_buffer()
    {
      {
        boost::lock_guard<boost::mutex> auto_lock(this->m_shared_buffer->m_lockstep_writers_timer_mutex);
        if (!this->m_shared_buffer->m_lockstep_writers_timer.is_running())
        {
          this->m_shared_buffer->m_lockstep_writers_timer.start();
        }
      }
      {
        boost::lock_guard<boost::mutex> auto_lock(this->m_shared_buffer->m_buffer_state_mutex);
        if (static_cast<int>(this->write_buffer().size()) != this->m_shared_buffer->m_batch_size)
        {
          this->write_buffer().resize(this->m_shared_buffer->m_batch_size);
        }
      }
    }

    SharedCounter * write_buffer_counter()
    { return &this->m_shared_buffer->m_lockstep_writers_counter; }

  protected:
    explicit SharedBatchBufferLockstepWriter(SharedBatchBuffer<T> * shared_buffer)
    : SharedBufferLockstepWriterBase<
        T,
        std::vector<T>,
        SharedBatchBuffer,
        core::SharedBatchBufferLockstepWriter>
        (shared_buffer)
    {}

    void done_writing_to_buffer_thread_0_task()
    {
      const int counter_value =
        this->m_shared_buffer->m_lockstep_writers_counter.get_value();
      if (counter_value > this->m_shared_buffer->m_batch_size)
      {
        /*
        Don't resize the buffer if more entries have been written to it than
        what its default size was. It will have been up to the calling
        streaming module to have resized the write buffer to accomodate the
        extra elements. An example of this is in streaming_main_module.cpp:
            // It could be possible to output more SIFT than the size of a typical batch.
            // For example, if each of the batch images match to a different single-image
            // cluster, then we would be saving twice the normal size of a batch.
            // if (buffer_index >= static_cast<int>(m_output_sift_data_writer->write_buffer().size()))
            {
              m_output_sift_data_writer->write_buffer().resize(buffer_index + 1);
            }
        */
        //this->write_buffer().resize(this->m_shared_buffer->m_batch_size);
      }
      else
      {
        this->write_buffer().resize(counter_value);
      }
      this->m_shared_buffer->m_lockstep_writers_counter.reset();

      boost::lock_guard<boost::mutex> auto_lock(this->m_shared_buffer->m_lockstep_writers_timer_mutex);
      this->m_shared_buffer->m_lockstep_writers_timer.stop();
      const double elapsed_seconds = this->m_shared_buffer->m_lockstep_writers_timer.elapsed();
      const int num_items = static_cast<int>(this->write_buffer().size());
      this->m_shared_buffer->m_lockstep_writers_speed_stats.add_timing(elapsed_seconds, num_items);
    }
};

template<typename T>
class SharedBatchBuffer
: public SharedBufferBase<
  T,
  std::vector<T>,
  SharedBatchBuffer,
  SharedBatchBufferLockstepReader,
  SharedBatchBufferLockstepWriter>
{
  public:
    friend class SharedBatchBufferLockstepReader<T>;
    friend class SharedBatchBufferLockstepWriter<T>;

    explicit SharedBatchBuffer(
      const int num_buffers,
      const int batch_size)
    : SharedBufferBase<
        T,
        std::vector<T>,
        core::SharedBatchBuffer,
        SharedBatchBufferLockstepReader,
        SharedBatchBufferLockstepWriter>
        (num_buffers),
      m_batch_size(batch_size)
    {
      m_lockstep_readers_timer.stop();
      m_lockstep_writers_timer.stop();

      for (int i = 0; i < num_buffers; ++i)
      {
        this->m_buffer[i].reserve(batch_size);
      }
    }

    const SharedSpeedStats * lockstep_readers_speed_stats() const
    { return &m_lockstep_readers_speed_stats; }

    const SharedSpeedStats * lockstep_writers_speed_stats() const
    { return &m_lockstep_writers_speed_stats; }

  protected:
    SharedCounter m_lockstep_readers_counter;
    SharedCounter m_lockstep_writers_counter;

    Timer m_lockstep_readers_timer;
    Timer m_lockstep_writers_timer;
    boost::mutex m_lockstep_readers_timer_mutex;
    boost::mutex m_lockstep_writers_timer_mutex;

    SharedSpeedStats m_lockstep_readers_speed_stats;
    SharedSpeedStats m_lockstep_writers_speed_stats;

    const int m_batch_size;
};

} // namespace core

#endif // SHARED_BATCH_BUFFER_H
