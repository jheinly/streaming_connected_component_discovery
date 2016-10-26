#pragma once
#ifndef SHARED_BUFFER_BASE_H
#define SHARED_BUFFER_BASE_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <assert/assert.h>
#include <core/signal.h>
#include <core/two_pass_checkpoint_with_thread_assignment.h>

template<
  typename T,
  typename BufferType,
  template<typename> class SharedBufferType,
  template<typename> class Derived>
class SharedBufferLockstepReaderBase
{
  public:
    virtual ~SharedBufferLockstepReaderBase()
    {}

    void wait_for_next_read_buffer()
    {
      for (;;)
      {
        {
          boost::lock_guard<boost::mutex> auto_lock(m_shared_buffer->m_buffer_state_mutex);
          if (m_shared_buffer->m_pending_lockstep_reader_reads[m_shared_buffer->m_lockstep_read_index] ||
              m_shared_buffer->m_num_active_lockstep_writers == 0)
          {
            return;
          }
        }
        m_shared_buffer->m_writer_done.wait();
      }
    }

    bool no_further_read_buffers_available()
    {
      boost::lock_guard<boost::mutex> auto_lock(m_shared_buffer->m_buffer_state_mutex);
      if (!m_shared_buffer->m_pending_lockstep_reader_reads[m_shared_buffer->m_lockstep_read_index])
      {
        ASSERT(m_shared_buffer->m_num_active_lockstep_writers == 0);
        return true;
      }
      return false;
    }

    const BufferType & read_buffer() const
    {
      return m_shared_buffer->m_buffer[m_shared_buffer->m_lockstep_read_index];
    }

    void done_reading_from_buffer()
    {
      const int thread_index =
        m_shared_buffer->m_two_pass_checkpoint_lockstep_readers.wait_for_thread_assignment();
      if (thread_index == 0)
      {
        {
          boost::lock_guard<boost::mutex> auto_lock(m_shared_buffer->m_buffer_state_mutex);

          const int current_index = m_shared_buffer->m_lockstep_read_index;
          m_shared_buffer->m_pending_lockstep_reader_reads[m_shared_buffer->m_lockstep_read_index] = false;

          static_cast<Derived<T> *>(this)->done_reading_from_buffer_thread_0_task();

          m_shared_buffer->m_lockstep_read_index = m_shared_buffer->next_index(current_index);
        }

        m_shared_buffer->m_reader_done.signal(m_shared_buffer->num_lockstep_writers());
      }
      m_shared_buffer->m_two_pass_checkpoint_lockstep_readers.wait_after_assignment();
    }

  protected:
    SharedBufferLockstepReaderBase(SharedBufferType<T> * shared_buffer)
    : m_shared_buffer(shared_buffer)
    {}

    SharedBufferLockstepReaderBase(const SharedBufferLockstepReaderBase &);
    SharedBufferLockstepReaderBase & operator=(const SharedBufferLockstepReaderBase &);

    void done_reading_from_buffer_thread_0_task()
    {}

    SharedBufferType<T> * m_shared_buffer;
};

template<
  typename T,
  typename BufferType,
  template<typename> class SharedBufferType,
  template<typename> class Derived>
class SharedBufferLockstepWriterBase
{
  public:
    virtual ~SharedBufferLockstepWriterBase()
    {}

    void wait_for_next_write_buffer()
    {
      for (;;)
      {
        {
          boost::lock_guard<boost::mutex> auto_lock(m_shared_buffer->m_buffer_state_mutex);
          if (!m_shared_buffer->m_pending_lockstep_reader_reads[m_shared_buffer->m_lockstep_write_index])
          {
            return;
          }
        }
        m_shared_buffer->m_reader_done.wait();
      }
    }

    const BufferType & write_buffer() const
    {
      return m_shared_buffer->m_buffer[m_shared_buffer->m_lockstep_write_index];
    }

    BufferType & write_buffer()
    {
      return m_shared_buffer->m_buffer[m_shared_buffer->m_lockstep_write_index];
    }

    void done_with_all_further_writes_and_wait_for_exit()
    {
      {
        boost::lock_guard<boost::mutex> auto_lock(m_shared_buffer->m_buffer_state_mutex);
        --m_shared_buffer->m_num_active_lockstep_writers;
      }
      for (;;)
      {
        done_writing_to_buffer();
        {
          boost::lock_guard<boost::mutex> auto_lock(m_shared_buffer->m_buffer_state_mutex);
          if (m_shared_buffer->m_num_active_lockstep_writers == 0)
          {
            return;
          }
        }
        wait_for_next_write_buffer();
      }
    }

    void done_writing_to_buffer()
    {
      const int thread_index =
        m_shared_buffer->m_two_pass_checkpoint_lockstep_writers.wait_for_thread_assignment();
      if (thread_index == 0)
      {
        {
          boost::lock_guard<boost::mutex> auto_lock(m_shared_buffer->m_buffer_state_mutex);

          const int current_index = m_shared_buffer->m_lockstep_write_index;
          m_shared_buffer->m_pending_lockstep_reader_reads[current_index] = true;

          static_cast<Derived<T> *>(this)->done_writing_to_buffer_thread_0_task();

          m_shared_buffer->m_lockstep_write_index = m_shared_buffer->next_index(current_index);
        }
        m_shared_buffer->m_writer_done.signal(m_shared_buffer->num_lockstep_readers());
      }
      m_shared_buffer->m_two_pass_checkpoint_lockstep_writers.wait_after_assignment();
    }

  protected:
    SharedBufferLockstepWriterBase(SharedBufferType<T> * shared_buffer)
    : m_shared_buffer(shared_buffer)
    {}

    SharedBufferLockstepWriterBase(const SharedBufferLockstepWriterBase &);
    SharedBufferLockstepWriterBase & operator=(const SharedBufferLockstepWriterBase &);

    void done_writing_to_buffer_thread_0_task()
    {}

    SharedBufferType<T> * m_shared_buffer;
};

template<
  typename T,
  typename BufferType,
  template<typename> class SharedBufferType,
  template<typename> class LockstepReaderType,
  template<typename> class LockstepWriterType>
class SharedBufferBase
{
  public:
    friend class SharedBufferLockstepReaderBase<
      T,
      BufferType,
      SharedBufferType,
      LockstepReaderType>;
    friend class SharedBufferLockstepWriterBase<
      T,
      BufferType,
      SharedBufferType,
      LockstepWriterType>;

    SharedBufferBase(const int num_buffers)
    : m_num_buffers(num_buffers),
      m_buffer(num_buffers),
      m_pending_lockstep_reader_reads(num_buffers, false),
      m_num_active_lockstep_writers(0),
      m_lockstep_read_index(0),
      m_lockstep_write_index(0)
    {}

    boost::shared_ptr<LockstepReaderType<T> > get_new_lockstep_reader()
    {
      boost::lock_guard<boost::mutex> auto_lock(m_buffer_state_mutex);

      boost::shared_ptr<LockstepReaderType<T> > new_lockstep_reader =
        boost::shared_ptr<LockstepReaderType<T> >(new LockstepReaderType<T>(
          static_cast<SharedBufferType<T> *>(this)));
      m_lockstep_readers.push_back(new_lockstep_reader);
      m_two_pass_checkpoint_lockstep_readers.set_num_threads(num_lockstep_readers());
      return new_lockstep_reader;
    }

    boost::shared_ptr<LockstepWriterType<T> > get_new_lockstep_writer()
    {
      boost::lock_guard<boost::mutex> auto_lock(m_buffer_state_mutex);

      boost::shared_ptr<LockstepWriterType<T> > new_lockstep_writer =
        boost::shared_ptr<LockstepWriterType<T> >(new LockstepWriterType<T>(
          static_cast<SharedBufferType<T> *>(this)));
      m_lockstep_writers.push_back(new_lockstep_writer);
      m_two_pass_checkpoint_lockstep_writers.set_num_threads(num_lockstep_writers());
      ++m_num_active_lockstep_writers;
      return new_lockstep_writer;
    }

    virtual int num_buffers() const
    { return m_num_buffers; }

    virtual const BufferType & manually_access_buffers(const int index) const
    { return m_buffer[index]; }

    virtual BufferType & manually_access_buffers(const int index)
    { return m_buffer[index]; }

  protected:
    SharedBufferBase(const SharedBufferBase &);
    SharedBufferBase & operator=(const SharedBufferBase &);

    int next_index(const int current_index) const
    { return (current_index < m_num_buffers - 1) ? current_index + 1 : 0; }

    int num_lockstep_readers() const
    { return static_cast<int>(m_lockstep_readers.size()); }

    int num_lockstep_writers() const
    { return static_cast<int>(m_lockstep_writers.size()); }
    
    const int m_num_buffers;
    std::vector<BufferType> m_buffer;
    std::vector<bool> m_pending_lockstep_reader_reads;
    int m_num_active_lockstep_writers;
    boost::mutex m_buffer_state_mutex;

    std::vector<boost::shared_ptr<LockstepReaderType<T> > > m_lockstep_readers;
    std::vector<boost::shared_ptr<LockstepWriterType<T> > > m_lockstep_writers;
    
    int m_lockstep_read_index;
    int m_lockstep_write_index;
    
    core::TwoPassCheckpointWithThreadAssignment m_two_pass_checkpoint_lockstep_readers;
    core::TwoPassCheckpointWithThreadAssignment m_two_pass_checkpoint_lockstep_writers;
    
    core::Signal m_reader_done;
    core::Signal m_writer_done;
};

#endif // SHARED_BUFFER_BASE_H
