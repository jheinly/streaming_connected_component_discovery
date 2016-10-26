#pragma once
#ifndef TWO_PASS_CHECKPOINT_WITH_THREAD_ASSIGNMENT_H
#define TWO_PASS_CHECKPOINT_WITH_THREAD_ASSIGNMENT_H

#include <core/checkpoint.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

namespace core {

class TwoPassCheckpointWithThreadAssignment
{
  public:
    explicit TwoPassCheckpointWithThreadAssignment(const int num_threads = 0)
    : m_checkpoint(num_threads),
      m_mutex(),
      m_index(0)
    {}
    
    ~TwoPassCheckpointWithThreadAssignment()
    {}
    
    void set_num_threads(const int num_threads)
    {
      m_checkpoint.set_num_threads(num_threads);
    }
    
    int wait_for_thread_assignment()
    {
      m_checkpoint.wait();
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      const int assigned_index = m_index;
      ++m_index;
      return assigned_index;
    }
    
    void wait_after_assignment()
    {
      m_checkpoint.wait();
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      m_index = 0;
    }
    
  private:
    TwoPassCheckpointWithThreadAssignment(const TwoPassCheckpointWithThreadAssignment &);
    TwoPassCheckpointWithThreadAssignment & operator=(const TwoPassCheckpointWithThreadAssignment &);

    Checkpoint m_checkpoint;
    boost::mutex m_mutex;
    int m_index;
};

} // namespace core

#endif // TWO_PASS_CHECKPOINT_WITH_THREAD_ASSIGNMENT_H
