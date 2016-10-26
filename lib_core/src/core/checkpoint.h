#pragma once
#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <core/signal.h>
#include <iostream>

/*
This class implements the idea of a barrier or a rendezvous. The idea is that
threads will call wait(), and will sleep until a predefined number of threads
are all sleeping. At that point, the threads will be woken up so that they can
continue. This is useful to syncrhonize iterations of a loop between threads.
NOTE: it is important that a checkpoint instance is not called with more threads
      than it is expecting.
http://www.cs.ucr.edu/~kishore/papers/semaphores.pdf, Section 3.6.7
*/

namespace core {

class Checkpoint
{
  public:
    explicit Checkpoint(const int num_threads = 0)
    : m_mutex(),
      m_signal_phase1(),
      m_signal_phase2(),
      m_num_threads(num_threads),
      m_num_threads_waiting(0),
      m_already_signaled(false)
    {}

    ~Checkpoint()
    {}

    void set_num_threads(const int num_threads)
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      if (m_already_signaled)
      {
        std::cerr << "ERROR: Checkpoint - attempting to call set_num_threads() after the checkpoint" << std::endl;
        std::cerr << "       has already been signaled" << std::endl;
        return;
      }
      m_num_threads = num_threads;
    }

    void wait()
    {
      // By dividing the wait into two phases, we guarantee that one of the threads in the group
      // can't exit the wait() call and loop around and pass through again before other threads
      // have even exited for the first time. This is enforced by the fact that when a thread
      // exits the wait call, the internal counter (m_num_threads_waiting) will be at zero, such
      // that all of the threads must increment the counter (phase 1) before being let out of
      // the function again.

      // Phase 1: Wait for all of the threads to enter the checkpoint.
      {
        boost::lock_guard<boost::mutex> auto_lock(m_mutex);
        ++m_num_threads_waiting;
        if (m_num_threads_waiting == m_num_threads)
        {
          m_already_signaled = true;
          m_signal_phase1.signal(m_num_threads);
        }
      }
      m_signal_phase1.wait();

      // Phase 2: Wait for all the threads to be ready to exit the checkpoint.
      {
        boost::lock_guard<boost::mutex> auto_lock(m_mutex);
        --m_num_threads_waiting;
        if (m_num_threads_waiting == 0)
        {
          m_signal_phase2.signal(m_num_threads);
        }
      }
      m_signal_phase2.wait();
    }

  private:
    Checkpoint(const Checkpoint &);
    Checkpoint & operator=(const Checkpoint &);

    boost::mutex m_mutex;
    Signal m_signal_phase1;
    Signal m_signal_phase2;
    int m_num_threads;
    int m_num_threads_waiting;
    bool m_already_signaled;
};

} // namespace core

#endif // CHECKPOINT_H
