#pragma once
#ifndef SIGNAL_H
#define SIGNAL_H

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/lock_types.hpp>

/*
This class implements something similar to a semaphore. A thread that calls
wait() will block until a separate thread calls signal(). This can be used
to avoid busy-waits, as the waiting thread will only execute once signal()
has been called. This class also allows multiple threads to be signaled.
However, it does not guarantee that each thread has the chance to receive
exactly one signal. If a waiting thread repeatedly calls wait, it could
consume all of the signals before other threads have a chance to wake.
*/

namespace core {

class Signal
{
  public:
    explicit Signal()
    : m_mutex(),
      m_condition_variable(),
      m_num_signals(0)
    {}

    ~Signal()
    {}

    void wait()
    {
      boost::unique_lock<boost::mutex> auto_lock(m_mutex);
      while (m_num_signals == 0)
      {
        m_condition_variable.wait(auto_lock);
      }
      --m_num_signals;
    }

    void signal(const int num_signals = 1)
    {
      boost::unique_lock<boost::mutex> auto_lock(m_mutex);
      m_num_signals += num_signals;
      if (m_num_signals == 1)
      {
        m_condition_variable.notify_one();
      }
      else
      {
        m_condition_variable.notify_all();
      }
    }

    void clear()
    {
      boost::unique_lock<boost::mutex> auto_lock(m_mutex);
      m_num_signals = 0;
    }

  private:
    Signal(const Signal &);
    Signal & operator=(const Signal &);

    boost::mutex m_mutex;
    boost::condition_variable m_condition_variable;
    int m_num_signals;
};

} // namespace core

#endif // SIGNAL_H
