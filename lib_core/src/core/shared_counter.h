#pragma once
#ifndef SHARED_COUNTER_H
#define SHARED_COUNTER_H

#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

/*
The SharedCounter class represents an integer index that starts at zero and
can be incremented. Access to the integer index is wrapped in a mutex so that
it can be shared among several threads.

Basic usage is as follows:
--------------------------
  SharedCounter counter;
  counter.reset();

  // Spawn several threads...

  // Each thread executes the following loop in parallel:
  while (true)
  {
    int index = counter.get_value_and_increment();
    if (index >= max_index)
    {
      break;
    }

    // Process using index...
  }
*/

namespace core {

class SharedCounter
{
  public:
    explicit SharedCounter()
    : m_mutex(),
      m_value(0)
    {}

    ~SharedCounter()
    {}

    int get_value() const
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      return m_value;
    }

    int get_value_and_increment()
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      const int value = m_value;
      ++m_value;
      return value;
    }

    void reset()
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      m_value = 0;
    }

  private:
    SharedCounter(const SharedCounter &);
    SharedCounter & operator=(const SharedCounter &);

    mutable boost::mutex m_mutex;
    int m_value;
};

} // namespace core

#endif // SHARED_COUNTER_H
