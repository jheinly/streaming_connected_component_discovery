#pragma once
#ifndef SHARED_VALUE_H
#define SHARED_VALUE_H

#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

namespace core {

template<typename T>
class SharedValue
{
  public:
    explicit SharedValue()
    : m_mutex()
    {}

    explicit SharedValue(const T & value)
    : m_mutex(),
      m_value(value)
    {}

    // Return by value (instead of const reference) because access to the
    // internal value needs to be protected by the mutex.
    T get() const
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      return m_value;
    }

    void set(const T & value)
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      m_value = value;
    }

  private:
    SharedValue(const SharedValue &);
    SharedValue & operator=(const SharedValue &);

    mutable boost::mutex m_mutex;
    T m_value;
};

} // namespace core

#endif // SHARED_VALUE_H
