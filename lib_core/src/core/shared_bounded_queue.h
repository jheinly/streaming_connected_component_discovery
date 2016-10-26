#pragma once
#ifndef SHARED_BOUNDED_QUEUE_H
#define SHARED_BOUNDED_QUEUE_H

#include <assert/assert.h>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_types.hpp>
#include <cstddef> // included to define NULL

namespace core {

template<typename T>
class SharedBoundedQueue
{
  public:
    explicit SharedBoundedQueue(
      const int max_queue_size)
    : m_max_size(max_queue_size),
      m_queue(NULL),
      m_size(0),
      m_write_index(0),
      m_read_index(0),
      m_done_with_all_further_pushes(false),
      m_mutex(),
      m_push_performed(),
      m_pop_performed()
    {
      ASSERT(max_queue_size > 0);
      m_queue = new T[max_queue_size];
    }

    ~SharedBoundedQueue()
    {
      if (m_queue != NULL)
      {
        delete [] m_queue;
        m_queue = NULL;
      }
    }

    void push(const T & t)
    {
      boost::unique_lock<boost::mutex> auto_lock(m_mutex);
      
      // Keep looping while the queue is full.
      while (m_size == m_max_size)
      {
        m_pop_performed.wait(auto_lock);
      }

      // The queue is not full, so push the element.
      m_queue[m_write_index] = t;
      m_write_index = next_index(m_write_index);
      ++m_size;

      // Unlock the queue and notify any waiting threads.
      auto_lock.unlock();
      m_push_performed.notify_one();
    }

    bool pop(T & t)
    {
      boost::unique_lock<boost::mutex> auto_lock(m_mutex);

      // Keep looping while the queue is empty.
      while (m_size == 0)
      {
        if (m_done_with_all_further_pushes)
        {
          return false;
        }
        m_push_performed.wait(auto_lock);
      }

      // The queue is not empty, so pop an element.
      t = m_queue[m_read_index];
      m_read_index = next_index(m_read_index);
      --m_size;

      // Unlock the queue and notify any waiting threads.
      auto_lock.unlock();
      m_pop_performed.notify_one();
      return true;
    }

    void done_with_all_further_pushes()
    {
      boost::unique_lock<boost::mutex> auto_lock(m_mutex);
      m_done_with_all_further_pushes = true;

      // Unlock the queue and notify all waiting threads.
      auto_lock.unlock();
      m_push_performed.notify_all();
    }

    void reset()
    {
      boost::unique_lock<boost::mutex> auto_lock(m_mutex);
      m_size = 0;
      m_write_index = 0;
      m_read_index = 0;
      m_done_with_all_further_pushes = false;
    }

  private:
    SharedBoundedQueue(const SharedBoundedQueue &);
    SharedBoundedQueue & operator=(const SharedBoundedQueue &);

    int next_index(const int index) const
    { return (index < m_max_size - 1) ? index + 1 : 0; }

    const int m_max_size;
    T * m_queue;
    int m_size;
    int m_write_index;
    int m_read_index;
    bool m_done_with_all_further_pushes;
    boost::mutex m_mutex;
    boost::condition_variable m_push_performed;
    boost::condition_variable m_pop_performed;
};

} // namespace core

#endif // SHARED_BOUNDED_QUEUE_H
