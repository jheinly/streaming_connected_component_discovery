#pragma once
#ifndef MANAGER_WORKER_SYNCHRONIZATION_H
#define MANAGER_WORKER_SYNCHRONIZATION_H

#include <core/signal.h>
#include <core/checkpoint.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <iostream>

/*
This class handles syncrhonization in the circumstances when there is a
one-to-many relationship between threads. For instance, there is one
producer (writer) and several comsumers (readers), or one manager thread
and several worker threads.

==============
Usage Examples
==============

NOTE: wait_for_workers_to_be_ready_to_be_signaled() must be called before
      signal_workers() can be called.

---------------------------
Use Case 1: Manager/Workers
---------------------------

  ManagerWorkerSynchronization sync;
  const int num_workers = ...;
  sync.set_num_workers(num_workers);

  ///////////
  // Manager
  sync.wait_for_workers_to_be_ready_to_be_signaled();
  while (true)
  {
    // If there is more work to do...
    sync.signal_workers();
    // Wait for work to be done...
    sync.wait_for_workers_to_be_ready_to_be_signaled();
  }

  ///////////
  // Workers
  while (true)
  {
    sync.wait_for_signal_from_manager();
    // Do work...
  }

-----------------------------------------
Use Case 2: Writer/Readers, Single Buffer
-----------------------------------------

  ManagerWorkerSynchronization sync;
  const int num_readers = ...;
  sync.set_num_workers(num_readers);

  //////////
  // Writer
  while (true)
  {
    sync.wait_for_workers_to_be_ready_to_be_signaled();
    // write to buffer...
    sync.signal_workers()
  }

  ///////////
  // Readers
  while (true)
  {
    sync.wait_for_signal_from_manager();
    // read from buffer...
  }

-----------------------------------------
Use Case 3: Writer/Readers, Double Buffer
-----------------------------------------

  ManagerWorkerSynchronization sync;
  const int num_readers = ...;
  sync.set_num_workers(num_readers);

  //////////
  // Writer
  while (true)
  {
    // write to buffer...
    sync.wait_for_workers_to_be_ready_to_be_signaled();
    // swap buffers...
    sync.signal_workers()
  }

  ///////////
  // Readers
  while (true)
  {
    sync.wait_for_signal_from_manager();
    // read from buffer...
  }
*/

namespace core {

class ManagerWorkerSynchronization
{
  public:
    explicit ManagerWorkerSynchronization(const int num_workers = 0)
    : m_signal(),
      m_checkpoint(num_workers + 1),
      m_mutex(),
      m_num_workers(num_workers),
      m_already_signaled(false)
    {}
    
    ~ManagerWorkerSynchronization()
    {}

    void set_num_workers(const int num_workers)
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      if (m_already_signaled)
      {
        std::cerr << "ERROR: ManagerWorkerSynchronization - attempting to call set_num_workers()" << std::endl;
        std::cerr << "       after the workers have already been signaled" << std::endl;
        return;
      }
      m_num_workers = num_workers;
      
      // The checkpoint will block for the N workers as well as the manager, so
      // we need to add 1 to account for the manager.
      m_checkpoint.set_num_threads(num_workers + 1);
    }

    void wait_for_workers_to_be_ready_to_be_signaled()
    {
      m_checkpoint.wait();
    }

    void signal_workers()
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      m_already_signaled = true;
      m_signal.signal(m_num_workers);
    }

    void wait_for_signal_from_manager()
    {
      m_checkpoint.wait();
      m_signal.wait();
    }
    
  private:
    ManagerWorkerSynchronization(const ManagerWorkerSynchronization &);
    ManagerWorkerSynchronization & operator=(const ManagerWorkerSynchronization &);
    
    Signal m_signal;
    Checkpoint m_checkpoint;
    boost::mutex m_mutex;
    int m_num_workers;
    bool m_already_signaled;
};

} // namespace core

#endif // MANAGER_WORKER_SYNCHRONIZATION_H
