#pragma once
#ifndef STREAMING_MODULE_INTERFACE_H
#define STREAMING_MODULE_INTERFACE_H

#include <core/manager_worker_synchronization.h>
#include <string>

namespace core {

class StreamingModuleInterface
{
  public:
    explicit StreamingModuleInterface(const int num_worker_threads)
    : m_start_synchronization(num_worker_threads)
    {}

    virtual ~StreamingModuleInterface()
    {}

    virtual void wait_until_ready_to_start()
    { m_start_synchronization.wait_for_workers_to_be_ready_to_be_signaled(); }

    virtual void start()
    { m_start_synchronization.signal_workers(); }

    virtual void generate_summary_speed_report(std::string & report) = 0;
    virtual void generate_detailed_speed_report(std::string & report) = 0;

    virtual void wait_until_finished() = 0;

  protected:
    void worker_thread_wait_for_start()
    { m_start_synchronization.wait_for_signal_from_manager(); }

    ManagerWorkerSynchronization m_start_synchronization;
};

} // namespace core

#endif // STREAMING_MODULE_INTERFACE_H
