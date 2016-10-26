#pragma once
#ifndef SHARED_AVERAGE_H
#define SHARED_AVERAGE_H

#include <core/speed_stats.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

namespace core {

class SharedSpeedStats : public SpeedStats
{
  public:
    SharedSpeedStats(const int rolling_average_length = 100)
    : SpeedStats(rolling_average_length),
      m_mutex()
    {}

    void add_timing(const double elapsed_seconds, const int num_items)
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      SpeedStats::add_timing(elapsed_seconds, num_items);
    }

    double get_most_recent_speed_hz() const
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      return SpeedStats::get_most_recent_speed_hz();
    }

    double get_rolling_average_speed_hz() const
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      return SpeedStats::get_rolling_average_speed_hz();
    }

    double get_overall_average_speed_hz() const
    {
      boost::lock_guard<boost::mutex> auto_lock(m_mutex);
      return SpeedStats::get_overall_average_speed_hz();
    }

  private:
    mutable boost::mutex m_mutex;
};

} // namespace core

#endif // SHARED_AVERAGE_H
