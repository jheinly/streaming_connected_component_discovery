#pragma once
#ifndef AVERAGE_H
#define AVERAGE_H

#include <assert/assert.h>
#include <vector>

// TODO: combine this class with Timer? Also create a SharedTimer
//       class that combines Timer and SharedSpeedStats?

namespace core {

class SpeedStats
{
  public:
    SpeedStats(const int rolling_average_length = 100)
    : m_newest_index(-1),
      m_rolling_elapsed_seconds(),
      m_rolling_num_items(),
      m_rolling_average_length(rolling_average_length),
      m_oldest_index(0),
      m_total_elapsed_seconds(0.0),
      m_total_num_items(0)
    {
      m_rolling_elapsed_seconds.reserve(rolling_average_length);
      m_rolling_num_items.reserve(rolling_average_length);
    }

    ~SpeedStats()
    {}

    void add_timing(const double elapsed_seconds, const int num_items)
    {
      const int num_timings = static_cast<int>(m_rolling_elapsed_seconds.size());
      if (num_timings < m_rolling_average_length)
      {
        m_newest_index = num_timings;
        m_rolling_elapsed_seconds.push_back(elapsed_seconds);
        m_rolling_num_items.push_back(num_items);
      }
      else
      {
        m_newest_index = m_oldest_index;
        m_rolling_elapsed_seconds[m_oldest_index] = elapsed_seconds;
        m_rolling_num_items[m_oldest_index] = num_items;
        ++m_oldest_index;
        if (m_oldest_index == m_rolling_average_length)
        {
          m_oldest_index = 0;
        }
      }

      m_total_elapsed_seconds += elapsed_seconds;
      m_total_num_items += num_items;
    }

    double get_most_recent_speed_hz() const
    {
      if (m_newest_index == -1)
      {
        return 0.0;
      }
      if (m_rolling_elapsed_seconds[m_newest_index] == 0.0)
      {
        return 0.0;
      }
      return m_rolling_num_items[m_newest_index] / m_rolling_elapsed_seconds[m_newest_index];
    }

    double get_rolling_average_speed_hz() const
    {
      double elapsed_seconds_sum = 0.0;
      int num_items_sum = 0;
      for (size_t i = 0; i < m_rolling_elapsed_seconds.size(); ++i)
      {
        elapsed_seconds_sum += m_rolling_elapsed_seconds[i];
        num_items_sum += m_rolling_num_items[i];
      }
      if (elapsed_seconds_sum == 0.0)
      {
        return 0.0;
      }
      return static_cast<double>(num_items_sum) / elapsed_seconds_sum;
    }

    double get_overall_average_speed_hz() const
    {
      if (m_total_elapsed_seconds == 0.0)
      {
        return 0.0;
      }
      return static_cast<double>(m_total_num_items) / m_total_elapsed_seconds;
    }

  private:
    int m_newest_index;

    // Used to compute the rolling average.
    std::vector<double> m_rolling_elapsed_seconds;
    std::vector<int> m_rolling_num_items;
    int m_rolling_average_length;
    int m_oldest_index;

    // Used to the compute the overall average.
    double m_total_elapsed_seconds;
    int m_total_num_items;
};

} // namespace core

#endif // AVERAGE_H
