#pragma once
#ifndef TIMER_H
#define TIMER_H

#include <boost/timer/timer.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <iostream>
#include <string>

namespace core {

class Timer
{
  public:
    // The timer will automatically start when it is created.
    explicit Timer()
    : m_timer()
    {}

    ~Timer()
    {}

    inline void start()
    { m_timer.start(); }
    
    inline void stop()
    { m_timer.stop(); }

    inline bool is_running()
    { return !m_timer.is_stopped(); }
    
    // Return the elapsed time (wall time) in seconds (resolution 1~3 microseconds).
    double elapsed()
    { return elapsed_wall(); }

    // Return the elapsed wall time in seconds (resolution 1~3 microseconds).
    double elapsed_wall()
    { return elapsed_to_seconds(m_timer.elapsed().wall); }

    // Return the elapsed user time in seconds (resolution 10~15 milliseconds).
    double elapsed_user()
    { return elapsed_to_seconds(m_timer.elapsed().user); }

    // Return the elapsed system time in seconds (resolution 10~15 milliseconds).
    double elapsed_system()
    { return elapsed_to_seconds(m_timer.elapsed().system); }

    template<typename T>
    double compute_items_per_second(const T num_items)
    { return double(num_items) / elapsed(); }

    void print(
      const std::string & message = "Elapsed",
      std::ostream & out = std::cout,
      const int message_width = 0)
    {
      const double elapsed_sec = elapsed();
      const std::string new_message = message + ":";
      out.setf(std::ios_base::fixed, std::ios_base::floatfield);
      out.precision(6);
      out.width(message_width);
      out << std::left << new_message << " " << elapsed_sec << " sec" << std::endl;
    }

    template<typename T>
    void print_items_per_second(
      const T num_items,
      const std::string & message = "Speed",
      std::ostream & out = std::cout)
    {
      const double items_per_second = compute_items_per_second(num_items);
      out << message << ": " << items_per_second << " Hz" << std::endl;
    }

  private:
    Timer(const Timer &);
    Timer & operator=(const Timer &);
    
    inline double elapsed_to_seconds(boost::timer::nanosecond_type elapsed)
    {
      return double(elapsed) / double(1000000000);
    }

    boost::timer::cpu_timer m_timer;
};

} // namespace core

#endif // TIMER_H
