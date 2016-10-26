#pragma once
#ifndef THREAD_HELPER_H
#define THREAD_HELPER_H

#include <boost/thread/thread.hpp>

namespace core {

namespace thread_helper {

inline void sleep_for_seconds(const double seconds)
{
  const long long int microseconds =
    static_cast<long long int>(1000000.0 * seconds + 0.5);
  boost::this_thread::sleep_for(boost::chrono::microseconds(microseconds));
}

} // namespace thread_helper

} // namespace core

#endif // THREAD_HELPER_H
