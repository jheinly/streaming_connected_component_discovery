#pragma once
#ifndef SIFT_MATCHER_THREAD_GROUP_H
#define SIFT_MATCHER_THREAD_GROUP_H

#include <core/shared_bounded_queue.h>
#include <core/signal.h>
#include <boost/thread/thread.hpp>
#include <vector>
#include <utility>

class SiftMatcherThreadGroup
{
  public:
    SiftMatcherThreadGroup(
      const int num_threads,
      const std::vector<int> & gpu_nums,
      const int max_num_features,
      const float min_matching_distance,
      const float max_matching_ratio);

    ~SiftMatcherThreadGroup();

    int match_descriptors(
      const unsigned char * descriptors1,
      const int num_descriptors1,
      const unsigned char * descriptors2,
      const int num_descriptors2,
      std::vector<std::pair<int, int> > & matches);

  private:
    void thread_run(const int thread_num);

    struct MatchTask
    {
      void reset()
      {
        descriptors1 = NULL;
        num_descriptors1 = 0;
        descriptors2 = NULL;
        num_descriptors2 = 0;
        matches = NULL;
      }

      const unsigned char * descriptors1;
      int num_descriptors1;
      const unsigned char * descriptors2;
      int num_descriptors2;
      std::vector<std::pair<int, int> > * matches;
    };

    boost::thread * m_threads;
    const int m_num_threads;
    std::vector<int> m_gpu_nums;
    const int m_max_num_features;
    const float m_min_matching_distance;
    const float m_max_matching_ratio;
    core::SharedBoundedQueue<int> m_available_thread_indices;
    core::Signal * m_pending_task_signals;
    core::Signal * m_completed_task_signals;
    std::vector<MatchTask> m_thread_match_tasks;
};

#endif // SIFT_MATCHER_THREAD_GROUP_H
