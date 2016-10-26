#include "sift_matcher_thread_group.h"
#include <assert/assert.h>
#include <cuda_helper/cuda_helper.h>
#include <descriptor_matcher/dot_product_matcher_uchar.h>
#include <features/sift_descriptor.h>

SiftMatcherThreadGroup::SiftMatcherThreadGroup(
  const int num_threads,
  const std::vector<int> & gpu_nums,
  const int max_num_features,
  const float min_matching_distance,
  const float max_matching_ratio)
: m_threads(NULL),
  m_num_threads(num_threads),
  m_gpu_nums(num_threads, 0),
  m_max_num_features(max_num_features),
  m_min_matching_distance(min_matching_distance),
  m_max_matching_ratio(max_matching_ratio),
  m_available_thread_indices(num_threads),
  m_pending_task_signals(NULL),
  m_completed_task_signals(NULL),
  m_thread_match_tasks(num_threads)
{
  if (gpu_nums.size() > 0)
  {
    for (int i = 0; i < num_threads; ++i)
    {
      m_gpu_nums[i] = gpu_nums[i % gpu_nums.size()];
    }
  }

  m_threads = new boost::thread[num_threads];
  m_pending_task_signals = new core::Signal[num_threads];
  m_completed_task_signals = new core::Signal[num_threads];

  for (int i = 0; i < num_threads; ++i)
  {
    m_threads[i] = boost::thread(&SiftMatcherThreadGroup::thread_run, this, i);
    m_available_thread_indices.push(i);
    m_thread_match_tasks[i].reset();
  }
}

SiftMatcherThreadGroup::~SiftMatcherThreadGroup()
{
  if (m_threads != NULL)
  {
    delete [] m_threads;
    m_threads = NULL;
  }
  if (m_pending_task_signals != NULL)
  {
    delete [] m_pending_task_signals;
    m_pending_task_signals = NULL;
  }
  if (m_completed_task_signals != NULL)
  {
    delete [] m_completed_task_signals;
    m_completed_task_signals = NULL;
  }
}

int SiftMatcherThreadGroup::match_descriptors(
  const unsigned char * descriptors1,
  const int num_descriptors1,
  const unsigned char * descriptors2,
  const int num_descriptors2,
  std::vector<std::pair<int, int> > & matches)
{
  matches.clear();
  
  int thread_index = -1;
  const bool success = m_available_thread_indices.pop(thread_index);
  ASSERT(success == true);

  m_thread_match_tasks[thread_index].descriptors1 = descriptors1;
  m_thread_match_tasks[thread_index].num_descriptors1 = num_descriptors1;
  m_thread_match_tasks[thread_index].descriptors2 = descriptors2;
  m_thread_match_tasks[thread_index].num_descriptors2 = num_descriptors2;
  m_thread_match_tasks[thread_index].matches = &matches;

  m_pending_task_signals[thread_index].signal();
  m_completed_task_signals[thread_index].wait();

  m_thread_match_tasks[thread_index].reset();
  m_available_thread_indices.push(thread_index);

  return static_cast<int>(matches.size());
}

void SiftMatcherThreadGroup::thread_run(const int thread_num)
{
  cuda_helper::set_device(m_gpu_nums[thread_num]);
  cuda_helper::initialize_cuda_context();
  cudaStream_t cuda_stream;
  CUDA_CALL(cudaStreamCreate(&cuda_stream));

  descriptor_matcher::DotProductMatcherUChar dot_matcher(
    m_max_num_features,
    features::SiftDescriptorUChar::num_uchars_per_descriptor,
    512,
    m_min_matching_distance,
    m_max_matching_ratio);

  dot_matcher.initialize_for_matching_host_row_major_descriptors_gpu();

  core::Signal * pending_task_signal = &m_pending_task_signals[thread_num];
  core::Signal * completed_task_signal = &m_completed_task_signals[thread_num];
  MatchTask * match_task = &m_thread_match_tasks[thread_num];

  for (;;)
  {
    pending_task_signal->wait();

    const int num_matches = dot_matcher.match_host_row_major_descriptors_gpu(
      match_task->descriptors1,
      match_task->num_descriptors1,
      match_task->descriptors2,
      match_task->num_descriptors2,
      *(match_task->matches),
      cuda_stream);
    ASSERT(num_matches == static_cast<int>(match_task->matches->size()));

    completed_task_signal->signal();
  }
}
