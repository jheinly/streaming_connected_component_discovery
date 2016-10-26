#include "main_module_thread_handler.h"
#include "logger.h"
#include <cuda_helper/cuda_helper.h>
#include <estimator/check_for_degenerate_match.h>
#include <features/sift_keypoint.h>
#include <features/sift_descriptor.h>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/lock_types.hpp>
#include <boost/thread/lock_algorithms.hpp>
#include <algorithm>
#include <sstream>

#define TIME_CODE
#define TIME_CODE_RATE 12

template<typename Keypoint1, typename Keypoint2>
struct FeatureMatchesAccessor
{
  inline const int num_matches() const
  { return m_num_matches; }

  inline const float x1(const int i) const
  { return (*m_keypoints1).at((*m_matches).at(i).first).x; }

  inline const float y1(const int i) const
  { return (*m_keypoints1).at((*m_matches).at(i).first).y; }

  inline const float x2(const int i) const
  { return (*m_keypoints2).at((*m_matches).at(i).second).x; }

  inline const float y2(const int i) const
  { return (*m_keypoints2).at((*m_matches).at(i).second).y; }

  int m_num_matches;
  const std::vector<Keypoint1> * m_keypoints1;
  const std::vector<Keypoint2> * m_keypoints2;
  const std::vector<std::pair<int, int> > * m_matches;
};

struct InlierIndicesAccessor
{
  InlierIndicesAccessor(
    const int num_inlier_matches,
    const std::vector<std::pair<int, int> > * matches_ptr,
    const std::vector<int> * inlier_match_indices_ptr)
    : m_num_inliers(num_inlier_matches),
    m_matches(matches_ptr),
    m_inlier_match_indices(inlier_match_indices_ptr)
  {}

  inline int num_inliers() const
  { return m_num_inliers; }

  inline int inlier_index1(const int index) const
  { return (*m_matches).at((*m_inlier_match_indices).at(index)).first; }

  inline int inlier_index2(const int index) const
  { return (*m_matches).at((*m_inlier_match_indices).at(index)).second; }

  int m_num_inliers;
  const std::vector<std::pair<int, int> > * m_matches;
  const std::vector<int> * m_inlier_match_indices;
};

struct InlierIndicesAccessorColmap
{
  InlierIndicesAccessorColmap(
    const int num_inlier_matches,
    colmap::FeatureMatches * colmap_feature_matches_ptr)
    : m_num_inliers(num_inlier_matches),
    m_colmap_feature_matches(colmap_feature_matches_ptr)
  {}

  inline int num_inliers() const
  { return m_num_inliers; }

  inline int inlier_index1(const int index) const
  { return (*m_colmap_feature_matches)(index, 0); }

  inline int inlier_index2(const int index) const
  { return (*m_colmap_feature_matches)(index, 1); }

  int m_num_inliers;
  const colmap::FeatureMatches * m_colmap_feature_matches;
};

template<typename InliersAccessor>
void update_cached_image_visual_words(
  const std::vector<int> & visual_words,
  ImageCacheEntry & cached_image,
  const InliersAccessor & inliers_accessor,
  std::vector<int> & new_visual_words,
  const MainArgs & main_args)
{
  new_visual_words.clear();

  if (!main_args.augmented_visual_words_enabled)
  {
    return;
  }

  boost::lock_guard<boost::mutex> auto_lock(*cached_image.visual_words_mutex);

  const int num_inliers = inliers_accessor.num_inliers();
  for (int i = 0; i < num_inliers; ++i)
  {
    const int idx1 = inliers_accessor.inlier_index1(i);
    const int idx2 = inliers_accessor.inlier_index2(i);

    const int new_visual_word = visual_words.at(idx1);
    bool found = false;

    for (size_t j = 0; j < cached_image.visual_words.at(idx2).size(); ++j)
    {
      if (new_visual_word == cached_image.visual_words.at(idx2).at(j))
      {
        found = true;
        break;
      }
    }

    if (found)
    {
      continue;
    }

    cached_image.visual_words.at(idx2).push_back(new_visual_word);

    std::sort(cached_image.visual_words.at(idx2).begin(), cached_image.visual_words.at(idx2).end());
    new_visual_words.push_back(new_visual_word);
  }
}

template<typename InliersAccessor>
void update_cached_image_visual_words(
  ImageCacheEntry & cached_image1,
  ImageCacheEntry & cached_image2,
  const InliersAccessor & inliers_accessor,
  std::vector<int> & new_visual_words1,
  std::vector<int> & new_visual_words2,
  const MainArgs & main_args)
{
  new_visual_words1.clear();
  new_visual_words2.clear();

  if (!main_args.augmented_visual_words_enabled)
  {
    return;
  }

  boost::lock(
    *cached_image1.visual_words_mutex,
    *cached_image2.visual_words_mutex);

  std::vector<int> new_words1;
  std::vector<int> new_words2;

  const int num_inliers = inliers_accessor.num_inliers();
  for (int i = 0; i < num_inliers; ++i)
  {
    const int idx1 = inliers_accessor.inlier_index1(i);
    const int idx2 = inliers_accessor.inlier_index2(i);

    // These will already be sorted, so they can directly be used in std::set_difference()
    std::vector<int> & words1 = cached_image1.visual_words.at(idx1);
    std::vector<int> & words2 = cached_image2.visual_words.at(idx2);

    new_words1.resize(words2.size());
    std::vector<int>::iterator iter1 =
      std::set_difference(words2.begin(), words2.end(), words1.begin(), words1.end(), new_words1.begin());
    new_words1.resize(iter1 - new_words1.begin());

    new_words2.resize(words1.size());
    std::vector<int>::iterator iter2 =
      std::set_difference(words1.begin(), words1.end(), words2.begin(), words2.end(), new_words2.begin());
    new_words2.resize(iter2 - new_words2.begin());

    words1.insert(
      words1.end(),
      new_words1.begin(),
      new_words1.end());
    words2.insert(
      words2.end(),
      new_words2.begin(),
      new_words2.end());

    std::sort(words1.begin(), words1.end());
    std::sort(words2.begin(), words2.end());

    new_visual_words1.insert(
      new_visual_words1.end(),
      new_words1.begin(),
      new_words1.end());
    new_visual_words2.insert(
      new_visual_words2.end(),
      new_words2.begin(),
      new_words2.end());
  }

  cached_image1.visual_words_mutex->unlock();
  cached_image2.visual_words_mutex->unlock();
}

MainModuleThreadHandler::MainModuleThreadHandler(
  const int num_cpu_threads,
  const int num_gpu_threads,
  const std::vector<int> & gpu_nums,
  const MainArgs & main_args,
  VocabTree * vocab_tree,
  FeatureDatabaseWrapper * feature_database,
  boost::unordered_map<int, ImageCacheEntry> * image_cache)
: m_thread_command(Exit),
  m_threads(NULL),
  m_num_threads(num_cpu_threads),
  m_shared_counter(),
  m_synchronization(num_cpu_threads),
  m_sift_matcher_thread_group(NULL),
  m_query_knn_voc_tree_speed_stats(),
  m_add_images_to_voc_tree_speed_stats(),
  m_remove_images_from_voc_tree_speed_stats(),
  m_geometric_verification_speed_stats(),
  m_main_args(main_args),
  m_vocab_tree(vocab_tree),
  m_feature_database(feature_database),
  m_image_cache(image_cache),
  m_successful_pairs_mutex(),
  m_timer(),
  m_batch_images_arg(NULL),
  m_query_results_arg(NULL),
  m_batch_subset_indices_arg(NULL),
  m_image_indices_arg(NULL),
  m_batch_match_tasks_arg(NULL),
  m_batch_successful_match_tasks_arg(NULL),
  m_image_indice_pairs_arg(NULL),
  m_successful_pairs_arg(NULL),
  m_logger(my_logger::get()),
  m_stream(),
  m_geometric_verification_call_count(0)
{
  m_timer.stop();
  m_stream.setf(std::ios_base::fixed, std::ios_base::floatfield);
  m_stream.precision(6);

  m_sift_matcher_thread_group = new SiftMatcherThreadGroup(
    num_gpu_threads,
    gpu_nums,
    main_args.max_num_features,
    main_args.min_matching_distance,
    main_args.max_matching_ratio);

  // Create the threads.
  m_threads = new boost::thread[num_cpu_threads];

  for (int i = 0; i < num_cpu_threads; ++i)
  {
    m_threads[i] = boost::thread(&MainModuleThreadHandler::thread_run, this);
  }
}

MainModuleThreadHandler::~MainModuleThreadHandler()
{
  if (m_threads != NULL)
  {
    delete [] m_threads;
    m_threads = NULL;
  }

  if (m_sift_matcher_thread_group != NULL)
  {
    delete m_sift_matcher_thread_group;
    m_sift_matcher_thread_group = NULL;
  }
}

void MainModuleThreadHandler::wait_until_threads_are_ready()
{
  m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
}

void MainModuleThreadHandler::generate_summary_speed_report(std::string & report)
{
  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);
  stream << "MainModuleThreadHandler" << std::endl;
  stream << "  Query KNN:        ";
  stream.width(7);
  stream << m_query_knn_voc_tree_speed_stats.get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << m_query_knn_voc_tree_speed_stats.get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << m_query_knn_voc_tree_speed_stats.get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;
  stream << "  Add to Voc-Tree:  ";
  stream.width(7);
  stream << m_add_images_to_voc_tree_speed_stats.get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << m_add_images_to_voc_tree_speed_stats.get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << m_add_images_to_voc_tree_speed_stats.get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;
  stream << "  Remove from Tree: ";
  stream.width(7);
  stream << m_remove_images_from_voc_tree_speed_stats.get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << m_remove_images_from_voc_tree_speed_stats.get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << m_remove_images_from_voc_tree_speed_stats.get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;
  stream << "  Geometric Verif:  ";
  stream.width(7);
  stream << m_geometric_verification_speed_stats.get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << m_geometric_verification_speed_stats.get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << m_geometric_verification_speed_stats.get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;
  report = stream.str();
}

void MainModuleThreadHandler::generate_detailed_speed_report(std::string & report)
{
  generate_summary_speed_report(report);
}

void MainModuleThreadHandler::add_whole_batch_to_voc_tree(
  const std::vector<buffer_data::SiftVisualWordsData> & batch_images)
{
  m_timer.start();
  if (batch_images.size() > 0)
  {
    m_batch_images_arg = &batch_images;
    m_thread_command = AddWholeBatchToVocTree;
    m_shared_counter.reset();
    m_synchronization.signal_workers();
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
    m_batch_images_arg = NULL;
  }
  m_timer.stop();
  m_add_images_to_voc_tree_speed_stats.add_timing(
    m_timer.elapsed(),
    static_cast<int>(batch_images.size()));
}

void MainModuleThreadHandler::get_batch_knn_from_voc_tree(
  const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
  std::vector<std::vector<VocabTree::QueryResult> > & query_results)
{
  m_timer.start();
  query_results.resize(batch_images.size());
  if (batch_images.size() > 0)
  {
    m_batch_images_arg = &batch_images;
    m_query_results_arg = &query_results;
    m_thread_command = GetBatchKnnFromVocTree;
    m_shared_counter.reset();
    m_synchronization.signal_workers();
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
    m_batch_images_arg = NULL;
    m_query_results_arg = NULL;
  }
  m_timer.stop();
  m_query_knn_voc_tree_speed_stats.add_timing(
    m_timer.elapsed(),
    static_cast<int>(batch_images.size()));
}

void MainModuleThreadHandler::add_batch_subset_to_voc_tree(
  const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
  const std::vector<int> & batch_subset_indices)
{
  m_timer.start();
  if (batch_subset_indices.size() > 0)
  {
    m_batch_images_arg = &batch_images;
    m_batch_subset_indices_arg = &batch_subset_indices;
    m_thread_command = AddBatchSubsetToVocTree;
    m_shared_counter.reset();
    m_synchronization.signal_workers();
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
    m_batch_images_arg = NULL;
    m_batch_subset_indices_arg = NULL;
  }
  m_add_images_to_voc_tree_speed_stats.add_timing(
    m_timer.elapsed(),
    static_cast<int>(batch_subset_indices.size()));
}

void MainModuleThreadHandler::add_cached_images_to_voc_tree(
  const std::vector<int> & image_indices)
{
  // Don't time this as it will only be called with a few images.
  if (image_indices.size() > 0)
  {
    m_image_indices_arg = &image_indices;
    m_thread_command = AddCachedImagesToVocTree;
    m_shared_counter.reset();
    m_synchronization.signal_workers();
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
    m_image_indices_arg = NULL;
  }
}

void MainModuleThreadHandler::remove_cached_images_from_voc_tree(
  const std::vector<int> & image_indices)
{
  m_timer.start();
  if (image_indices.size() > 0)
  {
    m_image_indices_arg = &image_indices;
    m_thread_command = RemoveCachedImagesFromVocTree;
    m_shared_counter.reset();
    m_synchronization.signal_workers();
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
    m_image_indices_arg = NULL;
    m_timer.stop();
  }
  m_remove_images_from_voc_tree_speed_stats.add_timing(
    m_timer.elapsed(),
    static_cast<int>(image_indices.size()));
}

void MainModuleThreadHandler::run_batch_geometric_verification(
  const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
  const std::vector<dataset::FocalLookup::Result> & batch_image_focals,
  const std::vector<std::vector<BatchMatchTask> > & batch_match_tasks,
  std::vector<std::vector<GeometricVerificationResult> > & batch_successful_match_tasks)
{
#ifdef TIME_CODE
  m_time_geometric_verification = m_geometric_verification_call_count % TIME_CODE_RATE == 0;
  if (m_time_geometric_verification)
  {
    m_existing_match_time = 0.0;
    m_sift_match_time = 0.0;
    m_duplicate_check_time = 0.0;
    m_essential_time = 0.0;
    m_border_check_time = 0.0;
    m_save_inliers_time = 0.0;
    m_update_success_time = 0.0;
  }
  ++m_geometric_verification_call_count;
#endif
  m_timer.start();
  ASSERT(batch_images.size() == batch_image_focals.size());
  ASSERT(batch_images.size() == batch_match_tasks.size());
  batch_successful_match_tasks.resize(batch_images.size());
  if (batch_match_tasks.size() > 0)
  {
    m_batch_images_arg = &batch_images;
    m_batch_image_focals_arg = &batch_image_focals;
    m_batch_match_tasks_arg = &batch_match_tasks;
    m_batch_successful_match_tasks_arg = &batch_successful_match_tasks;
    m_thread_command = RunBatchGeometricVerification;
    m_shared_counter.reset();
    m_synchronization.signal_workers();
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
    m_batch_images_arg = NULL;
    m_batch_image_focals_arg = NULL;
    m_batch_match_tasks_arg = NULL;
    m_batch_successful_match_tasks_arg = NULL;
  }
  m_timer.stop();
  m_geometric_verification_speed_stats.add_timing(
    m_timer.elapsed(),
    static_cast<int>(batch_images.size()));
#ifdef TIME_CODE
  if (m_time_geometric_verification)
  {
    const int width = 9;
    m_stream.str("");
    m_stream.clear();
    m_stream << "BATCH GEOMETRIC VERIFICATION" << std::endl;
    m_stream << "Existing Matches: ";
    m_stream.width(width);
    m_stream << m_existing_match_time << " sec" << std::endl;
    m_stream << "SIFT Matches:     ";
    m_stream.width(width);
    m_stream << m_sift_match_time << " sec" << std::endl;
    m_stream << "Duplicate Check:  ";
    m_stream.width(width);
    m_stream << m_duplicate_check_time << " sec" << std::endl;
    m_stream << "Essential Matrix: ";
    m_stream.width(width);
    m_stream << m_essential_time << " sec" << std::endl;
    m_stream << "Border Check:     ";
    m_stream.width(width);
    m_stream << m_border_check_time << " sec" << std::endl;
    m_stream << "Save Inliers:     ";
    m_stream.width(width);
    m_stream << m_save_inliers_time << " sec" << std::endl;
    m_stream << "Update Success:   ";
    m_stream.width(width);
    m_stream << m_update_success_time << " sec" << std::endl;
    m_stream << std::endl;
    BOOST_LOG(m_logger) << m_stream.str();
  }
#endif
}

void MainModuleThreadHandler::run_geometric_verification_cached_pairs(
  const std::vector<std::pair<int, int> > & image_indice_pairs,
  std::vector<GeometricVerificationResult> & successful_pairs)
{
  // Don't time this as it will only be run on a few pairs.
  successful_pairs.clear();
  if (image_indice_pairs.size() > 0)
  {
    m_image_indice_pairs_arg = &image_indice_pairs;
    m_successful_pairs_arg = &successful_pairs;
    m_thread_command = RunGeometricVerificationCachedPairs;
    m_shared_counter.reset();
    m_synchronization.signal_workers();
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();
    m_image_indice_pairs_arg = NULL;
    m_successful_pairs_arg = NULL;
  }
}

void MainModuleThreadHandler::thread_run()
{
  VocabTree::ThreadSafeStorage thread_safe_storage(m_vocab_tree);

  std::vector<int> all_visual_words;

  estimator::EssentialMatrixEstimator essential_estimator(
    m_main_args.max_num_ransac_iterations,
    m_main_args.min_num_inliers_for_database,
    m_main_args.max_pixel_error_ransac);

  for (;;)
  {
    m_synchronization.wait_for_signal_from_manager();

    switch (m_thread_command)
    {
      case AddWholeBatchToVocTree:
        thread_add_whole_batch_to_voc_tree(
          thread_safe_storage);
        break;
      case GetBatchKnnFromVocTree:
        thread_get_batch_knn_from_voc_tree(
          thread_safe_storage);
        break;
      case AddBatchSubsetToVocTree:
        thread_add_batch_subset_to_voc_tree(
          thread_safe_storage);
        break;
      case AddCachedImagesToVocTree:
        thread_add_cached_images_to_voc_tree(
          thread_safe_storage,
          all_visual_words);
        break;
      case RemoveCachedImagesFromVocTree:
        thread_remove_cached_images_from_voc_tree(
          thread_safe_storage,
          all_visual_words);
        break;
      case RunBatchGeometricVerification:
        thread_run_batch_geometric_verification(
          essential_estimator,
          thread_safe_storage);
        break;
      case RunGeometricVerificationCachedPairs:
        thread_run_geometric_verification_cached_pairs(
          essential_estimator,
          thread_safe_storage);
        break;
      case Exit:
        return;
    }
  }
}

void MainModuleThreadHandler::thread_add_whole_batch_to_voc_tree(
  VocabTree::ThreadSafeStorage & thread_safe_storage)
{
  for (;;)
  {
    const int index_within_batch = m_shared_counter.get_value_and_increment();
    if (index_within_batch >= static_cast<int>(m_batch_images_arg->size()))
    {
      break;
    }

    const buffer_data::SiftVisualWordsData & batch_image =
      (*m_batch_images_arg).at(index_within_batch);
    m_vocab_tree->v2_add_image_to_database_thread_safe(
      batch_image.image_index,
      batch_image.visual_words,
      true,
      thread_safe_storage);
  }
}

void MainModuleThreadHandler::thread_get_batch_knn_from_voc_tree(
  VocabTree::ThreadSafeStorage & thread_safe_storage)
{
  for (;;)
  {
    const int index_within_batch = m_shared_counter.get_value_and_increment();
    if (index_within_batch >= static_cast<int>(m_batch_images_arg->size()))
    {
      break;
    }

    const buffer_data::SiftVisualWordsData & batch_image =
      (*m_batch_images_arg).at(index_within_batch);
    std::vector<VocabTree::QueryResult> & results =
      (*m_query_results_arg).at(index_within_batch);
    m_vocab_tree->v2_query_database_thread_safe(
      batch_image.visual_words,
      results,
      m_main_args.vocab_tree_num_knn,
      true,
      thread_safe_storage);
  }
}

void MainModuleThreadHandler::thread_add_batch_subset_to_voc_tree(
  VocabTree::ThreadSafeStorage & thread_safe_storage)
{
  for (;;)
  {
    const int index_with_batch_subset = m_shared_counter.get_value_and_increment();
    if (index_with_batch_subset >= static_cast<int>(m_batch_subset_indices_arg->size()))
    {
      break;
    }

    const int subset_index = (*m_batch_subset_indices_arg).at(index_with_batch_subset);

    const buffer_data::SiftVisualWordsData & batch_image =
      (*m_batch_images_arg).at(subset_index);
    m_vocab_tree->v2_add_image_to_database_thread_safe(
      batch_image.image_index,
      batch_image.visual_words,
      true,
      thread_safe_storage);
  }
}

void MainModuleThreadHandler::thread_add_cached_images_to_voc_tree(
  VocabTree::ThreadSafeStorage & thread_safe_storage,
  std::vector<int> & all_visual_words)
{
  for (;;)
  {
    const int index = m_shared_counter.get_value_and_increment();
    if (index >= static_cast<int>(m_image_indices_arg->size()))
    {
      break;
    }

    const int image_index = (*m_image_indices_arg).at(index);

    ASSERT(m_image_cache->find(image_index) != m_image_cache->end());
    const ImageCacheEntry & image = (*m_image_cache)[image_index];
    image.concatenate_all_visual_words(all_visual_words);

    m_vocab_tree->v2_add_image_to_database_thread_safe(
      image_index,
      all_visual_words,
      true,
      thread_safe_storage);
  }
}

void MainModuleThreadHandler::thread_remove_cached_images_from_voc_tree(
  VocabTree::ThreadSafeStorage & thread_safe_storage,
  std::vector<int> & all_visual_words)
{
  for (;;)
  {
    const int index = m_shared_counter.get_value_and_increment();
    if (index >= static_cast<int>(m_image_indices_arg->size()))
    {
      break;
    }

    const int image_index = (*m_image_indices_arg).at(index);

    ASSERT(m_image_cache->find(image_index) != m_image_cache->end());
    const ImageCacheEntry & image = (*m_image_cache)[image_index];
    image.concatenate_all_visual_words(all_visual_words);

    m_vocab_tree->v2_remove_image_from_database_thread_safe(
      image_index,
      all_visual_words,
      thread_safe_storage);
  }
}

void MainModuleThreadHandler::thread_run_batch_geometric_verification(
  estimator::EssentialMatrixEstimator & essential_estimator,
  VocabTree::ThreadSafeStorage & thread_safe_storage)
{
  std::map<int, int> matched_connected_components;
  std::vector<std::pair<int, int> > matches;
  std::vector<int> inlier_match_indices;
  std::vector<int> new_visual_words;

#ifdef TIME_CODE
  core::Timer timer;
  double existing_match_time = 0.0;
  double sift_match_time = 0.0;
  double duplicate_check_time = 0.0;
  double essential_time = 0.0;
  double border_check_time = 0.0;
  double save_inliers_time = 0.0;
  double update_success_time = 0.0;
#endif

  for (;;)
  {
    const int index_within_batch = m_shared_counter.get_value_and_increment();
    if (index_within_batch >= static_cast<int>(m_batch_images_arg->size()))
    {
      break;
    }

    const buffer_data::SiftVisualWordsData & batch_image =
      (*m_batch_images_arg).at(index_within_batch);

    const std::vector<BatchMatchTask> & match_tasks =
      (*m_batch_match_tasks_arg).at(index_within_batch);

    std::vector<GeometricVerificationResult> & successful_match_tasks =
      (*m_batch_successful_match_tasks_arg).at(index_within_batch);
    successful_match_tasks.clear();

    const int image_index1 = batch_image.image_index;
    const core::ImageDimensions dimensions1 = batch_image.dimensions;
    const float focal1 = (*m_batch_image_focals_arg).at(index_within_batch).focal_in_pixels;

    matched_connected_components.clear();
    int num_match_attempts = 0;

    const int num_match_tasks = static_cast<int>(match_tasks.size());
    for (int match_task_index = 0; match_task_index < num_match_tasks; ++match_task_index)
    {
      if (num_match_attempts >= m_main_args.max_match_attempts_per_image)
      {
        break;
      }

      const BatchMatchTask & match_task = match_tasks.at(match_task_index);

      // If we've already matched to this connected component the maximum number
      // of times, skip the match task.
      std::map<int, int>::iterator matched_connected_component =
        matched_connected_components.find(match_task.connected_component_index);
      if (matched_connected_component != matched_connected_components.end() &&
        matched_connected_component->second >= m_main_args.max_matches_per_connected_component)
      {
        continue;
      }

      const int image_index2 = match_task.image_index;

      boost::unordered_map<int, ImageCacheEntry>::iterator found_image =
        m_image_cache->find(image_index2);
      ASSERT(found_image != m_image_cache->end());
      ImageCacheEntry & image2 = found_image->second;

      // Attempt to find an existing match between these images.
      // NOTE: this could only happen if the match tasks have duplicate entries,
      //       or if the program is being run with a database that has already
      //       been populated.
      {
#ifdef TIME_CODE
        if (m_time_geometric_verification)
        {
          timer.start();
        }
#endif
        boost::unique_lock<boost::mutex> database_lock(m_feature_database->mutex);
        const bool already_exists = m_feature_database->database.exists_inlier_matches(
          colmap::image_t(image_index1),
          colmap::image_t(image_index2));
#ifdef TIME_CODE
        if (m_time_geometric_verification)
        {
          existing_match_time += timer.elapsed();
        }
#endif
        if (already_exists)
        {
          colmap::TwoViewGeometry colmap_two_view_geometry =
            m_feature_database->database.read_inlier_matches(
              colmap::image_t(image_index1),
              colmap::image_t(image_index2));

          // Unlock the database as we don't need to access it anymore in this code block.
          database_lock.unlock();

          const int num_inliers = static_cast<int>(colmap_two_view_geometry.inlier_matches.rows());
          if (num_inliers >= m_main_args.min_num_inliers_for_successful_match)
          {
            successful_match_tasks.push_back(
              GeometricVerificationResult(match_task_index, num_inliers));
            if (matched_connected_component != matched_connected_components.end())
            {
              matched_connected_component->second += 1;
            }
            else
            {
              matched_connected_components[match_task.connected_component_index] = 1;
            }

            update_cached_image_visual_words(
              batch_image.visual_words,
              image2,
              InlierIndicesAccessorColmap(num_inliers, &colmap_two_view_geometry.inlier_matches),
              new_visual_words,
              m_main_args);

            if (new_visual_words.size() > 0)
            {
              // NOTE: image_index2 is already in the vocab_tree
              m_vocab_tree->v2_add_visual_words_to_image_thread_safe(
                image_index2,
                new_visual_words,
                true,
                thread_safe_storage);
            }
          }
          ++num_match_attempts;
          continue;
        }
      }

      matches.clear();
      int num_matches = 0;

#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        timer.start();
      }
#endif
      num_matches = m_sift_matcher_thread_group->match_descriptors(
        batch_image.descriptors_uchar.at(0).uchars,
        batch_image.num_features,
        image2.descriptors_uchar.at(0).uchars,
        image2.num_features,
        matches);
      ASSERT(num_matches == static_cast<int>(matches.size()));
      for (int i = 0; i < num_matches; ++i)
      {
        ASSERT(matches.at(i).first < batch_image.num_features);
        ASSERT(matches.at(i).second < image2.num_features);
      }
#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        sift_match_time += timer.elapsed();
      }
#endif

      // TODO: revisit this and figure out how to deal with duplicate images
#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        timer.start();
      }
#endif
      const bool are_duplicate_images = estimator::check_matches_for_duplicate_images(
        batch_image.keypoints,
        image2.keypoints,
        matches);
#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        duplicate_check_time += timer.elapsed();
      }
#endif
      if (are_duplicate_images)
      {
        num_matches = 0;
        matches.clear();
      }

      // If this pair has too few matches, skip to the next match task.
      if (num_matches < m_main_args.min_num_inliers_for_database)
      {
        colmap::TwoViewGeometry colmap_two_view_geometry;
        colmap_two_view_geometry.config = colmap::TwoViewGeometry::CALIBRATED;
        colmap_two_view_geometry.inlier_matches.resize(0, 0);
        {
          boost::lock_guard<boost::mutex> database_lock(m_feature_database->mutex);
          m_feature_database->database.write_inlier_matches(
            colmap::image_t(image_index1),
            colmap::image_t(image_index2),
            colmap_two_view_geometry);
        }

        ++num_match_attempts;
        continue;
      }

      const core::ImageDimensions dimensions2 = image2.dimensions;
      const float focal2 = image2.focal;
      inlier_match_indices.clear();

      FeatureMatchesAccessor<features::SiftKeypoint, features::SiftKeypoint> feature_matches_accessor;
      feature_matches_accessor.m_num_matches = num_matches;
      feature_matches_accessor.m_keypoints1 = &batch_image.keypoints;
      feature_matches_accessor.m_keypoints2 = &image2.keypoints;
      feature_matches_accessor.m_matches = &matches;

#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        timer.start();
      }
#endif
      float essential[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
      int num_inliers = essential_estimator.estimate(
        feature_matches_accessor,
        focal1,
        dimensions1.width,
        dimensions1.height,
        focal2,
        dimensions2.width,
        dimensions2.height,
        inlier_match_indices,
        essential);
      ASSERT(num_inliers == static_cast<int>(inlier_match_indices.size()));
      for (int i = 0; i < num_inliers; ++i)
      {
        ASSERT(inlier_match_indices.at(i) < num_matches);
      }
#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        essential_time += timer.elapsed();
      }
#endif

      ++num_match_attempts;

      if (num_inliers < m_main_args.min_num_inliers_for_database)
      {
        num_inliers = 0;
        inlier_match_indices.clear();
      }

#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        timer.start();
      }
#endif
      const bool is_border_match = estimator::check_inliers_for_border_matches(
        dimensions1,
        dimensions2,
        batch_image.keypoints,
        image2.keypoints,
        matches,
        inlier_match_indices);
#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        border_check_time += timer.elapsed();
      }
#endif
      if (is_border_match)
      {
        num_inliers = 0;
        inlier_match_indices.clear();
      }

#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        timer.start();
      }
#endif
      // Save inlier matches to the database.
      colmap::TwoViewGeometry colmap_two_view_geometry;
      colmap_two_view_geometry.config = colmap::TwoViewGeometry::CALIBRATED;
      colmap_two_view_geometry.inlier_matches.resize(num_inliers, 2);
      for (int i = 0; i < num_inliers; ++i)
      {
        colmap_two_view_geometry.inlier_matches(i, 0) = matches.at(inlier_match_indices.at(i)).first;
        colmap_two_view_geometry.inlier_matches(i, 1) = matches.at(inlier_match_indices.at(i)).second;
      }
      {
        boost::lock_guard<boost::mutex> database_lock(m_feature_database->mutex);
        m_feature_database->database.write_inlier_matches(
          colmap::image_t(image_index1),
          colmap::image_t(image_index2),
          colmap_two_view_geometry);
      }
#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        save_inliers_time += timer.elapsed();
      }
#endif

#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        timer.start();
      }
#endif
      if (num_inliers >= m_main_args.min_num_inliers_for_successful_match)
      {
        successful_match_tasks.push_back(
          GeometricVerificationResult(match_task_index, num_inliers));

        if (matched_connected_component != matched_connected_components.end())
        {
          matched_connected_component->second += 1;
        }
        else
        {
          matched_connected_components[match_task.connected_component_index] = 1;
        }

        update_cached_image_visual_words(
          batch_image.visual_words,
          image2,
          InlierIndicesAccessor(num_inliers, &matches, &inlier_match_indices),
          new_visual_words,
          m_main_args);

        if (new_visual_words.size() > 0)
        {
          // NOTE: image_index2 is already in the vocab_tree
          m_vocab_tree->v2_add_visual_words_to_image_thread_safe(
            image_index2,
            new_visual_words,
            true,
            thread_safe_storage);
        }
      }
#ifdef TIME_CODE
      if (m_time_geometric_verification)
      {
        update_success_time += timer.elapsed();
      }
#endif
    } // End loop over batch image's match tasks.

    if (num_match_attempts < m_main_args.max_match_attempts_per_image)
    {
      BOOST_LOG(m_logger) << "WARNING: not all match attempts were used, vocab_tree_num_knn is probably too low" << std::endl;
    }
  } // End loop over entire batch.

#ifdef TIME_CODE
  if (m_time_geometric_verification)
  {
    boost::lock_guard<boost::mutex> auto_lock(m_times_mutex);
    m_existing_match_time += existing_match_time;
    m_sift_match_time += sift_match_time;
    m_duplicate_check_time += duplicate_check_time;
    m_essential_time += essential_time;
    m_border_check_time += border_check_time;
    m_save_inliers_time += save_inliers_time;
    m_update_success_time += update_success_time;
  }
#endif
}

void MainModuleThreadHandler::thread_run_geometric_verification_cached_pairs(
  estimator::EssentialMatrixEstimator & essential_estimator,
  VocabTree::ThreadSafeStorage & thread_safe_storage)
{
  std::vector<std::pair<int, int> > matches;
  std::vector<int> inlier_match_indices;
  std::vector<int> new_visual_words1;
  std::vector<int> new_visual_words2;

  for (;;)
  {
    const int index_within_pairs = m_shared_counter.get_value_and_increment();
    if (index_within_pairs >= static_cast<int>(m_image_indice_pairs_arg->size()))
    {
      break;
    }

    const std::pair<int, int> & image_indice_pair =
      (*m_image_indice_pairs_arg).at(index_within_pairs);

    const int image_index1 = image_indice_pair.first;
    const int image_index2 = image_indice_pair.second;

    boost::unordered_map<int, ImageCacheEntry>::iterator found_image1 =
      m_image_cache->find(image_index1);
    boost::unordered_map<int, ImageCacheEntry>::iterator found_image2 =
      m_image_cache->find(image_index2);
    ASSERT(found_image1 != m_image_cache->end());
    ASSERT(found_image2 != m_image_cache->end());
    ImageCacheEntry & image1 = found_image1->second;
    ImageCacheEntry & image2 = found_image2->second;

    // Attempt to find an existing match between these images.
    {
      boost::unique_lock<boost::mutex> database_lock(m_feature_database->mutex);
      const bool already_exists = m_feature_database->database.exists_inlier_matches(
        colmap::image_t(image_index1),
        colmap::image_t(image_index2));
      if (already_exists)
      {
        colmap::TwoViewGeometry colmap_two_view_geometry =
          m_feature_database->database.read_inlier_matches(
            colmap::image_t(image_index1),
            colmap::image_t(image_index2));

        // Unlock the database as we don't need it anymore in this code block.
        database_lock.unlock();

        const int num_inliers = static_cast<int>(colmap_two_view_geometry.inlier_matches.rows());
        if (num_inliers >= m_main_args.min_num_inliers_for_successful_match)
        {
          {
            boost::lock_guard<boost::mutex> auto_lock(m_successful_pairs_mutex);
            m_successful_pairs_arg->push_back(
              GeometricVerificationResult(index_within_pairs, num_inliers));
          }

          update_cached_image_visual_words(
            image1,
            image2,
            InlierIndicesAccessorColmap(num_inliers, &colmap_two_view_geometry.inlier_matches),
            new_visual_words1,
            new_visual_words2,
            m_main_args);

          if (new_visual_words1.size() > 0 &&
              m_vocab_tree->v2_is_image_in_database(image_index1))
          {
            m_vocab_tree->v2_add_visual_words_to_image_thread_safe(
              image_index1,
              new_visual_words1,
              true,
              thread_safe_storage);
          }
          if (new_visual_words2.size() > 0 &&
              m_vocab_tree->v2_is_image_in_database(image_index2))
          {
            m_vocab_tree->v2_add_visual_words_to_image_thread_safe(
              image_index2,
              new_visual_words2,
              true,
              thread_safe_storage);
          }
        }
        continue;
      }
    }

    matches.clear();
    int num_matches = 0;

    num_matches = m_sift_matcher_thread_group->match_descriptors(
      image1.descriptors_uchar.at(0).uchars,
      image1.num_features,
      image2.descriptors_uchar.at(0).uchars,
      image2.num_features,
      matches);
    ASSERT(num_matches == static_cast<int>(matches.size()));
    for (int i = 0; i < num_matches; ++i)
    {
      ASSERT(matches.at(i).first < image1.num_features);
      ASSERT(matches.at(i).second < image2.num_features);
    }

    // TODO: revisit this and figure out how to deal with duplicate images
    const bool are_duplicate_images = estimator::check_matches_for_duplicate_images(
      image1.keypoints,
      image2.keypoints,
      matches);
    if (are_duplicate_images)
    {
      num_matches = 0;
      matches.clear();
    }

    // If this pair has too few matches, skip to the next match task.
    if (num_matches < m_main_args.min_num_inliers_for_database)
    {
      colmap::TwoViewGeometry colmap_two_view_geometry;
      colmap_two_view_geometry.config = colmap::TwoViewGeometry::CALIBRATED;
      colmap_two_view_geometry.inlier_matches.resize(0, 0);
      {
        boost::lock_guard<boost::mutex> database_lock(m_feature_database->mutex);
        m_feature_database->database.write_inlier_matches(
          colmap::image_t(image_index1),
          colmap::image_t(image_index2),
          colmap_two_view_geometry);
      }

      continue;
    }

    inlier_match_indices.clear();

    FeatureMatchesAccessor<features::SiftKeypoint, features::SiftKeypoint> feature_matches_accessor;
    feature_matches_accessor.m_num_matches = num_matches;
    feature_matches_accessor.m_keypoints1 = &image1.keypoints;
    feature_matches_accessor.m_keypoints2 = &image2.keypoints;
    feature_matches_accessor.m_matches = &matches;

    float essential[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int num_inliers = essential_estimator.estimate(
      feature_matches_accessor,
      image1.focal,
      image1.dimensions.width,
      image1.dimensions.height,
      image2.focal,
      image2.dimensions.width,
      image2.dimensions.height,
      inlier_match_indices,
      essential);
    ASSERT(num_inliers == static_cast<int>(inlier_match_indices.size()));
    for (int i = 0; i < num_inliers; ++i)
    {
      ASSERT(inlier_match_indices.at(i) < num_matches);
    }

    if (num_inliers < m_main_args.min_num_inliers_for_database)
    {
      num_inliers = 0;
      inlier_match_indices.clear();
    }

    const bool is_border_match = estimator::check_inliers_for_border_matches(
      image1.dimensions,
      image2.dimensions,
      image1.keypoints,
      image2.keypoints,
      matches,
      inlier_match_indices);
    if (is_border_match)
    {
      num_inliers = 0;
      inlier_match_indices.clear();
    }

    // Save inlier matches to the database.
    colmap::TwoViewGeometry colmap_two_view_geometry;
    colmap_two_view_geometry.config = colmap::TwoViewGeometry::CALIBRATED;
    colmap_two_view_geometry.inlier_matches.resize(num_inliers, 2);
    for (int i = 0; i < num_inliers; ++i)
    {
      colmap_two_view_geometry.inlier_matches(i, 0) = matches.at(inlier_match_indices.at(i)).first;
      colmap_two_view_geometry.inlier_matches(i, 1) = matches.at(inlier_match_indices.at(i)).second;
    }
    {
      boost::lock_guard<boost::mutex> database_lock(m_feature_database->mutex);
      m_feature_database->database.write_inlier_matches(
        colmap::image_t(image_index1),
        colmap::image_t(image_index2),
        colmap_two_view_geometry);
    }

    if (num_inliers >= m_main_args.min_num_inliers_for_successful_match)
    {
      {
        boost::lock_guard<boost::mutex> auto_lock(m_successful_pairs_mutex);
        m_successful_pairs_arg->push_back(
          GeometricVerificationResult(index_within_pairs, num_inliers));
      }

      update_cached_image_visual_words(
        image1,
        image2,
        InlierIndicesAccessorColmap(num_inliers, &colmap_two_view_geometry.inlier_matches),
        new_visual_words1,
        new_visual_words2,
        m_main_args);

      if (new_visual_words1.size() > 0 &&
          m_vocab_tree->v2_is_image_in_database(image_index1))
      {
        m_vocab_tree->v2_add_visual_words_to_image_thread_safe(
          image_index1,
          new_visual_words1,
          true,
          thread_safe_storage);
      }
      if (new_visual_words2.size() > 0 &&
          m_vocab_tree->v2_is_image_in_database(image_index2))
      {
        m_vocab_tree->v2_add_visual_words_to_image_thread_safe(
          image_index2,
          new_visual_words2,
          true,
          thread_safe_storage);
      }
    }
  } // End loop over image pairs.
}
