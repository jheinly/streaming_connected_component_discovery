#pragma once
#ifndef MAIN_MODULE_THREAD_HANDLER_H
#define MAIN_MODULE_THREAD_HANDLER_H

#include "image_cache_entry.h"
#include "batch_match_task.h"
#include "geometric_verification_result.h"
#include "logger.h"
#include <main_module/sift_matcher_thread_group.h>
#include <main_module/feature_database_wrapper.h>
#include <main_module/main_args.h>
#include <buffer_data/sift_visual_words_data.h>
#include <core/manager_worker_synchronization.h>
#include <core/shared_counter.h>
#include <core/shared_speed_stats.h>
#include <core/timer.h>
#include <dataset/focal_lookup.h>
#include <descriptor_matcher/dot_product_matcher_uchar.h>
#include <estimator/essential_matrix_estimator.h>
#include <vocab_tree/VocabTree.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <utility>

class MainModuleThreadHandler
{
  public:
    MainModuleThreadHandler(
      const int num_cpu_threads,
      const int num_gpu_threads,
      const std::vector<int> & gpu_nums,
      const MainArgs & main_args,
      VocabTree * vocab_tree,
      FeatureDatabaseWrapper * feature_database,
      boost::unordered_map<int, ImageCacheEntry> * image_cache);

    ~MainModuleThreadHandler();

    void wait_until_threads_are_ready();

    void generate_summary_speed_report(std::string & report);
    void generate_detailed_speed_report(std::string & report);

    void add_whole_batch_to_voc_tree(
      const std::vector<buffer_data::SiftVisualWordsData> & batch_images);

    void get_batch_knn_from_voc_tree(
      const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
      std::vector<std::vector<VocabTree::QueryResult> > & query_results);

    void add_batch_subset_to_voc_tree(
      const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
      const std::vector<int> & batch_subset_indices);

    void add_cached_images_to_voc_tree(
      const std::vector<int> & image_indices);

    void remove_cached_images_from_voc_tree(
      const std::vector<int> & image_indices);

    void run_batch_geometric_verification(
      const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
      const std::vector<dataset::FocalLookup::Result> & batch_image_focals,
      const std::vector<std::vector<BatchMatchTask> > & batch_match_tasks,
      std::vector<std::vector<GeometricVerificationResult> > & batch_successful_match_tasks);

    void run_geometric_verification_cached_pairs(
      const std::vector<std::pair<int, int> > & image_indice_pairs,
      std::vector<GeometricVerificationResult> & successful_pais);

  private:
    MainModuleThreadHandler(const MainModuleThreadHandler &);
    MainModuleThreadHandler & operator=(const MainModuleThreadHandler &);

    void thread_run();

    void thread_add_whole_batch_to_voc_tree(
      VocabTree::ThreadSafeStorage & thread_safe_storage);

    void thread_get_batch_knn_from_voc_tree(
      VocabTree::ThreadSafeStorage & thread_safe_storage);

    void thread_add_batch_subset_to_voc_tree(
      VocabTree::ThreadSafeStorage & thread_safe_storage);

    void thread_add_cached_images_to_voc_tree(
      VocabTree::ThreadSafeStorage & thread_safe_storage,
      std::vector<int> & all_visual_words);

    void thread_remove_cached_images_from_voc_tree(
      VocabTree::ThreadSafeStorage & thread_safe_storage,
      std::vector<int> & all_visual_words);

    void thread_run_batch_geometric_verification(
      estimator::EssentialMatrixEstimator & essential_estimator,
      VocabTree::ThreadSafeStorage & thread_safe_storage);

    void thread_run_geometric_verification_cached_pairs(
      estimator::EssentialMatrixEstimator & essential_estimator,
      VocabTree::ThreadSafeStorage & thread_safe_storage);

    enum ThreadCommand {
      AddWholeBatchToVocTree,
      GetBatchKnnFromVocTree,
      AddBatchSubsetToVocTree,
      AddCachedImagesToVocTree,
      RemoveCachedImagesFromVocTree,
      RunBatchGeometricVerification,
      RunGeometricVerificationCachedPairs,
      Exit
    };
    ThreadCommand m_thread_command;

    boost::thread * m_threads;
    const int m_num_threads;
    core::SharedCounter m_shared_counter;
    core::ManagerWorkerSynchronization m_synchronization;
    SiftMatcherThreadGroup * m_sift_matcher_thread_group;

    core::SharedSpeedStats m_query_knn_voc_tree_speed_stats;
    core::SharedSpeedStats m_add_images_to_voc_tree_speed_stats;
    core::SharedSpeedStats m_remove_images_from_voc_tree_speed_stats;
    core::SharedSpeedStats m_geometric_verification_speed_stats;

    const MainArgs m_main_args;
    VocabTree * m_vocab_tree;
    FeatureDatabaseWrapper * m_feature_database;
    boost::unordered_map<int, ImageCacheEntry> * m_image_cache;
    boost::mutex m_successful_pairs_mutex;
    core::Timer m_timer;

    const std::vector<buffer_data::SiftVisualWordsData> * m_batch_images_arg;
    std::vector<std::vector<VocabTree::QueryResult> > * m_query_results_arg;
    const std::vector<int> * m_batch_subset_indices_arg;
    const std::vector<int> * m_image_indices_arg;
    const std::vector<dataset::FocalLookup::Result> * m_batch_image_focals_arg;
    const std::vector<std::vector<BatchMatchTask> > * m_batch_match_tasks_arg;
    std::vector<std::vector<GeometricVerificationResult> > * m_batch_successful_match_tasks_arg;
    const std::vector<std::pair<int, int> > * m_image_indice_pairs_arg;
    std::vector<GeometricVerificationResult> * m_successful_pairs_arg;

    boost::log::sources::logger_mt & m_logger;
    std::stringstream m_stream;
    int m_geometric_verification_call_count;
    bool m_time_geometric_verification;
    boost::mutex m_times_mutex;
    double m_existing_match_time;
    double m_sift_match_time;
    double m_duplicate_check_time;
    double m_essential_time;
    double m_border_check_time;
    double m_save_inliers_time;
    double m_update_success_time;
};

#endif // MAIN_MODULE_THREAD_HANDLER_H
