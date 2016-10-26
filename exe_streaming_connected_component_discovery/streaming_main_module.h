#pragma once
#ifndef STREAMING_MAIN_MODULE_H
#define STREAMING_MAIN_MODULE_H

#include "image_cache_entry.h"
#include "main_module_thread_handler.h"
#include "streaming_state.h"
#include <main_module/streaming_ipc2sfm_types.h>
#include <main_module/feature_database_wrapper.h>
#include <main_module/main_args.h>
#include <buffer_data/sift_visual_words_data.h>
#include <buffer_data/output_sift_features_data.h>
#include <core/manager_worker_synchronization.h>
#include <core/streaming_module_interface.h>
#include <core/shared_batch_buffer.h>
#include <core/shared_value.h>
#include <core/timer.h>
#include <dataset/focal_lookup.h>
#include <v3d_support/pcdb.h>
#include <vocab_tree/VocabTree.h>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <vector>

class StreamingMainModule : public core::StreamingModuleInterface
{
  public:
    explicit StreamingMainModule(
      const int num_cpu_threads,
      const int num_gpu_threads,
      const std::vector<int> & gpu_nums,
      core::SharedBatchBuffer<buffer_data::SiftVisualWordsData> * input_visual_words_buffer,
      core::SharedBatchBuffer<buffer_data::OutputSiftFeaturesData> * output_sift_buffer,
      const MainArgs & main_args,
      VocabTree * vocab_tree,
      FeatureDatabaseWrapper * feature_database,
      const dataset::FocalLookup * focal_lookup,
      const v3d_support::PCDB * pcdb);

    ~StreamingMainModule();

    void generate_summary_speed_report(std::string & report);
    void generate_detailed_speed_report(std::string & report);

    const core::SharedValue<bool> * is_finished() const
    { return &m_is_finished; }

    void wait_until_finished();

  private:
    StreamingMainModule(const StreamingMainModule &);
    StreamingMainModule & operator=(const StreamingMainModule &);

    void thread_main();

    void database_thread_main();

    void add_batch_images_to_image_cache(
      const std::vector<int> & batch_subset_indices,
      const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
      const std::vector<dataset::FocalLookup::Result> & batch_image_focals);

    void remove_images_from_image_cache(
      const std::vector<image_index_t> & image_indices);

    void create_new_cluster_and_connected_component_for_image(
      const image_index_t image_index,
      const int batch_index);

    int compute_connected_component_size(
      const component_index_t connected_component_index);

    bool is_component_size_bigger_than_or_equal_to(
      const component_index_t connected_component_index,
      const int size_threshold);

    void get_connected_component_images(
      const component_index_t connected_component_index,
      std::vector<image_index_t> & image_indices);

    void update_connected_component_index(
      const component_index_t old_connected_component_index,
      const component_index_t new_connected_component_index);

    void compute_and_display_registration_stats();

    void save_streaming_state_visualization(
      const int batch_index);

    void save_streaming_state_backup(
      const int batch_index);

    static bool compare_pair_first_greater_second_less(
      const std::pair<int, int> & a,
      const std::pair<int, int> & b);

    // Input visual words buffer reader.
    const core::SharedBatchBuffer<buffer_data::SiftVisualWordsData> * m_input_visual_words_buffer;
    boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::SiftVisualWordsData> > m_visual_words_data_reader;

    // Output sift buffer writer.
    const core::SharedBatchBuffer<buffer_data::OutputSiftFeaturesData> * m_output_sift_buffer;
    boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::OutputSiftFeaturesData> > m_output_sift_data_writer;

    boost::thread m_thread;
    core::SharedValue<bool> m_is_finished;

    boost::thread m_database_thread;
    core::ManagerWorkerSynchronization m_database_thread_syncrhonization;
    const std::vector<buffer_data::SiftVisualWordsData> * m_database_thread_batch_images;
    const std::vector<dataset::FocalLookup::Result> * m_database_thread_batch_focals;
    core::SharedValue<bool> m_database_thread_should_exit;
    core::Timer m_database_thread_timer;

    const MainArgs m_main_args;
    FeatureDatabaseWrapper * m_feature_database;
    boost::unordered_map<image_index_t, ImageCacheEntry> m_image_cache;
    boost::unordered_set<image_index_t> m_sift_saved_image_indices;
    const dataset::FocalLookup * m_focal_lookup;
    const v3d_support::PCDB * m_pcdb;
    boost::unordered_map<image_index_t, std::string> m_image_index_to_image_name;
    MainModuleThreadHandler * m_main_thread_handler;
    StreamingState m_streaming_state;
};

#endif // STREAMING_MAIN_MODULE_H
