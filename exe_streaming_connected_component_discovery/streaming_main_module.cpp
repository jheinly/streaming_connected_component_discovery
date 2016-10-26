#include "streaming_main_module.h"
#include "logger.h"
#include <core/file_helper.h>
#include <base3d/camera_models.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <windows.h>
#include <psapi.h>

#define TIME_CODE
#define TIME_CODE_BATCH_RATE 12

struct MergeClusterCandidate
{
  MergeClusterCandidate(
    const cluster_index_t cluster_idx1,
    const cluster_index_t cluster_idx2,
    const int cluster_sz1,
    const int cluster_sz2)
  {
    if (cluster_sz1 <= cluster_sz2)
    {
      cluster_index1 = cluster_idx1;
      cluster_index2 = cluster_idx2;
      cluster_size1 = cluster_sz1;
      cluster_size2 = cluster_sz2;
    }
    else
    {
      cluster_index1 = cluster_idx2;
      cluster_index2 = cluster_idx1;
      cluster_size1 = cluster_sz2;
      cluster_size2 = cluster_sz1;
    }
  }

  cluster_index_t cluster_index1; // cluster with smaller size
  cluster_index_t cluster_index2; // cluster with larger size
  int cluster_size1;
  int cluster_size2;
};

bool compare_merge_cluster_candidates(
  const MergeClusterCandidate & a,
  const MergeClusterCandidate & b)
{
  if (a.cluster_size1 == b.cluster_index1)
  {
    return a.cluster_index2 < b.cluster_index2;
  }
  else
  {
    return a.cluster_size1 < b.cluster_size1;
  }
}

StreamingMainModule::StreamingMainModule(
  const int num_cpu_threads,
  const int num_gpu_threads,
  const std::vector<int> & gpu_nums,
  core::SharedBatchBuffer<buffer_data::SiftVisualWordsData> * input_visual_words_buffer,
  core::SharedBatchBuffer<buffer_data::OutputSiftFeaturesData> * output_sift_buffer,
  const MainArgs & main_args,
  VocabTree * vocab_tree,
  FeatureDatabaseWrapper * feature_database,
  const dataset::FocalLookup * focal_lookup,
  const v3d_support::PCDB * pcdb)
: core::StreamingModuleInterface(1),
  m_input_visual_words_buffer(input_visual_words_buffer),
  m_visual_words_data_reader(),
  m_output_sift_buffer(output_sift_buffer),
  m_output_sift_data_writer(),
  m_thread(),
  m_is_finished(false),
  m_database_thread(),
  m_database_thread_syncrhonization(1),
  m_database_thread_batch_images(NULL),
  m_database_thread_batch_focals(NULL),
  m_database_thread_should_exit(false),
  m_database_thread_timer(),
  m_main_args(main_args),
  m_feature_database(feature_database),
  m_image_cache(),
  m_sift_saved_image_indices(),
  m_focal_lookup(focal_lookup),
  m_pcdb(pcdb),
  m_image_index_to_image_name(),
  m_main_thread_handler(NULL),
  m_streaming_state()
{
  m_visual_words_data_reader = input_visual_words_buffer->get_new_lockstep_reader();
  m_output_sift_data_writer = output_sift_buffer->get_new_lockstep_writer();

  m_main_thread_handler = new MainModuleThreadHandler(
    num_cpu_threads,
    num_gpu_threads,
    gpu_nums,
    main_args,
    vocab_tree,
    feature_database,
    &m_image_cache);

  m_thread = boost::thread(&StreamingMainModule::thread_main, this);
  m_database_thread = boost::thread(&StreamingMainModule::database_thread_main, this);
}

StreamingMainModule::~StreamingMainModule()
{
  if (m_main_thread_handler != NULL)
  {
    delete m_main_thread_handler;
    m_main_thread_handler = NULL;
  }
}

void StreamingMainModule::generate_summary_speed_report(std::string & report)
{
  const core::SharedSpeedStats * visual_words_readers_speed_stats =
    m_input_visual_words_buffer->lockstep_readers_speed_stats();

  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);
  stream << "StreamingMainModule" << std::endl;
  stream << "  Input:            ";
  stream.width(7);
  stream << visual_words_readers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << visual_words_readers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << visual_words_readers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;

  std::string thread_report;
  m_main_thread_handler->generate_summary_speed_report(thread_report);
  stream << thread_report;
  report = stream.str();
}

void StreamingMainModule::generate_detailed_speed_report(std::string & report)
{
  generate_summary_speed_report(report);
}

void StreamingMainModule::wait_until_finished()
{
  m_thread.join();
}

void StreamingMainModule::thread_main()
{
  std::vector<dataset::FocalLookup::Result> batch_image_focals;
  std::vector<std::vector<VocabTree::QueryResult> > query_results;
  std::vector<std::vector<BatchMatchTask> > batch_match_tasks;
  std::vector<std::vector<GeometricVerificationResult> > batch_successful_match_tasks;
  std::vector<int> batch_subset_to_add_to_voc_tree;
  std::vector<int> batch_subset_to_add_to_image_cache;
  std::vector<cluster_index_t> matched_cluster_indices;
  std::set<component_index_t> matched_component_indices;
  std::vector<std::pair<int, component_index_t> > matched_connected_component_sizes_and_indices;
  std::vector<int> batch_subset_to_add_to_output_sift;
  std::vector<image_index_t> cached_images_to_add_to_output_sift;
  std::vector<cluster_index_t> cluster_indices_to_discard;
  std::vector<image_index_t> cached_images_to_remove_from_voc_tree;
  std::vector<image_index_t> image_indices_to_remove_from_image_cache;
  std::vector<cluster_index_t> cluster_indices_for_iconic_selection;
  std::vector<cluster_index_t> valid_cluster_indices_for_iconic_selection;
  std::vector<std::pair<image_index_t, image_index_t> > image_pair_match_tasks;
  std::vector<GeometricVerificationResult> image_pair_match_results;
  std::vector<int> match_tasks_num_inliers;
  std::vector<image_index_t> cached_images_to_add_to_voc_tree;
  std::vector<image_index_t> common_image_indices;
  std::vector<image_index_t> new_image_indices;
  std::vector<std::vector<cluster_index_t> > discard_cluster_indices_per_batch;
  std::vector<cluster_index_t> new_cluster_indices;
  std::vector<cluster_index_t> existing_cluster_indices;
  std::vector<std::pair<cluster_index_t, cluster_index_t> > candidate_cluster_pairs_to_merge;
  std::map<component_index_t, std::vector<cluster_index_t> > clusters_in_matched_components;
  std::vector<MergeClusterCandidate> merge_cluster_candidates;

  if (m_main_args.cluster_merging_enabled)
  {
    ASSERT(MainArgs::num_knn_in_same_component_to_merge <= m_main_args.vocab_tree_num_knn);
  }
  
  const int cluster_discard_rate_in_batches = static_cast<int>(
    float(m_main_args.cluster_discard_rate_in_images) /
    float(m_main_args.batch_size)) + 1;
  discard_cluster_indices_per_batch.resize(cluster_discard_rate_in_batches);

  int batch_index = -1;
  int discard_cluster_index = -1;

#ifdef TIME_CODE
  core::Timer timer;
  std::stringstream stream;
  const int message_width = 31;
  GET_LOGGER();
#endif

  m_main_thread_handler->wait_until_threads_are_ready();

  m_database_thread_syncrhonization.wait_for_workers_to_be_ready_to_be_signaled();
  
  worker_thread_wait_for_start();

  FILE * memory_stats_file =
    core::file_helper::open_file(m_main_args.output_log_folder + "/memory_stats.txt", "w");
  FILE * num_clusters_file =
    core::file_helper::open_file(m_main_args.output_log_folder + "/num_clusters_stats.txt", "w");
  int num_valid_images_processed = 0;
  {
    // Print and save memory usage statistics.

    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);

    PROCESS_MEMORY_COUNTERS_EX pmc;
    BOOL success = GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    if (success == 0)
    {
      throw std::runtime_error("Call to GetProcessMemoryInfo has failed.");
    }

    // Total Virtual Memory
    DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;

    // Total Physical Memory (RAM)
    DWORDLONG totalPhysMem = memInfo.ullTotalPhys;

    // Virtual Memory currently used
    DWORDLONG virtualMemUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;

    // Physical Memory currently used
    DWORDLONG physMemUsed = memInfo.ullTotalPhys - memInfo.ullAvailPhys;

    // Virtual Memory currently used by current process
    SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;

    // Physical Memory currently used by current process
    SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

    std::stringstream memory_stream;
    memory_stream <<
      num_valid_images_processed << " " <<
      totalVirtualMem << " " <<
      totalPhysMem << " " <<
      virtualMemUsed << " " <<
      physMemUsed << " " <<
      virtualMemUsedByMe << " " <<
      physMemUsedByMe;
    const std::string line = memory_stream.str();
    fprintf(memory_stats_file, "%s\n", line.c_str());
    fflush(memory_stats_file);
    LOGGER << "#Memory " << line << std::endl;
    
    const size_t num_clusters =
      m_streaming_state.cluster_index_to_batch_index.size();
    std::stringstream num_clusters_stream;
    num_clusters_stream <<
      num_valid_images_processed << " " <<
      num_clusters;
    const std::string line2 = num_clusters_stream.str();
    fprintf(num_clusters_file, "%s\n", line2.c_str());
    fflush(num_clusters_file);
    LOGGER << "#Clusters " << line2 << std::endl;
  }

  for (;;)
  {
    m_visual_words_data_reader->wait_for_next_read_buffer();
    m_output_sift_data_writer->wait_for_next_write_buffer();
    if (m_visual_words_data_reader->no_further_read_buffers_available())
    {
      {
        // Print and save memory usage statistics.

        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);

        PROCESS_MEMORY_COUNTERS_EX pmc;
        BOOL success = GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
        if (success == 0)
        {
          throw std::runtime_error("Call to GetProcessMemoryInfo has failed.");
        }

        // Total Virtual Memory
        DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;

        // Total Physical Memory (RAM)
        DWORDLONG totalPhysMem = memInfo.ullTotalPhys;

        // Virtual Memory currently used
        DWORDLONG virtualMemUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;

        // Physical Memory currently used
        DWORDLONG physMemUsed = memInfo.ullTotalPhys - memInfo.ullAvailPhys;

        // Virtual Memory currently used by current process
        SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;

        // Physical Memory currently used by current process
        SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

        std::stringstream memory_stream;
        memory_stream <<
          num_valid_images_processed << " " <<
          totalVirtualMem << " " <<
          totalPhysMem << " " <<
          virtualMemUsed << " " <<
          physMemUsed << " " <<
          virtualMemUsedByMe << " " <<
          physMemUsedByMe;
        const std::string line = memory_stream.str();
        fprintf(memory_stats_file, "%s\n", line.c_str());
        fflush(memory_stats_file);
        LOGGER << "#Memory " << line << std::endl;
        fclose(memory_stats_file);
        
        const size_t num_clusters =
          m_streaming_state.cluster_index_to_batch_index.size();
        std::stringstream num_clusters_stream;
        num_clusters_stream <<
          num_valid_images_processed << " " <<
          num_clusters;
        const std::string line2 = num_clusters_stream.str();
        fprintf(num_clusters_file, "%s\n", line2.c_str());
        fflush(num_clusters_file);
        LOGGER << "#Clusters " << line2 << std::endl;
        fclose(num_clusters_file);
      }

      compute_and_display_registration_stats();
      std::cout << "Saving visualizations..." << std::endl;
      save_streaming_state_visualization(batch_index + 1);
      std::cout << "Saving backup..." << std::endl;
      save_streaming_state_backup(batch_index + 1);

      m_database_thread_should_exit.set(true);
      m_database_thread_syncrhonization.signal_workers();

      m_output_sift_data_writer->done_with_all_further_writes_and_wait_for_exit();
      break;
    }

    m_visual_words_data_reader->starting_to_read_from_buffer();
    m_output_sift_data_writer->starting_to_write_to_buffer();

    const std::vector<buffer_data::SiftVisualWordsData> & batch_images =
      m_visual_words_data_reader->read_buffer();
    const int current_batch_size = static_cast<int>(batch_images.size());
    num_valid_images_processed += current_batch_size;

    ++batch_index;
    ++discard_cluster_index;
    if (discard_cluster_index >= cluster_discard_rate_in_batches)
    {
      discard_cluster_index = 0;
    }

#ifdef TIME_CODE
    const bool time_code = batch_index % TIME_CODE_BATCH_RATE == 0;
    if (time_code)
    {
      stream.str("");
      stream.clear();
      stream << "STREAMING MAIN MODULE" << std::endl;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Store the image names in the lookup table.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    for (int i = 0; i < current_batch_size; ++i)
    {
      const buffer_data::SiftVisualWordsData & image =
        batch_images.at(i);

      ASSERT(m_image_index_to_image_name.find(image.image_index) == m_image_index_to_image_name.end());
      m_image_index_to_image_name[image.image_index] =
        image.image_name;
    }
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Store Image Names", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Lookup the focal lengths for all of the batch images.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    batch_image_focals.resize(current_batch_size);
    for (int i = 0; i < current_batch_size; ++i)
    {
      const buffer_data::SiftVisualWordsData & image =
        batch_images.at(i);

      batch_image_focals.at(i) = m_focal_lookup->lookup_focal(
        image.image_name,
        image.dimensions,
        image.image_scale_factor);
    }
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Lookup Focals", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Add batch images to SQLite database.

    m_database_thread_batch_images = &batch_images;
    m_database_thread_batch_focals = &batch_image_focals;
    m_database_thread_syncrhonization.signal_workers();

    ///////////////////////////////////////////////////////////////////////////
    // If this is the first batch, use it to initialize the state of the clusters.

    if (batch_index == 0)
    {
      batch_subset_to_add_to_image_cache.clear();

      for (int i = 0; i < current_batch_size; ++i)
      {
        create_new_cluster_and_connected_component_for_image(
          batch_images.at(i).image_index,
          batch_index);

        discard_cluster_indices_per_batch.at(discard_cluster_index).push_back(
          batch_images.at(i).image_index);

        batch_subset_to_add_to_image_cache.push_back(i);
      }

      m_main_thread_handler->add_whole_batch_to_voc_tree(
        batch_images);

      add_batch_images_to_image_cache(
        batch_subset_to_add_to_image_cache,
        batch_images,
        batch_image_focals);

      m_database_thread_syncrhonization.wait_for_workers_to_be_ready_to_be_signaled();

      m_visual_words_data_reader->done_reading_from_buffer();
      m_output_sift_data_writer->done_writing_to_buffer();
      continue;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Get voc-tree query knn results for the current batch of images.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    m_main_thread_handler->get_batch_knn_from_voc_tree(
      batch_images,
      query_results);
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Get KNN", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Run geometric verification between the batch images and their query results.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    // Create the batch match tasks.
    batch_match_tasks.resize(query_results.size());
    for (size_t i = 0; i < query_results.size(); ++i)
    {
      batch_match_tasks.at(i).resize(query_results.at(i).size());
      for (size_t j = 0; j < query_results.at(i).size(); ++j)
      {
        const image_index_t image_index = query_results.at(i).at(j).image_index;
        const cluster_index_t cluster_index =
          m_streaming_state.image_index_to_cluster_index[image_index];
        const component_index_t connected_component_index =
          m_streaming_state.cluster_index_to_component_index[cluster_index];

        batch_match_tasks.at(i).at(j) =
          BatchMatchTask(image_index, connected_component_index);
      }
    }

    // Wait for the batch images to be added to the database before performing
    // geometric verification.
    m_database_thread_syncrhonization.wait_for_workers_to_be_ready_to_be_signaled();
#ifdef TIME_CODE
    if (time_code)
    {
      m_database_thread_timer.print("Add Images to Database", stream, message_width);
    }
#endif

    m_main_thread_handler->run_batch_geometric_verification(
      batch_images,
      batch_image_focals,
      batch_match_tasks,
      batch_successful_match_tasks);
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Geometric Verification", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Update streaming state.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    batch_subset_to_add_to_voc_tree.clear();
    batch_subset_to_add_to_image_cache.clear();
    batch_subset_to_add_to_output_sift.clear();
    cached_images_to_add_to_output_sift.clear();
    cluster_indices_for_iconic_selection.clear();
    cached_images_to_remove_from_voc_tree.clear();
    image_indices_to_remove_from_image_cache.clear();
    cached_images_to_add_to_voc_tree.clear();
    new_cluster_indices.clear();
    candidate_cluster_pairs_to_merge.clear();

    for (int index_within_batch = 0; index_within_batch < current_batch_size; ++index_within_batch)
    {
      const buffer_data::SiftVisualWordsData & image =
        batch_images.at(index_within_batch);

      const int num_successful_matches =
        static_cast<int>(batch_successful_match_tasks.at(index_within_batch).size());
      if (num_successful_matches > 0)
      {
        // This batch image successfully matched to at least one other image.

        // Because this image matched to at least one other image, indicate that this
        // image's SIFT features should be saved to disk.
        batch_subset_to_add_to_output_sift.push_back(index_within_batch);

        // If at least two of the first N (MainArgs::num_knn_in_same_component_to_merge)
        // knn results came from the same connected component, and at least one of the
        // images within that component was matched to, then add those images' clusters
        // as potential ones to merge.
        if (m_main_args.cluster_merging_enabled)
        {
          // Find the connected components that were successfully matched.
          matched_component_indices.clear();
          for (int match_index = 0; match_index < num_successful_matches; ++match_index)
          {
            const int matched_task_index =
              batch_successful_match_tasks.at(index_within_batch).at(match_index).successful_task_index;
            if (matched_task_index < MainArgs::num_knn_in_same_component_to_merge)
            {
              const image_index_t matched_image_index =
                batch_match_tasks.at(index_within_batch).at(matched_task_index).image_index;
              const cluster_index_t matched_cluster_index =
                m_streaming_state.image_index_to_cluster_index[matched_image_index];
              const component_index_t matched_component_index =
                m_streaming_state.cluster_index_to_component_index[matched_cluster_index];
              matched_component_indices.insert(matched_component_index);
            }
            else
            {
              break;
            }
          }

          if (matched_component_indices.size() > 0)
          {
            // Create a list of the cluster indices that belonged to each of the matched
            // connected components.
            clusters_in_matched_components.clear();
            for (int i = 0; i < MainArgs::num_knn_in_same_component_to_merge; ++i)
            {
              const image_index_t image_index =
                batch_match_tasks.at(index_within_batch).at(i).image_index;
              const cluster_index_t cluster_index =
                m_streaming_state.image_index_to_cluster_index[image_index];
              const component_index_t component_index =
                m_streaming_state.cluster_index_to_component_index[cluster_index];

              std::set<component_index_t>::const_iterator iter = matched_component_indices.find(
                component_index);
              if (iter != matched_component_indices.end())
              {
                clusters_in_matched_components[component_index].push_back(cluster_index);
              }
            }

            // Iterate over each of the matched components.
            for (std::map<component_index_t, std::vector<cluster_index_t> >::const_iterator iter = clusters_in_matched_components.begin();
              iter != clusters_in_matched_components.end();
              ++iter)
            {
              // If this matched component had at least two clusters that belonged to it,
              // add those clusters as potential ones to merge.
              const std::vector<cluster_index_t> & cluster_indices = iter->second;
              if (cluster_indices.size() >= 2)
              {
                // Enumerate the cluster pair combinations.
                for (size_t i = 0; i < cluster_indices.size() - 1; ++i)
                {
                  for (size_t j = i + 1; j < cluster_indices.size(); ++j)
                  {
                    candidate_cluster_pairs_to_merge.push_back(std::make_pair(
                      cluster_indices.at(i),
                      cluster_indices.at(j)));
                  }
                }
              }
            }
          }
        } // End if cluster merging is enabled.
        
        // Get the clusters that this image matched to, as well as determine
        // the cluster with which this image had the most inliers.
        int best_match_index = 0;
        matched_cluster_indices.resize(num_successful_matches);
        for (int match_index = 0; match_index < num_successful_matches; ++match_index)
        {
          const int matched_task_index =
            batch_successful_match_tasks.at(index_within_batch).at(match_index).successful_task_index;
          const image_index_t matched_image_index =
            batch_match_tasks.at(index_within_batch).at(matched_task_index).image_index;
          const cluster_index_t matched_cluster_index =
            m_streaming_state.image_index_to_cluster_index[matched_image_index];
          matched_cluster_indices.at(match_index) = matched_cluster_index;

          // For each image that this image matched to, make sure that the
          // SIFT features for that image have been saved to disk.
          cached_images_to_add_to_output_sift.push_back(matched_image_index);

          if (batch_successful_match_tasks.at(index_within_batch).at(match_index).num_inliers >
            batch_successful_match_tasks.at(index_within_batch).at(best_match_index).num_inliers)
          {
            best_match_index = match_index;
          }
        }
        const cluster_index_t matched_cluster_index =
          matched_cluster_indices.at(best_match_index);

        // Add the image to the best matching cluster.
        m_streaming_state.image_index_to_cluster_index[image.image_index] =
          matched_cluster_index;
        std::vector<image_index_t> & cluster_image_indices =
          m_streaming_state.cluster_index_to_image_indices[matched_cluster_index];
        cluster_image_indices.push_back(image.image_index);

        if (m_main_args.iconic_selection_enabled)
        {
          const int cluster_size = static_cast<int>(cluster_image_indices.size());
          
          // If the cluster size is smaller than the number of images used for iconic
          // selection, add this image to the image cache as it may become an iconic.
          if (cluster_size <= m_main_args.num_images_for_iconic_selection)
          {
            batch_subset_to_add_to_image_cache.push_back(index_within_batch);
          }

          // If the new cluster size is equal to the number of images required for iconic
          // selection, flag that this cluster should perform iconic selection.
          if (cluster_size == m_main_args.num_images_for_iconic_selection)
          {
            cluster_indices_for_iconic_selection.push_back(matched_cluster_index);
          }
        }

        // If this image matched to more than one cluster, indicate that we should
        // attempt to merge those clusters.
        if (m_main_args.cluster_merging_enabled && num_successful_matches > 1)
        {
          // Enumerate the possible cluster combinations.
          for (int i = 0; i < num_successful_matches - 1; ++i)
          {
            for (int j = i + 1; j < num_successful_matches; ++j)
            {
              candidate_cluster_pairs_to_merge.push_back(std::make_pair(
                matched_cluster_indices.at(i),
                matched_cluster_indices.at(j)));
            }
          }
        }

        // If this image matched to more than one cluster, connect the clusters
        // into one connected component.
        if (num_successful_matches > 1)
        {
          matched_component_indices.clear();

          // Create the set of connected components to which each of the matched
          // clusters belonged.
          for (int i = 0; i < num_successful_matches; ++i)
          {
            const cluster_index_t cluster_index =
              matched_cluster_indices.at(i);
            const component_index_t connected_component_index =
              m_streaming_state.cluster_index_to_component_index[cluster_index];
            matched_component_indices.insert(
              connected_component_index);
          }

          // Test to see if the clusters actually belonged to more than one
          // connected component.
          if (matched_component_indices.size() > 1)
          {
            // Get a list of the sizes and indices of the matched connected components.
            matched_connected_component_sizes_and_indices.resize(
              matched_component_indices.size());
            std::set<component_index_t>::const_iterator iter = matched_component_indices.begin();
            for (int i = 0;
              iter != matched_component_indices.end();
              ++iter, ++i)
            {
              matched_connected_component_sizes_and_indices.at(i) = std::make_pair(
                compute_connected_component_size(*iter),
                *iter);
            }

            // Get the index of the largest connected component.
            std::vector<std::pair<int, component_index_t> >::const_iterator largest_connected_component =
              std::max_element(
              matched_connected_component_sizes_and_indices.begin(),
              matched_connected_component_sizes_and_indices.end());
            const component_index_t largest_connected_component_index =
              largest_connected_component->second;

            // Update all of the matched connected components to have the index
            // of the largest connected component (to reduce the number of required
            // update operations).
            for (size_t i = 0; i < matched_connected_component_sizes_and_indices.size(); ++i)
            {
              update_connected_component_index(
                matched_connected_component_sizes_and_indices.at(i).second,
                largest_connected_component_index);
            }
          }
        } // End connecting matched clusters into one connected component.
      }
      else
      {
        // This batch image did not successfully match to any images.
        // Create a new cluster and connected component for this image.

        create_new_cluster_and_connected_component_for_image(
          image.image_index,
          batch_index);

        new_cluster_indices.push_back(image.image_index);

        batch_subset_to_add_to_voc_tree.push_back(index_within_batch);
        batch_subset_to_add_to_image_cache.push_back(index_within_batch);
      }
    }
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Update State", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Add batch images to voc-tree.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    m_main_thread_handler->add_batch_subset_to_voc_tree(
      batch_images,
      batch_subset_to_add_to_voc_tree);
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Add Images to Voc-Tree", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Add batch images to image cache.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    add_batch_images_to_image_cache(
      batch_subset_to_add_to_image_cache,
      batch_images,
      batch_image_focals);
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Add Images to Image Cache", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Merge clusters.

    if (m_main_args.cluster_merging_enabled)
    {
#ifdef TIME_CODE
      if (time_code)
      {
        timer.start();
      }
#endif
      // Enforce that the first index in the cluster pairs is smaller than the second.
      for (size_t i = 0; i < candidate_cluster_pairs_to_merge.size(); ++i)
      {
        ASSERT(candidate_cluster_pairs_to_merge[i].first != candidate_cluster_pairs_to_merge[i].second);
        if (candidate_cluster_pairs_to_merge[i].first >
          candidate_cluster_pairs_to_merge[i].second)
        {
          std::swap(
            candidate_cluster_pairs_to_merge[i].first,
            candidate_cluster_pairs_to_merge[i].second);
        }
      }

      // Sort the pairs so that we can use std::unique.
      std::sort(
        candidate_cluster_pairs_to_merge.begin(),
        candidate_cluster_pairs_to_merge.end());

      // Remove any duplicate pairs from the list.
      std::vector<std::pair<cluster_index_t, cluster_index_t> >::const_iterator iter = std::unique(
        candidate_cluster_pairs_to_merge.begin(),
        candidate_cluster_pairs_to_merge.end());
      candidate_cluster_pairs_to_merge.resize(
        iter - candidate_cluster_pairs_to_merge.begin());

      // Form the match tasks.
      image_pair_match_tasks.clear();
      for (size_t i = 0; i < candidate_cluster_pairs_to_merge.size(); ++i)
      {
        const std::pair<cluster_index_t, cluster_index_t> & cluster_indices =
          candidate_cluster_pairs_to_merge[i];

        const std::vector<image_index_t> & cluster_images1 =
          m_streaming_state.cluster_index_to_image_indices[cluster_indices.first];
        const std::vector<image_index_t> & cluster_images2 =
          m_streaming_state.cluster_index_to_image_indices[cluster_indices.second];

        const image_index_t image_index1 = cluster_images1.at(0);
        const image_index_t image_index2 = cluster_images2.at(0);

        image_pair_match_tasks.push_back(std::make_pair(
          image_index1, image_index2));
      }

      // Run geometric verification.
      m_main_thread_handler->run_geometric_verification_cached_pairs(
        image_pair_match_tasks,
        image_pair_match_results);

      // Create a list of the successful cluster pairs and their sizes.
      merge_cluster_candidates.clear();
      for (size_t i = 0; i < image_pair_match_results.size(); ++i)
      {
        const int pair_index =
          image_pair_match_results.at(i).successful_task_index;
        const image_index_t image_index1 =
          image_pair_match_tasks.at(pair_index).first;
        const image_index_t image_index2 =
          image_pair_match_tasks.at(pair_index).second;
        const cluster_index_t cluster_index1 =
          m_streaming_state.image_index_to_cluster_index[image_index1];
        const cluster_index_t cluster_index2 =
          m_streaming_state.image_index_to_cluster_index[image_index2];
        const int cluster_size1 = static_cast<int>(
          m_streaming_state.cluster_index_to_image_indices[cluster_index1].size());
        const int cluster_size2 = static_cast<int>(
          m_streaming_state.cluster_index_to_image_indices[cluster_index2].size());

        MergeClusterCandidate candidate(
          cluster_index1, cluster_index2,
          cluster_size1, cluster_size2);
        merge_cluster_candidates.push_back(candidate);
      }

      // Sort pairs by their sizes (sort cluster sizes within each pair,
      // and then between all successful pairs).
      std::sort(
        merge_cluster_candidates.begin(),
        merge_cluster_candidates.end(),
        compare_merge_cluster_candidates);

      // Iterate over sorted list.
      for (size_t i = 0; i < merge_cluster_candidates.size(); ++i)
      {
        const MergeClusterCandidate & candidate =
          merge_cluster_candidates.at(i);

        const cluster_index_t cluster_index1 = candidate.cluster_index1;
        const cluster_index_t cluster_index2 = candidate.cluster_index2;

        // If either of the clusters no longer exist, continue to the next
        // merge candidate.
        if (m_streaming_state.cluster_index_to_image_indices.find(cluster_index1) ==
          m_streaming_state.cluster_index_to_image_indices.end())
        {
          continue;
        }
        if (m_streaming_state.cluster_index_to_image_indices.find(cluster_index2) ==
          m_streaming_state.cluster_index_to_image_indices.end())
        {
          continue;
        }

        std::vector<image_index_t> & cluster_images1 =
          m_streaming_state.cluster_index_to_image_indices[cluster_index1];
        std::vector<image_index_t> & cluster_images2 =
          m_streaming_state.cluster_index_to_image_indices[cluster_index2];

        // Merge the smaller cluster into the larger one (cluster_index1 is
        // the cluster with the smaller size, as defined by the
        // MergeClusterCandidate constructor).

        // Remove the smaller cluster's iconic from the voc-tree.
        ASSERT(cluster_images1.size() > 0);
        cached_images_to_remove_from_voc_tree.push_back(
          cluster_images1.at(0));

        if (m_main_args.iconic_selection_enabled)
        {
          // If the newly merge cluster is now large enough for iconic selection,
          // add it to the list.
          const int old_cluster_size =
            static_cast<int>(cluster_images2.size());
          const int new_cluster_size =
            static_cast<int>(cluster_images1.size() + cluster_images2.size());
          if (old_cluster_size < MainArgs::num_images_for_iconic_selection &&
            new_cluster_size >= MainArgs::num_images_for_iconic_selection)
          {
            cluster_indices_for_iconic_selection.push_back(cluster_index2);
          }

          if (old_cluster_size >= MainArgs::num_images_for_iconic_selection)
          {
            // If the original cluster already has enough images for iconic
            // selection (or it was already performed), remove the smaller
            // cluster's images from the image cache.
            image_indices_to_remove_from_image_cache.insert(
              image_indices_to_remove_from_image_cache.end(),
              cluster_images1.begin(),
              cluster_images1.end());
          }
          else
          {
            // The original cluster did not have enough images for iconic selection,
            // but it may have enough now that the smaller cluster has been added.
            // Determine now many extra images the new cluster has, and remove
            // those from the image cache.
            const int num_images_to_remove =
              new_cluster_size - MainArgs::num_images_for_iconic_selection;
            if (num_images_to_remove > 0)
            {
              ASSERT(cluster_images1.end() - num_images_to_remove >= cluster_images1.begin());
              image_indices_to_remove_from_image_cache.insert(
                image_indices_to_remove_from_image_cache.end(),
                cluster_images1.end() - num_images_to_remove,
                cluster_images1.end());
            }
          }
        } // End if iconic selection enabled.

        // Update cluster image indices.
        for (size_t i = 0; i < cluster_images1.size(); ++i)
        {
          const image_index_t image_index = cluster_images1.at(i);
          m_streaming_state.image_index_to_cluster_index[image_index] =
            cluster_index2;
        }
        cluster_images2.insert(
          cluster_images2.end(),
          cluster_images1.begin(),
          cluster_images1.end());
        cluster_images1.clear();
        m_streaming_state.cluster_index_to_image_indices.erase(cluster_index1);

        // Update cluster batch index.
        const int batch_index1 =
          m_streaming_state.cluster_index_to_batch_index[cluster_index1];
        const int batch_index2 =
          m_streaming_state.cluster_index_to_batch_index[cluster_index2];
        m_streaming_state.cluster_index_to_batch_index[cluster_index2] =
          std::min(batch_index1, batch_index2);
        m_streaming_state.cluster_index_to_batch_index.erase(cluster_index1);

        // Update connected component cluster indices.
        const component_index_t component_index =
          m_streaming_state.cluster_index_to_component_index[cluster_index1];
        ASSERT(component_index == m_streaming_state.cluster_index_to_component_index[cluster_index2]);
        m_streaming_state.cluster_index_to_component_index.erase(cluster_index1);
        std::vector<cluster_index_t> & component_cluster_indices =
          m_streaming_state.component_index_to_cluster_indices[component_index];
        std::vector<cluster_index_t>::iterator found = std::find(
          component_cluster_indices.begin(),
          component_cluster_indices.end(),
          cluster_index1);
        ASSERT(found != component_cluster_indices.end());
        component_cluster_indices.erase(found);
      }
#ifdef TIME_CODE
      if (time_code)
      {
        timer.print("Merge Clusters", stream, message_width);
      }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Select new cluster iconics.

    if (m_main_args.iconic_selection_enabled)
    {
#ifdef TIME_CODE
      if (time_code)
      {
        timer.start();
      }
#endif

      // Check to see which clusters still exist and haven't been merged
      // into another cluster.
      valid_cluster_indices_for_iconic_selection.clear();
      for (size_t i = 0; i < cluster_indices_for_iconic_selection.size(); ++i)
      {
        const cluster_index_t cluster_index =
          cluster_indices_for_iconic_selection.at(i);

        boost::unordered_map<cluster_index_t, std::vector<image_index_t> >::const_iterator found =
          m_streaming_state.cluster_index_to_image_indices.find(cluster_index);

        if (found != m_streaming_state.cluster_index_to_image_indices.end())
        {
          valid_cluster_indices_for_iconic_selection.push_back(cluster_index);
        }
      }

      image_pair_match_tasks.clear();

      // Iterate over the cluster indices, and create match tasks for each cluster.
      for (size_t selection_task_index = 0;
        selection_task_index < valid_cluster_indices_for_iconic_selection.size();
        ++selection_task_index)
      {
        const cluster_index_t cluster_index =
          valid_cluster_indices_for_iconic_selection.at(selection_task_index);

        const std::vector<image_index_t> & cluster_image_indices =
          m_streaming_state.cluster_index_to_image_indices[cluster_index];

        ASSERT(cluster_image_indices.size() >= 3);
        COMPILER_ASSERT(MainArgs::num_images_for_iconic_selection == 3);
        image_pair_match_tasks.push_back(std::make_pair(
          cluster_image_indices.at(0), cluster_image_indices.at(1)));
        image_pair_match_tasks.push_back(std::make_pair(
          cluster_image_indices.at(0), cluster_image_indices.at(2)));
        image_pair_match_tasks.push_back(std::make_pair(
          cluster_image_indices.at(1), cluster_image_indices.at(2)));
      }

      // Perform geometric verification on the image pairs.
      m_main_thread_handler->run_geometric_verification_cached_pairs(
        image_pair_match_tasks,
        image_pair_match_results);

      // This will store the number of inliers for each of the match task pairs.
      // By default, set them all to a value of zero.
      match_tasks_num_inliers.clear();
      match_tasks_num_inliers.resize(image_pair_match_tasks.size(), 0);

      // Geometric verification will output successful match task indices in an
      // arbitrary order (and will only output the indices of the successful
      // tasks), so figure out which original match tasks actually were successful.
      for (size_t i = 0; i < image_pair_match_results.size(); ++i)
      {
        // Get the index of the original match task that ended up being successful.
        const int successful_index = image_pair_match_results.at(i).successful_task_index;

        // Indicate that the original match task succeeded, and store its number of inliers.
        ASSERT(successful_index >= 0 && successful_index < static_cast<int>(match_tasks_num_inliers.size()));
        match_tasks_num_inliers.at(successful_index) = image_pair_match_results.at(i).num_inliers;
      }

      for (size_t selection_task_index = 0;
        selection_task_index < valid_cluster_indices_for_iconic_selection.size();
        ++selection_task_index)
      {
        const cluster_index_t cluster_index =
          valid_cluster_indices_for_iconic_selection.at(selection_task_index);

        COMPILER_ASSERT(MainArgs::num_images_for_iconic_selection == 3);
        const int task_offset = 3 * static_cast<int>(selection_task_index);

        ASSERT(task_offset >= 0 && task_offset < static_cast<int>(match_tasks_num_inliers.size() - 2));
        const int num_inliers12 = match_tasks_num_inliers.at(task_offset);
        const int num_inliers13 = match_tasks_num_inliers.at(task_offset + 1);
        const int num_inliers23 = match_tasks_num_inliers.at(task_offset + 2);

        // If not all match tasks for this cluster succeeded, skip this cluster.
        if (num_inliers12 == 0 || num_inliers13 == 0 || num_inliers23 == 0)
        {
          ASSERT(task_offset + 2 < static_cast<int>(image_pair_match_tasks.size()));
          // Remove from the image cache the 2nd and 3rd images in the cluster.
          image_indices_to_remove_from_image_cache.push_back(
            image_pair_match_tasks.at(task_offset + 2).first);
          image_indices_to_remove_from_image_cache.push_back(
            image_pair_match_tasks.at(task_offset + 2).second);

          continue;
        }

        // For each image, sum the number of inliers of the match pairs it was involved in.
        const int num_inliers1 = num_inliers12 + num_inliers13;
        const int num_inliers2 = num_inliers12 + num_inliers23;
        const int num_inliers3 = num_inliers13 + num_inliers23;

        if (num_inliers1 >= num_inliers2 && num_inliers1 >= num_inliers3)
        {
          // The original iconic is the best, so keep it.

          ASSERT(task_offset + 2 < static_cast<int>(image_pair_match_tasks.size()));
          // Remove from the image cache the 2nd and 3rd images in the cluster.
          image_indices_to_remove_from_image_cache.push_back(
            image_pair_match_tasks.at(task_offset + 2).first);
          image_indices_to_remove_from_image_cache.push_back(
            image_pair_match_tasks.at(task_offset + 2).second);
        }
        else
        {
          std::vector<image_index_t> & cluster_image_indices =
            m_streaming_state.cluster_index_to_image_indices[cluster_index];

          ASSERT(cluster_image_indices.size() >= 3);

          if (num_inliers2 >= num_inliers3)
          {
            // The second image is the best, so make it the iconic.

            image_indices_to_remove_from_image_cache.push_back(
              cluster_image_indices.at(0));
            image_indices_to_remove_from_image_cache.push_back(
              cluster_image_indices.at(2));

            cached_images_to_remove_from_voc_tree.push_back(
              cluster_image_indices.at(0));
            cached_images_to_add_to_voc_tree.push_back(
              cluster_image_indices.at(1));

            std::swap(cluster_image_indices[0], cluster_image_indices[1]);
          }
          else
          {
            // The third image is the best, so make it the iconic.

            image_indices_to_remove_from_image_cache.push_back(
              cluster_image_indices.at(0));
            image_indices_to_remove_from_image_cache.push_back(
              cluster_image_indices.at(1));

            cached_images_to_remove_from_voc_tree.push_back(
              cluster_image_indices.at(0));
            cached_images_to_add_to_voc_tree.push_back(
              cluster_image_indices.at(2));

            std::swap(cluster_image_indices[0], cluster_image_indices[2]);
          }
        }
      }

#ifdef TIME_CODE
      if (time_code)
      {
        timer.print("Iconic Selection", stream, message_width);
      }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Discard clusters that aren't growing fast enough in size.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    const float inv_discard_rate = 1.0f / float(m_main_args.cluster_discard_rate_in_images);

    existing_cluster_indices.clear();
    for (size_t i = 0;
      i < discard_cluster_indices_per_batch.at(discard_cluster_index).size();
      ++i)
    {
      const cluster_index_t cluster_index = discard_cluster_indices_per_batch.at(discard_cluster_index).at(i);

      boost::unordered_map<cluster_index_t, int>::const_iterator iter =
        m_streaming_state.cluster_index_to_batch_index.find(cluster_index);

      // Check if the cluster still exists, and if not, skip it.
      if (iter == m_streaming_state.cluster_index_to_batch_index.end())
      {
        continue;
      }

      const int batch_difference = batch_index - iter->second;
      const int num_images_difference = batch_difference * m_main_args.batch_size;
      const int expected_size = 1 +
        static_cast<int>(float(num_images_difference) * inv_discard_rate);

      const std::vector<image_index_t> & image_indices =
        m_streaming_state.cluster_index_to_image_indices[cluster_index];
      const int cluster_size =
        static_cast<int>(image_indices.size());

      if (cluster_size < expected_size)
      {
        m_streaming_state.cluster_index_to_batch_index.erase(cluster_index);

        const image_index_t iconic_image_index = image_indices.at(0);
        cached_images_to_remove_from_voc_tree.push_back(iconic_image_index);

        image_indices_to_remove_from_image_cache.insert(
          image_indices_to_remove_from_image_cache.end(),
          image_indices.begin(),
          image_indices.end());

        // TODO: can probably discard images that occur in m_sift_saved_image_indices
      }
      else
      {
        existing_cluster_indices.push_back(cluster_index);
      }
    }

    discard_cluster_indices_per_batch.at(discard_cluster_index).clear();
    discard_cluster_indices_per_batch.at(discard_cluster_index).insert(
      discard_cluster_indices_per_batch[discard_cluster_index].end(),
      existing_cluster_indices.begin(),
      existing_cluster_indices.end());
    discard_cluster_indices_per_batch.at(discard_cluster_index).insert(
      discard_cluster_indices_per_batch.at(discard_cluster_index).end(),
      new_cluster_indices.begin(),
      new_cluster_indices.end());

#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Find Slow Clusters", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Unify the remove and add requests to the voc-tree.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    {
      std::vector<image_index_t>::iterator iter;

      // Find the indices common to both the add and remove lists.

      std::sort(
        cached_images_to_remove_from_voc_tree.begin(),
        cached_images_to_remove_from_voc_tree.end());
      std::sort(
        cached_images_to_add_to_voc_tree.begin(),
        cached_images_to_add_to_voc_tree.end());

      common_image_indices.resize(cached_images_to_remove_from_voc_tree.size());
      iter = std::set_intersection(
        cached_images_to_remove_from_voc_tree.begin(),
        cached_images_to_remove_from_voc_tree.end(),
        cached_images_to_add_to_voc_tree.begin(),
        cached_images_to_add_to_voc_tree.end(),
        common_image_indices.begin());
      common_image_indices.resize(iter - common_image_indices.begin());

      if (common_image_indices.size() > 0)
      {
        // Remove the indices from the remove list.
        new_image_indices.resize(cached_images_to_remove_from_voc_tree.size());
        iter = std::set_difference(
          cached_images_to_remove_from_voc_tree.begin(),
          cached_images_to_remove_from_voc_tree.end(),
          common_image_indices.begin(),
          common_image_indices.end(),
          new_image_indices.begin());
        new_image_indices.resize(iter - new_image_indices.begin());
        cached_images_to_remove_from_voc_tree = new_image_indices;

        // Remove the indices from the add list.
        new_image_indices.resize(cached_images_to_add_to_voc_tree.size());
        iter = std::set_difference(
          cached_images_to_add_to_voc_tree.begin(),
          cached_images_to_add_to_voc_tree.end(),
          common_image_indices.begin(),
          common_image_indices.end(),
          new_image_indices.begin());
        new_image_indices.resize(iter - new_image_indices.begin());
        cached_images_to_add_to_voc_tree = new_image_indices;
      }
    }
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Unify Voc-Tree Tasks", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Remove images from voc-tree.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    m_main_thread_handler->remove_cached_images_from_voc_tree(
      cached_images_to_remove_from_voc_tree);
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Remove Images from Voc-Tree", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Add images to voc-tree.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    m_main_thread_handler->add_cached_images_to_voc_tree(
      cached_images_to_add_to_voc_tree);
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Add Images to Voc-Tree", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Save SIFT features for verified images to a streaming output buffer.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    // Iterate over the batch images to save (those batch images that registered
    // to one or more other images.
    for (size_t i = 0; i < batch_subset_to_add_to_output_sift.size(); ++i)
    {
      const int buffer_index =
        m_output_sift_data_writer->write_buffer_counter()->get_value_and_increment();

      buffer_data::OutputSiftFeaturesData & output_sift =
        m_output_sift_data_writer->write_buffer().at(buffer_index);

      const buffer_data::SiftVisualWordsData & batch_image =
        batch_images.at(batch_subset_to_add_to_output_sift.at(i));

      output_sift.image_index = batch_image.image_index;
      output_sift.image_name = batch_image.image_name;
      output_sift.num_features = batch_image.num_features;
      output_sift.keypoints = batch_image.keypoints;
      output_sift.descriptors_uchar = batch_image.descriptors_uchar;
      output_sift.image_scale_factor = batch_image.image_scale_factor;

      m_sift_saved_image_indices.insert(batch_image.image_index);
    }

    // Iterate over the cached images to save (those images that were matched
    // to by a batch image).
    for (size_t i = 0; i < cached_images_to_add_to_output_sift.size(); ++i)
    {
      const image_index_t image_index = cached_images_to_add_to_output_sift.at(i);
      
      boost::unordered_set<image_index_t>::iterator found =
        m_sift_saved_image_indices.find(image_index);
      if (found != m_sift_saved_image_indices.end())
      {
        continue;
      }

      const int buffer_index =
        m_output_sift_data_writer->write_buffer_counter()->get_value_and_increment();

      // It could be possible to output more SIFT than the size of a typical batch.
      // For example, if each of the batch images match to a different single-image
      // cluster, then we would be saving twice the normal size of a batch.
      if (buffer_index >= static_cast<int>(m_output_sift_data_writer->write_buffer().size()))
      {
        m_output_sift_data_writer->write_buffer().resize(buffer_index + 1);
      }

      buffer_data::OutputSiftFeaturesData & output_sift =
        m_output_sift_data_writer->write_buffer().at(buffer_index);

      boost::unordered_map<image_index_t, ImageCacheEntry>::const_iterator found_image =
        m_image_cache.find(image_index);
      ASSERT(found_image != m_image_cache.end());
      const ImageCacheEntry & image = found_image->second;

      output_sift.image_index = image.image_index;
      output_sift.image_name = image.image_name;
      output_sift.num_features = image.num_features;
      output_sift.keypoints = image.keypoints;
      output_sift.descriptors_uchar = image.descriptors_uchar;
      output_sift.image_scale_factor = image.image_scale_factor;

      m_sift_saved_image_indices.insert(image_index);
    }
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Find Images to Save", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Remove images from image cache. This should be at the end of the main
    // loop in case any of the cached images need to be accessed.

#ifdef TIME_CODE
    if (time_code)
    {
      timer.start();
    }
#endif
    // Remove images from image cache (images from slow-growing clusters).
    remove_images_from_image_cache(
      image_indices_to_remove_from_image_cache);
#ifdef TIME_CODE
    if (time_code)
    {
      timer.print("Remove Images from Image Cache", stream, message_width);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////

#ifdef TIME_CODE
    if (time_code)
    {
      stream << std::endl;
      LOGGER << stream.str();
    }
#endif

    m_visual_words_data_reader->done_reading_from_buffer();
    m_output_sift_data_writer->done_writing_to_buffer();

    ///////////////////////////////////////////////////////////////////////////
    // Display stats, save visualizations, and backup data.

    if (batch_index % m_main_args.main_registration_stats_display_rate_in_batches == 0 &&
      batch_index > 0)
    {
      compute_and_display_registration_stats();
    }
    if (batch_index % m_main_args.main_visualization_rate_in_batches == 0 &&
      batch_index > 0)
    {
      std::cout << std::endl;
      std::cout << "Saving visualizations..." << std::endl;
      save_streaming_state_visualization(batch_index);
      std::cout << std::endl;
    }
    if (batch_index % m_main_args.main_backup_rate_in_batches == 0 &&
      batch_index > 0)
    {
      std::cout << std::endl;
      std::cout << "Saving backup..." << std::endl;
      save_streaming_state_backup(batch_index);
      std::cout << std::endl;
    }

    const int rate = 10000;
    const int previous = (num_valid_images_processed - current_batch_size) / rate;
    const int current = num_valid_images_processed / rate;
    if (current > previous)
    {
      // Print and save memory usage statistics.

      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);

      PROCESS_MEMORY_COUNTERS_EX pmc;
      BOOL success = GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
      if (success == 0)
      {
        throw std::runtime_error("Call to GetProcessMemoryInfo has failed.");
      }

      // Total Virtual Memory
      DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;

      // Total Physical Memory (RAM)
      DWORDLONG totalPhysMem = memInfo.ullTotalPhys;

      // Virtual Memory currently used
      DWORDLONG virtualMemUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;

      // Physical Memory currently used
      DWORDLONG physMemUsed = memInfo.ullTotalPhys - memInfo.ullAvailPhys;

      // Virtual Memory currently used by current process
      SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;

      // Physical Memory currently used by current process
      SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

      std::stringstream memory_stream;
      memory_stream <<
        num_valid_images_processed << " " <<
        totalVirtualMem << " " <<
        totalPhysMem << " " <<
        virtualMemUsed << " " <<
        physMemUsed << " " <<
        virtualMemUsedByMe << " " <<
        physMemUsedByMe;
      const std::string line = memory_stream.str();
      fprintf(memory_stats_file, "%s\n", line.c_str());
      fflush(memory_stats_file);
      LOGGER << "#Memory " << line << std::endl;
      
      const size_t num_clusters =
        m_streaming_state.cluster_index_to_batch_index.size();
      std::stringstream num_clusters_stream;
      num_clusters_stream <<
        num_valid_images_processed << " " <<
        num_clusters;
      const std::string line2 = num_clusters_stream.str();
      fprintf(num_clusters_file, "%s\n", line2.c_str());
      fflush(num_clusters_file);
      LOGGER << "#Clusters " << line2 << std::endl;
    }

  } // End main loop.

  m_is_finished.set(true);
}

void StreamingMainModule::database_thread_main()
{
  for (;;)
  {
    m_database_thread_syncrhonization.wait_for_signal_from_manager();

    if (m_database_thread_should_exit.get())
    {
      return;
    }

    m_database_thread_timer.start();

    boost::lock_guard<boost::mutex> database_lock(m_feature_database->mutex);
    m_feature_database->database.begin_transaction();

    for (size_t i = 0; i < m_database_thread_batch_images->size(); ++i)
    {
      const buffer_data::SiftVisualWordsData & image =
        (*m_database_thread_batch_images).at(i);

      colmap::Camera camera;
      camera.camera_id = static_cast<colmap::camera_t>(image.image_index);
      camera.model = colmap::SimpleRadialCameraModel::code;
      camera.prior_focal = (*m_database_thread_batch_focals).at(i).is_focal_from_exif;
      camera.init(
        (*m_database_thread_batch_focals).at(i).focal_in_pixels,
        image.dimensions.width,
        image.dimensions.height);

      const colmap::camera_t camera_id =
        m_feature_database->database.add_camera(camera, true);
      ASSERT(camera_id == static_cast<colmap::camera_t>(image.image_index));

      colmap::ImageData image_data;
      image_data.image_id = static_cast<colmap::image_t>(image.image_index);
      image_data.camera_id = static_cast<colmap::camera_t>(image.image_index);
      const std::string & name = image.image_name;
      image_data.name = name.substr(0, 3) + "/" + name.substr(3, 2) + "/" + name + ".jpg";

      ASSERT(m_feature_database->database.exists_image(image_data.image_id) == false);
      m_feature_database->database.add_image(image_data, true);
    }

    m_feature_database->database.end_transaction();

    m_database_thread_timer.stop();
  }
}

void StreamingMainModule::add_batch_images_to_image_cache(
  const std::vector<int> & batch_subset_indices,
  const std::vector<buffer_data::SiftVisualWordsData> & batch_images,
  const std::vector<dataset::FocalLookup::Result> & batch_image_focals)
{
  for (size_t i = 0; i < batch_subset_indices.size(); ++i)
  {
    const buffer_data::SiftVisualWordsData & batch_image =
      batch_images.at(batch_subset_indices.at(i));

    ASSERT(m_image_cache.find(batch_image.image_index) == m_image_cache.end());
    ImageCacheEntry & image = m_image_cache[batch_image.image_index];

    image.image_index = batch_image.image_index;
    image.image_name = batch_image.image_name;
    image.dimensions = batch_image.dimensions;
    image.focal = batch_image_focals.at(batch_subset_indices.at(i)).focal_in_pixels;
    image.num_features = batch_image.num_features;
    image.keypoints = batch_image.keypoints;
    image.descriptors_uchar = batch_image.descriptors_uchar;
    image.visual_words.resize(batch_image.visual_words.size());
    for (int j = 0; j < batch_image.num_features; ++j)
    {
      image.visual_words.at(j).resize(1);
      image.visual_words.at(j).at(0) = batch_image.visual_words.at(j);
    }
    image.visual_words_mutex =
      boost::shared_ptr<boost::mutex>(new boost::mutex());
    image.image_scale_factor = batch_image.image_scale_factor;
  }
}

void StreamingMainModule::remove_images_from_image_cache(
  const std::vector<image_index_t> & image_indices)
{
  for (size_t i = 0; i < image_indices.size(); ++i)
  {
    const image_index_t image_index = image_indices.at(i);

    boost::unordered_map<image_index_t, ImageCacheEntry>::const_iterator found_image =
      m_image_cache.find(image_index);

    if (found_image != m_image_cache.end())
    {
      m_image_cache.erase(image_index);
    }
  }
}

void StreamingMainModule::create_new_cluster_and_connected_component_for_image(
  const image_index_t image_index,
  const int batch_index)
{
  m_streaming_state.image_index_to_cluster_index[image_index] = image_index;
  m_streaming_state.cluster_index_to_component_index[image_index] = image_index;

  m_streaming_state.cluster_index_to_image_indices[image_index].push_back(image_index);
  m_streaming_state.component_index_to_cluster_indices[image_index].push_back(image_index);

  m_streaming_state.cluster_index_to_batch_index[image_index] = batch_index;
}

int StreamingMainModule::compute_connected_component_size(
  const component_index_t connected_component_index)
{
  size_t size = 0;

  // Get the clusters that belong to this connected component.
  const std::vector<cluster_index_t> & cluster_indices =
    m_streaming_state.component_index_to_cluster_indices[connected_component_index];

  // Sum the number of images that belong to each cluster.
  for (size_t i = 0; i < cluster_indices.size(); ++i)
  {
    size +=
      m_streaming_state.cluster_index_to_image_indices[cluster_indices.at(i)].size();
  }

  return static_cast<int>(size);
}

bool StreamingMainModule::is_component_size_bigger_than_or_equal_to(
  const component_index_t connected_component_index,
  const int size_threshold)
{
  const size_t threshold = static_cast<size_t>(size_threshold);
  size_t size = 0;

  // Get the clusters that belong to this connected component.
  const std::vector<cluster_index_t> & cluster_indices =
    m_streaming_state.component_index_to_cluster_indices[connected_component_index];

  // Sum the number of images that belong to each cluster.
  for (size_t i = 0; i < cluster_indices.size(); ++i)
  {
    size +=
      m_streaming_state.cluster_index_to_image_indices[cluster_indices.at(i)].size();
    if (size >= threshold)
    {
      return true;
    }
  }

  return false;
}

void StreamingMainModule::get_connected_component_images(
  const component_index_t connected_component_index,
  std::vector<image_index_t> & image_indices)
{
  image_indices.clear();

  // Get the clusters that belong to this connected component.
  const std::vector<cluster_index_t> & cluster_indices =
    m_streaming_state.component_index_to_cluster_indices[connected_component_index];

  // Sum the number of images that belong to each cluster.
  for (size_t i = 0; i < cluster_indices.size(); ++i)
  {
    const std::vector<image_index_t> & cluster_images =
      m_streaming_state.cluster_index_to_image_indices[cluster_indices.at(i)];

    image_indices.insert(
      image_indices.end(),
      cluster_images.begin(),
      cluster_images.end());
  }
}

void StreamingMainModule::update_connected_component_index(
  const component_index_t old_connected_component_index,
  const component_index_t new_connected_component_index)
{
  if (old_connected_component_index == new_connected_component_index)
  {
    return;
  }

  std::vector<cluster_index_t> & old_cluster_indices =
    m_streaming_state.component_index_to_cluster_indices[old_connected_component_index];

  // Update the cluster to connected component indices.
  for (size_t i = 0; i < old_cluster_indices.size(); ++i)
  {
    m_streaming_state.cluster_index_to_component_index[old_cluster_indices.at(i)] =
      new_connected_component_index;
  }

  // Update the connected component to cluster indices.
  std::vector<cluster_index_t> & new_cluster_indices =
    m_streaming_state.component_index_to_cluster_indices[new_connected_component_index];
  new_cluster_indices.insert(
    new_cluster_indices.end(),
    old_cluster_indices.begin(),
    old_cluster_indices.end());

  // Remove the old connected component.
  old_cluster_indices.clear(); // This is unnecessary given that we erase the entry in the next line.
  m_streaming_state.component_index_to_cluster_indices.erase(
    old_connected_component_index);
}

void StreamingMainModule::compute_and_display_registration_stats()
{
  int total_num_images = 0;
  int total_num_registered = 0;
  std::vector<int> component_sizes;
  component_sizes.reserve(m_streaming_state.component_index_to_cluster_indices.size());
  for (const auto & iter : m_streaming_state.component_index_to_cluster_indices)
  {
    const int component_size = compute_connected_component_size(iter.first);
    ASSERT(component_size > 0);
    total_num_images += component_size;
    component_sizes.push_back(component_size);
    if (component_size > 1)
    {
      total_num_registered += component_size;
    }
  }

  const double inv_total_num_images =
    1.0 / static_cast<double>(total_num_images);
  double component_entropy = 0;
  for (size_t i = 0; i < component_sizes.size(); ++i)
  {
    const double component_probability =
      static_cast<double>(component_sizes.at(i)) *
      inv_total_num_images;
    component_entropy -= component_probability * log(component_probability);
  }

  std::stringstream stream;
  stream.setf(std::ios_base::fixed, std::ios_base::floatfield);
  stream.precision(6);

  stream << "REGISTRATION STATS" << std::endl;
  stream << "Num Registered: " << total_num_registered << " / " << total_num_images <<
    " = " << (100.0 * double(total_num_registered) / double(total_num_images)) << "%" << std::endl;
  stream << "Component Entropy: " << component_entropy << std::endl;
  stream << std::endl;

  GET_LOGGER();
  LOGGER << stream.str();
}

void StreamingMainModule::save_streaming_state_visualization(
  const int batch_index)
{
  ///////////////////////////////////////////////////////////////////////////
  // Output clusters file.

  std::stringstream clusters_name_stream;
  clusters_name_stream << "clusters_batch_";
  clusters_name_stream.fill('0');
  clusters_name_stream.width(6);
  clusters_name_stream << batch_index;

  std::stringstream clusters_path_stream;
  clusters_path_stream << m_main_args.output_visualization_folder;
  clusters_path_stream << "/";
  clusters_path_stream << clusters_name_stream.str();
  clusters_path_stream << ".html";

  const std::string clusters_path = clusters_path_stream.str();

  std::ofstream clusters_file;
  core::file_helper::open_file(clusters_path, clusters_file);

  clusters_file << "<html>" << std::endl;
  clusters_file << "<head>" << std::endl;
  clusters_file << "<title>" << clusters_name_stream.str() << "</title>" << std::endl;
  clusters_file << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-1\">" << std::endl;
  clusters_file << "</head>" << std::endl;

  std::vector<std::pair<int, cluster_index_t> > cluster_sizes_and_indices;
  cluster_sizes_and_indices.reserve(
    m_streaming_state.cluster_index_to_image_indices.size());

  for (boost::unordered_map<cluster_index_t, std::vector<image_index_t> >::const_iterator iter =
    m_streaming_state.cluster_index_to_image_indices.begin();
    iter != m_streaming_state.cluster_index_to_image_indices.end();
    ++iter)
  {
    cluster_sizes_and_indices.push_back(
      std::make_pair(static_cast<int>(iter->second.size()), iter->first));
  }

  std::sort(
    cluster_sizes_and_indices.begin(),
    cluster_sizes_and_indices.end(),
    compare_pair_first_greater_second_less);

  for (int i = 0;
    i < m_main_args.main_visualization_num_clusters &&
    i < static_cast<int>(cluster_sizes_and_indices.size());
    ++i)
  {
    const cluster_index_t cluster_index = cluster_sizes_and_indices.at(i).second;
    const int cluster_size = cluster_sizes_and_indices.at(i).first;

    clusters_file << "<h1 style=\"display:inline;\">";
    clusters_file << cluster_size << " Images, Cluster " << cluster_index;
    clusters_file << "</h1> <br/>" << std::endl;

    const std::vector<image_index_t> & cluster_images =
      m_streaming_state.cluster_index_to_image_indices[cluster_index];

    for (int j = 0;
      j < m_main_args.main_visualization_num_images_per_cluster &&
      j < static_cast<int>(cluster_images.size());
      ++j)
    {
      ASSERT(m_image_index_to_image_name.find(cluster_images.at(j)) != m_image_index_to_image_name.end());
      const std::string & image_name =
        m_image_index_to_image_name[cluster_images.at(j)];

      const std::string thumbnail_path =
        m_pcdb->get_indexed_file(v3d_support::PCDB::thumbnail, image_name);

      clusters_file << "<img src=\"file:///" << thumbnail_path << "\"/>" << std::endl;
    }
    
    clusters_file << "<br/> <hr />" << std::endl;
  }

  clusters_file << "</html>" << std::endl;
  clusters_file.close();

  ///////////////////////////////////////////////////////////////////////////
  // Output components file.

  std::stringstream components_name_stream;
  components_name_stream << "components_batch_";
  components_name_stream.fill('0');
  components_name_stream.width(6);
  components_name_stream << batch_index;

  std::stringstream components_path_stream;
  components_path_stream << m_main_args.output_visualization_folder;
  components_path_stream << "/";
  components_path_stream << components_name_stream.str();
  components_path_stream << ".html";

  const std::string components_path = components_path_stream.str();

  std::ofstream components_file;
  core::file_helper::open_file(components_path, components_file);

  components_file << "<html>" << std::endl;
  components_file << "<head>" << std::endl;
  components_file << "<title>" << components_name_stream.str() << "</title>" << std::endl;
  components_file << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-1\">" << std::endl;
  components_file << "</head>" << std::endl;

  std::vector<std::pair<component_index_t, int> > component_sizes_and_indices;
  component_sizes_and_indices.reserve(
    m_streaming_state.component_index_to_cluster_indices.size());

  for (boost::unordered_map<component_index_t, std::vector<cluster_index_t> >::const_iterator iter =
    m_streaming_state.component_index_to_cluster_indices.begin();
    iter != m_streaming_state.component_index_to_cluster_indices.end();
    ++iter)
  {
    component_sizes_and_indices.push_back(
      std::make_pair(compute_connected_component_size(iter->first), iter->first));
  }

  std::sort(
    component_sizes_and_indices.begin(),
    component_sizes_and_indices.end(),
    compare_pair_first_greater_second_less);

  std::vector<image_index_t> component_images;

  for (int i = 0;
    i < m_main_args.main_visualization_num_components &&
    i < static_cast<int>(component_sizes_and_indices.size());
    ++i)
  {
    const component_index_t component_index = component_sizes_and_indices.at(i).second;
    const int component_size = component_sizes_and_indices.at(i).first;

    components_file << "<h1 style=\"display:inline;\">";
    components_file << component_size << " Images, Component " << component_index;
    components_file << "</h1> <br/>" << std::endl;

    get_connected_component_images(
      component_index,
      component_images);

    const int num_images_to_display =
      std::min(m_main_args.main_visualization_num_images_per_component,
      static_cast<int>(component_images.size()));

    std::partial_sort(
      component_images.begin(),
      component_images.begin() + num_images_to_display,
      component_images.end());

    for (int j = 0; j < num_images_to_display; ++j)
    {
      ASSERT(m_image_index_to_image_name.find(component_images.at(j)) != m_image_index_to_image_name.end());
      const std::string & image_name =
        m_image_index_to_image_name[component_images.at(j)];

      const std::string thumbnail_path =
        m_pcdb->get_indexed_file(v3d_support::PCDB::thumbnail, image_name);

      components_file << "<img src=\"file:///" << thumbnail_path << "\"/>" << std::endl;
    }

    components_file << "<br/> <hr />" << std::endl;
  }

  components_file << "</html>" << std::endl;
  components_file.close();
}

void StreamingMainModule::save_streaming_state_backup(
  const int batch_index)
{
  ///////////////////////////////////////////////////////////////////////////
  // Output clusters file.

  std::stringstream clusters_path_stream;
  clusters_path_stream << m_main_args.output_backup_folder;
  clusters_path_stream << "/clusters_batch_";
  clusters_path_stream.fill('0');
  clusters_path_stream.width(6);
  clusters_path_stream << batch_index;
  clusters_path_stream << ".txt";

  const std::string clusters_path = clusters_path_stream.str();

  std::ofstream clusters_file;
  core::file_helper::open_file(clusters_path, clusters_file);

  for (boost::unordered_map<cluster_index_t, std::vector<image_index_t> >::const_iterator iter =
    m_streaming_state.cluster_index_to_image_indices.begin();
    iter != m_streaming_state.cluster_index_to_image_indices.end();
    ++iter)
  {
    const cluster_index_t cluster_index = iter->first;
    const component_index_t component_index =
      m_streaming_state.cluster_index_to_component_index[cluster_index];

    if (is_component_size_bigger_than_or_equal_to(component_index,
      m_main_args.main_backup_min_images_per_component))
    {
      clusters_file << "#" << cluster_index << std::endl;

      const std::vector<image_index_t> & cluster_images = iter->second;
      for (size_t i = 0; i < cluster_images.size(); ++i)
      {
        clusters_file << cluster_images.at(i) << std::endl;
      }
    }
  }

  clusters_file.close();

  ///////////////////////////////////////////////////////////////////////////
  // Output components file.

  std::stringstream components_path_stream;
  components_path_stream << m_main_args.output_backup_folder;
  components_path_stream << "/components_batch_";
  components_path_stream.fill('0');
  components_path_stream.width(6);
  components_path_stream << batch_index;
  components_path_stream << ".txt";

  const std::string components_path = components_path_stream.str();

  std::ofstream components_file;
  core::file_helper::open_file(components_path, components_file);

  for (boost::unordered_map<component_index_t, std::vector<cluster_index_t> >::const_iterator iter =
    m_streaming_state.component_index_to_cluster_indices.begin();
    iter != m_streaming_state.component_index_to_cluster_indices.end();
    ++iter)
  {
    const component_index_t component_index = iter->first;

    if (is_component_size_bigger_than_or_equal_to(component_index,
      m_main_args.main_backup_min_images_per_component))
    {
      components_file << "#" << component_index << std::endl;

      const std::vector<cluster_index_t> & cluster_indices = iter->second;
      for (size_t i = 0; i < cluster_indices.size(); ++i)
      {
        components_file << cluster_indices.at(i) << std::endl;
      }
    }
  }

  components_file.close();
}

bool StreamingMainModule::compare_pair_first_greater_second_less(
  const std::pair<int, int> & a,
  const std::pair<int, int> & b)
{
  if (a.first == b.first)
  {
    return a.second < b.second;
  }
  else
  {
    return a.first > b.first;
  }
}
