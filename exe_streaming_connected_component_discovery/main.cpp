#include "logger.h"
#include "streaming_main_module.h"
#include <main_module/feature_database_wrapper.h>
#include <main_module/main_args.h>
#include <streaming_balanced_pcdb_file_reader/streaming_balanced_pcdb_file_reader.h>
#include <streaming_image_loader/streaming_image_loader.h>
#include <streaming_cuda_sift/streaming_cuda_sift.h>
#include <streaming_visual_words/streaming_visual_words.h>
#include <streaming_sift_file_writer/streaming_sift_file_writer.h>
#include <google_breakpad_wrapper/google_breakpad_wrapper.h>
#include <core/file_helper.h>
#include <core/speed_stats.h>
#include <core/thread_helper.h>
#include <core/timer.h>
#include <dataset/focal_lookup.h>
#include <v3d_support/pcdb.h>
#include <vocab_tree/VocabTree.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main(const int argc, const char ** argv)
{
  std::cout << std::endl;
  std::cout << "STREAMING_IPC2SFM" << std::endl;
  std::cout << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  // Parse command-line and config-file arguments.
  
  MainArgs main_args;
  handle_main_args(argc, argv, main_args);

  /////////////////////////////////////////////////////////////////////////////
  // Set up the exception handler.

  //GoogleBreakpadWrapper google_breakpad_wrapper(
  //  main_args.output_log_folder);

  /////////////////////////////////////////////////////////////////////////////
  // Set up the logger.

  init_logger(main_args.output_log_folder);
  GET_LOGGER();

  /////////////////////////////////////////////////////////////////////////////
  // Load dataset.

  core::Timer timer;
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(6);

  std::cout << std::endl;
  std::cout << "Loading..." << std::endl;
  std::cout << std::endl;

  // Open SQLite database.
  std::cout << "Opening SQLite database..." << std::endl;
  timer.start();
  if (core::file_helper::path_exists(main_args.sqlite_database_path))
  {
    std::cerr << "ERROR: SQLite database already exists at the specified path." << std::endl;
    std::cerr << "       A new, empty database will be created, so remove the existing one." << std::endl;
    std::cerr << main_args.sqlite_database_path << std::endl;
    return EXIT_FAILURE;
  }
  FeatureDatabaseWrapper feature_database;
  feature_database.database.init(main_args.sqlite_database_path);
  timer.print();
  std::cout << std::endl;
  
  // Load PCDB file.
  std::cout << "Loading PCDB..." << std::endl;
  timer.start();
  v3d_support::PCDB pcdb(main_args.pcdb_path);
  timer.print();
  std::cout << std::endl;

  // Load paths to each of the image lists.
  std::vector<std::string> image_list_names;
  {
    std::ifstream image_lists_file;
    core::file_helper::open_file(main_args.image_lists_path, image_lists_file);
    std::string line;
    for (;;)
    {
      line.clear();
      std::getline(image_lists_file, line);
      if (line.length() == 0)
      {
        break;
      }
      image_list_names.push_back(line);
    }
    image_lists_file.close();
  }

  // Start loading the image focal length information.
  dataset::FocalLookup focal_lookup;
  if (main_args.dim_list_path.length() > 0)
  {
    std::cout << "Starting to load image dimensions..." << std::endl;
    focal_lookup.load_dimension_list_with_thread(main_args.dim_list_path);
    std::cout << std::endl;
  }
  if (main_args.focal_list_path.length() > 0)
  {
    std::cout << "Starting to load image focals..." << std::endl;
    focal_lookup.load_focal_list_with_thread(main_args.focal_list_path);
    std::cout << std::endl;
  }

  // Load the vocabulary tree.
  std::cout << "Loading vocabulary tree..." << std::endl;
  timer.start();
  VocabTree vocab_tree;
  vocab_tree.Read(main_args.vocab_tree_path);
  vocab_tree.set_distance_type(DistanceMin);
  vocab_tree.SetInteriorNodeWeight(0.0f);
  vocab_tree.SetConstantLeafWeights();
  vocab_tree.ClearDatabase();
  timer.print();
  std::cout << std::endl;
  std::cout << "Loaded " << vocab_tree.get_num_visual_words() << " visual words." << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  // Create shared buffers.
  
  std::cout << std::endl;
  std::cout << "Creating streaming buffers..." << std::endl;
  std::cout << std::endl;

  core::SharedBatchBuffer<buffer_data::FileData> file_data_buffer(
    main_args.num_file_buffers,
    main_args.batch_size);
  
  core::SharedBatchBuffer<buffer_data::ImageData> image_data_buffer(
    main_args.num_image_buffers,
    main_args.batch_size);
  
  core::SharedBatchBuffer<buffer_data::SiftFeaturesData> sift_features_data_buffer(
    main_args.num_sift_features_buffers,
    main_args.batch_size);

  core::SharedBatchBuffer<buffer_data::SiftVisualWordsData> visual_words_data_buffer(
    main_args.num_visual_words_buffers,
    main_args.batch_size);

  core::SharedBatchBuffer<buffer_data::OutputSiftFeaturesData> output_sift_data_buffer(
    main_args.num_output_sift_buffers,
    main_args.batch_size);

  /////////////////////////////////////////////////////////////////////////////
  // Create streaming modules.
  
  std::cout << std::endl;
  std::cout << "Creating streaming modules..." << std::endl;
  std::cout << std::endl;

  StreamingBalancedPcdbFileReader streaming_balanced_pcdb_file_reader(
    &pcdb,
    v3d_support::PCDB::image,
    image_list_names,
    &file_data_buffer,
    main_args.num_files_to_skip);
  
  StreamingImageLoader streaming_image_loader(
    main_args.num_image_loader_threads,
    &file_data_buffer,
    &image_data_buffer,
    core::colorspace::Grayscale,
    image_resizer::ImageResizer::Resized,
    main_args.max_image_dimension,
    main_args.min_image_dimension,
    main_args.max_aspect_ratio,
    main_args.max_image_dimension);
  
  StreamingCudaSift streaming_cuda_sift(
    main_args.num_cuda_sift_threads,
    main_args.sift_gpu_nums,
    &image_data_buffer,
    &sift_features_data_buffer,
    main_args.max_image_dimension,
    main_args.max_num_features,
    main_args.min_num_features);

  StreamingVisualWords streaming_visual_words(
    main_args.num_visual_words_threads,
    &vocab_tree,
    &sift_features_data_buffer,
    &visual_words_data_buffer);

  StreamingMainModule streaming_main_module(
    main_args.num_main_cpu_threads,
    main_args.num_main_gpu_threads,
    main_args.main_gpu_nums,
    &visual_words_data_buffer,
    &output_sift_data_buffer,
    main_args,
    &vocab_tree,
    &feature_database,
    &focal_lookup,
    &pcdb);

  StreamingSiftFileWriter streaming_sift_file_writer(
    main_args.num_output_sift_threads,
    &pcdb,
    &output_sift_data_buffer);

  /////////////////////////////////////////////////////////////////////////////
  // Wait for dataset to load.

  if (main_args.dim_list_path.length() > 0)
  {
    focal_lookup.wait_for_thread_to_load_dimension_list();
  }
  if (main_args.focal_list_path.length() > 0)
  {
    focal_lookup.wait_for_thread_to_load_focal_list();
  }

  std::cout << std::endl;
  std::cout << "Loaded " << focal_lookup.num_dimension_entries() << " image dimensions." << std::endl;
  std::cout << "Loaded " << focal_lookup.num_focal_entries() << " image focals." << std::endl;
  std::cout << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  // Start.

  std::cout << std::endl;
  std::cout << "Waiting for initialization..." << std::endl;
  std::cout << std::endl;

  streaming_balanced_pcdb_file_reader.wait_until_ready_to_start();
  streaming_image_loader.wait_until_ready_to_start();
  streaming_cuda_sift.wait_until_ready_to_start();
  streaming_visual_words.wait_until_ready_to_start();
  streaming_main_module.wait_until_ready_to_start();
  streaming_sift_file_writer.wait_until_ready_to_start();

  std::cout << std::endl;
  std::cout << "Starting..." << std::endl;
  std::cout << std::endl;

  streaming_balanced_pcdb_file_reader.start();
  streaming_image_loader.start();
  streaming_cuda_sift.start();
  streaming_visual_words.start();
  streaming_main_module.start();
  streaming_sift_file_writer.start();

  /////////////////////////////////////////////////////////////////////////////
  // Loop while StreamingMainModule runs.

  core::Timer total_timer;
  total_timer.start();

  std::cout << std::endl;
  std::cout << "Running..." << std::endl;
  std::cout << std::endl;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(1);

  const double rolling_average_duration = 150; // seconds
  const double stats_sleep_duration = 15; // seconds
  int rolling_average_length =
    static_cast<int>(rolling_average_duration / stats_sleep_duration + 0.5);
  if (rolling_average_length < 2)
  {
    rolling_average_length = 2;
  }

  timer.start();
  core::SpeedStats speed_stats(rolling_average_length);
  int previous_num_files_remaining =
    streaming_balanced_pcdb_file_reader.num_files_remaining()->get();
  const int total_num_files =
    streaming_balanced_pcdb_file_reader.total_num_files()->get();
  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);

  std::string report;
  for (;;)
  {
    core::thread_helper::sleep_for_seconds(stats_sleep_duration);
    if (streaming_main_module.is_finished()->get())
    {
      break;
    }

    timer.stop();
    const double elapsed_seconds = timer.elapsed();
    timer.start();

    stream.str("");
    stream.clear();
    
    const int num_files_remaining =
      streaming_balanced_pcdb_file_reader.num_files_remaining()->get();
    const int num_files_processed = total_num_files - num_files_remaining;
    double progress = 0.0;
    if (total_num_files != 0)
    {
      progress =
        static_cast<double>(num_files_processed) /
        static_cast<double>(total_num_files);
      progress *= 100.0;
    }

    stream << "MAIN" << std::endl;
    stream << "Progress: " <<
      num_files_processed << " / " << total_num_files <<
      " = " << progress << " %" << std::endl;

    const int num_files_processed_difference =
      previous_num_files_remaining - num_files_remaining;
    previous_num_files_remaining = num_files_remaining;

    speed_stats.add_timing(elapsed_seconds, num_files_processed_difference);

    stream << "  Speed:            ";
    stream.width(7);
    stream << speed_stats.get_most_recent_speed_hz() << " Hz";
    stream << "      (rolling avg: ";
    stream.width(7);
    stream << speed_stats.get_rolling_average_speed_hz() << " Hz)";
    stream << "      (overall avg: ";
    stream.width(7);
    stream << speed_stats.get_overall_average_speed_hz() << " Hz)";
    stream << std::endl;
#if 1
    streaming_balanced_pcdb_file_reader.generate_summary_speed_report(report);
    stream << report;
    streaming_image_loader.generate_summary_speed_report(report);
    stream << report;
    streaming_cuda_sift.generate_summary_speed_report(report);
    stream << report;
    streaming_visual_words.generate_summary_speed_report(report);
    stream << report;
    streaming_main_module.generate_summary_speed_report(report);
    stream << report;
    streaming_sift_file_writer.generate_summary_speed_report(report);
    stream << report;
#else
    streaming_balanced_pcdb_file_reader.generate_detailed_speed_report(report);
    stream << report;
    streaming_image_loader.generate_detailed_speed_report(report);
    stream << report;
    streaming_cuda_sift.generate_detailed_speed_report(report);
    stream << report;
    streaming_visual_words.generate_detailed_speed_report(report);
    stream << report;
    streaming_main_module.generate_detailed_speed_report(report);
    stream << report;
    streaming_sift_file_writer.generate_detailed_speed_report(report);
    stream << report;
#endif
    stream << std::endl;
    LOGGER << stream.str();
  }

  /////////////////////////////////////////////////////////////////////////////
  // Wait until finished.

  std::cout << std::endl;
  std::cout << "Waiting for termination..." << std::endl;
  std::cout << std::endl;

  streaming_balanced_pcdb_file_reader.wait_until_finished();
  streaming_image_loader.wait_until_finished();
  streaming_cuda_sift.wait_until_finished();
  streaming_main_module.wait_until_finished();
  streaming_sift_file_writer.wait_until_finished();

  /////////////////////////////////////////////////////////////////////////////

  const double total_elapsed = total_timer.elapsed();
  std::cout << std::endl;
  LOGGER << "TOTAL ELAPSED: " << total_elapsed << " sec" << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;
  LOGGER << "DONE." << std::endl;
  std::cout << std::endl;

  return 0;
}
