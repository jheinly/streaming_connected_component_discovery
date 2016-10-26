#include <streaming_cuda_sift/streaming_cuda_sift.h>
#include <cuda_helper/cuda_helper.h>
#include <core/thread_helper.h>
#include <core/timer.h>

StreamingCudaSift::StreamingCudaSift(
  const int num_threads,
  const std::vector<int> & gpu_nums,
  core::SharedBatchBuffer<buffer_data::ImageData> * input_image_data_buffer,
  core::SharedBatchBuffer<buffer_data::SiftFeaturesData> * output_sift_features_data_buffer,
  const int max_image_dimension,
  const int max_num_features,
  const int min_allowed_num_features)
: core::StreamingModuleInterface(num_threads),
  m_input_image_data_buffer(input_image_data_buffer),
  m_image_data_readers(num_threads),
  m_output_sift_features_data_buffer(output_sift_features_data_buffer),
  m_sift_features_data_writers(num_threads),
  m_threads(NULL),
  m_num_threads(num_threads),
  m_gpu_nums(num_threads, 0),
  m_gpu_names(num_threads),
  m_cuda_sifts_speed_stats(NULL),
  m_max_image_dimension(max_image_dimension),
  m_max_num_features(max_num_features),
  m_min_allowed_num_features(min_allowed_num_features)
{
  if (gpu_nums.size() > 0)
  {
    for (int i = 0; i < num_threads; ++i)
    {
      m_gpu_nums[i] = gpu_nums[i % gpu_nums.size()];
      m_gpu_names[i] = cuda_helper::get_device_name(m_gpu_nums[i]);
    }
  }

  m_cuda_sifts_speed_stats = new core::SharedSpeedStats[num_threads];

  // Create the threads.
  m_threads = new boost::thread[num_threads];

  for (int i = 0; i < num_threads; ++i)
  {
    m_image_data_readers[i] =
      input_image_data_buffer->get_new_lockstep_reader();
    m_sift_features_data_writers[i] =
      output_sift_features_data_buffer->get_new_lockstep_writer();

    m_threads[i] = boost::thread(&StreamingCudaSift::thread_run, this, i);
    core::thread_helper::sleep_for_seconds(0.2); // stagger starting the SiftGPU servers
  }
}

StreamingCudaSift::~StreamingCudaSift()
{
  if (m_threads != NULL)
  {
    delete [] m_threads;
    m_threads = NULL;
  }

  if (m_cuda_sifts_speed_stats != NULL)
  {
    delete [] m_cuda_sifts_speed_stats;
    m_cuda_sifts_speed_stats = NULL;
  }
}

void StreamingCudaSift::generate_summary_speed_report(std::string & report)
{
  const core::SharedSpeedStats * image_readers_speed_stats =
    m_input_image_data_buffer->lockstep_readers_speed_stats();
  const core::SharedSpeedStats * sift_writers_speed_stats =
    m_output_sift_features_data_buffer->lockstep_writers_speed_stats();

  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);
  stream << "StreamingCudaSift" << std::endl;
  stream << "  Input:            ";
  stream.width(7);
  stream << image_readers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << image_readers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << image_readers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;
  stream << "  Output:           ";
  stream.width(7);
  stream << sift_writers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << sift_writers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << sift_writers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;

  report = stream.str();
}

void StreamingCudaSift::generate_detailed_speed_report(std::string & report)
{
  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);

  generate_summary_speed_report(report);
  stream << report;

  for (int i = 0; i < m_num_threads; ++i)
  {
    stream << "    - GPU #" << m_gpu_nums[i] << ": " << m_gpu_names[i] << std::endl;
    stream << "                    ";
    stream.width(7);
    stream << m_cuda_sifts_speed_stats[i].get_most_recent_speed_hz() << " Hz";
    stream << "      (rolling avg: ";
    stream.width(7);
    stream << m_cuda_sifts_speed_stats[i].get_rolling_average_speed_hz() << " Hz)";
    stream << "      (overall avg: ";
    stream.width(7);
    stream << m_cuda_sifts_speed_stats[i].get_overall_average_speed_hz() << " Hz)";
    stream << std::endl;
  }

  report = stream.str();
}

void StreamingCudaSift::wait_until_finished()
{
  for (int i = 0; i < m_num_threads; ++i)
  {
    m_threads[i].join();
  }
}

void StreamingCudaSift::thread_run(const int thread_num)
{
  cuda_sift::CudaSift cuda_sift(
    m_max_image_dimension,
    m_max_num_features,
    m_gpu_nums[thread_num],
    thread_num);

  boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::ImageData> > image_data_reader =
    m_image_data_readers[thread_num];
  boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::SiftFeaturesData> > sift_features_data_writer =
    m_sift_features_data_writers[thread_num];

  worker_thread_wait_for_start();

  // Start the main loop.
  core::Timer timer;
  timer.stop();
  for (;;)
  {
    image_data_reader->wait_for_next_read_buffer();
    sift_features_data_writer->wait_for_next_write_buffer();
    if (image_data_reader->no_further_read_buffers_available())
    {
      sift_features_data_writer->done_with_all_further_writes_and_wait_for_exit();
      break;
    }
    image_data_reader->starting_to_read_from_buffer();
    sift_features_data_writer->starting_to_write_to_buffer();

    timer.start();
    int num_entries_processed = 0;

    // Continue looping until all of the images in the image buffer
    // have been processed.
    for (;;)
    {
      // Get the next index to process from the image buffer.
      const int image_index =
        image_data_reader->read_buffer_counter()->get_value_and_increment();

      // Exit the loop if all of the images from the image buffer have already been assigned.
      if (image_index >= static_cast<int>(image_data_reader->read_buffer().size()))
      {
        break;
      }

      ++num_entries_processed;

      // Get the image data that this thread should handle.
      const buffer_data::ImageData & image_data =
        image_data_reader->read_buffer()[image_index];

      const int num_features = cuda_sift.compute_using_host_image(
        &image_data.data[0],
        image_data.dimensions.width,
        image_data.dimensions.height);

      if (num_features < m_min_allowed_num_features)
      {
        continue;
      }

      // Get the sift features buffer entry where the final features should be stored.
      const int sift_features_buffer_index =
        sift_features_data_writer->write_buffer_counter()->get_value_and_increment();
      buffer_data::SiftFeaturesData & sift_features_data =
        sift_features_data_writer->write_buffer()[sift_features_buffer_index];

      sift_features_data.num_features = num_features;
      sift_features_data.keypoints = cuda_sift.host_keypoints();
      sift_features_data.descriptors = cuda_sift.host_descriptors();

      sift_features_data.image_index = image_data.image_index;
      sift_features_data.image_name = image_data.image_name;
      sift_features_data.dimensions = image_data.dimensions;
      sift_features_data.image_scale_factor = image_data.scale_factor;
    } // end processing input image buffer

    timer.stop();
    m_cuda_sifts_speed_stats[thread_num].add_timing(timer.elapsed(), num_entries_processed);

    image_data_reader->done_reading_from_buffer();
    sift_features_data_writer->done_writing_to_buffer();
  } // end main loop
}
