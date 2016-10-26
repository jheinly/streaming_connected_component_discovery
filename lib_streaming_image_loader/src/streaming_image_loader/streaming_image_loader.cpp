#include <streaming_image_loader/streaming_image_loader.h>

StreamingImageLoader::StreamingImageLoader(
  const int num_threads,
  core::SharedBatchBuffer<buffer_data::FileData> * input_file_data_buffer,
  core::SharedBatchBuffer<buffer_data::ImageData> * output_image_data_buffer,
  const core::colorspace::Colorspace output_image_colorspace,
  const image_resizer::ImageResizer::ResizeMode output_image_resize_mode,
  const int resized_image_dimension,
  const int minimum_allowed_image_dimension,
  const float maximum_allowed_aspect_ratio,
  const int maximum_loaded_image_dimension_hint)
: core::StreamingModuleInterface(num_threads),
  m_input_file_data_buffer(input_file_data_buffer),
  m_output_image_data_buffer(output_image_data_buffer),
  m_file_data_readers(num_threads),
  m_image_data_writers(num_threads),
  m_threads(NULL),
  m_num_threads(num_threads),
  m_image_loaders(num_threads, NULL),
  m_output_image_colorspace(output_image_colorspace),
  m_output_image_resize_mode(output_image_resize_mode),
  m_resized_image_dimension(resized_image_dimension),
  m_minimum_allowed_image_dimension(minimum_allowed_image_dimension),
  m_maximum_allowed_aspect_ratio(maximum_allowed_aspect_ratio),
  m_maximum_loaded_image_dimension_hint(maximum_loaded_image_dimension_hint),
  m_maximum_image_size(0)
{
  for (int i = 0; i < num_threads; ++i)
  {
    m_file_data_readers[i] = input_file_data_buffer->get_new_lockstep_reader();
    m_image_data_writers[i] = output_image_data_buffer->get_new_lockstep_writer();

    m_image_loaders[i] = new image_loader::ImageLoader(
      output_image_colorspace,
      output_image_resize_mode,
      resized_image_dimension,
      minimum_allowed_image_dimension,
      maximum_allowed_aspect_ratio,
      maximum_loaded_image_dimension_hint);
  }

  m_maximum_image_size = resized_image_dimension * resized_image_dimension *
    core::colorspace::num_bytes_per_pixel(output_image_colorspace);

  // Create the threads.
  m_threads = new boost::thread[m_num_threads];
  for (int i = 0; i < m_num_threads; ++i)
  {
    m_threads[i] = boost::thread(&StreamingImageLoader::thread_run, this, i);
  }
}

StreamingImageLoader::~StreamingImageLoader()
{
  if (m_threads != NULL)
  {
    delete [] m_threads;
    m_threads = NULL;
  }

  for (size_t i = 0; i < m_image_loaders.size(); ++i)
  {
    if (m_image_loaders[i] != NULL)
    {
      delete m_image_loaders[i];
      m_image_loaders[i] = NULL;
    }
  }
  m_image_loaders.resize(0);
}

void StreamingImageLoader::generate_summary_speed_report(std::string & report)
{
  const core::SharedSpeedStats * file_readers_speed_stats =
    m_input_file_data_buffer->lockstep_readers_speed_stats();
  const core::SharedSpeedStats * image_writers_speed_stats =
    m_output_image_data_buffer->lockstep_writers_speed_stats();

  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);
  stream << "StreamingImageLoader" << std::endl;
  stream << "  Input:            ";
  stream.width(7);
  stream << file_readers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << file_readers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << file_readers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;
  stream << "  Output:           ";
  stream.width(7);
  stream << image_writers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << image_writers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << image_writers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;

  report = stream.str();
}

void StreamingImageLoader::generate_detailed_speed_report(std::string & report)
{
  generate_summary_speed_report(report);
}

void StreamingImageLoader::wait_until_finished()
{
  for (int i = 0; i < m_num_threads; ++i)
  {
    m_threads[i].join();
  }
}

void StreamingImageLoader::thread_run(const int thread_num)
{
  boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::FileData> > file_data_reader =
    m_file_data_readers[thread_num];
  boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::ImageData> > image_data_writer =
    m_image_data_writers[thread_num];
  image_loader::ImageLoader * current_image_loader = m_image_loaders[thread_num];

  worker_thread_wait_for_start();

  // Start the main loop.
  for (;;)
  {
    file_data_reader->wait_for_next_read_buffer();
    image_data_writer->wait_for_next_write_buffer();
    if (file_data_reader->no_further_read_buffers_available())
    {
      image_data_writer->done_with_all_further_writes_and_wait_for_exit();
      break;
    }
    file_data_reader->starting_to_read_from_buffer();
    image_data_writer->starting_to_write_to_buffer();

    // Continue looping until all of the files in the file buffer
    // have been processed.
    for (;;)
    {
      // Get the next index to process from the file buffer.
      const int file_index =
        file_data_reader->read_buffer_counter()->get_value_and_increment();

      // Exit the loop if all of the files from the file buffer have already been assigned.
      if (file_index >= static_cast<int>(file_data_reader->read_buffer().size()))
      {
        break;
      }

      // Get the file data that this thread should handle.
      const buffer_data::FileData & file_data =
        file_data_reader->read_buffer()[file_index];

      if (file_data.data.size() == 0)
      {
        continue;
      }

      const bool success = current_image_loader->start_loading_jpeg_from_data(
        &file_data.data[0],
        static_cast<int>(file_data.data.size()));

      if (!success)
      {
        continue;
      }

      // Get the image data where the final image should be stored.
      const int image_index =
        image_data_writer->write_buffer_counter()->get_value_and_increment();
      buffer_data::ImageData & image_data =
        image_data_writer->write_buffer()[image_index];
      image_data.data.resize(m_maximum_image_size);

      const image_resizer::ImageResizer::ResizeResult resize_result =
        current_image_loader->finish_loading_jpeg(
          &image_data.data[0],
          m_maximum_image_size);

      const int num_image_bytes =
        resize_result.width * resize_result.height *
        core::colorspace::num_bytes_per_pixel(m_output_image_colorspace);
      image_data.data.resize(num_image_bytes);

      image_data.dimensions.width = resize_result.width;
      image_data.dimensions.height = resize_result.height;
      image_data.scale_factor = resize_result.scale;
      image_data.image_index = file_data.file_index;
      image_data.image_name = file_data.file_name;
      image_data.colorspace = m_output_image_colorspace;
    } // end processing input file buffer

    file_data_reader->done_reading_from_buffer();
    image_data_writer->done_writing_to_buffer();
  } // end main loop
}
