#include <streaming_sift_file_writer/streaming_sift_file_writer.h>
#include <v3d_support/sift_parser.h>

//#define ENABLE_SCALED_KEYPOINTS

StreamingSiftFileWriter::StreamingSiftFileWriter(
  const int num_threads,
  const v3d_support::PCDB * pcdb,
  core::SharedBatchBuffer<buffer_data::OutputSiftFeaturesData> * output_sift_data_buffer)
: core::StreamingModuleInterface(num_threads),
  m_output_sift_data_buffer(output_sift_data_buffer),
  m_output_sift_data_readers(num_threads),
  m_threads(NULL),
  m_num_threads(num_threads),
  m_pcdb(pcdb)
{
  // Create the threads.
  m_threads = new boost::thread[num_threads];

  for (int i = 0; i < num_threads; ++i)
  {
    m_output_sift_data_readers[i] =
      output_sift_data_buffer->get_new_lockstep_reader();

    m_threads[i] = boost::thread(&StreamingSiftFileWriter::thread_run, this, i);
  }
}

StreamingSiftFileWriter::~StreamingSiftFileWriter()
{
  if (m_threads != NULL)
  {
    delete [] m_threads;
    m_threads = NULL;
  }
}

void StreamingSiftFileWriter::generate_summary_speed_report(std::string & report)
{
  const core::SharedSpeedStats * output_sfit_readers_speed_stats =
    m_output_sift_data_buffer->lockstep_readers_speed_stats();

  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);
  stream << "StreamingSiftFileWriter" << std::endl;
  stream << "  Input:            ";
  stream.width(7);
  stream << output_sfit_readers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << output_sfit_readers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << output_sfit_readers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;

  report = stream.str();
}

void StreamingSiftFileWriter::generate_detailed_speed_report(std::string & report)
{
  generate_summary_speed_report(report);
}

void StreamingSiftFileWriter::wait_until_finished()
{
  for (int i = 0; i < m_num_threads; ++i)
  {
    m_threads[i].join();
  }
}

void StreamingSiftFileWriter::thread_run(const int thread_num)
{
  boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::OutputSiftFeaturesData> > output_sift_data_reader =
    m_output_sift_data_readers[thread_num];

#ifdef ENABLE_SCALED_KEYPOINTS
  std::vector<SiftKeypoint> scaled_keypoints;
#endif

  worker_thread_wait_for_start();

  for (;;)
  {
    output_sift_data_reader->wait_for_next_read_buffer();
    if (output_sift_data_reader->no_further_read_buffers_available())
    {
      break;
    }
    output_sift_data_reader->starting_to_read_from_buffer();

    for (;;)
    {
      const int output_sift_data_index =
        output_sift_data_reader->read_buffer_counter()->get_value_and_increment();

      // Exit the loop if all of the buffer entries have already been assigned.
      if (output_sift_data_index >= static_cast<int>(output_sift_data_reader->read_buffer().size()))
      {
        break;
      }

      // Get the output sift data entry that this thread should handle.
      const buffer_data::OutputSiftFeaturesData & output_sift_data =
        output_sift_data_reader->read_buffer()[output_sift_data_index];

      const std::string output_path = m_pcdb->get_indexed_file(
        v3d_support::PCDB::sift,
        output_sift_data.image_name);

#ifdef ENABLE_SCALED_KEYPOINTS
      scaled_keypoints = output_sift_data.keypoints;
      const float inv_scale = 1.0f / output_sift_data.image_scale_factor;
      for (size_t i = 0; i < scaled_keypoints.size(); ++i)
      {
        scaled_keypoints[i].x *= inv_scale;
        scaled_keypoints[i].y *= inv_scale;
      }
#endif

      v3d_support::sift_parser::write_to_file(
        output_path,
#ifdef ENABLE_SCALED_KEYPOINTS
        scaled_keypoints,
#else
        output_sift_data.keypoints,
#endif
        output_sift_data.descriptors_uchar);
    } // end processing current batch

    output_sift_data_reader->done_reading_from_buffer();
  } // end main loop
}
