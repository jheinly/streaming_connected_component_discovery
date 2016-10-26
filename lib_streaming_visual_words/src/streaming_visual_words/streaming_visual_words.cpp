#include <streaming_visual_words/streaming_visual_words.h>
#include <features/sift_support.h>

StreamingVisualWords::StreamingVisualWords(
  const int num_threads,
  const VocabTree * vocab_tree,
  core::SharedBatchBuffer<buffer_data::SiftFeaturesData> * input_sift_features_data_buffer,
  core::SharedBatchBuffer<buffer_data::SiftVisualWordsData> * output_visual_words_data_buffer)
: core::StreamingModuleInterface(num_threads),
  m_input_sift_features_data_buffer(input_sift_features_data_buffer),
  m_sift_features_data_readers(num_threads),
  m_output_visual_words_data_buffer(output_visual_words_data_buffer),
  m_visual_words_data_writers(num_threads),
  m_threads(NULL),
  m_num_threads(num_threads),
  m_vocab_tree(vocab_tree)
{
  // Create the threads.
  m_threads = new boost::thread[num_threads];

  for (int i = 0; i < num_threads; ++i)
  {
    m_sift_features_data_readers[i] =
      input_sift_features_data_buffer->get_new_lockstep_reader();
    m_visual_words_data_writers[i] =
      output_visual_words_data_buffer->get_new_lockstep_writer();

    m_threads[i] = boost::thread(&StreamingVisualWords::thread_run, this, i);
  }
}

StreamingVisualWords::~StreamingVisualWords()
{
  if (m_threads != NULL)
  {
    delete [] m_threads;
    m_threads = NULL;
  }
}

void StreamingVisualWords::generate_summary_speed_report(std::string & report)
{
  const core::SharedSpeedStats * sift_readers_speed_stats =
    m_input_sift_features_data_buffer->lockstep_readers_speed_stats();
  const core::SharedSpeedStats * words_writers_speed_stats =
    m_output_visual_words_data_buffer->lockstep_writers_speed_stats();

  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);
  stream << "StreamingVisualWords" << std::endl;
  stream << "  Input:            ";
  stream.width(7);
  stream << sift_readers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << sift_readers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << sift_readers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;
  stream << "  Output:           ";
  stream.width(7);
  stream << words_writers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << words_writers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << words_writers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;

  report = stream.str();
}

void StreamingVisualWords::generate_detailed_speed_report(std::string & report)
{
  generate_summary_speed_report(report);
}

void StreamingVisualWords::wait_until_finished()
{
  for (int i = 0; i < m_num_threads; ++i)
  {
    m_threads[i].join();
  }
}

void StreamingVisualWords::thread_run(const int thread_num)
{
  boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::SiftFeaturesData> > sift_features_data_reader =
    m_sift_features_data_readers[thread_num];
  boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::SiftVisualWordsData> > visual_words_data_writer =
    m_visual_words_data_writers[thread_num];

  worker_thread_wait_for_start();

  for (;;)
  {
    sift_features_data_reader->wait_for_next_read_buffer();
    visual_words_data_writer->wait_for_next_write_buffer();
    if (sift_features_data_reader->no_further_read_buffers_available())
    {
      visual_words_data_writer->done_with_all_further_writes_and_wait_for_exit();
      break;
    }
    sift_features_data_reader->starting_to_read_from_buffer();
    visual_words_data_writer->starting_to_write_to_buffer();

    for (;;)
    {
      // Get the next index to process from the sift features buffer.
      const int sift_features_index =
        sift_features_data_reader->read_buffer_counter()->get_value_and_increment();

      // Exit the loop if all of the sift features buffer entries have already been assigned.
      if (sift_features_index >= static_cast<int>(sift_features_data_reader->read_buffer().size()))
      {
        break;
      }

      // Get the sift features data entry that this thread should handle.
      const buffer_data::SiftFeaturesData & sift_features_data =
        sift_features_data_reader->read_buffer()[sift_features_index];

      // Get the visual words buffer entry where the visual words should be stored.
      const int visual_words_index =
        visual_words_data_writer->write_buffer_counter()->get_value_and_increment();
      buffer_data::SiftVisualWordsData & visual_words_data =
        visual_words_data_writer->write_buffer()[visual_words_index];

      features::sift_support::convert_descriptors_from_float_to_uchar(
        sift_features_data.descriptors,
        visual_words_data.descriptors_uchar);

      m_vocab_tree->v2_convert_features_to_visual_words_thread_safe(
        sift_features_data.num_features,
        visual_words_data.descriptors_uchar[0].uchars,
        visual_words_data.visual_words);

      visual_words_data.image_index = sift_features_data.image_index;
      visual_words_data.image_name = sift_features_data.image_name;
      visual_words_data.dimensions = sift_features_data.dimensions;
      visual_words_data.num_features = sift_features_data.num_features;
      visual_words_data.keypoints = sift_features_data.keypoints;
      visual_words_data.descriptors_float = sift_features_data.descriptors;
      visual_words_data.image_scale_factor = sift_features_data.image_scale_factor;
    } // end processing input buffer

    sift_features_data_reader->done_reading_from_buffer();
    visual_words_data_writer->done_writing_to_buffer();
  } // end main loop
}
